"""
ML Functions for MedFind Drug Lookup
Contains all machine learning and search functions

UPDATED: Phonetic-as-Feature Implementation
- 12 features (was 9)
- No phonetic filtering
- Searches all 71,885 drugs

FIXED: Score direction - HIGHER score = BETTER match (was inverted!)
"""

import pandas as pd
import numpy as np
from rapidfuzz.distance import Levenshtein, JaroWinkler, Jaro
import jellyfish
import re
import unicodedata

# ────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────

def normalize(text):
    """Normalize text for matching"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_features(query_norm, term_norm, source):
    """
    Extract 12 features for ML model
    
    CRITICAL: Returns features in EXACT order expected by model
    """
    if not term_norm or not query_norm:
        return {
            'lev_dist': 999,
            'lev_norm': 1.0,
            'jw_sim': 0.0,
            'jaro_sim': 0.0,
            'len_ratio': 1.0,
            'is_generic': 0.0,
            'prefix_match': 0.0,
            'soundex_match': 0.0,
            'metaphone_match': 0.0,
            'nysiis_match': 0.0,
            'match_rating': 0.0,
            'phonetic_score': 0.0
        }
    
    # Basic string similarity
    lev_d = Levenshtein.distance(query_norm, term_norm)
    max_len = max(len(query_norm), len(term_norm))
    
    # Phonetic features
    soundex_match = 1.0 if jellyfish.soundex(query_norm) == jellyfish.soundex(term_norm) else 0.0
    metaphone_match = 1.0 if jellyfish.metaphone(query_norm) == jellyfish.metaphone(term_norm) else 0.0
    nysiis_match = 1.0 if jellyfish.nysiis(query_norm) == jellyfish.nysiis(term_norm) else 0.0
    
    try:
        match_rating_result = jellyfish.match_rating_comparison(query_norm, term_norm)
        match_rating = 1.0 if match_rating_result else 0.0
    except:
        match_rating = 0.0
    
    # Combined phonetic score
    phonetic_score = (soundex_match + metaphone_match + nysiis_match + match_rating) / 4.0
    
    # Return in EXACT order (important for model!)
    return {
        'lev_dist': lev_d,
        'lev_norm': lev_d / max_len if max_len > 0 else 0.0,
        'jw_sim': JaroWinkler.similarity(query_norm, term_norm),
        'jaro_sim': Jaro.similarity(query_norm, term_norm),
        'len_ratio': len(term_norm) / len(query_norm) if len(query_norm) > 0 else 1.0,
        'is_generic': 1.0 if source == 'generic' else 0.0,
        'prefix_match': 1.0 if term_norm.startswith(query_norm[:3]) else 0.0,
        'soundex_match': soundex_match,
        'metaphone_match': metaphone_match,
        'nysiis_match': nysiis_match,
        'match_rating': match_rating,
        'phonetic_score': phonetic_score
    }


# ────────────────────────────────────────────────────────────────
#  RANKING FUNCTIONS
# ────────────────────────────────────────────────────────────────

def get_edit_distance_ranking(query, lookup_df, top_k=5):
    """
    Pure edit distance ranking
    NO phonetic filtering - searches all drugs
    DEDUPLICATION: Only returns one result per unique drug name
    """
    q_norm = normalize(query)
    
    if not q_norm:
        return pd.DataFrame(columns=['canonical', 'term', 'edit_dist', 'edit_confidence'])
    
    # Search ALL drugs (no filtering!)
    candidates = lookup_df.copy()
    
    # Calculate edit distances for all
    candidates['edit_dist'] = candidates['term'].apply(
        lambda t: Levenshtein.distance(q_norm, t) if t else 999
    )
    
    # Sort by edit distance (smallest = best)
    candidates = candidates.sort_values('edit_dist', ascending=True)
    
    # ════════════════════════════════════════════════════════════
    # DEDUPLICATION: Only keep first (best) result per drug name
    # ════════════════════════════════════════════════════════════
    seen_drugs = set()
    unique_results = []
    
    for _, row in candidates.iterrows():
        drug_name = row['canonical'].lower()
        if drug_name not in seen_drugs:
            seen_drugs.add(drug_name)
            unique_results.append(row)
            if len(unique_results) >= top_k:
                break
    
    top_candidates = pd.DataFrame(unique_results)
    
    if top_candidates.empty:
        return pd.DataFrame(columns=['canonical', 'term', 'edit_dist', 'edit_confidence'])
    
    # Calculate confidence: 1 - (distance / max_length)
    for idx in top_candidates.index:
        row = top_candidates.loc[idx]
        term_len = len(row['term'])
        query_len = len(q_norm)
        max_len = max(term_len, query_len)
        
        if max_len > 0:
            top_candidates.at[idx, 'edit_confidence'] = 1.0 - (row['edit_dist'] / max_len)
        else:
            top_candidates.at[idx, 'edit_confidence'] = 0.0
    
    return top_candidates[['canonical', 'term', 'edit_dist', 'edit_confidence']].reset_index(drop=True)


def get_ml_ranking(query, lookup_df, model, top_k=5):
    """
    ML-based ranking using LightGBM model
    NO phonetic filtering - phonetic matching is now a feature
    
    CRITICAL: Features must be in EXACT order model was trained with
    
    FIX: HIGHER score = BETTER match (LightGBM LambdaRank convention)
    FIX: Deduplicate results - only show one result per unique drug name
    """
    q_norm = normalize(query)
    
    if not q_norm:
        return pd.DataFrame(columns=['canonical', 'term', 'ml_score', 'ml_confidence'])
    
    # Search ALL drugs (no filtering!)
    candidates = lookup_df.copy()
    
    # Extract features for ALL candidates
    feature_list = []
    for _, row in candidates.iterrows():
        feats = extract_features(q_norm, row['term'], row['source'])
        feature_list.append(feats)
    
    if not feature_list:
        return pd.DataFrame(columns=['canonical', 'term', 'ml_score', 'ml_confidence'])
    
    # ════════════════════════════════════════════════════════════
    # CRITICAL: Feature order MUST match training!
    # ════════════════════════════════════════════════════════════
    FEATURE_ORDER = [
        'lev_dist',
        'lev_norm',
        'jw_sim',
        'jaro_sim',
        'len_ratio',
        'is_generic',
        'prefix_match',
        'soundex_match',
        'metaphone_match',
        'nysiis_match',
        'match_rating',
        'phonetic_score'
    ]
    
    # Build feature matrix in EXACT order
    X = []
    for feat_dict in feature_list:
        row = [feat_dict[fn] for fn in FEATURE_ORDER]
        X.append(row)
    
    X = np.array(X, dtype=np.float32)
    
    # Predict scores with model
    raw_scores = model.predict(X)
    
    # Add scores to candidates
    candidates = candidates.copy()
    candidates['ml_score'] = raw_scores
    
    # ════════════════════════════════════════════════════════════
    # FIX: HIGHER score = BETTER match
    # ════════════════════════════════════════════════════════════
    
    # Sort ALL candidates by score (descending - best first)
    candidates = candidates.sort_values('ml_score', ascending=False)
    
    # ════════════════════════════════════════════════════════════
    # DEDUPLICATION: Only keep first (best) result per drug name
    # This ensures we show top_k UNIQUE drugs, not top_k rows
    # ════════════════════════════════════════════════════════════
    seen_drugs = set()
    unique_results = []
    
    for _, row in candidates.iterrows():
        drug_name = row['canonical'].lower()  # Case-insensitive dedup
        if drug_name not in seen_drugs:
            seen_drugs.add(drug_name)
            unique_results.append(row)
            if len(unique_results) >= top_k:
                break
    
    top_candidates = pd.DataFrame(unique_results)
    
    if top_candidates.empty:
        return pd.DataFrame(columns=['canonical', 'term', 'ml_score', 'ml_confidence'])
    
    # Convert raw scores to confidence (0-100%)
    min_score = top_candidates['ml_score'].min()
    max_score = top_candidates['ml_score'].max()
    
    if max_score > min_score:
        # HIGHER score = HIGHER confidence (direct mapping)
        top_candidates['ml_confidence'] = (
            (top_candidates['ml_score'] - min_score) / (max_score - min_score)
        )
    else:
        # All same score (perfect tie)
        top_candidates['ml_confidence'] = 1.0
    
    return top_candidates[['canonical', 'term', 'ml_score', 'ml_confidence']].reset_index(drop=True)


def hybrid_ensemble_search_v2(query, lookup_df, model, strategy='confidence_weighted', top_k=5):
    """
    Hybrid ensemble: combines edit distance + ML ranking
    
    Strategies:
    - confidence_weighted: Average both confidences (default)
    - max_confidence: Take maximum confidence
    - conservative: Take minimum confidence
    
    NOTE: Both edit_distance and ML rankings are already deduplicated,
    so this function just merges them and sorts by ensemble score.
    """
    # Get results from both methods (already deduplicated)
    edit_results = get_edit_distance_ranking(query, lookup_df, top_k=top_k)
    ml_results = get_ml_ranking(query, lookup_df, model, top_k=top_k)
    
    # Handle empty results
    if edit_results.empty and ml_results.empty:
        return pd.DataFrame(columns=['canonical', 'ensemble_score', 'agreement', 
                                     'edit_confidence', 'ml_confidence'])
    
    # Merge results on canonical name (outer join to keep all)
    merged = pd.merge(
        edit_results[['canonical', 'edit_dist', 'edit_confidence']],
        ml_results[['canonical', 'ml_score', 'ml_confidence']],
        on='canonical',
        how='outer'
    )
    
    # Fill missing values with 0 (drug only ranked by one method)
    merged['edit_confidence'] = merged['edit_confidence'].fillna(0.0)
    merged['ml_confidence'] = merged['ml_confidence'].fillna(0.0)
    
    # Agreement flag: both methods ranked this drug in their top K
    merged['agreement'] = (
        (merged['edit_confidence'] > 0) & 
        (merged['ml_confidence'] > 0)
    ).astype(int)
    
    # Calculate ensemble score based on strategy
    if strategy == 'confidence_weighted':
        # Average of both confidences
        merged['ensemble_score'] = (
            merged['edit_confidence'] + merged['ml_confidence']
        ) / 2.0
        
        # Small boost if both methods agree (5% bonus)
        agreement_boost = 0.05
        merged.loc[merged['agreement'] == 1, 'ensemble_score'] += agreement_boost
        
    elif strategy == 'max_confidence':
        # Take maximum confidence (optimistic)
        merged['ensemble_score'] = merged[['edit_confidence', 'ml_confidence']].max(axis=1)
        
    elif strategy == 'conservative':
        # Take minimum confidence (pessimistic)
        merged['ensemble_score'] = merged[['edit_confidence', 'ml_confidence']].min(axis=1)
    
    else:
        # Default to confidence_weighted
        merged['ensemble_score'] = (
            merged['edit_confidence'] + merged['ml_confidence']
        ) / 2.0
    
    # Safety cap: max 100% confidence
    merged['ensemble_score'] = merged['ensemble_score'].clip(upper=1.0)
    
    # Sort by ensemble score (descending - highest first)
    merged = merged.sort_values('ensemble_score', ascending=False)
    
    # ════════════════════════════════════════════════════════════
    # DEDUPLICATION: Ensure unique drug names in final output
    # (Should already be unique from sub-functions, but double-check)
    # ════════════════════════════════════════════════════════════
    merged = merged.drop_duplicates(subset=['canonical'], keep='first')
    
    # Return top K results
    result = merged.head(top_k)[[
        'canonical',
        'ensemble_score',
        'agreement',
        'edit_confidence',
        'ml_confidence'
    ]].reset_index(drop=True)
    
    return result
