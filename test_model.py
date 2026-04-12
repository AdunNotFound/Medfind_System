import pandas as pd
import numpy as np
import pickle
import time
from ml_functions import hybrid_ensemble_search_v2, get_edit_distance_ranking, get_ml_ranking

# Load model and data
with open('Models/drug_ranker_model.pkl', 'rb') as f:
    model = pickle.load(f)
lookup_df = pd.read_pickle('Models/lookup_df.pkl')

# Test cases: (misspelled, correct)
test_cases = [
    # 1-character errors
    ("amoxicilln", "amoxicillin"),
    ("ibuprofn", "ibuprofen"),
    ("asprin", "aspirin"),
    ("metformn", "metformin"),
    ("cetirizne", "cetirizine"),
    ("omeprazol", "omeprazole"),
    ("losartn", "losartan"),
    ("sertralin", "sertraline"),
    ("albuterl", "albuterol"),
    ("prednisne", "prednisone"),
    # 2-character errors
    ("amoxisilin", "amoxicillin"),
    ("parastamol", "paracetamol"),
    ("atorvstatin", "atorvastatin"),
    ("lisinoprl", "lisinopril"),
    ("gabapntin", "gabapentin"),
    ("azithromcin", "azithromycin"),
    ("ciproflxacin", "ciprofloxacin"),
    ("hydrocortisne", "hydrocortisone"),
    ("amlodipne", "amlodipine"),
    ("tramadl", "tramadol"),
    # 3+ character / phonetic errors
    ("fenasitamol", "paracetamol"),
    ("amoksisilin", "amoxicillin"),
    ("ibeuprofin", "ibuprofen"),
    ("metoprolo", "metoprolol"),
    ("omeprazzole", "omeprazole"),
]

print("=" * 70)
print("MEDFIND MODEL TESTING")
print("=" * 70)

# ── Hit@K Testing ──
for k in [1, 3, 5]:
    hits = 0
    for query, expected in test_cases:
        result = hybrid_ensemble_search_v2(query, lookup_df, model, top_k=k)
        top_names = [r.lower() for r in result['canonical'].tolist()]
        if expected.lower() in top_names:
            hits += 1
    accuracy = hits / len(test_cases) * 100
    print(f"\nHit@{k}: {accuracy:.1f}% ({hits}/{len(test_cases)})")

# ── Mean Reciprocal Rank (MRR) ──
reciprocal_ranks = []
for query, expected in test_cases:
    result = hybrid_ensemble_search_v2(query, lookup_df, model, top_k=5)
    top_names = [r.lower() for r in result['canonical'].tolist()]
    if expected.lower() in top_names:
        rank = top_names.index(expected.lower()) + 1
        reciprocal_ranks.append(1.0 / rank)
    else:
        reciprocal_ranks.append(0.0)
mrr = np.mean(reciprocal_ranks)
print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")

# ── NDCG@5 ──
ndcg_scores = []
for query, expected in test_cases:
    result = hybrid_ensemble_search_v2(query, lookup_df, model, top_k=5)
    top_names = [r.lower() for r in result['canonical'].tolist()]
    # Binary relevance: 1 if correct, 0 if not
    relevance = [1.0 if name == expected.lower() else 0.0 for name in top_names]
    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))
    # Ideal DCG (best case: correct result at position 1)
    ideal = 1.0 / np.log2(2)  # 1/log2(2) = 1.0
    ndcg = dcg / ideal if ideal > 0 else 0.0
    ndcg_scores.append(ndcg)
ndcg_avg = np.mean(ndcg_scores)
print(f"NDCG@5: {ndcg_avg:.4f}")

# ── Breakdown by error severity ──
print("\n" + "=" * 70)
print("ACCURACY BY ERROR SEVERITY")
print("=" * 70)

categories = {
    "1-character errors": test_cases[:10],
    "2-character errors": test_cases[10:20],
    "3+ / phonetic errors": test_cases[20:25],
}

for category, cases in categories.items():
    hits = 0
    for query, expected in cases:
        result = hybrid_ensemble_search_v2(query, lookup_df, model, top_k=5)
        top_names = [r.lower() for r in result['canonical'].tolist()]
        if expected.lower() in top_names:
            hits += 1
    print(f"  {category}: {hits}/{len(cases)} ({hits/len(cases)*100:.1f}%)")

# ── Per-query detail ──
print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)

for query, expected in test_cases:
    result = hybrid_ensemble_search_v2(query, lookup_df, model, top_k=5)
    top_names = [r.lower() for r in result['canonical'].tolist()]
    found = expected.lower() in top_names
    if found:
        rank = top_names.index(expected.lower()) + 1
        conf = result.iloc[rank-1]['ensemble_score'] * 100
        print(f"  ✓ '{query}' → '{expected}' at rank {rank} ({conf:.1f}%)")
    else:
        print(f"  ✗ '{query}' → '{expected}' NOT FOUND. Got: {top_names}")