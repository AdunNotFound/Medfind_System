"""
Merge FDA Brand Names into Lookup Database - FIXED VERSION 2
=============================================================
This script adds brand names from the FDA NDC database to your lookup_df,
so users can search for "Advil" and find "Ibuprofen".

FIXES:
1. Normalizes brand names to lowercase
2. Strips salt forms (HYDROCHLORIDE, CALCIUM, etc.) to match DrugBank generics

Run this AFTER build_drug_relationships.py

Usage:
    python merge_brands_into_lookup.py
"""

import pandas as pd
import pickle
import os
import re
import unicodedata
import jellyfish

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

LOOKUP_PATH = 'Models/lookup_df.pkl'
DRUG_REL_PATH = 'Models/drug_relationships.pkl'
OUTPUT_PATH = 'Models/lookup_df.pkl'
BACKUP_PATH = 'Models/lookup_df_backup.pkl'

# Common salt forms to strip from generic names
SALT_FORMS = [
    'HYDROCHLORIDE', 'HCL', 'DIHYDROCHLORIDE',
    'CALCIUM', 'SODIUM', 'POTASSIUM', 'MAGNESIUM',
    'SULFATE', 'SULPHATE', 'PHOSPHATE', 'NITRATE',
    'ACETATE', 'CITRATE', 'MALEATE', 'FUMARATE', 'SUCCINATE',
    'TARTRATE', 'BESYLATE', 'MESYLATE', 'TOSYLATE',
    'BROMIDE', 'CHLORIDE', 'IODIDE',
    'TRIHYDRATE', 'DIHYDRATE', 'MONOHYDRATE', 'HYDRATE',
    'ANHYDROUS', 'HEMIHYDRATE',
    'DISODIUM', 'TRISODIUM', 'DIPOTASSIUM',
    'GLUCONATE', 'LACTATE', 'MALATE', 'OXALATE',
    'HYDROBROMINE', 'HYDROBROMIDE',
    'PAMOATE', 'STEARATE', 'PROPIONATE',
    'AND DIPHENHYDRAMINE CITRATE',  # For combination products
    'AND DIPHENHYDRAMINE',
]

# ════════════════════════════════════════════════════════════════
# FUNCTIONS
# ════════════════════════════════════════════════════════════════

def normalize(text):
    """Normalize text for matching - converts to lowercase"""
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def strip_salt_form(generic_name):
    """
    Strip salt forms from generic name to get base compound.
    E.g., "ATORVASTATIN CALCIUM" -> "ATORVASTATIN"
          "SERTRALINE HYDROCHLORIDE" -> "SERTRALINE"
    """
    name = generic_name.upper().strip()
    
    # Sort by length (longest first) to match multi-word salts first
    sorted_salts = sorted(SALT_FORMS, key=len, reverse=True)
    
    for salt in sorted_salts:
        if name.endswith(' ' + salt):
            name = name[:-len(salt)-1].strip()
        elif name.endswith(salt):
            name = name[:-len(salt)].strip()
    
    return name

def merge_brands():
    """Merge FDA brand names into lookup database"""
    
    print("="*60)
    print("MERGING FDA BRAND NAMES INTO LOOKUP DATABASE")
    print("="*60)
    
    # Load existing lookup_df
    print(f"\n1. Loading lookup database from {LOOKUP_PATH}...")
    lookup_df = pd.read_pickle(LOOKUP_PATH)
    original_count = len(lookup_df)
    print(f"   Current entries: {original_count:,}")
    
    # Create backup
    print(f"   Creating backup at {BACKUP_PATH}...")
    lookup_df.to_pickle(BACKUP_PATH)
    
    # Load drug relationships (has brand -> generic mappings)
    print(f"\n2. Loading drug relationships from {DRUG_REL_PATH}...")
    with open(DRUG_REL_PATH, 'rb') as f:
        rel_data = pickle.load(f)
    
    brand_to_generic = rel_data['brand_to_generic']
    print(f"   Brand names available: {len(brand_to_generic):,}")
    
    # Get existing terms (normalized to lowercase)
    existing_terms = set(lookup_df['term'].str.lower().tolist())
    print(f"   Existing unique terms: {len(existing_terms):,}")
    
    # Build a mapping from normalized generic names to canonical names
    # Include both exact matches and base names (without salt forms)
    generic_to_canonical = {}
    for _, row in lookup_df.iterrows():
        term_lower = row['term'].lower() if row['term'] else ''
        canonical = row['canonical']
        if term_lower and term_lower not in generic_to_canonical:
            generic_to_canonical[term_lower] = canonical
    
    print(f"   Generic name mappings: {len(generic_to_canonical):,}")
    
    # Add brand names that map to drugs we have
    new_rows = []
    matched = 0
    matched_via_base = 0
    skipped_exists = 0
    skipped_no_match = 0
    
    print(f"\n3. Processing brand names...")
    
    for brand, generic in brand_to_generic.items():
        # Normalize brand to lowercase
        brand_norm = normalize(brand)
        
        # Skip if already exists
        if brand_norm in existing_terms:
            skipped_exists += 1
            continue
        
        # Skip empty or too short
        if not brand_norm or len(brand_norm) < 2:
            continue
        
        # Try to find the canonical name for this generic
        canonical = None
        
        # 1. First try exact match on normalized generic
        generic_norm = normalize(generic)
        if generic_norm in generic_to_canonical:
            canonical = generic_to_canonical[generic_norm]
        
        # 2. If not found, try stripping salt form
        if not canonical:
            base_name = strip_salt_form(generic)
            base_norm = normalize(base_name)
            if base_norm in generic_to_canonical:
                canonical = generic_to_canonical[base_norm]
                matched_via_base += 1
        
        # 3. Try first word only (for combination products)
        if not canonical:
            first_word = generic_norm.split()[0] if generic_norm else ''
            if first_word and len(first_word) > 3 and first_word in generic_to_canonical:
                canonical = generic_to_canonical[first_word]
                matched_via_base += 1
        
        if not canonical:
            skipped_no_match += 1
            continue
        
        # Add this brand name
        new_row = {
            'canonical': canonical,
            'term': brand_norm,
            'source': 'brand',
        }
        
        # Add optional columns if they exist in original
        if 'drugbank_id' in lookup_df.columns:
            new_row['drugbank_id'] = ''
        
        # Add phonetic codes
        try:
            new_row['soundex'] = jellyfish.soundex(brand_norm)
            new_row['metaphone'] = jellyfish.metaphone(brand_norm)
            new_row['nysiis'] = jellyfish.nysiis(brand_norm)
        except:
            new_row['soundex'] = ''
            new_row['metaphone'] = ''
            new_row['nysiis'] = ''
        
        new_rows.append(new_row)
        existing_terms.add(brand_norm)
        matched += 1
        
        # Progress indicator
        if matched % 5000 == 0:
            print(f"   Processed {matched:,} brand names...")
    
    print(f"\n4. Results:")
    print(f"   Brand names matched: {matched:,}")
    print(f"     - Matched via salt stripping: {matched_via_base:,}")
    print(f"   Skipped (already exists): {skipped_exists:,}")
    print(f"   Skipped (no generic match): {skipped_no_match:,}")
    
    if new_rows:
        # Create DataFrame
        new_df = pd.DataFrame(new_rows)
        
        # Ensure columns match
        for col in lookup_df.columns:
            if col not in new_df.columns:
                new_df[col] = ''
        
        # Reorder columns to match original
        new_df = new_df[lookup_df.columns]
        
        # Append to original
        lookup_df_merged = pd.concat([lookup_df, new_df], ignore_index=True)
        
        # Remove duplicates by term (keep first occurrence)
        before_dedup = len(lookup_df_merged)
        lookup_df_merged = lookup_df_merged.drop_duplicates(subset=['term'], keep='first')
        after_dedup = len(lookup_df_merged)
        
        if before_dedup > after_dedup:
            print(f"   Removed {before_dedup - after_dedup} duplicates")
        
        # Save
        print(f"\n5. Saving merged database to {OUTPUT_PATH}...")
        lookup_df_merged.to_pickle(OUTPUT_PATH)
        
        final_count = len(lookup_df_merged)
        print(f"\n" + "="*60)
        print("[OK] MERGE COMPLETE!")
        print("="*60)
        print(f"   Original entries: {original_count:,}")
        print(f"   New brand entries: {matched:,}")
        print(f"   Final total: {final_count:,}")
        
        # Test some searches
        print(f"\n" + "="*60)
        print("TESTING BRAND NAME SEARCHES")
        print("="*60)
        
        test_brands = [
            'advil', 'tylenol', 'motrin', 'lipitor', 'zoloft', 
            'prozac', 'viagra', 'nexium', 'aspirin', 'benadryl',
            'xanax', 'ambien', 'prilosec', 'synthroid', 'crestor'
        ]
        found = 0
        for brand in test_brands:
            matches = lookup_df_merged[lookup_df_merged['term'].str.lower() == brand.lower()]
            if not matches.empty:
                canonical = matches.iloc[0]['canonical']
                print(f"   [OK] '{brand}' -> {canonical}")
                found += 1
            else:
                # Check if it exists in drug_relationships
                if brand.upper() in brand_to_generic:
                    generic = brand_to_generic[brand.upper()]
                    base = strip_salt_form(generic)
                    print(f"   [--] '{brand}' not found (FDA: {generic} -> base: {base})")
                else:
                    print(f"   [--] '{brand}' not in FDA database")
        
        print(f"\n   Found {found}/{len(test_brands)} test brands")
        
        return lookup_df_merged
        
    else:
        print("\n[WARN] No new brand names to add!")
        return lookup_df

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not os.path.exists(LOOKUP_PATH):
        print(f"[ERROR] {LOOKUP_PATH} not found!")
        print("   Run the notebook first to create lookup_df.pkl")
        exit(1)
    
    if not os.path.exists(DRUG_REL_PATH):
        print(f"[ERROR] {DRUG_REL_PATH} not found!")
        print("   Run build_drug_relationships.py first")
        exit(1)
    
    merge_brands()