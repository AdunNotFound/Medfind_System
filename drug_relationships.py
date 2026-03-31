"""
MedFind Drug Relationship System
================================
Links brand names ↔ generic names ↔ related products

Data Sources:
- FDA NDC (National Drug Code) Directory: Brand names, generics, manufacturers
- FDA Orange Book: Therapeutic equivalents
- DrugBank: Scientific synonyms, identifiers

Usage:
    from drug_relationships import DrugRelationshipDB
    
    db = DrugRelationshipDB()
    db.load_fda_data('products.csv')  # From FDA NDC
    
    # When user clicks on a search result:
    info = db.get_drug_info('Advil')
    # Returns:
    # {
    #     'searched': 'Advil',
    #     'generic_name': 'Ibuprofen',
    #     'brand_names': ['Advil', 'Motrin', 'Nurofen', ...],
    #     'manufacturers': ['Pfizer', 'Johnson & Johnson', ...],
    #     'strengths': ['200mg', '400mg', '600mg'],
    #     'forms': ['Tablet', 'Capsule', 'Liquid']
    # }
"""

import pandas as pd
import re
import os
from collections import defaultdict


class DrugRelationshipDB:
    """
    Database linking brand names, generics, and related products
    """
    
    def __init__(self):
        # Core mappings
        self.brand_to_generic = {}          # 'Advil' -> 'IBUPROFEN'
        self.generic_to_brands = defaultdict(set)  # 'IBUPROFEN' -> {'Advil', 'Motrin', ...}
        self.generic_to_manufacturers = defaultdict(set)
        self.generic_to_strengths = defaultdict(set)
        self.generic_to_forms = defaultdict(set)
        
        # For search matching
        self.all_names = {}  # normalized_name -> {'type': 'brand'/'generic', 'canonical': '...'}
        
    def normalize(self, text):
        """Normalize drug name for matching"""
        if pd.isna(text) or not text:
            return ""
        text = str(text).upper().strip()
        # Remove common suffixes
        text = re.sub(r'\s+(TABLET|CAPSULE|INJECTION|SOLUTION|CREAM|GEL|OINTMENT|SYRUP|SUSPENSION)S?$', '', text)
        # Remove strength info
        text = re.sub(r'\s+\d+(\.\d+)?\s*(MG|MCG|ML|%|IU).*$', '', text)
        return text.strip()
    
    def load_fda_ndc_data(self, filepath):
        """
        Load FDA NDC Directory product.csv
        Download from: https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory
        
        Key columns:
        - PROPRIETARYNAME: Brand name (e.g., "Advil")
        - NONPROPRIETARYNAME: Generic name (e.g., "Ibuprofen")
        - LABELERNAME: Manufacturer
        - DOSAGEFORMNAME: Form (Tablet, Capsule, etc.)
        - ACTIVE_NUMERATOR_STRENGTH: Strength
        - ACTIVE_INGRED_UNIT: Unit (mg, ml, etc.)
        """
        print(f"Loading FDA NDC data from {filepath}...")
        
        # FDA file uses tab separator and latin-1 encoding (not UTF-8)
        try:
            df = pd.read_csv(filepath, sep='\t', dtype=str, low_memory=False, encoding='latin-1')
        except:
            # Fallback to cp1252 (Windows encoding)
            df = pd.read_csv(filepath, sep='\t', dtype=str, low_memory=False, encoding='cp1252')
        
        print(f"  Loaded {len(df)} product entries")
        
        # Process each row
        for _, row in df.iterrows():
            brand = self.normalize(row.get('PROPRIETARYNAME', ''))
            generic = self.normalize(row.get('NONPROPRIETARYNAME', ''))
            manufacturer = str(row.get('LABELERNAME', '')).strip()
            form = str(row.get('DOSAGEFORMNAME', '')).strip()
            strength = str(row.get('ACTIVE_NUMERATOR_STRENGTH', '')).strip()
            unit = str(row.get('ACTIVE_INGRED_UNIT', '')).strip()
            
            if not generic:
                continue
                
            # Build relationships
            if brand and brand != generic:
                self.brand_to_generic[brand] = generic
                self.generic_to_brands[generic].add(brand)
                self.all_names[brand] = {'type': 'brand', 'generic': generic}
            
            self.all_names[generic] = {'type': 'generic', 'generic': generic}
            
            if manufacturer:
                self.generic_to_manufacturers[generic].add(manufacturer)
            if form:
                self.generic_to_forms[generic].add(form)
            if strength and unit:
                self.generic_to_strengths[generic].add(f"{strength} {unit}")
        
        print(f"  Unique generics: {len(self.generic_to_brands)}")
        print(f"  Unique brands: {len(self.brand_to_generic)}")
        print(f"  Total searchable names: {len(self.all_names)}")
        
    def load_drugbank_mappings(self, drugbank_df):
        """
        Add DrugBank synonyms to the relationship database
        """
        print("Adding DrugBank synonyms...")
        
        for _, row in drugbank_df.iterrows():
            generic = self.normalize(row.get('Common name', ''))
            if not generic:
                continue
                
            # Add generic name
            self.all_names[generic] = {'type': 'generic', 'generic': generic}
            
            # Add synonyms as potential brand names
            synonyms = row.get('Synonyms', '')
            if pd.notna(synonyms):
                for syn in synonyms.split('|'):
                    syn_norm = self.normalize(syn.strip())
                    if syn_norm and syn_norm != generic:
                        if syn_norm not in self.all_names:
                            self.all_names[syn_norm] = {'type': 'synonym', 'generic': generic}
                            self.generic_to_brands[generic].add(syn_norm)
        
        print(f"  Total searchable names: {len(self.all_names)}")
    
    def get_drug_info(self, query):
        """
        Get complete drug information for a search result
        
        Returns:
        {
            'searched': 'Advil',
            'type': 'brand',
            'generic_name': 'IBUPROFEN',
            'brand_names': ['Advil', 'Motrin', 'Nurofen', ...],
            'manufacturers': ['Pfizer', 'Johnson & Johnson', ...],
            'strengths': ['200 MG', '400 MG', '600 MG'],
            'forms': ['TABLET', 'CAPSULE', 'ORAL SUSPENSION']
        }
        """
        query_norm = self.normalize(query)
        
        # Find the drug in our database
        if query_norm not in self.all_names:
            # Try fuzzy match (first word)
            first_word = query_norm.split()[0] if query_norm else ''
            matches = [k for k in self.all_names.keys() if k.startswith(first_word)]
            if matches:
                query_norm = matches[0]
            else:
                return {
                    'searched': query,
                    'type': 'unknown',
                    'generic_name': query,
                    'brand_names': [],
                    'manufacturers': [],
                    'strengths': [],
                    'forms': [],
                    'message': 'Drug not found in relationship database'
                }
        
        drug_info = self.all_names[query_norm]
        generic = drug_info['generic']
        
        # Get all related brands (excluding the searched term itself for cleaner display)
        brands = sorted(self.generic_to_brands.get(generic, set()))
        
        return {
            'searched': query,
            'searched_normalized': query_norm,
            'type': drug_info['type'],
            'generic_name': generic,
            'brand_names': brands,
            'other_brands': [b for b in brands if b != query_norm],  # Exclude searched term
            'manufacturers': sorted(self.generic_to_manufacturers.get(generic, set())),
            'strengths': sorted(self.generic_to_strengths.get(generic, set())),
            'forms': sorted(self.generic_to_forms.get(generic, set()))
        }
    
    def save(self, filepath):
        """Save relationship database to pickle"""
        import pickle
        data = {
            'brand_to_generic': dict(self.brand_to_generic),
            'generic_to_brands': {k: list(v) for k, v in self.generic_to_brands.items()},
            'generic_to_manufacturers': {k: list(v) for k, v in self.generic_to_manufacturers.items()},
            'generic_to_strengths': {k: list(v) for k, v in self.generic_to_strengths.items()},
            'generic_to_forms': {k: list(v) for k, v in self.generic_to_forms.items()},
            'all_names': self.all_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to {filepath}")
    
    def load(self, filepath):
        """Load relationship database from pickle"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.brand_to_generic = data['brand_to_generic']
        self.generic_to_brands = defaultdict(set, {k: set(v) for k, v in data['generic_to_brands'].items()})
        self.generic_to_manufacturers = defaultdict(set, {k: set(v) for k, v in data['generic_to_manufacturers'].items()})
        self.generic_to_strengths = defaultdict(set, {k: set(v) for k, v in data['generic_to_strengths'].items()})
        self.generic_to_forms = defaultdict(set, {k: set(v) for k, v in data['generic_to_forms'].items()})
        self.all_names = data['all_names']
        print(f"Loaded from {filepath}")
        print(f"  {len(self.all_names)} drug names")
        print(f"  {len(self.generic_to_brands)} generics with brand mappings")


# ════════════════════════════════════════════════════════════════
# DOWNLOAD FDA DATA INSTRUCTIONS
# ════════════════════════════════════════════════════════════════

def print_download_instructions():
    """Print instructions for downloading FDA data"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           FDA DATA DOWNLOAD INSTRUCTIONS                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. FDA NDC (National Drug Code) Directory:                     ║
║     URL: https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory
║                                                                  ║
║     - Click "NDC Database File - Text Version (Zip)"            ║
║     - Extract and use "product.txt"                              ║
║     - Contains: Brand names, generics, manufacturers, forms      ║
║                                                                  ║
║  2. FDA Orange Book (Therapeutic Equivalents):                   ║
║     URL: https://www.fda.gov/drugs/drug-approvals-and-databases/approved-drug-products-therapeutic-equivalence-evaluations-orange-book
║                                                                  ║
║     - Contains therapeutic equivalence ratings                   ║
║     - Useful for "same as" relationships                        ║
║                                                                  ║
║  3. OpenFDA API (Alternative - no download needed):             ║
║     URL: https://open.fda.gov/apis/drug/ndc/                    ║
║                                                                  ║
║     - Free API for drug information                              ║
║     - Can query in real-time                                    ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


# ════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_download_instructions()
    
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    
    # Example code
    example_code = '''
# Initialize the relationship database
db = DrugRelationshipDB()

# Load FDA NDC data (download first - see instructions above)
db.load_fda_ndc_data('Data/FDA/product.txt')

# Optionally add DrugBank data
import pandas as pd
drugbank = pd.read_csv('Data/drugbank_vocabulary.csv')
db.load_drugbank_mappings(drugbank)

# Save for quick loading later
db.save('Models/drug_relationships.pkl')

# ─────────────────────────────────────────────────
# In your backend, when user clicks a search result:
# ─────────────────────────────────────────────────

info = db.get_drug_info('Advil')
print(info)
# Output:
# {
#     'searched': 'Advil',
#     'type': 'brand',
#     'generic_name': 'IBUPROFEN',
#     'brand_names': ['ADVIL', 'MOTRIN', 'NUROFEN', ...],
#     'other_brands': ['MOTRIN', 'NUROFEN', ...],  # Excludes 'ADVIL'
#     'manufacturers': ['Pfizer Consumer Healthcare', ...],
#     'strengths': ['200 MG', '400 MG', '600 MG', '800 MG'],
#     'forms': ['TABLET', 'CAPSULE', 'ORAL SUSPENSION']
# }
'''
    print(example_code)
