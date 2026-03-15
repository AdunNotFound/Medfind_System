"""
Build Drug Relationship Database
=================================
Downloads FDA data and creates the relationship database.

Run this once to set up the brand/generic mappings.

Usage:
    python build_drug_relationships.py
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
from drug_relationships import DrugRelationshipDB

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

FDA_NDC_URL = "https://www.accessdata.fda.gov/cder/ndctext.zip"
DATA_DIR = "Data/FDA"
OUTPUT_DIR = "Models"
DRUGBANK_CSV = "Data/drugbank_vocabulary.csv"

# ════════════════════════════════════════════════════════════════
# DOWNLOAD FDA DATA
# ════════════════════════════════════════════════════════════════

def download_fda_data():
    """Download and extract FDA NDC data"""
    
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ndctext.zip")
    
    # Check if already downloaded
    product_file = os.path.join(DATA_DIR, "product.txt")
    if os.path.exists(product_file):
        print(f"✓ FDA data already exists at {product_file}")
        return product_file
    
    print(f"Downloading FDA NDC data from {FDA_NDC_URL}...")
    print("  (This is ~15MB, may take a minute)")
    
    try:
        urllib.request.urlretrieve(FDA_NDC_URL, zip_path)
        print(f"✓ Downloaded to {zip_path}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual download instructions:")
        print(f"  1. Go to: https://www.fda.gov/drugs/drug-approvals-and-databases/national-drug-code-directory")
        print(f"  2. Download 'NDC Database File - Text Version (Zip)'")
        print(f"  3. Extract to {DATA_DIR}/")
        print(f"  4. Rename 'product.txt' if needed")
        sys.exit(1)
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Find the product file (might be product.txt or Product.txt)
    for f in os.listdir(DATA_DIR):
        if f.lower() == 'product.txt':
            product_file = os.path.join(DATA_DIR, f)
            break
    
    if os.path.exists(product_file):
        print(f"✓ Extracted to {product_file}")
        return product_file
    else:
        print("✗ Could not find product.txt in zip file")
        print(f"  Files in {DATA_DIR}: {os.listdir(DATA_DIR)}")
        sys.exit(1)

# ════════════════════════════════════════════════════════════════
# BUILD RELATIONSHIP DATABASE
# ════════════════════════════════════════════════════════════════

def build_database():
    """Build the drug relationship database"""
    
    # Download FDA data if needed
    fda_product_file = download_fda_data()
    
    # Initialize database
    print("\n" + "="*60)
    print("BUILDING DRUG RELATIONSHIP DATABASE")
    print("="*60)
    
    db = DrugRelationshipDB()
    
    # Load FDA NDC data
    print(f"\n1. Loading FDA NDC data from {fda_product_file}...")
    db.load_fda_ndc_data(fda_product_file)
    
    # Load DrugBank data if available
    if os.path.exists(DRUGBANK_CSV):
        print(f"\n2. Adding DrugBank synonyms from {DRUGBANK_CSV}...")
        drugbank = pd.read_csv(DRUGBANK_CSV)
        db.load_drugbank_mappings(drugbank)
    else:
        print(f"\n2. Skipping DrugBank (file not found: {DRUGBANK_CSV})")
    
    # Save database
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "drug_relationships.pkl")
    print(f"\n3. Saving database to {output_path}...")
    db.save(output_path)
    
    # Print summary
    print("\n" + "="*60)
    print("DATABASE BUILT SUCCESSFULLY!")
    print("="*60)
    print(f"""
    Statistics:
    - Generic drugs: {len(db.generic_to_brands)}
    - Brand names: {len(db.brand_to_generic)}
    - Total searchable names: {len(db.all_names)}
    
    Output file: {output_path}
    
    Next steps:
    1. Copy {output_path} to your Backend/Models/ folder
    2. Add the drug details endpoint to medfind_backend.py
    3. Update your frontend to show drug details on click
    """)
    
    # Test a few lookups
    print("\n" + "="*60)
    print("TESTING LOOKUPS")
    print("="*60)
    
    test_drugs = ['ADVIL', 'IBUPROFEN', 'TYLENOL', 'ACETAMINOPHEN', 'LIPITOR', 'AMOXICILLIN']
    
    for drug in test_drugs:
        info = db.get_drug_info(drug)
        print(f"\n{drug}:")
        print(f"  Type: {info['type']}")
        print(f"  Generic: {info['generic_name']}")
        brands = info.get('other_brands', [])[:5]
        if brands:
            print(f"  Other brands: {', '.join(brands)}" + (" ..." if len(info.get('brand_names', [])) > 5 else ""))
        forms = info.get('forms', [])[:3]
        if forms:
            print(f"  Forms: {', '.join(forms)}")

# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    build_database()
