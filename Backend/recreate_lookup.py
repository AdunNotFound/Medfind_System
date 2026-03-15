import os
import glob

print("Searching for pickle files...\n")

# Search starting from MedFind folder
search_path = r'C:\Users\User\OneDrive - University of Westminster\Medfind'

# Find all .pkl files
for root, dirs, files in os.walk(search_path):
    for file in files:
        if file.endswith('.pkl'):
            full_path = os.path.join(root, file)
            size_mb = os.path.getsize(full_path) / 1024 / 1024
            print(f"✓ Found: {full_path}")
            print(f"  Size: {size_mb:.2f} MB")
            print()