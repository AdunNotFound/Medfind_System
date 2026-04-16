import time
import pandas as pd
import pickle
from ml_functions import hybrid_ensemble_search_v2

with open('Models/drug_ranker_model.pkl', 'rb') as f:
    model = pickle.load(f)
lookup_df = pd.read_pickle('Models/lookup_df.pkl')

queries = ["amoxisilin", "parastamol", "ibuprofn", "cetrizine", "asprin",
           "metformn", "losartn", "gabapntin", "fenasitamol", "tramadl"]

print("NFR02 - PERFORMANCE TESTING")
print("=" * 50)

times = []
for q in queries:
    start = time.time()
    result = hybrid_ensemble_search_v2(q, lookup_df, model, top_k=5)
    elapsed = (time.time() - start) * 1000
    times.append(elapsed)
    print(f"  '{q}' → {elapsed:.1f}ms")

print(f"\nMin:     {min(times):.1f}ms")
print(f"Max:     {max(times):.1f}ms")
print(f"Average: {sum(times)/len(times):.1f}ms")
print(f"Target:  <2000ms")

if max(times) < 2000:
    print("\n✅ NFR02 PASSED - All queries under 2000ms")
else:
    print("\n❌ NFR02 FAILED")