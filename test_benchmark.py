import pandas as pd
import numpy as np
import pickle
from ml_functions import hybrid_ensemble_search_v2, get_edit_distance_ranking, get_ml_ranking

with open('Models/drug_ranker_model.pkl', 'rb') as f:
    model = pickle.load(f)
lookup_df = pd.read_pickle('Models/lookup_df.pkl')

test_cases = [
    ("amoxicilln", "amoxicillin"),
    ("ibuprofn", "ibuprofen"),
    ("asprin", "acetylsalicylic acid"),
    ("metformn", "metformin"),
    ("cetirizne", "cetirizine"),
    ("amoxisilin", "amoxicillin"),
    ("parastamol", "acetaminophen"),
    ("atorvstatin", "atorvastatin"),
    ("lisinoprl", "lisinopril"),
    ("gabapntin", "gabapentin"),
    ("azithromcin", "azithromycin"),
    ("ciproflxacin", "ciprofloxacin"),
    ("hydrocortisne", "hydrocortisone"),
    ("fenasitamol", "acetaminophen"),
    ("amoksisilin", "amoxicillin"),
    ("ibeuprofin", "ibuprofen"),
    ("omeprazol", "omeprazole"),
    ("losartn", "losartan"),
    ("sertralin", "sertraline"),
    ("tramadl", "tramadol"),
]

print("=" * 70)
print("BENCHMARKING: HYBRID vs EDIT DISTANCE vs ML ALONE")
print("=" * 70)

methods = {
    "Edit Distance Only": lambda q: get_edit_distance_ranking(q, lookup_df, top_k=5),
    "ML (LightGBM) Only": lambda q: get_ml_ranking(q, lookup_df, model, top_k=5),
    "Hybrid Ensemble": lambda q: hybrid_ensemble_search_v2(q, lookup_df, model, top_k=5),
}

for method_name, method_fn in methods.items():
    hits_at_1 = 0
    hits_at_5 = 0
    reciprocal_ranks = []

    for query, expected in test_cases:
        result = method_fn(query)
        top_names = [r.lower() for r in result['canonical'].tolist()]

        # Hit@1
        if top_names and top_names[0] == expected.lower():
            hits_at_1 += 1

        # Hit@5
        if expected.lower() in top_names:
            hits_at_5 += 1
            rank = top_names.index(expected.lower()) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)
    print(f"\n{method_name}:")
    print(f"  Hit@1: {hits_at_1/len(test_cases)*100:.1f}%")
    print(f"  Hit@5: {hits_at_5/len(test_cases)*100:.1f}%")
    print(f"  MRR:   {mrr:.4f}")