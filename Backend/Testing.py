"""
MedFind Testing Suite
=====================
Run all tests for Chapter 8 documentation.

Usage:
    python run_tests.py

This will generate:
    - Model evaluation metrics (accuracy, precision, recall, F1)
    - Confusion matrix
    - Performance benchmarks
    - Functional test results
"""

import pandas as pd
import numpy as np
import pickle
import time
import random
import json
from datetime import datetime
import requests
import sys

# ════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════

MODEL_PATH = 'Models/drug_ranker_model.pkl'
LOOKUP_PATH = 'Models/lookup_df.pkl'
API_BASE = 'http://127.0.0.1:5000'
OUTPUT_DIR = 'TestResults'

# ════════════════════════════════════════════════════════════════
# 8.3 MODEL TESTING
# ════════════════════════════════════════════════════════════════

def generate_misspelling(word):
    """Generate realistic misspelling of a drug name"""
    if len(word) < 3:
        return word
    
    word = word.lower()
    error_type = random.choice(['delete', 'swap', 'replace', 'insert', 'phonetic'])
    
    if error_type == 'delete' and len(word) > 3:
        # Delete a random character
        idx = random.randint(1, len(word) - 2)
        return word[:idx] + word[idx+1:]
    
    elif error_type == 'swap' and len(word) > 2:
        # Swap two adjacent characters
        idx = random.randint(0, len(word) - 2)
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    
    elif error_type == 'replace':
        # Replace with nearby keyboard character
        idx = random.randint(0, len(word) - 1)
        replacements = {
            'a': 's', 'b': 'v', 'c': 'x', 'd': 's', 'e': 'r',
            'f': 'g', 'g': 'h', 'h': 'j', 'i': 'o', 'j': 'k',
            'k': 'l', 'l': 'k', 'm': 'n', 'n': 'm', 'o': 'i',
            'p': 'o', 'q': 'w', 'r': 't', 's': 'a', 't': 'r',
            'u': 'i', 'v': 'b', 'w': 'e', 'x': 'z', 'y': 'u', 'z': 'x'
        }
        char = word[idx]
        new_char = replacements.get(char, char)
        return word[:idx] + new_char + word[idx+1:]
    
    elif error_type == 'insert':
        # Insert a random character
        idx = random.randint(1, len(word) - 1)
        char = random.choice('aeiou')
        return word[:idx] + char + word[idx:]
    
    elif error_type == 'phonetic':
        # Common phonetic substitutions
        phonetic_subs = [
            ('ph', 'f'), ('f', 'ph'), ('c', 'k'), ('k', 'c'),
            ('tion', 'shun'), ('x', 'ks'), ('qu', 'kw')
        ]
        for old, new in phonetic_subs:
            if old in word:
                return word.replace(old, new, 1)
    
    return word


def run_model_testing(lookup_df, model, n_samples=300):
    """
    Run ML model evaluation tests
    Returns metrics for Chapter 8.3
    """
    print("\n" + "="*60)
    print("8.3 MODEL TESTING")
    print("="*60)
    
    from ml_functions import hybrid_ensemble_search_v2
    
    # Generate test set
    print(f"\nGenerating {n_samples} test queries...")
    random.seed(42)  # Reproducibility
    
    unique_drugs = lookup_df[lookup_df['source'] == 'generic']['canonical'].unique()
    sampled_drugs = random.sample(list(unique_drugs), min(n_samples, len(unique_drugs)))
    
    test_pairs = []
    for drug in sampled_drugs:
        if random.random() < 0.5:  # 50% misspelled
            misspelled = generate_misspelling(drug)
            test_pairs.append((misspelled, drug, 'misspelled'))
        else:
            test_pairs.append((drug, drug, 'exact'))
    
    print(f"   Test set: {len(test_pairs)} queries")
    print(f"   Exact matches: {sum(1 for _, _, t in test_pairs if t == 'exact')}")
    print(f"   Misspelled: {sum(1 for _, _, t in test_pairs if t == 'misspelled')}")
    
    # Evaluate
    print("\nRunning evaluation...")
    results = []
    start_time = time.time()
    
    for i, (query, expected, query_type) in enumerate(test_pairs):
        try:
            search_results = hybrid_ensemble_search_v2(
                query=query,
                lookup_df=lookup_df,
                model=model,
                strategy='confidence_weighted',
                top_k=5
            )
            
            if search_results.empty:
                results.append({
                    'query': query,
                    'expected': expected,
                    'type': query_type,
                    'top1': None,
                    'top1_correct': False,
                    'top5_correct': False,
                    'confidence': 0
                })
            else:
                top1 = search_results.iloc[0]['canonical']
                top5 = search_results['canonical'].tolist()
                confidence = search_results.iloc[0]['ensemble_score']
                
                results.append({
                    'query': query,
                    'expected': expected,
                    'type': query_type,
                    'top1': top1,
                    'top1_correct': top1.lower() == expected.lower(),
                    'top5_correct': expected.lower() in [d.lower() for d in top5],
                    'confidence': confidence
                })
        except Exception as e:
            results.append({
                'query': query,
                'expected': expected,
                'type': query_type,
                'top1': None,
                'top1_correct': False,
                'top5_correct': False,
                'confidence': 0,
                'error': str(e)
            })
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(test_pairs)}...")
    
    elapsed = time.time() - start_time
    
    # Calculate metrics
    results_df = pd.DataFrame(results)
    
    total = len(results_df)
    tp = results_df['top1_correct'].sum()
    fp = total - tp
    fn = total - results_df['top5_correct'].sum()
    
    metrics = {
        'total_queries': total,
        'top1_accuracy': tp / total,
        'top5_accuracy': results_df['top5_correct'].sum() / total,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'avg_time_per_query': elapsed / total,
        'total_time': elapsed
    }
    
    # F1 Score
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0
    
    # Print results
    print("\n" + "-"*60)
    print("EVALUATION RESULTS")
    print("-"*60)
    print(f"Total Test Queries: {metrics['total_queries']}")
    print(f"\nAccuracy Metrics:")
    print(f"   Top-1 Accuracy: {metrics['top1_accuracy']:.3f} ({metrics['top1_accuracy']*100:.1f}%)")
    print(f"   Top-5 Accuracy: {metrics['top5_accuracy']:.3f} ({metrics['top5_accuracy']*100:.1f}%)")
    print(f"\nClassification Metrics:")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall: {metrics['recall']:.3f}")
    print(f"   F1-Score: {metrics['f1_score']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"   True Positives: {metrics['tp']}")
    print(f"   False Positives: {metrics['fp']}")
    print(f"   False Negatives: {metrics['fn']}")
    print(f"\nPerformance:")
    print(f"   Total Time: {metrics['total_time']:.1f}s")
    print(f"   Avg per Query: {metrics['avg_time_per_query']*1000:.1f}ms")
    
    # NFR01 Check
    print("\n" + "-"*60)
    print("NFR01 COMPLIANCE (95% Accuracy Target)")
    print("-"*60)
    if metrics['top1_accuracy'] >= 0.95:
        print(f"[PASS] {metrics['top1_accuracy']*100:.1f}% >= 95%")
    else:
        print(f"[FAIL] {metrics['top1_accuracy']*100:.1f}% < 95%")
    
    # Save results
    results_df.to_csv(f'{OUTPUT_DIR}/model_test_results.csv', index=False)
    
    return metrics, results_df


# ════════════════════════════════════════════════════════════════
# 8.4 BENCHMARKING
# ════════════════════════════════════════════════════════════════

def run_benchmarking(lookup_df, model, n_samples=100):
    """
    Compare ML model vs baseline methods
    Returns comparison table for Chapter 8.4
    """
    print("\n" + "="*60)
    print("8.4 BENCHMARKING")
    print("="*60)
    
    from ml_functions import get_edit_distance_ranking, get_ml_ranking, hybrid_ensemble_search_v2
    
    # Generate test set
    random.seed(42)
    unique_drugs = lookup_df[lookup_df['source'] == 'generic']['canonical'].unique()
    sampled_drugs = random.sample(list(unique_drugs), min(n_samples, len(unique_drugs)))
    
    test_pairs = []
    for drug in sampled_drugs:
        misspelled = generate_misspelling(drug)
        test_pairs.append((misspelled, drug))
    
    print(f"\nBenchmarking {len(test_pairs)} misspelled queries...\n")
    
    # Test each method
    methods = {
        'Edit Distance': lambda q: get_edit_distance_ranking(q, lookup_df, top_k=5),
        'ML Only': lambda q: get_ml_ranking(q, lookup_df, model, top_k=5),
        'Hybrid Ensemble': lambda q: hybrid_ensemble_search_v2(q, lookup_df, model, top_k=5)
    }
    
    benchmark_results = {}
    
    for method_name, method_func in methods.items():
        print(f"Testing {method_name}...")
        correct_top1 = 0
        correct_top5 = 0
        total_time = 0
        
        for query, expected in test_pairs:
            start = time.time()
            try:
                results = method_func(query)
                total_time += time.time() - start
                
                if not results.empty:
                    top1 = results.iloc[0]['canonical']
                    top5 = results['canonical'].tolist()
                    
                    if top1.lower() == expected.lower():
                        correct_top1 += 1
                    if expected.lower() in [d.lower() for d in top5]:
                        correct_top5 += 1
            except Exception as e:
                total_time += time.time() - start
        
        benchmark_results[method_name] = {
            'top1_accuracy': correct_top1 / len(test_pairs),
            'top5_accuracy': correct_top5 / len(test_pairs),
            'avg_time_ms': (total_time / len(test_pairs)) * 1000
        }
    
    # Print comparison table
    print("\n" + "-"*60)
    print("BENCHMARK COMPARISON")
    print("-"*60)
    print(f"{'Method':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Avg Time':<12}")
    print("-"*60)
    for method, scores in benchmark_results.items():
        print(f"{method:<20} {scores['top1_accuracy']*100:>6.1f}%     {scores['top5_accuracy']*100:>6.1f}%     {scores['avg_time_ms']:>6.1f}ms")
    
    return benchmark_results


# ════════════════════════════════════════════════════════════════
# 8.7 FUNCTIONAL TESTING
# ════════════════════════════════════════════════════════════════

def run_functional_tests(api_base=API_BASE):
    """
    Run functional tests against the API
    Returns test results for Chapter 8.7
    """
    print("\n" + "="*60)
    print("8.7 FUNCTIONAL TESTING")
    print("="*60)
    
    test_results = []
    token = None
    
    # Helper function
    def run_test(test_id, requirement, description, test_func):
        try:
            passed, actual = test_func()
            test_results.append({
                'test_id': test_id,
                'requirement': requirement,
                'description': description,
                'expected': 'Pass',
                'actual': actual,
                'status': 'Pass' if passed else 'Fail'
            })
            status = "[PASS]" if passed else "[FAIL]"
            print(f"   {status} {test_id}: {description}")
            return passed
        except Exception as e:
            test_results.append({
                'test_id': test_id,
                'requirement': requirement,
                'description': description,
                'expected': 'Pass',
                'actual': f'Error: {str(e)}',
                'status': 'Fail'
            })
            print(f"   [FAIL] {test_id}: {description} - {e}")
            return False
    
    # ─────────────────────────────────────────────────────────
    # FR01: User Registration
    # ─────────────────────────────────────────────────────────
    print("\nFR01: User Registration")
    
    test_user = f"testuser_{int(time.time())}"
    
    def test_register():
        resp = requests.post(f"{api_base}/api/register", json={
            'username': test_user,
            'password': 'testpass123'
        })
        return resp.status_code == 201, f"Status: {resp.status_code}"
    
    run_test("FR01-TC1", "FR01", "Register new user", test_register)
    
    def test_register_duplicate():
        resp = requests.post(f"{api_base}/api/register", json={
            'username': test_user,
            'password': 'testpass123'
        })
        return resp.status_code == 400, f"Status: {resp.status_code}"
    
    run_test("FR01-TC2", "FR01", "Reject duplicate username", test_register_duplicate)
    
    def test_register_missing_fields():
        resp = requests.post(f"{api_base}/api/register", json={'username': 'test'})
        return resp.status_code == 400, f"Status: {resp.status_code}"
    
    run_test("FR01-TC3", "FR01", "Reject missing password", test_register_missing_fields)
    
    # ─────────────────────────────────────────────────────────
    # FR02: User Login
    # ─────────────────────────────────────────────────────────
    print("\nFR02: User Login")
    
    def test_login():
        nonlocal token
        resp = requests.post(f"{api_base}/api/login", json={
            'username': test_user,
            'password': 'testpass123'
        })
        if resp.status_code == 200:
            token = resp.json().get('token')
        return resp.status_code == 200 and token is not None, f"Status: {resp.status_code}"
    
    run_test("FR02-TC1", "FR02", "Login with valid credentials", test_login)
    
    def test_login_invalid():
        resp = requests.post(f"{api_base}/api/login", json={
            'username': test_user,
            'password': 'wrongpassword'
        })
        return resp.status_code == 401, f"Status: {resp.status_code}"
    
    run_test("FR02-TC2", "FR02", "Reject invalid password", test_login_invalid)
    
    # ─────────────────────────────────────────────────────────
    # FR03: Drug Search
    # ─────────────────────────────────────────────────────────
    print("\nFR03: Drug Search")
    
    def test_search(query, expected_top):
        resp = requests.post(f"{api_base}/api/search", 
            json={'query': query},
            headers={'Authorization': f'Bearer {token}'}
        )
        if resp.status_code == 200:
            results = resp.json().get('results', [])
            if results:
                top = results[0]['name'].lower()
                return expected_top.lower() in top, f"Got: {top}"
        return False, f"Status: {resp.status_code}"
    
    run_test("FR03-TC1", "FR03", "Search 'ibuprofen'", lambda: test_search('ibuprofen', 'ibuprofen'))
    run_test("FR03-TC2", "FR03", "Search 'amoxicillin'", lambda: test_search('amoxicillin', 'amoxicillin'))
    run_test("FR03-TC3", "FR03", "Search 'paracetamol'", lambda: test_search('paracetamol', 'acetaminophen'))
    
    # ─────────────────────────────────────────────────────────
    # FR04: Misspelling Tolerance
    # ─────────────────────────────────────────────────────────
    print("\nFR04: Misspelling Tolerance")
    
    run_test("FR04-TC1", "FR04", "Search 'amoxcilin' (misspelled)", lambda: test_search('amoxcilin', 'amoxicillin'))
    run_test("FR04-TC2", "FR04", "Search 'ibuprofn' (misspelled)", lambda: test_search('ibuprofn', 'ibuprofen'))
    run_test("FR04-TC3", "FR04", "Search 'acetominophen' (misspelled)", lambda: test_search('acetominophen', 'acetaminophen'))
    
    # ─────────────────────────────────────────────────────────
    # FR07: Brand/Generic Mapping
    # ─────────────────────────────────────────────────────────
    print("\nFR07: Brand/Generic Mapping")
    
    run_test("FR07-TC1", "FR07", "Search 'advil' -> ibuprofen", lambda: test_search('advil', 'ibuprofen'))
    run_test("FR07-TC2", "FR07", "Search 'tylenol' -> acetaminophen", lambda: test_search('tylenol', 'acetaminophen'))
    run_test("FR07-TC3", "FR07", "Search 'lipitor' -> atorvastatin", lambda: test_search('lipitor', 'atorvastatin'))
    
    # ─────────────────────────────────────────────────────────
    # FR09: Authentication Required
    # ─────────────────────────────────────────────────────────
    print("\nFR09: Session Management")
    
    def test_no_token():
        resp = requests.post(f"{api_base}/api/search", json={'query': 'test'})
        return resp.status_code == 401, f"Status: {resp.status_code}"
    
    run_test("FR09-TC1", "FR09", "Reject search without token", test_no_token)
    
    def test_invalid_token():
        resp = requests.post(f"{api_base}/api/search", 
            json={'query': 'test'},
            headers={'Authorization': 'Bearer invalid_token_here'}
        )
        return resp.status_code == 401, f"Status: {resp.status_code}"
    
    run_test("FR09-TC2", "FR09", "Reject invalid token", test_invalid_token)
    
    # ─────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────
    results_df = pd.DataFrame(test_results)
    passed = (results_df['status'] == 'Pass').sum()
    total = len(results_df)
    
    print("\n" + "-"*60)
    print(f"FUNCTIONAL TEST SUMMARY: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print("-"*60)
    
    results_df.to_csv(f'{OUTPUT_DIR}/functional_test_results.csv', index=False)
    
    return results_df


# ════════════════════════════════════════════════════════════════
# 8.8 NON-FUNCTIONAL TESTING
# ════════════════════════════════════════════════════════════════

def run_performance_tests(api_base=API_BASE, n_requests=50):
    """
    Test response time performance
    Returns metrics for Chapter 8.8
    """
    print("\n" + "="*60)
    print("8.8 NON-FUNCTIONAL TESTING - Performance")
    print("="*60)
    
    # Login first
    resp = requests.post(f"{api_base}/api/login", json={
        'username': 'admin',
        'password': 'admin123'
    })
    token = resp.json().get('token')
    
    if not token:
        print("[ERROR] Could not login for performance testing")
        return None
    
    # Test queries
    test_queries = [
        'ibuprofen', 'amoxicillin', 'paracetamol', 'aspirin',
        'advil', 'tylenol', 'lipitor', 'metformin',
        'amoxcilin', 'ibuprofn', 'acetominophen'
    ]
    
    response_times = []
    
    print(f"\nRunning {n_requests} search requests...")
    
    for i in range(n_requests):
        query = random.choice(test_queries)
        
        start = time.time()
        resp = requests.post(f"{api_base}/api/search",
            json={'query': query},
            headers={'Authorization': f'Bearer {token}'}
        )
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        response_times.append({
            'request': i + 1,
            'query': query,
            'status': resp.status_code,
            'time_ms': elapsed
        })
        
        if (i + 1) % 10 == 0:
            print(f"   Completed {i+1}/{n_requests}...")
    
    # Calculate metrics
    times = [r['time_ms'] for r in response_times]
    
    metrics = {
        'total_requests': n_requests,
        'avg_time_ms': np.mean(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'p50_time_ms': np.percentile(times, 50),
        'p90_time_ms': np.percentile(times, 90),
        'p95_time_ms': np.percentile(times, 95),
        'p99_time_ms': np.percentile(times, 99),
        'success_rate': sum(1 for r in response_times if r['status'] == 200) / n_requests
    }
    
    print("\n" + "-"*60)
    print("PERFORMANCE RESULTS")
    print("-"*60)
    print(f"Total Requests: {metrics['total_requests']}")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"\nResponse Time (ms):")
    print(f"   Average: {metrics['avg_time_ms']:.1f}")
    print(f"   Min: {metrics['min_time_ms']:.1f}")
    print(f"   Max: {metrics['max_time_ms']:.1f}")
    print(f"   P50 (Median): {metrics['p50_time_ms']:.1f}")
    print(f"   P90: {metrics['p90_time_ms']:.1f}")
    print(f"   P95: {metrics['p95_time_ms']:.1f}")
    print(f"   P99: {metrics['p99_time_ms']:.1f}")
    
    # NFR02 Check
    print("\n" + "-"*60)
    print("NFR02 COMPLIANCE (<500ms Response Time)")
    print("-"*60)
    if metrics['p95_time_ms'] < 500:
        print(f"[PASS] P95 = {metrics['p95_time_ms']:.1f}ms < 500ms")
    else:
        print(f"[FAIL] P95 = {metrics['p95_time_ms']:.1f}ms >= 500ms")
    
    pd.DataFrame(response_times).to_csv(f'{OUTPUT_DIR}/performance_test_results.csv', index=False)
    
    return metrics


# ════════════════════════════════════════════════════════════════
# MAIN - RUN ALL TESTS
# ════════════════════════════════════════════════════════════════

def main():
    """Run all tests and generate reports"""
    
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*60)
    print("MEDFIND TESTING SUITE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model and data
    print("\nLoading model and data...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        lookup_df = pd.read_pickle(LOOKUP_PATH)
        print(f"   Model loaded: {MODEL_PATH}")
        print(f"   Database loaded: {len(lookup_df)} entries")
    except Exception as e:
        print(f"[ERROR] Could not load model/data: {e}")
        return
    
    # Run tests
    all_results = {}
    
    # 8.3 Model Testing
    try:
        model_metrics, model_results = run_model_testing(lookup_df, model, n_samples=300)
        all_results['model_testing'] = model_metrics
    except Exception as e:
        print(f"[ERROR] Model testing failed: {e}")
    
    # 8.4 Benchmarking
    try:
        benchmark_results = run_benchmarking(lookup_df, model, n_samples=100)
        all_results['benchmarking'] = benchmark_results
    except Exception as e:
        print(f"[ERROR] Benchmarking failed: {e}")
    
    # Check if API is running
    try:
        resp = requests.get(f"{API_BASE}/", timeout=5)
        api_running = resp.status_code == 200
    except:
        api_running = False
    
    if api_running:
        # 8.7 Functional Testing
        try:
            functional_results = run_functional_tests()
            all_results['functional_testing'] = {
                'total': len(functional_results),
                'passed': (functional_results['status'] == 'Pass').sum(),
                'pass_rate': (functional_results['status'] == 'Pass').mean()
            }
        except Exception as e:
            print(f"[ERROR] Functional testing failed: {e}")
        
        # 8.8 Performance Testing
        try:
            perf_results = run_performance_tests(n_requests=50)
            all_results['performance_testing'] = perf_results
        except Exception as e:
            print(f"[ERROR] Performance testing failed: {e}")
    else:
        print("\n[WARNING] API not running - skipping functional and performance tests")
        print("   Start the backend with: python medfind_backend.py")
    
    # Save summary
    with open(f'{OUTPUT_DIR}/test_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print(f"Results saved to: {OUTPUT_DIR}/")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()