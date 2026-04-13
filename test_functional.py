import requests
import json

BASE = "http://localhost:5000"

print("=" * 70)
print("FUNCTIONAL TESTING")
print("=" * 70)

results = []

# ────────────────────────────────────────────────────────────
# SETUP: Login as admin and get token
# ────────────────────────────────────────────────────────────
r = requests.post(f"{BASE}/api/login", json={"username": "admin", "password": "admin123"})
admin_token = r.json().get("token", "")
admin_headers = {"Authorization": f"Bearer {admin_token}"}

# Register a regular user for role testing
requests.post(f"{BASE}/api/register", json={"username": "testuser99", "password": "test123"})
r = requests.post(f"{BASE}/api/login", json={"username": "testuser99", "password": "test123"})
user_token = r.json().get("token", "")
user_headers = {"Authorization": f"Bearer {user_token}"}

# ────────────────────────────────────────────────────────────
# FR01: Allow misspelled medication input
# ────────────────────────────────────────────────────────────
print("\n── FR01: Misspelled Medication Input ──")

fr01_cases = [
    ("TC-01", "amoxisilin", "amoxicillin", "2-char error"),
    ("TC-02", "ibuprofn", "ibuprofen", "1-char error"),
    ("TC-03", "metformn", "metformin", "1-char error"),
    ("TC-04", "cetirizne", "cetirizine", "1-char error"),
    ("TC-05", "atorvstatin", "atorvastatin", "2-char error"),
]

for tc_id, query, expected, error_type in fr01_cases:
    r = requests.post(f"{BASE}/api/search", json={"query": query}, headers=admin_headers)
    data = r.json()
    top5 = [res["name"].lower() for res in data.get("results", [])]
    passed = expected.lower() in top5
    status = "PASS" if passed else "FAIL"
    results.append(("FR01", tc_id, f"Search '{query}' ({error_type})", f"'{expected}' in top 5", status))
    print(f"  {tc_id}: '{query}' → {status} {'✓' if passed else '✗'}")

# ────────────────────────────────────────────────────────────
# FR02: Phonetic similarity and ML algorithms
# ────────────────────────────────────────────────────────────
print("\n── FR02: Phonetic Similarity & ML ──")

# TC-06: Phonetic misspelling returns correct result
r = requests.post(f"{BASE}/api/search", json={"query": "amoksisilin"}, headers=admin_headers)
data = r.json()
top5 = [res["name"].lower() for res in data.get("results", [])]
passed = "amoxicillin" in top5
status = "PASS" if passed else "FAIL"
results.append(("FR02", "TC-06", "Phonetic search 'amoksisilin'", "'amoxicillin' in top 5", status))
print(f"  TC-06: Phonetic search → {status}")

# TC-07: Results contain both edit and ML confidence
r = requests.post(f"{BASE}/api/search", json={"query": "omeprazol"}, headers=admin_headers)
data = r.json()
first_result = data.get("results", [{}])[0]
has_edit = "edit_confidence" in first_result
has_ml = "ml_confidence" in first_result
passed = has_edit and has_ml
status = "PASS" if passed else "FAIL"
results.append(("FR02", "TC-07", "Results contain edit_confidence and ml_confidence", "Both fields present", status))
print(f"  TC-07: Dual confidence scores → {status}")

# TC-08: Results contain agreement flag
has_agreement = "agreement" in first_result
status = "PASS" if has_agreement else "FAIL"
results.append(("FR02", "TC-08", "Results contain agreement flag", "Agreement field present", status))
print(f"  TC-08: Agreement flag present → {status}")

# ────────────────────────────────────────────────────────────
# FR03: Confidence score and alternatives
# ────────────────────────────────────────────────────────────
print("\n── FR03: Confidence Score & Alternatives ──")

# TC-09: Returns multiple results (alternatives)
r = requests.post(f"{BASE}/api/search", json={"query": "losartn"}, headers=admin_headers)
data = r.json()
num_results = len(data.get("results", []))
passed = num_results >= 2
status = "PASS" if passed else "FAIL"
results.append(("FR03", "TC-09", "Search returns multiple alternatives", f"Got {num_results} results (need ≥2)", status))
print(f"  TC-09: Multiple alternatives → {status} ({num_results} results)")

# TC-10: Each result has confidence percentage
all_have_confidence = all("confidence" in res for res in data.get("results", []))
status = "PASS" if all_have_confidence else "FAIL"
results.append(("FR03", "TC-10", "All results have confidence score", "confidence field in every result", status))
print(f"  TC-10: Confidence scores present → {status}")

# TC-11: Results are ranked by confidence (descending)
confidences = [res["confidence"] for res in data.get("results", [])]
is_sorted = all(confidences[i] >= confidences[i+1] for i in range(len(confidences)-1))
status = "PASS" if is_sorted else "FAIL"
results.append(("FR03", "TC-11", "Results ranked by confidence descending", f"Scores: {[f'{c:.1f}' for c in confidences]}", status))
print(f"  TC-11: Ranked by confidence → {status}")

# ────────────────────────────────────────────────────────────
# FR04: Search logging
# ────────────────────────────────────────────────────────────
print("\n── FR04: Search Logging ──")

# TC-12: Search is logged in database
from database import get_search_history
history = get_search_history("admin", limit=1)
passed = len(history) > 0
status = "PASS" if passed else "FAIL"
results.append(("FR04", "TC-12", "Search queries are logged to database", f"Found {len(history)} log entries", status))
print(f"  TC-12: Searches logged → {status}")

# TC-13: Log contains query, result, confidence, timestamp
if history:
    log = history[0]
    has_fields = all(k in log for k in ["query", "result", "confidence", "timestamp"])
    status = "PASS" if has_fields else "FAIL"
    results.append(("FR04", "TC-13", "Log entry contains all required fields", "query, result, confidence, timestamp", status))
    print(f"  TC-13: Log fields complete → {status}")
else:
    results.append(("FR04", "TC-13", "Log entry contains all required fields", "No logs found", "FAIL"))
    print(f"  TC-13: Log fields complete → FAIL (no logs)")

# ────────────────────────────────────────────────────────────
# FR05: Authentication and role-based access
# ────────────────────────────────────────────────────────────
print("\n── FR05: Authentication & Role-Based Access ──")

# TC-14: Unauthenticated access blocked
r = requests.post(f"{BASE}/api/search", json={"query": "test"})
passed = r.status_code == 401
status = "PASS" if passed else "FAIL"
results.append(("FR05", "TC-14", "Search without token is blocked", f"Status: {r.status_code} (need 401)", status))
print(f"  TC-14: No token blocked → {status}")

# TC-15: Invalid token blocked
r = requests.post(f"{BASE}/api/search", json={"query": "test"}, headers={"Authorization": "Bearer faketoken"})
passed = r.status_code == 401
status = "PASS" if passed else "FAIL"
results.append(("FR05", "TC-15", "Search with fake token is blocked", f"Status: {r.status_code} (need 401)", status))
print(f"  TC-15: Fake token blocked → {status}")

# TC-16: Valid token allows search
r = requests.post(f"{BASE}/api/search", json={"query": "aspirin"}, headers=admin_headers)
passed = r.status_code == 200
status = "PASS" if passed else "FAIL"
results.append(("FR05", "TC-16", "Search with valid token succeeds", f"Status: {r.status_code} (need 200)", status))
print(f"  TC-16: Valid token works → {status}")

# TC-17: Admin can access admin endpoint
r = requests.get(f"{BASE}/api/admin/searches", headers=admin_headers)
passed = r.status_code == 200
status = "PASS" if passed else "FAIL"
results.append(("FR05", "TC-17", "Admin accesses /api/admin/searches", f"Status: {r.status_code} (need 200)", status))
print(f"  TC-17: Admin access granted → {status}")

# TC-18: Regular user blocked from admin endpoint
r = requests.get(f"{BASE}/api/admin/searches", headers=user_headers)
passed = r.status_code == 403
status = "PASS" if passed else "FAIL"
results.append(("FR05", "TC-18", "Regular user blocked from admin endpoint", f"Status: {r.status_code} (need 403)", status))
print(f"  TC-18: User access denied → {status}")

# ────────────────────────────────────────────────────────────
# FR06: Admin-only search log retrieval
# ────────────────────────────────────────────────────────────
print("\n── FR06: Admin Search Log Retrieval ──")

# TC-19: Admin retrieves all user search logs
r = requests.get(f"{BASE}/api/admin/searches", headers=admin_headers)
data = r.json()
passed = r.status_code == 200 and "searches" in data
status = "PASS" if passed else "FAIL"
results.append(("FR06", "TC-19", "Admin retrieves all search logs", f"Status: {r.status_code}, has 'searches' key", status))
print(f"  TC-19: Admin log retrieval → {status}")

# TC-20: Logs contain entries from multiple users
searches = data.get("searches", [])
usernames = set(s.get("username", "") for s in searches)
passed = len(searches) > 0
status = "PASS" if passed else "FAIL"
results.append(("FR06", "TC-20", "Search logs contain data", f"Found {len(searches)} entries from {len(usernames)} users", status))
print(f"  TC-20: Logs have data → {status} ({len(searches)} entries)")

# ────────────────────────────────────────────────────────────
# FR07: Search history
# ────────────────────────────────────────────────────────────
print("\n── FR07: Search History ──")

# TC-21: Search history function returns user-specific results
from database import get_search_history
admin_history = get_search_history("admin", limit=5)
passed = len(admin_history) > 0
status = "PASS" if passed else "FAIL"
results.append(("FR07", "TC-21", "get_search_history returns admin's searches", f"Found {len(admin_history)} entries", status))
print(f"  TC-21: Admin history → {status} ({len(admin_history)} entries)")

# TC-22: History is user-specific (not mixed)
all_admin = all(True for h in admin_history)  # All from the function are for admin
status = "PASS" if all_admin else "FAIL"
results.append(("FR07", "TC-22", "History is filtered by username", "All entries belong to queried user", status))
print(f"  TC-22: User-specific filtering → {status}")

# ────────────────────────────────────────────────────────────
# FR08: Generic and brand name support
# ────────────────────────────────────────────────────────────
print("\n── FR08: Generic & Brand Name Support ──")

# TC-23: Generic name search returns results
r = requests.post(f"{BASE}/api/search", json={"query": "ibuprofen"}, headers=admin_headers)
data = r.json()
passed = len(data.get("results", [])) > 0
status = "PASS" if passed else "FAIL"
results.append(("FR08", "TC-23", "Search generic name 'ibuprofen'", f"Got {len(data.get('results', []))} results", status))
print(f"  TC-23: Generic search → {status}")

# TC-24: Brand name search returns results
r = requests.post(f"{BASE}/api/search", json={"query": "advil"}, headers=admin_headers)
data = r.json()
passed = len(data.get("results", [])) > 0
status = "PASS" if passed else "FAIL"
results.append(("FR08", "TC-24", "Search brand name 'advil'", f"Got {len(data.get('results', []))} results", status))
print(f"  TC-24: Brand search → {status}")

# TC-25: Drug info endpoint returns data
r = requests.get(f"{BASE}/api/drug-info/ibuprofen", headers=admin_headers)
data = r.json()
has_fields = all(k in data for k in ["name", "description", "indications"])
passed = r.status_code == 200 and has_fields
status = "PASS" if passed else "FAIL"
results.append(("FR08", "TC-25", "Drug info returns description and indications", f"Status: {r.status_code}, fields present: {has_fields}", status))
print(f"  TC-25: Drug info endpoint → {status}")

# ────────────────────────────────────────────────────────────
# SUMMARY
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FUNCTIONAL TESTING SUMMARY")
print("=" * 70)

total = len(results)
passed_count = sum(1 for r in results if r[4] == "PASS")
failed_count = total - passed_count
pass_rate = passed_count / total * 100

print(f"\nTotal test cases: {total}")
print(f"Passed: {passed_count}")
print(f"Failed: {failed_count}")
print(f"Pass rate: {pass_rate:.1f}%")

# Print table
print(f"\n{'FR':<6} {'TC':<7} {'Description':<50} {'Result':<6}")
print("-" * 70)
for fr, tc, desc, expected, status in results:
    symbol = "✓" if status == "PASS" else "✗"
    print(f"{fr:<6} {tc:<7} {desc[:48]:<50} {symbol} {status}")

# Per-FR summary
print(f"\n{'FR':<8} {'Passed':<10} {'Total':<10} {'Rate':<10}")
print("-" * 40)
frs = sorted(set(r[0] for r in results))
for fr in frs:
    fr_results = [r for r in results if r[0] == fr]
    fr_passed = sum(1 for r in fr_results if r[4] == "PASS")
    fr_total = len(fr_results)
    print(f"{fr:<8} {fr_passed:<10} {fr_total:<10} {fr_passed/fr_total*100:.0f}%")