import requests

BASE = "http://localhost:5000"

# Login first
r = requests.post(f"{BASE}/api/login", json={"username": "admin", "password": "admin123"})
token = r.json()["token"]
headers = {"Authorization": f"Bearer {token}"}

print("NFR04 - RELIABILITY TESTING")
print("=" * 50)

# Test 1: Empty query
r = requests.post(f"{BASE}/api/search", json={"query": ""}, headers=headers)
print(f"1. Empty query → Status: {r.status_code} ({'PASS' if r.status_code == 400 else 'FAIL'})")

# Test 2: Gibberish
r = requests.post(f"{BASE}/api/search", json={"query": "xyzqqq"}, headers=headers)
print(f"2. Gibberish input → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 3: Very long input
r = requests.post(f"{BASE}/api/search", json={"query": "a" * 500}, headers=headers)
print(f"3. Very long input → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 4: Special characters
r = requests.post(f"{BASE}/api/search", json={"query": "!@#$%^&*()"}, headers=headers)
print(f"4. Special characters → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 5: Numbers only
r = requests.post(f"{BASE}/api/search", json={"query": "12345"}, headers=headers)
print(f"5. Numbers only → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 6: Rapid successive searches
import time
start = time.time()
for i in range(20):
    r = requests.post(f"{BASE}/api/search", json={"query": f"aspirin"}, headers=headers)
elapsed = time.time() - start
print(f"6. 20 rapid searches → All completed in {elapsed:.1f}s ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 7: Non-existent drug info
r = requests.get(f"{BASE}/api/drug-info/XYZNOTREAL", headers=headers)
print(f"7. Non-existent drug info → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")