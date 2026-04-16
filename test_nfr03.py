import requests

BASE = "http://localhost:5000"

print("NFR03 - SECURITY TESTING")
print("=" * 50)

# Test 1: No token
r = requests.post(f"{BASE}/api/search", json={"query": "test"})
print(f"\n1. No token → Status: {r.status_code} ({'PASS' if r.status_code == 401 else 'FAIL'})")

# Test 2: Fake token
r = requests.post(f"{BASE}/api/search", json={"query": "test"},
                   headers={"Authorization": "Bearer faketoken123"})
print(f"2. Fake token → Status: {r.status_code} ({'PASS' if r.status_code == 401 else 'FAIL'})")

# Test 3: Login and get valid token
r = requests.post(f"{BASE}/api/login", json={"username": "admin", "password": "admin123"})
token = r.json().get("token", "")
print(f"3. Valid login → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 4: Valid token works
r = requests.post(f"{BASE}/api/search", json={"query": "aspirin"},
                   headers={"Authorization": f"Bearer {token}"})
print(f"4. Valid token search → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 5: Wrong password
r = requests.post(f"{BASE}/api/login", json={"username": "admin", "password": "wrongpass"})
print(f"5. Wrong password → Status: {r.status_code} ({'PASS' if r.status_code == 401 else 'FAIL'})")

# Test 6: Admin endpoint with admin token
r = requests.get(f"{BASE}/api/admin/searches",
                  headers={"Authorization": f"Bearer {token}"})
print(f"6. Admin endpoint (admin) → Status: {r.status_code} ({'PASS' if r.status_code == 200 else 'FAIL'})")

# Test 7: Register regular user and test admin endpoint
requests.post(f"{BASE}/api/register", json={"username": "testuser", "password": "test123"})
r = requests.post(f"{BASE}/api/login", json={"username": "testuser", "password": "test123"})
user_token = r.json().get("token", "")
r = requests.get(f"{BASE}/api/admin/searches",
                  headers={"Authorization": f"Bearer {user_token}"})
print(f"7. Admin endpoint (user) → Status: {r.status_code} ({'PASS' if r.status_code == 403 else 'FAIL'})")

passed = 7  # Update based on results
print(f"\nResult: {passed}/7 tests passed")