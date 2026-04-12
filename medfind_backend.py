from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import os
from database import init_db, create_user, verify_user, log_search
from ml_functions import hybrid_ensemble_search_v2
import jwt
import datetime
from functools import wraps
import requests

# ────────────────────────────────────────────────────────────────
#  FLASK APP SETUP
# ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'medfind-secret-key-2024')

# Enable CORS for frontend
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "OPTIONS"]
    }
})

# ────────────────────────────────────────────────────────────────
#  LOAD ML MODEL AND DATABASE
# ────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("[*] MEDFIND BACKEND - LOADING...")
print("="*70)

# Model paths
MODEL_PATH = 'Models/drug_ranker_model.pkl'
LOOKUP_PATH = 'Models/lookup_df.pkl'
DRUG_REL_PATH = 'Models/drug_relationships.pkl'

# Load ML model
try:
    print(f"\n[+] Loading ML model from: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        ranker_model = pickle.load(f)
    print(f"[OK] Model loaded: {type(ranker_model)}")
    print(f"[OK] Model features: {ranker_model.num_feature()}")
except FileNotFoundError:
    print(f"[ERROR] Model file not found at {MODEL_PATH}")
    print(f"   Run the Jupyter notebook to train and save the model first!")
    exit(1)
except Exception as e:
    print(f"[ERROR] Loading model: {e}")
    exit(1)

# Load lookup database
try:
    print(f"\n[+] Loading lookup database from: {LOOKUP_PATH}")
    lookup_df = pd.read_pickle(LOOKUP_PATH)
    print(f"[OK] Database loaded: {len(lookup_df)} entries")
    print(f"[OK] Columns: {list(lookup_df.columns)}")
    
    # Verify required columns
    required_cols = ['canonical', 'term', 'source']
    missing = [col for col in required_cols if col not in lookup_df.columns]
    if missing:
        print(f"[WARN] Missing columns: {missing}")
    else:
        print(f"[OK] All required columns present")
        
except FileNotFoundError:
    print(f"[ERROR] Lookup database not found at {LOOKUP_PATH}")
    print(f"   Run the Jupyter notebook to generate lookup_df.pkl first!")
    exit(1)
except Exception as e:
    print(f"[ERROR] Loading database: {e}")
    exit(1)

# Load drug relationship database (optional - for brand/generic lookups)
drug_rel_db = None
try:
    if os.path.exists(DRUG_REL_PATH):
        print(f"\n[+] Loading drug relationships from: {DRUG_REL_PATH}")
        from drug_relationships import DrugRelationshipDB
        drug_rel_db = DrugRelationshipDB()
        drug_rel_db.load(DRUG_REL_PATH)
        print(f"[OK] Drug relationships loaded")
    else:
        print(f"\n[WARN] Drug relationships not found at {DRUG_REL_PATH}")
        print(f"   Run build_drug_relationships.py to enable brand/generic lookups")
except Exception as e:
    print(f"[WARN] Could not load drug relationships: {e}")
    drug_rel_db = None

# Initialize SQLite database
print(f"\n[+] Initializing user database...")
init_db()
print(f"[OK] User database ready")

print("\n" + "="*70)
print("[OK] BACKEND READY!")
print("="*70)
print(f"\n[*] Search mode: PHONETIC-AS-FEATURE (no filtering)")
print(f"[*] Model features: 12")
print(f"[*] Database size: {len(lookup_df):,} drugs")
print(f"[*] Drug relationships: {'Loaded' if drug_rel_db else 'Not available'}")
print("\n" + "="*70 + "\n")


# ────────────────────────────────────────────────────────────────
#  JWT AUTHENTICATION
# ────────────────────────────────────────────────────────────────

def token_required(f):
    """Decorator to require valid JWT token"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        
        try:
            # Remove 'Bearer ' prefix if present
            if token.startswith('Bearer '):
                token = token[7:]
            
            # Decode token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['username']
            
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated


# ────────────────────────────────────────────────────────────────
#  API ROUTES
# ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'version': '2.0-phonetic-as-feature',
        'database_size': len(lookup_df),
        'model_features': ranker_model.num_feature(),
        'search_mode': 'no_phonetic_filtering',
        'drug_relationships': drug_rel_db is not None
    })


@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    """Register new user"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Create user
        if create_user(username, password):
            return jsonify({'message': 'User created successfully'}), 201
        else:
            return jsonify({'error': 'Username already exists'}), 400
            
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    """Login and get JWT token"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Verify credentials
        if verify_user(username, password):
            # Generate JWT token (expires in 24 hours)
            token = jwt.encode({
                'username': username,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, app.config['SECRET_KEY'], algorithm='HS256')
            
            return jsonify({
                'token': token,
                'username': username,
                'message': 'Login successful'
            }), 200
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search', methods=['POST', 'OPTIONS'])
def search_drug():
    """Search for drugs using hybrid ensemble method"""
    
    # Handle CORS preflight FIRST (before token check)
    if request.method == 'OPTIONS':
        return '', 200
    
    # NOW check token (only for POST requests)
    token = request.headers.get('Authorization')
    
    if not token:
        return jsonify({'error': 'Token is missing'}), 401
    
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token[7:]
        
        # Decode token
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        current_user = data['username']
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    
    # NOW do the actual search
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"\n[SEARCH] Query: '{query}' (user: {current_user})")
        
        # Perform hybrid ensemble search
        result = hybrid_ensemble_search_v2(
            query=query,
            lookup_df=lookup_df,
            model=ranker_model,
            strategy='confidence_weighted',
            top_k=5
        )
        
        # Log search
        try:
            if not result.empty:
                top_result = result.iloc[0]['canonical']
                confidence = float(result.iloc[0]['ensemble_score'])
                log_search(current_user, query, top_result, confidence)
        except Exception as log_error:
            print(f"[WARN] Logging error: {log_error}")
        
        # Convert to JSON
        if result.empty:
            print(f"[WARN] No results found for '{query}'")
            return jsonify({'results': []}), 200
        
        results_list = []
        for _, row in result.iterrows():
            results_list.append({
                'name': row['canonical'],
                'confidence': float(row['ensemble_score']) * 100,
                'agreement': bool(row['agreement']),
                'edit_confidence': float(row['edit_confidence']) * 100,
                'ml_confidence': float(row['ml_confidence']) * 100
            })
        
        print(f"[OK] Found {len(results_list)} results")
        print(f"  Top result: {results_list[0]['name']} ({results_list[0]['confidence']:.1f}%)")
        
        return jsonify({'results': results_list}), 200
        
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/drug-info/<drug_name>', methods=['GET', 'OPTIONS'])
def get_drug_info(drug_name):
    """Get detailed drug information from FDA API"""
    
    # Handle CORS preflight FIRST
    if request.method == 'OPTIONS':
        return '', 200
    
    # NOW check token (only for GET requests)
    token = request.headers.get('Authorization')
    
    if not token:
        return jsonify({'error': 'Token is missing'}), 401
    
    try:
        if token.startswith('Bearer '):
            token = token[7:]
        
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        current_user = data['username']
        
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    
    # NOW fetch drug info
    try:
        print(f"\n[INFO] Fetching drug info for: {drug_name}")
        
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:\"{drug_name}\"+openfda.generic_name:\"{drug_name}\"&limit=1"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 404:
            print(f"[WARN] Drug not found in FDA database: {drug_name}")
            return jsonify({
                'name': drug_name,
                'description': 'Information not available in FDA database.',
                'indications': 'Consult a healthcare professional.',
                'warnings': 'Always consult your doctor or pharmacist.',
                'brand_names': []
            }), 200
        
        if response.status_code != 200:
            print(f"[WARN] FDA API error: {response.status_code}")
            return jsonify({'error': 'FDA API error'}), response.status_code
        
        data = response.json()
        
        if 'results' not in data or len(data['results']) == 0:
            print(f"[WARN] No FDA data for: {drug_name}")
            return jsonify({
                'name': drug_name,
                'description': 'Information not available.',
                'indications': 'Consult a healthcare professional.',
                'warnings': 'Always consult your doctor or pharmacist.',
                'brand_names': []
            }), 200
        
        drug_data = data['results'][0]
        
        # Extract relevant fields
        info = {
            'name': drug_name,
            'description': drug_data.get('description', ['N/A'])[0] if 'description' in drug_data else 'N/A',
            'indications': drug_data.get('indications_and_usage', ['N/A'])[0] if 'indications_and_usage' in drug_data else 'N/A',
            'dosage': drug_data.get('dosage_and_administration', ['N/A'])[0] if 'dosage_and_administration' in drug_data else 'N/A',
            'warnings': drug_data.get('warnings', ['N/A'])[0] if 'warnings' in drug_data else 'N/A',
            'brand_name': drug_data.get('openfda', {}).get('brand_name', ['N/A'])[0] if 'openfda' in drug_data else 'N/A',
            'manufacturer': drug_data.get('openfda', {}).get('manufacturer_name', ['N/A'])[0] if 'openfda' in drug_data else 'N/A'
        }
        
        print(f"[OK] FDA data retrieved for: {drug_name}")
        return jsonify(info), 200
        
    except requests.Timeout:
        print(f"[WARN] FDA API timeout for: {drug_name}")
        return jsonify({'error': 'FDA API timeout'}), 504
    except Exception as e:
        print(f"Drug info error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/drug-details/<drug_name>', methods=['GET', 'OPTIONS'])
def get_drug_details(drug_name):
    """
    Get detailed information about a drug including:
    - Generic name
    - Other brand names
    - Available strengths
    - Dosage forms
    - Manufacturers
    
    Called when user clicks on a search result.
    """
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    # Check if drug relationships database is loaded
    if drug_rel_db is None:
        return jsonify({
            'success': False,
            'error': 'Drug relationships database not available. Run build_drug_relationships.py first.'
        }), 503
    
    try:
        info = drug_rel_db.get_drug_info(drug_name)
        
        return jsonify({
            'success': True,
            'data': {
                'searched': info['searched'],
                'type': info['type'],  # 'brand', 'generic', or 'synonym'
                'generic_name': info['generic_name'],
                'is_generic': info['type'] == 'generic',
                'brand_names': info.get('other_brands', [])[:10],  # Top 10 brands
                'all_brands_count': len(info.get('brand_names', [])),
                'manufacturers': info.get('manufacturers', [])[:5],  # Top 5 manufacturers
                'strengths': info.get('strengths', []),
                'forms': info.get('forms', [])
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/app')
def serve_frontend():
    return app.send_static_file('frontend.html')


# ────────────────────────────────────────────────────────────────
#  RUN SERVER
# ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
