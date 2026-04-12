"""
Database Functions for MedFind
Handles user authentication and search logging
"""

import sqlite3
import hashlib
from datetime import datetime

DATABASE_PATH = 'medfind.db'

# ────────────────────────────────────────────────────────────────
#  DATABASE INITIALIZATION
# ────────────────────────────────────────────────────────────────

def init_db():
    """Initialize SQLite database with tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Search history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            query TEXT NOT NULL,
            result TEXT,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users(username)
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("✓ Database initialized successfully")


# ────────────────────────────────────────────────────────────────
#  USER MANAGEMENT
# ────────────────────────────────────────────────────────────────

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password, role = 'user'):
    """
    Create a new user
    Returns True if successful, False if username exists
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        
        cursor.execute(
            'INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)',
            (username, password_hash, role)
        )
        
        conn.commit()
        conn.close()
        
        print(f"✓ User created: {username}")
        return True
        
    except sqlite3.IntegrityError:
        # Username already exists
        print(f"⚠️  Username already exists: {username}")
        return False
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return False


def verify_user(username, password):
    """
    Verify user credentials
    Returns dict with username and role if correct, None otherwise
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute(
            'SELECT username, role FROM users WHERE username = ? AND password_hash = ?',
            (username, password_hash)
        )
        user = cursor.fetchone()
        conn.close()
        if user:
            print(f"✓ User authenticated: {username} (role: {user[1]})")
            return {'username': user[0], 'role': user[1]}
        else:
            print(f"⚠️ Invalid credentials for: {username}")
            return None
    except Exception as e:
        print(f"❌ Error verifying user: {e}")
        return None


def get_user(username):
    """
    Get user information
    Returns user dict or None
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, username, created_at FROM users WHERE username = ?',
            (username,)
        )
        
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'created_at': user[2]
            }
        return None
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return None


# ────────────────────────────────────────────────────────────────
#  SEARCH LOGGING
# ────────────────────────────────────────────────────────────────

def log_search(username, query, result, confidence):
    """
    Log a search query to the database
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            '''INSERT INTO search_history 
               (username, query, result, confidence) 
               VALUES (?, ?, ?, ?)''',
            (username, query, result, confidence)
        )
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error logging search: {e}")
        return False


def get_search_history(username, limit=10):
    """
    Get search history for a user
    Returns list of searches (most recent first)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT query, result, confidence, timestamp 
               FROM search_history 
               WHERE username = ? 
               ORDER BY timestamp DESC 
               LIMIT ?''',
            (username, limit)
        )
        
        history = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts
        results = []
        for row in history:
            results.append({
                'query': row[0],
                'result': row[1],
                'confidence': row[2],
                'timestamp': row[3]
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error getting search history: {e}")
        return []


def get_all_searches(limit=100):
    """
    Get all searches across all users (for admin/analytics)
    Returns list of searches (most recent first)
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            '''SELECT username, query, result, confidence, timestamp 
               FROM search_history 
               ORDER BY timestamp DESC 
               LIMIT ?''',
            (limit,)
        )
        
        searches = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts
        results = []
        for row in searches:
            results.append({
                'username': row[0],
                'query': row[1],
                'result': row[2],
                'confidence': row[3],
                'timestamp': row[4]
            })
        
        return results
        
    except Exception as e:
        print(f"❌ Error getting all searches: {e}")
        return []


# ────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────

def clear_search_history(username):
    """Clear search history for a user"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            'DELETE FROM search_history WHERE username = ?',
            (username,)
        )
        
        conn.commit()
        deleted = cursor.rowcount
        conn.close()
        
        print(f"✓ Deleted {deleted} search records for {username}")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing history: {e}")
        return False


def delete_user(username):
    """Delete a user and their search history"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Delete search history first
        cursor.execute(
            'DELETE FROM search_history WHERE username = ?',
            (username,)
        )
        
        # Delete user
        cursor.execute(
            'DELETE FROM users WHERE username = ?',
            (username,)
        )
        
        conn.commit()
        conn.close()
        
        print(f"✓ User deleted: {username}")
        return True
        
    except Exception as e:
        print(f"❌ Error deleting user: {e}")
        return False


# ────────────────────────────────────────────────────────────────
#  RUN INITIALIZATION
# ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """Initialize database and create default admin user"""
    print("Initializing MedFind database...")
    
    init_db()
    
    # Create default admin user
    if create_user('admin', 'admin123', role='admin'):
        print("✓ Default admin user created (username: admin, password: admin123, role: admin)")
    else:
        print("ℹ️ Admin user already exists")
    
    print("\n✅ Database setup complete!")
