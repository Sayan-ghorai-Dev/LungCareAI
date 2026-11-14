import sqlite3
import hashlib
from pathlib import Path

# Ensures users.db is always created in the main LungCareAI root directory
DB_PATH = Path(__file__).resolve().parent.parent / "users.db"

def make_hashes(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(str.encode(password)).hexdigest()

def create_user_table():
    """Create the users table if it does not already exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS userstable (
            username TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_userdata(username: str, name: str, password: str) -> bool:
    """Register a new user. Returns True if successful, False if username exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO userstable(username, name, password) VALUES (?,?,?)',
            (username.strip(), name.strip(), make_hashes(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username: str, password: str):
    """Verify user credentials. Returns user record if valid, else empty list."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        'SELECT * FROM userstable WHERE username = ? AND password = ?',
        (username.strip(), make_hashes(password))
    )
    data = c.fetchall()
    conn.close()
    return data