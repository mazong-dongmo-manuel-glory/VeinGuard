import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'veinguard.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Biometrics table
    c.execute('''
        CREATE TABLE IF NOT EXISTS biometrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            lbp_reference BLOB NOT NULL,
            pbbm_mask BLOB NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    ''')
    
    # Create Access Logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            method TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL
        )
    ''')

    # Create Audit Logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            meta TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create Settings table
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')
    
    # Create a default admin if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin@gmail.com'")
    if not c.fetchone():
        from werkzeug.security import generate_password_hash
        default_hash = generate_password_hash('admin1234', method='pbkdf2:sha256')
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                  ('admin@gmail.com', default_hash, 'admin'))
    
    # Pre-populate some audit logs for demonstration if empty
    c.execute("SELECT COUNT(*) FROM audit_logs")
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO audit_logs (level, title, description, meta) VALUES (?, ?, ?, ?)",
                  ('CRITICAL', 'SYSTEM MODERNIZATION', 'Transitioned to pure MQTT architecture', 'ID: VG-2024-AUTO'))
        c.execute("INSERT INTO audit_logs (level, title, description, meta) VALUES (?, ?, ?, ?)",
                  ('HIGH', 'DATABASE UPGRADE', 'Added secure audit logs table', 'Schema: v2.1'))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
    print(f"Database initialized at {DB_PATH}")
