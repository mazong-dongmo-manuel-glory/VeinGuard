from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import database
import os

app = Flask(__name__)
CORS(app)

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"status": "running", "service": "VeinGuard API (PC Mode - Sans Biométrie)"})

# --- CRUD & AUTHENTIFICATION ---

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing username or password"}), 400
        
    conn = database.get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (data['username'],)).fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], data['password']):
        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user['id'],
                "username": user['username'],
                "role": user['role']
            }
        }), 200
        
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    # En API sans état, l'app supprime le token localement. Endpoint ajouté pour complétude.
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/api/users', methods=['GET'])
def get_users():
    conn = database.get_db_connection()
    users = conn.execute('SELECT id, username, role, created_at FROM users').fetchall()
    conn.close()
    return jsonify([dict(u) for u in users])

@app.route('/api/users/enroll', methods=['POST'])
def enroll_user():
    data = request.json
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Missing required fields"}), 400
        
    conn = database.get_db_connection()
    try:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (data['username'],)).fetchone()
        if user:
            return jsonify({"error": "Username already exists"}), 409
            
        hashed_pw = generate_password_hash(data['password'], method='pbkdf2:sha256')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                       (data['username'], hashed_pw, data.get('role', 'user')))
        user_id = cursor.lastrowid
        
        conn.commit()
        return jsonify({"message": "User enrolled successfully (Without biometrics)", "user_id": user_id}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    conn = database.get_db_connection()
    try:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        new_role = data.get('role', user['role'])
        new_username = data.get('username', user['username'])
        cursor = conn.cursor()
        
        if 'password' in data and data['password']:
            hashed_pw = generate_password_hash(data['password'], method='pbkdf2:sha256')
            cursor.execute('UPDATE users SET username = ?, role = ?, password_hash = ? WHERE id = ?', 
                           (new_username, new_role, hashed_pw, user_id))
        else:
            cursor.execute('UPDATE users SET username = ?, role = ? WHERE id = ?', 
                           (new_username, new_role, user_id))
            
        conn.commit()
        return jsonify({"message": "User updated successfully"}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    conn = database.get_db_connection()
    try:
        user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404
            
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        return jsonify({"message": "User deleted successfully"}), 200
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# --- MOCKS BIOMÉTRIQUES (Pour ne pas crasher côté front-end) ---

@app.route('/api/scan', methods=['POST'])
def scan_biometrics():
    data = request.json
    if not data or 'user_id' not in data:
        return jsonify({"error": "Missing user_id"}), 400
        
    user_id = data['user_id']
    conn = database.get_db_connection()
    
    # Simuler un passage réussi dans les logs
    conn.execute("INSERT INTO access_logs (user_id, status, method) VALUES (?, ?, ?)",
                 (user_id, "GRANTED", "mock_pc_scan"))
    conn.commit()
    conn.close()
    
    return jsonify({
        "match": True,
        "score": 0.05,
        "status": "GRANTED",
        "message": "Simulated successful scan (PC Mode)"
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    conn = database.get_db_connection()
    logs = conn.execute('''
        SELECT a.id, a.timestamp, a.status, a.method, u.username 
        FROM access_logs a 
        LEFT JOIN users u ON a.user_id = u.id
        ORDER BY a.timestamp DESC
        LIMIT 100
    ''').fetchall()
    conn.close()
    return jsonify([dict(l) for l in logs])

if __name__ == '__main__':
    if not os.path.exists(database.DB_PATH):
        database.init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
