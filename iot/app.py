from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import check_password_hash

import database


app = Flask(__name__)
CORS(app)


@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'ok', 'service': 'veinguard-api'})


@app.route('/api/login', methods=['POST'])
def login():
    payload = request.get_json(silent=True) or {}
    username = payload.get('username')
    password = payload.get('password')

    if not username or not password:
        return jsonify({'error': 'Missing credentials'}), 400

    conn = database.get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()

    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    return jsonify(
        {
            'status': 'success',
            'user': {
                'id': user['id'],
                'username': user['username'],
                'role': user['role'],
            },
        }
    )


@app.route('/api/logout', methods=['POST'])
def logout():
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    database.init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)
