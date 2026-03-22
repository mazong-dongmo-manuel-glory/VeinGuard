import unittest
import base64
import json
import os
import tempfile
from app import app
import database

class TestVeinGuardAPI(unittest.TestCase):
    def setUp(self):
        # Create a temp DB for testing
        self.test_db_fd, database.DB_PATH = tempfile.mkstemp()
        database.init_db()
        app.config['TESTING'] = True
        self.client = app.test_client()

    def tearDown(self):
        os.close(self.test_db_fd)
        os.unlink(database.DB_PATH)

    def test_status(self):
        response = self.client.get('/api/status')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data)['status'], 'running')
        
    def test_login_success(self):
        # Default admin is created by init_db
        response = self.client.post('/api/login', 
                                   json={'username': 'admin', 'password': 'admin123'})
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['user']['username'], 'admin')
        
    def test_login_fail(self):
        response = self.client.post('/api/login', 
                                   json={'username': 'admin', 'password': 'wrongpassword'})
        self.assertEqual(response.status_code, 401)
        
    def test_user_enrollment(self):
        response = self.client.post('/api/users/enroll',
                                    json={'username': 'jon_doe', 'password': 'password123'})
        self.assertEqual(response.status_code, 201)
        
    def test_get_users(self):
        response = self.client.get('/api/users')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertGreaterEqual(len(data), 1)

if __name__ == '__main__':
    unittest.main()
