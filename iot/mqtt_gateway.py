import time
import base64
from werkzeug.security import check_password_hash
import database
from core.security_controller import SecurityController
import biometrics.biometrics_service as biometrics_service
import config
import json
import paho.mqtt.client as mqtt

class VeinGuardMQTTGateway:
    """
    The main entry point for the VeinGuard IoT backend.
    Handles all communication via MQTT, following the user's request to remove Flask.
    """
    def __init__(self):
        self.controller = SecurityController()
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        # Connect to the broker
        print(f"Connecting to MQTT Broker @ {config.MQTT_BROKER}...")
        self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, config.MQTT_KEEPALIVE)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected successfully to MQTT Broker")
            # Subscribe to all command topics
            self.client.subscribe(config.MQTT_TOPIC_CMD)
            self.publish_status("ONLINE")
        else:
            print(f"Connection failed with code {rc}")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode('utf-8', errors='ignore')
        print(f"Received message on {topic}: {payload[:50]}...")

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            data = {}

        if topic == "veinguard/cmd/scan":
            self.handle_scan_command(data)
        elif topic == "veinguard/cmd/enroll":
            self.handle_enroll_command(data)
        elif topic == "veinguard/cmd/auth/login":
            self.handle_login_command(data)
        elif topic == "veinguard/cmd/users/list":
            self.handle_users_list_command(data)
        elif topic == "veinguard/cmd/logs/list":
            self.handle_logs_list_command(data)
        elif topic == "veinguard/cmd/audit/list":
            self.handle_audit_list_command(data)
        elif topic == "veinguard/cmd/settings/update":
            self.handle_settings_update_command(data)
        elif topic == "veinguard/cmd/ping":
            self.publish_status("ALIVE")

    def handle_scan_command(self, data):
        """Processes a biometric scan request."""
        self.controller.handle_scanning()
        
        # In a real scenario, we'd wait for sensor data.
        # Here we assume the image is sent via MQTT (base64) or we use the mock.
        image_b64 = data.get("image")
        user_id = data.get("user_id")
        
        if not user_id:
            self.controller.handle_access_denied("UNKNOWN OPERATIVE")
            return

        # Simple verification logic (Integrating with database)
        conn = database.get_db_connection()
        user_bio = conn.execute("SELECT lbp_reference, pbbm_mask FROM biometrics WHERE user_id = ?", (user_id,)).fetchone()
        conn.close()

        if user_bio:
            # Use real biometrics logic if image provided
            if image_b64:
                image_bytes = base64.b64decode(image_b64)
                match, score = verify_user(image_bytes, user_bio['lbp_reference'], user_bio['pbbm_mask'])
                if match:
                    self.controller.handle_access_granted(f"ID {user_id}")
                    self.log_access(user_id, "GRANTED", "mqtt_app_scan")
                else:
                    self.controller.handle_access_denied("VASCULAR MISMATCH")
                    self.log_access(user_id, "DENIED", "mqtt_app_scan")
            else:
                # Mock result if no image sent (for testing)
                self.controller.handle_access_granted(f"ID {user_id}")
                self.log_access(user_id, "GRANTED", "mqtt_mock")
        else:
            self.controller.handle_access_denied("NO PROFILE FOUND")

    def handle_enroll_command(self, data):
        """Enrolls a new user using hardware camera or MQTT images."""
        images_b64 = data.get("images", [])
        user_id = data.get("user_id")
        
        self.controller.lcd.show_message("ENROLLING...", "FOLLOW SENSORS")

        # Capture from hardware if no images provided via MQTT
        if not images_b64:
            image_bytes_list = []
            for i in range(3): # Take 3 samples
                self.controller.lcd.show_message("ENROLLING...", f"SAMPLE {i+1}/3")
                image_bytes_list.append(biometrics_service.capture_image(f"enroll_{i}.jpg"))
                time.sleep(1)
        else:
            image_bytes_list = [base64.b64decode(img) for img in images_b64]
        
        if not user_id or not image_bytes_list:
            self.publish_status("ENROLL_FAILED: Missing Data")
            return

        ref_lbp, pbbm_mask = biometrics_service.enroll_user(image_bytes_list)
        if ref_lbp:
            conn = database.get_db_connection()
            conn.execute("INSERT OR REPLACE INTO biometrics (user_id, lbp_reference, pbbm_mask) VALUES (?, ?, ?)",
                         (user_id, ref_lbp, pbbm_mask))
            conn.commit()
            conn.close()
            self.controller.lcd.show_message("ENROLL SUCCESS", f"USER ID {user_id}")
            self.publish_status(f"ENROLL_SUCCESS: {user_id}")
        else:
            self.controller.lcd.show_message("ENROLL FAILED", "BAD SAMPLES")
            self.publish_status("ENROLL_FAILED: Biometric failure")

    def handle_login_command(self, data):
        """Processes a login request from the mobile app."""
        username = data.get("username")
        password = data.get("password")
        client_id = data.get("client_id", "anonymous")
        
        response_topic = f"veinguard/res/auth/login/{client_id}"
        
        if not username or not password:
            self.client.publish(response_topic, json.dumps({"error": "Missing credentials"}))
            return

        conn = database.get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            response = {
                "status": "success",
                "user": {"id": user['id'], "username": user['username'], "role": user['role']}
            }
            self.client.publish(response_topic, json.dumps(response))
            print(f"Login success for {username}")
        else:
            self.client.publish(response_topic, json.dumps({"status": "fail", "error": "Invalid credentials"}))
            print(f"Login failed for {username}")

    def handle_users_list_command(self, data):
        """Returns the list of operatives to the mobile app."""
        client_id = data.get("client_id", "anonymous")
        response_topic = f"veinguard/res/users/list/{client_id}"
        
        conn = database.get_db_connection()
        users = conn.execute('SELECT id, username, role, created_at FROM users').fetchall()
        conn.close()
        
        response = [dict(u) for u in users]
        self.client.publish(response_topic, json.dumps(response))

    def handle_logs_list_command(self, data):
        """Returns the access logs to the mobile app."""
        client_id = data.get("client_id", "anonymous")
        response_topic = f"veinguard/res/logs/list/{client_id}"
        
        conn = database.get_db_connection()
        logs = conn.execute('''
            SELECT al.*, u.username 
            FROM access_logs al 
            LEFT JOIN users u ON al.user_id = u.id 
            ORDER BY al.timestamp DESC LIMIT 50
        ''').fetchall()
        conn.close()
        
        response = [dict(l) for l in logs]
        self.client.publish(response_topic, json.dumps(response))

    def handle_audit_list_command(self, data):
        """Returns administrative audit logs."""
        client_id = data.get("client_id", "anonymous")
        response_topic = f"veinguard/res/audit/list/{client_id}"
        
        conn = database.get_db_connection()
        logs = conn.execute('SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 50').fetchall()
        conn.close()
        
        response = [dict(l) for l in logs]
        self.client.publish(response_topic, json.dumps(response))

    def handle_settings_update_command(self, data):
        """Updates system settings in the database."""
        conn = database.get_db_connection()
        for key, value in data.items():
            if key in ["broker_host", "tls_enabled", "biometric_override"]:
                conn.execute('INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)', (key, str(value)))
        conn.commit()
        conn.close()
        self.publish_status("SETTINGS_UPDATED")

    def publish_status(self, status):
        msg = json.dumps({"status": status, "timestamp": time.time()})
        self.client.publish(config.MQTT_TOPIC_STATUS, msg)

    def log_access(self, user_id, status, method):
        conn = database.get_db_connection()
        conn.execute("INSERT INTO access_logs (user_id, status, method) VALUES (?, ?, ?)",
                     (user_id, status, method))
        conn.commit()
        conn.close()
        
        # Broadcast logs to MQTT as well
        log_msg = json.dumps({"user_id": user_id, "status": status, "method": method})
        self.client.publish(config.MQTT_TOPIC_LOGS, log_msg)

    def run(self):
        """Main loop: Process MQTT messages and check hardware sensors."""
        self.client.loop_start()
        print("[Gateway] Background loop started. Monitoring sensors...")
        
        try:
            while True:
                # Example: Proximity-based auto-alert
                if self.controller.check_proximity(threshold=0.1): # 10cm
                    print("[Gateway] Hand detected! Proximity alert.")
                    self.publish_status("PROXIMITY_DETECTED")
                    # We could trigger a scan automatically here if desired
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("[Gateway] Shutting down...")
            self.client.loop_stop()

if __name__ == "__main__":
    gateway = VeinGuardMQTTGateway()
    gateway.run()
