from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from werkzeug.security import check_password_hash, generate_password_hash

import config
import database
from biometrics.biometrics_service import build_multimodal_profile, verify_multimodal
from cloud.firebase_service import FirebaseService
from core.security_controller import SecurityController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class BioGuardMQTTGateway:
    def __init__(self):
        database.init_db()
        self.controller = SecurityController()
        self.firebase = FirebaseService()
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        logger.info("Connecting to MQTT broker at %s:%s", config.MQTT_BROKER, config.MQTT_PORT)
        self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, config.MQTT_KEEPALIVE)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("MQTT connected")
            self.client.subscribe(config.MQTT_TOPIC_CMD_ALL)
            self.publish_status("ONLINE")
        else:
            logger.error("MQTT connection failed with code %s", rc)

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload = msg.payload.decode("utf-8", errors="ignore")
        logger.info("Received message on %s", topic)

        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {}

        if topic == config.MQTT_CMD_SCAN:
            self.handle_scan_command(data)
        elif topic == config.MQTT_CMD_ENROLL:
            self.handle_enroll_command(data)
        elif topic == config.MQTT_CMD_LOGIN:
            self.handle_login_command(data)
        elif topic == config.MQTT_CMD_USERS:
            self.handle_users_list_command(data)
        elif topic == config.MQTT_CMD_LOGS:
            self.handle_logs_list_command(data)
        elif topic == config.MQTT_CMD_AUDIT:
            self.handle_audit_list_command(data)
        elif topic == config.MQTT_CMD_SETTINGS:
            self.handle_settings_update_command(data)
        elif topic == config.MQTT_CMD_PING:
            self.publish_status("ALIVE")

    def handle_scan_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("access/scan", client_id)
        user_id = data.get("user_id")

        if not user_id:
            self.controller.handle_access_denied("ID requis")
            self.client.publish(response_topic, json.dumps({"status": "error", "error": "Missing user_id"}))
            return

        try:
            capture = self.controller.capture_attempt(claimed_user_id=user_id)
        except Exception as exc:
            self.controller.handle_access_denied("Capture invalide")
            self.client.publish(response_topic, json.dumps({"status": "error", "error": str(exc)}))
            return

        stored_profile = database.get_biometric_profile(user_id) or self.firebase.get_biometric_profile(user_id)
        if not stored_profile:
            self.controller.handle_access_denied("Profil absent")
            self._record_access(
                user_id=user_id,
                username=data.get("username"),
                status="DENIED",
                score=None,
                reason="PROFILE_NOT_FOUND",
                method="multimodal_scan",
                modalities={"telemetry": capture["telemetry"]},
            )
            self.client.publish(response_topic, json.dumps({"status": "fail", "reason": "PROFILE_NOT_FOUND"}))
            return

        result = verify_multimodal(capture["frame"], stored_profile)
        if result["match"]:
            self.controller.handle_access_granted(user_id, result["score"])
            status = "GRANTED"
            reason = "MATCH"
        else:
            self.controller.handle_access_denied("Mismatch")
            status = "DENIED"
            reason = "BIOMETRIC_MISMATCH"

        event = self._record_access(
            user_id=user_id,
            username=data.get("username"),
            status=status,
            score=result["score"],
            reason=reason,
            method="multimodal_scan",
            modalities={
                "components": result["components"],
                "telemetry": capture["telemetry"],
                "preview_path": capture["preview_path"],
            },
        )

        response = {
            "status": "success",
            "result": status,
            "score": result["score"],
            "threshold": result["threshold"],
            "components": result["components"],
            "event": event,
        }
        self.client.publish(response_topic, json.dumps(response))

    def handle_enroll_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("users/enroll", client_id)
        user_id = data.get("user_id")
        username = data.get("username", user_id or "unknown")
        role = data.get("role", "operator")
        department = data.get("department", "")
        password = data.get("password", "Temp1234!")

        if not user_id:
            self.client.publish(response_topic, json.dumps({"status": "error", "error": "Missing user_id"}))
            return

        self.controller.handle_enrollment(user_id)

        try:
            capture = self.controller.capture_attempt(claimed_user_id=user_id)
            profile = build_multimodal_profile(capture["frame"])
        except Exception as exc:
            self.client.publish(response_topic, json.dumps({"status": "error", "error": str(exc)}))
            return

        database.upsert_user(
            user_id=user_id,
            username=username,
            password_hash=generate_password_hash(password, method="pbkdf2:sha256"),
            role=role,
            department=department,
        )
        database.save_biometric_profile(user_id, profile)
        self.firebase.save_user_profile(
            user_id,
            {"username": username, "role": role, "department": department, "device_id": config.DEVICE_ID},
        )
        self.firebase.save_biometric_profile(user_id, profile)
        database.log_audit("INFO", "USER_ENROLLED", f"Profil multimodal cree pour {username}", user_id)
        self.publish_status("ENROLLMENT_COMPLETED")

        self.client.publish(
            response_topic,
            json.dumps(
                {
                    "status": "success",
                    "user_id": user_id,
                    "username": username,
                    "profile_modalities": profile["modalities"],
                }
            ),
        )
        self.controller.reset_idle()

    def handle_login_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("auth/login", client_id)
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            self.client.publish(response_topic, json.dumps({"status": "fail", "error": "Missing credentials"}))
            return

        conn = database.get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password_hash"], password):
            self.client.publish(
                response_topic,
                json.dumps(
                    {
                        "status": "success",
                        "user": {
                            "id": user["id"],
                            "username": user["username"],
                            "role": user["role"],
                            "department": user["department"],
                        },
                    }
                ),
            )
        else:
            self.client.publish(response_topic, json.dumps({"status": "fail", "error": "Invalid credentials"}))

    def handle_users_list_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("users/list", client_id)
        self.client.publish(response_topic, json.dumps(database.list_users()))

    def handle_logs_list_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("access/logs", client_id)
        self.client.publish(response_topic, json.dumps(database.list_access_events()))

    def handle_audit_list_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("audit/list", client_id)
        self.client.publish(response_topic, json.dumps(database.list_audit_logs()))

    def handle_settings_update_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("settings/update", client_id)
        applied = self.controller.apply_remote_settings(data)

        for key, value in applied.items():
            database.update_device_state(key, value)

        database.log_audit("INFO", "SETTINGS_UPDATED", "Configuration mobile synchronisee", json.dumps(data))
        self.client.publish(
            response_topic,
            json.dumps(
                {
                    "status": "success",
                    "settings": {k: v for k, v in applied.items() if k != "telemetry"},
                    "telemetry": applied["telemetry"],
                }
            ),
        )
        self.publish_status("SETTINGS_UPDATED")

    def publish_status(self, status: str) -> None:
        payload = {
            "status": status,
            "device_id": config.DEVICE_ID,
            "app": config.APP_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.client.publish(config.MQTT_TOPIC_STATUS, json.dumps(payload))

    def publish_telemetry(self) -> None:
        payload = self.controller.sensor_snapshot()
        database.update_device_state("last_telemetry", payload)
        self.firebase.save_telemetry(config.DEVICE_ID, payload)
        self.client.publish(config.MQTT_TOPIC_TELEMETRY, json.dumps(payload))

    def _record_access(
        self,
        user_id: str | None,
        username: str | None,
        status: str,
        score: float | None,
        reason: str,
        method: str,
        modalities: dict,
    ) -> dict:
        event = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "username": username or user_id,
            "status": status,
            "score": score,
            "reason": reason,
            "method": method,
            "modalities": modalities,
            "device_id": config.DEVICE_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        database.log_access_event(
            event_id=event["id"],
            user_id=user_id,
            username=event["username"],
            status=status,
            score=score,
            reason=reason,
            method=method,
            modalities=modalities,
            synced=self.firebase.save_access_event(event["id"], event),
        )
        self.client.publish(config.MQTT_TOPIC_EVENTS, json.dumps(event))
        return event

    def run(self) -> None:
        self.client.loop_start()
        logger.info("Gateway running")
        try:
            while True:
                self.publish_telemetry()
                time.sleep(3)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.client.loop_stop()
            self.controller.close()


VeinGuardMQTTGateway = BioGuardMQTTGateway


if __name__ == "__main__":
    gateway = BioGuardMQTTGateway()
    gateway.run()
