from __future__ import annotations

import json
import logging
import socket
import time
import uuid
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from werkzeug.security import check_password_hash, generate_password_hash

import config
import database
from biometrics.biometrics_service import build_enrollment_profile, verify_live_profile
from cloud.firebase_service import FirebaseService
from core.security_controller import SecurityController

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def resolve_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"


class BioGuardMQTTGateway:
    def __init__(self):
        database.init_db()
        self.controller = SecurityController()
        self.firebase = FirebaseService()
        self.local_ip = resolve_local_ip()
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        if config.MQTT_USERNAME:
            self.client.username_pw_set(config.MQTT_USERNAME, config.MQTT_PASSWORD)
            logger.info("Using MQTT credentials for user %s", config.MQTT_USERNAME)

        logger.info("Connecting to MQTT broker at %s:%s", config.MQTT_BROKER, config.MQTT_PORT)
        logger.info("Broker local TCP endpoint: mqtt://%s:%s", self.local_ip, config.MQTT_PORT)
        logger.info("Broker local WebSocket endpoint: ws://%s:%s", self.local_ip, config.MQTT_WS_PORT)
        self.client.connect(config.MQTT_BROKER, config.MQTT_PORT, config.MQTT_KEEPALIVE)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logger.info("MQTT connected")
            self.client.subscribe(config.MQTT_TOPIC_CMD_ALL)
            self.publish_status("ONLINE")
        else:
            logger.error("MQTT connection failed with code %s", rc)
            if rc == 5:
                logger.error("MQTT broker refused the connection: verify username, password and broker ACL.")

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
        elif topic == config.MQTT_CMD_USERS_UPDATE:
            self.handle_user_update_command(data)
        elif topic == config.MQTT_CMD_USERS_DELETE:
            self.handle_user_delete_command(data)
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

        try:
            capture = self.controller.capture_attempt(claimed_user_id=user_id)
        except Exception as exc:
            self.controller.handle_access_denied("Main absente")
            self._record_access(
                user_id=user_id,
                username=data.get("username"),
                status="DENIED",
                score=None,
                reason="INVALID_CAPTURE",
                method="multimodal_scan",
                modalities={"error": str(exc)},
            )
            self.client.publish(
                response_topic,
                json.dumps({"status": "fail", "reason": "INVALID_CAPTURE", "error": str(exc)}),
            )
            return

        matched_user = None
        if user_id:
            stored_profile = database.get_biometric_profile(user_id) or self.firebase.get_biometric_profile(user_id)
            if stored_profile:
                matched_user = {
                    "user_id": user_id,
                    "username": data.get("username") or user_id,
                    "profile": stored_profile,
                }
        else:
            candidates = database.list_biometric_profiles()
            best_candidate = None
            for candidate in candidates:
                result = verify_live_profile(capture["profile"], candidate["profile"])
                if best_candidate is None or result["score"] < best_candidate["result"]["score"]:
                    best_candidate = {"candidate": candidate, "result": result}
            if best_candidate and best_candidate["result"]["match"]:
                matched_user = {
                    "user_id": best_candidate["candidate"]["user_id"],
                    "username": best_candidate["candidate"]["username"],
                    "profile": best_candidate["candidate"]["profile"],
                    "prefetched_result": best_candidate["result"],
                }

        if not matched_user:
            self.controller.handle_access_denied("Profil absent")
            self._record_access(
                user_id=user_id,
                username=data.get("username"),
                status="DENIED",
                score=None,
                reason="PROFILE_NOT_FOUND" if user_id else "NO_MATCH_FOUND",
                method="multimodal_scan",
                modalities={"telemetry": capture["telemetry"]},
            )
            self.client.publish(response_topic, json.dumps({"status": "fail", "reason": "PROFILE_NOT_FOUND" if user_id else "NO_MATCH_FOUND"}))
            return

        result = matched_user.get("prefetched_result") or verify_live_profile(capture["profile"], matched_user["profile"])
        if result["match"]:
            self.controller.handle_access_granted(matched_user["username"], result["score"])
            status = "GRANTED"
            reason = "MATCH"
        else:
            self.controller.handle_access_denied("Mismatch")
            status = "DENIED"
            reason = "BIOMETRIC_MISMATCH"

        event = self._record_access(
            user_id=matched_user["user_id"],
            username=matched_user["username"],
            status=status,
            score=result["score"],
            reason=reason,
            method="multimodal_scan",
            modalities={
                "components": result["components"],
                "biometric_key": result["live_profile"]["biometric_key"],
                "quality": result["live_profile"]["palmprint"].get("quality"),
                "quality_gate_passed": result.get("quality_gate_passed"),
                "quality_reason": result.get("quality_reason"),
                "telemetry": capture["telemetry"],
                "preview_path": capture["preview_path"],
            },
        )

        response = {
            "status": "success",
            "result": status,
            "user_id": matched_user["user_id"],
            "username": matched_user["username"],
            "biometric_key": result["live_profile"]["biometric_key"],
            "score": result["score"],
            "threshold": result["threshold"],
            "quality_gate_passed": result.get("quality_gate_passed"),
            "quality_reason": result.get("quality_reason"),
            "quality": result["live_profile"]["palmprint"].get("quality"),
            "components": result["components"],
            "event": event,
        }
        self.client.publish(response_topic, json.dumps(response))

    def handle_enroll_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("users/enroll", client_id)
        user_id = data.get("user_id") or database.generate_user_id()
        username = data.get("username", user_id or "unknown")
        role = data.get("role", "operator")
        department = data.get("department", "")
        password = data.get("password", "Temp1234!")

        self.controller.handle_enrollment(user_id)

        enrollment_frames = []
        preview_paths = []
        attempts = 0
        try:
            while len(enrollment_frames) < config.ENROLLMENT_SAMPLE_COUNT and attempts < config.ENROLLMENT_MAX_ATTEMPTS:
                attempts += 1
                sample_index = len(enrollment_frames) + 1
                self.publish_status(
                    "ONLINE",
                    phase="ENROLLMENT",
                    sample_index=sample_index,
                    sample_count=config.ENROLLMENT_SAMPLE_COUNT,
                    attempts=attempts,
                )
                self.controller.lcd.show_message(
                    f"Angle {sample_index}/{config.ENROLLMENT_SAMPLE_COUNT}",
                    user_id[: config.LCD_COLS],
                )
                try:
                    capture = self.controller.capture_attempt(claimed_user_id=f"{user_id}_{sample_index}")
                    enrollment_frames.append(capture["frame"])
                    if capture["preview_path"]:
                        preview_paths.append(capture["preview_path"])
                except Exception as exc:
                    logger.warning("Enrollment sample rejected for %s: %s", user_id, exc)
                    self.publish_status(
                        "ONLINE",
                        phase="ENROLLMENT_RETRY",
                        sample_index=sample_index,
                        sample_count=config.ENROLLMENT_SAMPLE_COUNT,
                        attempts=attempts,
                        reason=str(exc),
                    )
                    self.controller.lcd.show_message("Repositionne", "la main")
                time.sleep(0.4)
            if len(enrollment_frames) < config.ENROLLMENT_SAMPLE_COUNT:
                raise ValueError("Nombre d'echantillons valides insuffisant pour l'enrolement.")
            profile = build_enrollment_profile(enrollment_frames)
        except Exception as exc:
            self.client.publish(response_topic, json.dumps({"status": "error", "error": str(exc)}))
            return

        database.upsert_user(
            user_id=user_id,
            username=username,
            password_hash=generate_password_hash(password, method="pbkdf2:sha256"),
            role=role,
            department=department,
            email=data.get("email", ""),
        )
        database.save_biometric_profile(user_id, profile)
        self.firebase.save_user_profile(
            user_id,
            {
                "username": username,
                "email": data.get("email", ""),
                "role": role,
                "department": department,
                "device_id": config.DEVICE_ID,
            },
        )
        self.firebase.save_biometric_profile(user_id, profile)
        database.log_audit("INFO", "USER_ENROLLED", f"Profil multimodal cree pour {username}", user_id)
        self.publish_status("ONLINE", phase="ENROLLMENT_COMPLETED", enrolled_user_id=user_id)

        self.client.publish(
            response_topic,
            json.dumps(
                {
                    "status": "success",
                    "user_id": user_id,
                    "username": username,
                    "biometric_key": profile["biometric_key"],
                    "sample_count": profile.get("sample_count", 1),
                    "profile_modalities": profile["modalities"],
                    "profile": profile,
                    "preview_paths": preview_paths,
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

    def handle_user_update_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("users/update", client_id)
        user_id = data.get("user_id")

        if not user_id:
            self.client.publish(response_topic, json.dumps({"status": "error", "error": "Missing user_id"}))
            return

        existing = database.get_user_by_id(user_id)
        if not existing:
            self.client.publish(response_topic, json.dumps({"status": "error", "error": "USER_NOT_FOUND"}))
            return

        username = data.get("username", existing["username"])
        role = data.get("role", existing["role"])
        department = data.get("department", existing["department"])
        email = data.get("email", existing.get("email", ""))

        database.update_user(user_id, username=username, role=role, department=department, email=email)
        self.firebase.save_user_profile(
            user_id,
            {
                "username": username,
                "email": email,
                "role": role,
                "department": department,
                "device_id": config.DEVICE_ID,
            },
        )
        database.log_audit("INFO", "USER_UPDATED", f"Utilisateur {user_id} modifie", json.dumps(data))
        self.client.publish(
            response_topic,
            json.dumps(
                {
                    "status": "success",
                    "user": database.get_user_by_id(user_id),
                }
            ),
        )

    def handle_user_delete_command(self, data: dict) -> None:
        client_id = data.get("client_id", "anonymous")
        response_topic = config.response_topic("users/delete", client_id)
        user_id = data.get("user_id")

        if not user_id:
            self.client.publish(response_topic, json.dumps({"status": "error", "error": "Missing user_id"}))
            return

        existing = database.get_user_by_id(user_id)
        if not existing:
            self.client.publish(response_topic, json.dumps({"status": "error", "error": "USER_NOT_FOUND"}))
            return

        database.delete_user(user_id)
        self.firebase.delete_user_profile(user_id)
        self.firebase.delete_biometric_profile(user_id)
        database.log_audit("WARNING", "USER_DELETED", f"Utilisateur {user_id} supprime", user_id)
        self.client.publish(
            response_topic,
            json.dumps(
                {
                    "status": "success",
                    "deleted_user_id": user_id,
                }
            ),
        )

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

    def publish_status(self, status: str, **extra) -> None:
        payload = {
            "status": status,
            "device_id": config.DEVICE_ID,
            "app": config.APP_NAME,
            "local_ip": self.local_ip,
            "mqtt_port": config.MQTT_PORT,
            "mqtt_ws_port": config.MQTT_WS_PORT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        payload.update(extra)
        self.client.publish(config.MQTT_TOPIC_STATUS, json.dumps(payload))

    def publish_telemetry(self) -> None:
        payload = self.controller.sensor_snapshot(include_preview=True)
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
                time.sleep(config.TELEMETRY_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.client.loop_stop()
            self.controller.close()


VeinGuardMQTTGateway = BioGuardMQTTGateway


if __name__ == "__main__":
    gateway = BioGuardMQTTGateway()
    gateway.run()
