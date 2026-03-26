from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CAPTURE_DIR = DATA_DIR / "captures"
TEMPLATE_FILE = DATA_DIR / "biometric_templates.json"
LOCAL_CACHE_DB = BASE_DIR / "veinguard.db"

for path in (DATA_DIR, CAPTURE_DIR):
    path.mkdir(parents=True, exist_ok=True)


APP_NAME = os.getenv("VG_APP_NAME", "BioGuard Access")
APP_SHORT_NAME = os.getenv("VG_APP_SHORT_NAME", "BioGuard")
DEVICE_ID = os.getenv("VG_DEVICE_ID", "rpi-entry-01")
MOCK_MODE = os.getenv("VG_MOCK_MODE", "1").lower() in {"1", "true", "yes", "on"}

# MQTT
MQTT_BROKER = os.getenv("VG_MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("VG_MQTT_PORT", "1883"))
MQTT_KEEPALIVE = int(os.getenv("VG_MQTT_KEEPALIVE", "60"))
MQTT_TOPIC_PREFIX = os.getenv("VG_TOPIC_PREFIX", "bioguard")


def topic(path: str) -> str:
    return f"{MQTT_TOPIC_PREFIX}/{path.strip('/')}"


MQTT_TOPIC_CMD_ALL = topic("cmd/#")
MQTT_TOPIC_STATUS = topic("status")
MQTT_TOPIC_TELEMETRY = topic("telemetry")
MQTT_TOPIC_EVENTS = topic("events")
MQTT_TOPIC_LOGS = topic("logs")

MQTT_CMD_SCAN = topic("cmd/access/scan")
MQTT_CMD_ENROLL = topic("cmd/users/enroll")
MQTT_CMD_LOGIN = topic("cmd/auth/login")
MQTT_CMD_USERS = topic("cmd/users/list")
MQTT_CMD_LOGS = topic("cmd/access/logs")
MQTT_CMD_AUDIT = topic("cmd/audit/list")
MQTT_CMD_SETTINGS = topic("cmd/settings/update")
MQTT_CMD_PING = topic("cmd/ping")


def response_topic(command: str, client_id: str) -> str:
    return topic(f"res/{command.strip('/')}/{client_id}")


# GPIO BCM pins
PIN_LED_GREEN = int(os.getenv("VG_PIN_LED_GREEN", "16"))
PIN_LED_RED = int(os.getenv("VG_PIN_LED_RED", "22"))
PIN_BUZZER = int(os.getenv("VG_PIN_BUZZER", "23"))
PIN_TOUCH = int(os.getenv("VG_PIN_TOUCH", "4"))
PIN_DISTANCE_TRIGGER = int(os.getenv("VG_PIN_DISTANCE_TRIGGER", "24"))
PIN_DISTANCE_ECHO = int(os.getenv("VG_PIN_DISTANCE_ECHO", "25"))
PIN_MOTION = int(os.getenv("VG_PIN_MOTION", "18"))

# LCD
LCD_I2C_ADDRESS = int(os.getenv("VG_LCD_I2C_ADDRESS", "0x27"), 16)
LCD_PORT = int(os.getenv("VG_LCD_PORT", "1"))
LCD_COLS = int(os.getenv("VG_LCD_COLS", "16"))
LCD_ROWS = int(os.getenv("VG_LCD_ROWS", "2"))

# Biometrics
MATCH_THRESHOLD = float(os.getenv("VG_MATCH_THRESHOLD", "0.33"))
MIN_HAND_AREA = int(os.getenv("VG_MIN_HAND_AREA", "3500"))
CAMERA_WIDTH = int(os.getenv("VG_CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("VG_CAMERA_HEIGHT", "480"))

# Firebase
FIREBASE_ENABLED = os.getenv("VG_FIREBASE_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
FIREBASE_CREDENTIALS = Path(
    os.getenv("VG_FIREBASE_CREDENTIALS", str(BASE_DIR / "firebase-service-account.json"))
)
FIREBASE_STORAGE_BUCKET = os.getenv("VG_FIREBASE_STORAGE_BUCKET", "")
FIREBASE_PROJECT_ID = os.getenv("VG_FIREBASE_PROJECT_ID", "")
FIREBASE_USERS_COLLECTION = os.getenv("VG_FIREBASE_USERS_COLLECTION", "users")
FIREBASE_BIOMETRICS_COLLECTION = os.getenv("VG_FIREBASE_BIOMETRICS_COLLECTION", "biometric_profiles")
FIREBASE_EVENTS_COLLECTION = os.getenv("VG_FIREBASE_EVENTS_COLLECTION", "access_events")
FIREBASE_TELEMETRY_COLLECTION = os.getenv("VG_FIREBASE_TELEMETRY_COLLECTION", "device_telemetry")

