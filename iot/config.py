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
MOCK_MODE = os.getenv("VG_MOCK_MODE", "0").lower() in {"1", "true", "yes", "on"}

# MQTT
MQTT_BROKER = os.getenv("VG_MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("VG_MQTT_PORT", "1883"))
MQTT_WS_PORT = int(os.getenv("VG_MQTT_WS_PORT", "9090"))
MQTT_KEEPALIVE = int(os.getenv("VG_MQTT_KEEPALIVE", "60"))
MQTT_TOPIC_PREFIX = os.getenv("VG_TOPIC_PREFIX", "bioguard")
MQTT_USERNAME = os.getenv("VG_MQTT_USERNAME", "admin").strip()
MQTT_PASSWORD = os.getenv("VG_MQTT_PASSWORD", "admin1234")


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
MQTT_CMD_USERS_UPDATE = topic("cmd/users/update")
MQTT_CMD_USERS_DELETE = topic("cmd/users/delete")
MQTT_CMD_LOGS = topic("cmd/access/logs")
MQTT_CMD_AUDIT = topic("cmd/audit/list")
MQTT_CMD_SETTINGS = topic("cmd/settings/update")
MQTT_CMD_PING = topic("cmd/ping")
MQTT_CMD_CAMERA_PREVIEW = topic("cmd/camera/preview")


def response_topic(command: str, client_id: str) -> str:
    return topic(f"res/{command.strip('/')}/{client_id}")


# GPIO BCM pins
PIN_LED_GREEN = int(os.getenv("VG_PIN_LED_GREEN", "16"))
PIN_LED_RED = int(os.getenv("VG_PIN_LED_RED", "22"))
PIN_BUZZER = int(os.getenv("VG_PIN_BUZZER", "23"))
PIN_LIGHT_SENSOR = int(os.getenv("VG_PIN_LIGHT_SENSOR", "4"))
PIN_LIGHT_LED_1 = int(os.getenv("VG_PIN_LIGHT_LED_1", "17"))
PIN_LIGHT_LED_2 = int(os.getenv("VG_PIN_LIGHT_LED_2", "27"))

# LCD
LCD_I2C_ADDRESS = int(os.getenv("VG_LCD_I2C_ADDRESS", "0x27"), 16)
LCD_PORT = int(os.getenv("VG_LCD_PORT", "1"))
LCD_COLS = int(os.getenv("VG_LCD_COLS", "16"))
LCD_ROWS = int(os.getenv("VG_LCD_ROWS", "2"))

# Lighting automation
LIGHT_SENSOR_TIMEOUT = float(os.getenv("VG_LIGHT_SENSOR_TIMEOUT", "0.5"))
LIGHT_SENSOR_SAMPLES = int(os.getenv("VG_LIGHT_SENSOR_SAMPLES", "5"))
LIGHT_SENSOR_DARK_RATIO = float(os.getenv("VG_LIGHT_SENSOR_DARK_RATIO", "1.25"))

# Biometrics
MATCH_THRESHOLD = float(os.getenv("VG_MATCH_THRESHOLD", "0.44"))
MIN_HAND_AREA = int(os.getenv("VG_MIN_HAND_AREA", "3500"))
CAMERA_WIDTH = int(os.getenv("VG_CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("VG_CAMERA_HEIGHT", "480"))
CAMERA_PREVIEW_WIDTH = int(os.getenv("VG_CAMERA_PREVIEW_WIDTH", "320"))
CAMERA_PREVIEW_QUALITY = int(os.getenv("VG_CAMERA_PREVIEW_QUALITY", "60"))
CAMERA_FRAME_DURATION_US = int(os.getenv("VG_CAMERA_FRAME_DURATION_US", "40000"))
NOIR_CAMERA_BRIGHTNESS = float(os.getenv("VG_NOIR_CAMERA_BRIGHTNESS", "-0.05"))
NOIR_CAMERA_CONTRAST = float(os.getenv("VG_NOIR_CAMERA_CONTRAST", "1.25"))
NOIR_CAMERA_SHARPNESS = float(os.getenv("VG_NOIR_CAMERA_SHARPNESS", "1.6"))
NOIR_CAMERA_EXPOSURE_VALUE = float(os.getenv("VG_NOIR_CAMERA_EXPOSURE_VALUE", "0.35"))
NOIR_CAMERA_WARMUP_SECONDS = float(os.getenv("VG_NOIR_CAMERA_WARMUP_SECONDS", "0.35"))
SCAN_PREVIEW_SECONDS = float(os.getenv("VG_SCAN_PREVIEW_SECONDS", "12"))
ENROLLMENT_PREVIEW_SECONDS = float(os.getenv("VG_ENROLLMENT_PREVIEW_SECONDS", "90"))
ENROLLMENT_SAMPLE_COUNT = int(os.getenv("VG_ENROLLMENT_SAMPLE_COUNT", "5"))
ENROLLMENT_MAX_ATTEMPTS = int(os.getenv("VG_ENROLLMENT_MAX_ATTEMPTS", "12"))
IDENTIFICATION_SAMPLE_COUNT = int(os.getenv("VG_IDENTIFICATION_SAMPLE_COUNT", "3"))
TELEMETRY_INTERVAL_SECONDS = float(os.getenv("VG_TELEMETRY_INTERVAL_SECONDS", "1.0"))
NOIR_CLAHE_CLIP_LIMIT = float(os.getenv("VG_NOIR_CLAHE_CLIP_LIMIT", "2.8"))
NOIR_CLAHE_GRID_SIZE = int(os.getenv("VG_NOIR_CLAHE_GRID_SIZE", "8"))
NOIR_BLACKHAT_SMALL = int(os.getenv("VG_NOIR_BLACKHAT_SMALL", "11"))
NOIR_BLACKHAT_LARGE = int(os.getenv("VG_NOIR_BLACKHAT_LARGE", "19"))
NOIR_ADAPTIVE_BLOCK_SIZE = int(os.getenv("VG_NOIR_ADAPTIVE_BLOCK_SIZE", "25"))
NOIR_ADAPTIVE_C = int(os.getenv("VG_NOIR_ADAPTIVE_C", "4"))
ORB_FEATURE_COUNT = int(os.getenv("VG_ORB_FEATURE_COUNT", "256"))
ORB_DESCRIPTOR_LIMIT = int(os.getenv("VG_ORB_DESCRIPTOR_LIMIT", "96"))
MIN_MASK_FILL_RATIO = float(os.getenv("VG_MIN_MASK_FILL_RATIO", "0.10"))
MAX_MASK_FILL_RATIO = float(os.getenv("VG_MAX_MASK_FILL_RATIO", "0.72"))
MAX_HAND_AREA_RATIO = float(os.getenv("VG_MAX_HAND_AREA_RATIO", "0.72"))
MIN_HAND_EXTENT = float(os.getenv("VG_MIN_HAND_EXTENT", "0.20"))
MIN_HAND_SOLIDITY = float(os.getenv("VG_MIN_HAND_SOLIDITY", "0.35"))
MAX_HAND_SOLIDITY = float(os.getenv("VG_MAX_HAND_SOLIDITY", "0.98"))
MIN_HAND_ASPECT_RATIO = float(os.getenv("VG_MIN_HAND_ASPECT_RATIO", "0.45"))
MAX_HAND_ASPECT_RATIO = float(os.getenv("VG_MAX_HAND_ASPECT_RATIO", "1.65"))
MAX_HAND_CENTER_DISTANCE = float(os.getenv("VG_MAX_HAND_CENTER_DISTANCE", "0.34"))
MAX_BORDER_TOUCHES = int(os.getenv("VG_MAX_BORDER_TOUCHES", "2"))
MIN_ORB_KEYPOINTS = int(os.getenv("VG_MIN_ORB_KEYPOINTS", "18"))
MIN_SHARPNESS = float(os.getenv("VG_MIN_SHARPNESS", "28.0"))
MIN_CAPTURE_QUALITY = float(os.getenv("VG_MIN_CAPTURE_QUALITY", "0.32"))
ENROLLMENT_MIN_MASK_FILL_RATIO = float(os.getenv("VG_ENROLLMENT_MIN_MASK_FILL_RATIO", "0.08"))
ENROLLMENT_MAX_HAND_CENTER_DISTANCE = float(os.getenv("VG_ENROLLMENT_MAX_HAND_CENTER_DISTANCE", "0.40"))
ENROLLMENT_MAX_BORDER_TOUCHES = int(os.getenv("VG_ENROLLMENT_MAX_BORDER_TOUCHES", "3"))
ENROLLMENT_MIN_ORB_KEYPOINTS = int(os.getenv("VG_ENROLLMENT_MIN_ORB_KEYPOINTS", "12"))
ENROLLMENT_MIN_SHARPNESS = float(os.getenv("VG_ENROLLMENT_MIN_SHARPNESS", "18.0"))
ENROLLMENT_MIN_CAPTURE_QUALITY = float(os.getenv("VG_ENROLLMENT_MIN_CAPTURE_QUALITY", "0.24"))

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
