from __future__ import annotations

import logging
import time

import config

logger = logging.getLogger(__name__)

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except ImportError:
    firebase_admin = None
    credentials = None
    firestore = None


class FirebaseService:
    def __init__(self):
        self.enabled = False
        self.db = None
        self._quota_cooldown_until = 0.0
        self._last_telemetry_sync_at: dict[str, float] = {}

        if (
            not config.FIREBASE_ENABLED
            or firebase_admin is None
            or credentials is None
            or firestore is None
            or not config.FIREBASE_CREDENTIALS.exists()
        ):
            logger.info("Firebase disabled or not configured; using local edge cache only.")
            return

        try:
            app = firebase_admin.get_app()
        except ValueError:
            app = firebase_admin.initialize_app(
                credentials.Certificate(str(config.FIREBASE_CREDENTIALS)),
                {
                    "storageBucket": config.FIREBASE_STORAGE_BUCKET or None,
                    "projectId": config.FIREBASE_PROJECT_ID or None,
                },
            )

        self.db = firestore.client(app=app)
        self.enabled = True
        logger.info("Firebase initialised")

    def _is_quota_error(self, error: Exception) -> bool:
        raw = f"{getattr(error, 'code', '')} {error}".lower()
        return "resource_exhausted" in raw or "resource-exhausted" in raw or "quota exceeded" in raw

    def _enter_quota_cooldown(self) -> None:
        self._quota_cooldown_until = time.monotonic() + max(1.0, config.FIREBASE_QUOTA_COOLDOWN_SECONDS)
        logger.warning(
            "Firestore quota reached; suspending cloud sync for %.0f seconds.",
            config.FIREBASE_QUOTA_COOLDOWN_SECONDS,
        )

    def _is_in_quota_cooldown(self) -> bool:
        return self._quota_cooldown_until > time.monotonic()

    def _sanitize_payload(self, payload):
        if isinstance(payload, list):
            return [self._sanitize_payload(item) for item in payload]

        if not isinstance(payload, dict):
            return payload

        sanitized = {}
        for key, value in payload.items():
            if "base64" in str(key).lower():
                continue
            sanitized[key] = self._sanitize_payload(value)
        return sanitized

    def _write_document(self, collection_name: str, document_id: str, payload: dict) -> bool:
        if not self.enabled or self._is_in_quota_cooldown() or not document_id:
            return False

        try:
            self.db.collection(collection_name).document(document_id).set(
                self._sanitize_payload(payload),
                merge=True,
            )
            return True
        except Exception as error:
            if self._is_quota_error(error):
                self._enter_quota_cooldown()
                return False
            raise

    def save_user_profile(self, user_id: str, payload: dict) -> bool:
        return self._write_document(config.FIREBASE_USERS_COLLECTION, user_id, payload)

    def delete_user_profile(self, user_id: str) -> bool:
        if not self.enabled or self._is_in_quota_cooldown() or not user_id:
            return False
        try:
            self.db.collection(config.FIREBASE_USERS_COLLECTION).document(user_id).delete()
            return True
        except Exception as error:
            if self._is_quota_error(error):
                self._enter_quota_cooldown()
                return False
            raise

    def save_biometric_profile(self, user_id: str, payload: dict) -> bool:
        return self._write_document(config.FIREBASE_BIOMETRICS_COLLECTION, user_id, payload)

    def delete_biometric_profile(self, user_id: str) -> bool:
        if not self.enabled or self._is_in_quota_cooldown() or not user_id:
            return False
        try:
            self.db.collection(config.FIREBASE_BIOMETRICS_COLLECTION).document(user_id).delete()
            return True
        except Exception as error:
            if self._is_quota_error(error):
                self._enter_quota_cooldown()
                return False
            raise

    def get_biometric_profile(self, user_id: str) -> dict | None:
        if not self.enabled or self._is_in_quota_cooldown() or not user_id:
            return None
        try:
            doc = self.db.collection(config.FIREBASE_BIOMETRICS_COLLECTION).document(user_id).get()
            return doc.to_dict() if doc.exists else None
        except Exception as error:
            if self._is_quota_error(error):
                self._enter_quota_cooldown()
                return None
            raise

    def save_access_event(self, event_id: str, payload: dict) -> bool:
        return self._write_document(config.FIREBASE_EVENTS_COLLECTION, event_id, payload)

    def save_telemetry(self, device_id: str, payload: dict) -> bool:
        if not self.enabled or self._is_in_quota_cooldown():
            return False

        now = time.monotonic()
        last_synced_at = self._last_telemetry_sync_at.get(device_id, 0.0)
        if now - last_synced_at < max(1.0, config.FIREBASE_TELEMETRY_SYNC_INTERVAL_SECONDS):
            return False

        persisted = self._write_document(config.FIREBASE_TELEMETRY_COLLECTION, device_id, payload)
        if persisted:
            self._last_telemetry_sync_at[device_id] = now
        return persisted
