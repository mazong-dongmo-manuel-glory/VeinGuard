from __future__ import annotations

import logging

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

    def save_user_profile(self, user_id: str, payload: dict) -> bool:
        if not self.enabled:
            return False
        self.db.collection(config.FIREBASE_USERS_COLLECTION).document(user_id).set(payload, merge=True)
        return True

    def save_biometric_profile(self, user_id: str, payload: dict) -> bool:
        if not self.enabled:
            return False
        self.db.collection(config.FIREBASE_BIOMETRICS_COLLECTION).document(user_id).set(payload, merge=True)
        return True

    def get_biometric_profile(self, user_id: str) -> dict | None:
        if not self.enabled:
            return None
        doc = self.db.collection(config.FIREBASE_BIOMETRICS_COLLECTION).document(user_id).get()
        return doc.to_dict() if doc.exists else None

    def save_access_event(self, event_id: str, payload: dict) -> bool:
        if not self.enabled:
            return False
        self.db.collection(config.FIREBASE_EVENTS_COLLECTION).document(event_id).set(payload, merge=True)
        return True

    def save_telemetry(self, device_id: str, payload: dict) -> bool:
        if not self.enabled:
            return False
        self.db.collection(config.FIREBASE_TELEMETRY_COLLECTION).document(device_id).set(payload, merge=True)
        return True

