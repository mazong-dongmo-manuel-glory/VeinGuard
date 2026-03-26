from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from werkzeug.security import generate_password_hash

import config


DB_PATH = Path(config.LOCAL_CACHE_DB)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'operator',
            department TEXT DEFAULT '',
            firebase_uid TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS biometric_profiles (
            user_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS access_events (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            username TEXT,
            status TEXT NOT NULL,
            score REAL,
            reason TEXT,
            method TEXT NOT NULL,
            modalities TEXT,
            synced INTEGER DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            meta TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS device_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute("SELECT 1 FROM users WHERE username = ?", ("admin@bioguard.local",))
    if not cur.fetchone():
        cur.execute(
            """
            INSERT INTO users (id, username, password_hash, role, department)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "admin-001",
                "admin@bioguard.local",
                generate_password_hash("Admin1234!", method="pbkdf2:sha256"),
                "admin",
                "Security",
            ),
        )

    cur.execute("SELECT COUNT(*) AS total FROM audit_logs")
    if cur.fetchone()["total"] == 0:
        cur.executemany(
            """
            INSERT INTO audit_logs (level, title, description, meta)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    "INFO",
                    "SYSTEM_BOOTSTRAP",
                    "Initialisation du cache edge et de la passerelle MQTT.",
                    config.DEVICE_ID,
                ),
                (
                    "INFO",
                    "PROJECT_PIVOT",
                    "Migration du projet vers la reconnaissance multimodale paume/doigts.",
                    config.APP_NAME,
                ),
            ],
        )

    conn.commit()
    conn.close()


def upsert_user(user_id: str, username: str, password_hash: str, role: str, department: str = "") -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO users (id, username, password_hash, role, department)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            username = excluded.username,
            password_hash = excluded.password_hash,
            role = excluded.role,
            department = excluded.department
        """,
        (user_id, username, password_hash, role, department),
    )
    conn.commit()
    conn.close()


def list_users() -> list[dict]:
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT u.id, u.username, u.role, u.department, u.created_at,
               CASE WHEN bp.user_id IS NULL THEN 0 ELSE 1 END AS has_biometrics
        FROM users u
        LEFT JOIN biometric_profiles bp ON bp.user_id = u.id
        ORDER BY u.created_at DESC
        """
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def save_biometric_profile(user_id: str, profile: dict) -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO biometric_profiles (user_id, profile_json, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            profile_json = excluded.profile_json,
            updated_at = CURRENT_TIMESTAMP
        """,
        (user_id, json.dumps(profile, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def get_biometric_profile(user_id: str) -> dict | None:
    conn = get_db_connection()
    row = conn.execute(
        "SELECT profile_json FROM biometric_profiles WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    conn.close()
    return json.loads(row["profile_json"]) if row else None


def log_access_event(
    event_id: str,
    user_id: str | None,
    username: str | None,
    status: str,
    score: float | None,
    reason: str,
    method: str,
    modalities: dict,
    synced: bool = False,
) -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO access_events (
            id, user_id, username, status, score, reason, method, modalities, synced
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            user_id,
            username,
            status,
            score,
            reason,
            method,
            json.dumps(modalities, ensure_ascii=False),
            int(synced),
        ),
    )
    conn.commit()
    conn.close()


def list_access_events(limit: int = 50) -> list[dict]:
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, user_id, username, status, score, reason, method, modalities, timestamp
        FROM access_events
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    events = []
    for row in rows:
        item = dict(row)
        item["modalities"] = json.loads(item["modalities"]) if item["modalities"] else {}
        events.append(item)
    return events


def list_audit_logs(limit: int = 50) -> list[dict]:
    conn = get_db_connection()
    rows = conn.execute(
        """
        SELECT id, level, title, description, meta, timestamp
        FROM audit_logs
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def log_audit(level: str, title: str, description: str, meta: str = "") -> None:
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO audit_logs (level, title, description, meta)
        VALUES (?, ?, ?, ?)
        """,
        (level, title, description, meta),
    )
    conn.commit()
    conn.close()


def update_device_state(key: str, value: dict | str) -> None:
    serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO device_state (key, value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = CURRENT_TIMESTAMP
        """,
        (key, serialized),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
