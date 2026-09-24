from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from dotenv import load_dotenv
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
DATABASE_PATH = Path(os.environ.get("DATABASE_PATH", "backend/audit_log.db"))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = PROJECT_ROOT / DATABASE_PATH


def _connect() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DATABASE_PATH)


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )


def log_prediction(image_hash: str, label: str, confidence: float) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO audit_log
                (timestamp, image_hash, predicted_label, confidence)
            VALUES (?, ?, ?, ?)
            """,
            (datetime.now(timezone.utc).isoformat(), image_hash, label, confidence),
        )
