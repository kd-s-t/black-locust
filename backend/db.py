"""SQLite persistence for dataset uploads."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "black-locust.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                folder TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL,
                task TEXT NOT NULL DEFAULT 'unknown',
                has_data_yaml INTEGER NOT NULL DEFAULT 0,
                data_yaml TEXT,
                ready INTEGER NOT NULL DEFAULT 0,
                original_filename TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_dataset(row: Dict[str, Any]) -> Dict[str, Any]:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO datasets (
                id, name, folder, path, task, has_data_yaml, data_yaml,
                ready, original_filename, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["id"],
                row["name"],
                row["folder"],
                row["path"],
                row["task"],
                1 if row.get("has_data_yaml") else 0,
                row.get("data_yaml"),
                1 if row.get("ready") else 0,
                row.get("original_filename"),
                row["created_at"],
            ),
        )
        conn.commit()
    return get_dataset(row["id"]) or row


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        row = cur.fetchone()
    return _row_to_dict(row) if row else None


def get_dataset_by_folder(folder: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM datasets WHERE folder = ?", (folder,))
        row = cur.fetchone()
    return _row_to_dict(row) if row else None


def delete_dataset(dataset_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        conn.commit()


def list_datasets() -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC")
        rows = cur.fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "folder": row["folder"],
        "path": row["path"],
        "task": row["task"],
        "has_data_yaml": bool(row["has_data_yaml"]),
        "data_yaml": row["data_yaml"],
        "ready": bool(row["ready"]),
        "original_filename": row["original_filename"],
        "created_at": row["created_at"],
    }
