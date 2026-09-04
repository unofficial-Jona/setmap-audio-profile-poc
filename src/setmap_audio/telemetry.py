from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Telemetry:
    """Tiny local event sink. Never records filenames, audio, vectors, IPs, or user agents."""

    def __init__(self, data_dir: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = data_dir / "telemetry.sqlite3"
        self._lock = threading.Lock()
        if enabled:
            data_dir.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL NOT NULL,
                        job_id TEXT NOT NULL,
                        event_name TEXT NOT NULL,
                        duration_ms INTEGER,
                        properties_json TEXT NOT NULL
                    )
                    """
                )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def record(
        self,
        job_id: str,
        event_name: str,
        *,
        duration_ms: int | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload = json.dumps(properties or {}, separators=(",", ":"), sort_keys=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO events(created_at, job_id, event_name, duration_ms, properties_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (time.time(), job_id, event_name, duration_ms, payload),
            )

