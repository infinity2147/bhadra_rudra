"""
Detector threshold config — persisted in SQLite alongside cases.

Each detector has a small set of tuning knobs that a compliance team would
realistically want to adjust as their fraud landscape changes. We expose
them as a single config table with key/value rows so adding new knobs
doesn't require schema migrations.

Defaults match the values hardcoded in the detectors before this change,
so behaviour is unchanged on first run.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    # Circular detection
    "circular_amount_tolerance": 0.15,    # max within-cycle amount deviation
    "circular_min_total_flow": 100_000,
    "circular_max_cycle_length": 8,
    "circular_max_alerts": 50,
    # Layering
    "layering_min_chain_length": 3,
    "layering_max_chains": 200,
    "layering_decrease_ratio": 0.85,      # next amount must be >= ratio * prev
    # Smurfing
    "smurfing_threshold": 200_000,        # ₹2L reporting threshold
    "smurfing_cluster_tolerance": 0.10,
    # Shell funnel
    "funnel_imbalance_threshold": 0.7,
    "funnel_min_in_degree": 3,
    # Dormant activation
    "dormant_threshold_days": 30,
    "dormant_z_score_threshold": 2.5,
    # ML
    "ml_alert_threshold": 0.6,
}


class ConfigStore:
    """File-backed key/value config — uses the same SQLite as CaseStore."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
        self._seed_defaults()

    def _init_schema(self) -> None:
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS config (
                key        TEXT PRIMARY KEY,
                value_json TEXT,
                updated_at TEXT
            )
        """)
        self.conn.commit()

    def _seed_defaults(self) -> None:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM config")
        if c.fetchone()[0] > 0:
            return
        from datetime import datetime
        now = datetime.now().isoformat()
        for k, v in DEFAULT_CONFIG.items():
            c.execute(
                "INSERT OR IGNORE INTO config (key, value_json, updated_at) VALUES (?, ?, ?)",
                (k, json.dumps(v), now),
            )
        self.conn.commit()

    def get(self, key: str, default: Any = None) -> Any:
        c = self.conn.cursor()
        c.execute("SELECT value_json FROM config WHERE key = ?", (key,))
        row = c.fetchone()
        if row is None:
            return default if default is not None else DEFAULT_CONFIG.get(key)
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return default

    def get_all(self) -> Dict[str, Any]:
        c = self.conn.cursor()
        c.execute("SELECT key, value_json FROM config")
        out = dict(DEFAULT_CONFIG)
        for row in c.fetchall():
            try:
                out[row["key"]] = json.loads(row["value_json"])
            except json.JSONDecodeError:
                pass
        return out

    def set(self, key: str, value: Any) -> None:
        from datetime import datetime
        c = self.conn.cursor()
        c.execute(
            """
            INSERT INTO config (key, value_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json,
                                            updated_at = excluded.updated_at
            """,
            (key, json.dumps(value), datetime.now().isoformat()),
        )
        self.conn.commit()

    def set_many(self, values: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in values.items():
            if k not in DEFAULT_CONFIG:
                # Reject unknown keys — prevents arbitrary writes
                raise ValueError(f"Unknown config key: {k}")
            self.set(k, v)
        return self.get_all()

    def reset(self) -> Dict[str, Any]:
        c = self.conn.cursor()
        c.execute("DELETE FROM config")
        self.conn.commit()
        self._seed_defaults()
        return self.get_all()
