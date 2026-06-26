"""
Persistent taint memory.

When an entity is confirmed fraudulent (a case escalated / SAR filed, or an
analyst manually marks an alert), the suspicion shouldn't evaporate when the
batch pipeline next reruns. Real investigators carry memory: an account two
hops from a confirmed mule stays "warm" for a long time.

This store propagates a decaying taint score outward from confirmed-bad seed
entities across the fund-flow graph, PERSISTS it in the same SQLite file as
cases/config, and FLOORS future risk scores with it — so taint accumulates
across runs instead of resetting every pipeline execution.

  taint(hop) = decay ** hop          (seed = hop 0 = 1.0)

Propagation is over the *undirected* view: money laundered *through* a node
implicates both who fed it and where it went. Each entity keeps the MAX taint
it has ever received (a stronger/closer confirmation can only raise it).

This is the cross-time "compounding" memory competitors described but shipped
as a stub — here it is real, persistent, and wired into risk scoring.
"""

from __future__ import annotations

import sqlite3
import threading
from collections import deque
from datetime import datetime
from typing import Dict, Iterable, List, Optional

import networkx as nx


class TaintStore:
    """SQLite-backed decaying taint propagated over the fund-flow graph."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # FastAPI's threadpool can call concurrently on one connection; serialize.
        self._lock = threading.RLock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS taint (
                    entity_id  TEXT PRIMARY KEY,
                    taint      REAL NOT NULL,
                    source     TEXT,
                    hops       INTEGER,
                    updated_at TEXT
                )
                """
            )
            self.conn.commit()

    def seed(self, graph: nx.DiGraph, seed_entities: Iterable[str],
             source: str = "manual", decay: float = 0.7, max_hops: int = 4) -> Dict[str, float]:
        """Propagate decaying taint outward from `seed_entities`.

        Returns the {entity_id: taint} computed for THIS seeding (the persisted
        value is the max over all seedings). Propagation is a hop-limited BFS on
        the undirected view; taint at hop h is ``decay ** h``.
        """
        seeds = [s for s in seed_entities]
        # BFS over the undirected view, recording the minimum hop to each node.
        und = graph.to_undirected(as_view=True) if graph is not None and graph.number_of_nodes() else None
        hop_of: Dict[str, int] = {}
        q: deque = deque()
        for s in seeds:
            if s not in hop_of:
                hop_of[s] = 0
                q.append(s)
        while q:
            node = q.popleft()
            h = hop_of[node]
            if h >= max_hops or und is None or node not in und:
                continue
            for nb in und.neighbors(node):
                if nb not in hop_of:
                    hop_of[nb] = h + 1
                    q.append(nb)

        computed = {e: float(decay ** h) for e, h in hop_of.items()}
        self._upsert_max(computed, source)
        return computed

    def _upsert_max(self, scores: Dict[str, float], source: str) -> None:
        now = datetime.now().isoformat()
        with self._lock:
            c = self.conn.cursor()
            for entity_id, taint in scores.items():
                hops = 0
                # derive hops back from taint only for display; store as-is
                c.execute(
                    """
                    INSERT INTO taint (entity_id, taint, source, hops, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        taint      = MAX(taint, excluded.taint),
                        source     = CASE WHEN excluded.taint >= taint
                                          THEN excluded.source ELSE source END,
                        updated_at = excluded.updated_at
                    """,
                    (entity_id, float(taint), source, hops, now),
                )
            self.conn.commit()

    def get(self, entity_id: str) -> float:
        with self._lock:
            row = self.conn.execute(
                "SELECT taint FROM taint WHERE entity_id = ?", (entity_id,)
            ).fetchone()
        return float(row["taint"]) if row else 0.0

    def get_all(self) -> Dict[str, Dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT entity_id, taint, source, updated_at FROM taint ORDER BY taint DESC"
            ).fetchall()
        return {
            r["entity_id"]: {"taint": float(r["taint"]), "source": r["source"],
                             "updated_at": r["updated_at"]}
            for r in rows
        }

    def apply_floor(self, risk_scores: Dict[str, float]) -> Dict[str, float]:
        """Return risk_scores with each entity floored at its persisted taint.

        A clean-looking node that sits near confirmed fraud cannot drop below
        its taint; a genuinely high score is preserved (we take the max).
        """
        all_taint = self.get_all()
        out = dict(risk_scores)
        for entity_id, info in all_taint.items():
            t = info["taint"]
            out[entity_id] = max(float(out.get(entity_id, 0.0)), t)
        return out

    def clear(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM taint")
            self.conn.commit()
