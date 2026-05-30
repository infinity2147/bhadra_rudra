"""Shared data shapes for the streaming layer."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class StreamTxn:
    """One transaction event on the wire.

    Mirrors the columns score_live_txn needs. Stored as plain JSON over Kafka
    (no Avro/schema-registry to keep the demo dep footprint small — in prod
    you'd swap this for an Avro or protobuf encoder behind a Schema Registry).
    """
    transaction_id: str
    sender_id: str
    receiver_id: str
    amount: float
    timestamp: str               # ISO-8601 string; consumer parses to pd.Timestamp
    channel: str = "NetBanking"
    transaction_type: str = "NEFT"
    currency: str = "INR"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScoredTxn:
    """One transaction after it has been through the live ML model."""
    txn: StreamTxn
    ml_score: Optional[float]
    latency_ms: Optional[Dict[str, float]]
    severity: Optional[str]                 # MEDIUM / HIGH / CRITICAL based on score
    received_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["txn"] = self.txn.to_dict()
        return d
