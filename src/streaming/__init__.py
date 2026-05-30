"""
Real streaming layer.

Two backends, one interface:

  - KafkaBackend       — real aiokafka producer + consumer talking to a broker
                         (single-node KRaft setup in docker-compose).
  - InprocBackend      — asyncio.Queue inside the FastAPI process; used as a
                         fallback when no Kafka broker is reachable, so local
                         dev / CI works without Docker.

The selection happens at startup: if STREAM_BACKEND env var is set, that
wins. Otherwise we probe the broker once and fall back gracefully.

The unit on the wire is a `StreamTxn` (see stream_types.py): a JSON-encoded
transaction event matching the columns our live scorer expects.

The consumer pipes each received event into `score_live_txn` (already exists
in src.live_scoring) and appends the scored result to an in-memory ring
buffer. The /api/stream/recent endpoint reads from that buffer.
"""

from .stream_types import StreamTxn, ScoredTxn
from .ingestor import StreamIngestor, get_ingestor

__all__ = ["StreamTxn", "ScoredTxn", "StreamIngestor", "get_ingestor"]
