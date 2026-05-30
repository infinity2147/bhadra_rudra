"""
Pathway windowed analytics — real stream processing on top of Kafka.

Why Pathway alongside Kafka?
    Kafka is the bus; Pathway is the engine. In a real PSB deployment the
    same Kafka topic that feeds ML scoring also feeds a stream processor that
    does *windowed* aggregations the per-event scorer can't see — things like
    "total fund volume per entity over the last 5 minutes," which is what
    catches velocity-based AML patterns (rapid layering, burst behaviour).

What this module does:
    Subscribes to KAFKA_TOPIC, treats each event as (sender_id, receiver_id,
    amount, timestamp). For each entity, computes a 5-minute sliding window
    (1-minute hop) of outgoing volume + transaction count. Emits a velocity
    alert when an entity's window-volume crosses VELOCITY_THRESHOLD_INR.

Output destinations:
    - Default: writes alerts as JSONL to data/streaming/velocity_alerts.jsonl
      (the backend tails this file via /api/stream/velocity_alerts).
    - Optionally publishes back to KAFKA_ALERT_TOPIC if PATHWAY_PUBLISH_KAFKA=1.

Runtime model:
    Pathway runs as a long-lived process. Two ways to use it:

      1. CLI (recommended for demo — judges see it as a separate worker):
            python -m src.streaming.pathway_engine
         Logs roll to stdout; alerts go to data/streaming/velocity_alerts.jsonl.

      2. Background thread, opt-in from the FastAPI startup:
            from streaming.pathway_engine import start_in_thread
            start_in_thread()
         Runs the Pathway dataflow in a daemon thread inside the backend
         process. Disabled by default (PATHWAY_INPROC=1 to enable) because
         Pathway grabs significant resources and one-process demos do fine
         without it.

If Pathway isn't installed (`pip install pathway`), this module imports
without error but every public entrypoint raises ImportError on call.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


VELOCITY_THRESHOLD_INR = float(os.getenv("VELOCITY_THRESHOLD_INR", "2000000"))   # ₹20L in 5 min
WINDOW_DURATION_MS = int(os.getenv("VELOCITY_WINDOW_MS", str(5 * 60 * 1000)))
WINDOW_HOP_MS = int(os.getenv("VELOCITY_HOP_MS", str(60 * 1000)))                 # 1-min hop

ALERTS_PATH = Path(
    os.getenv("PATHWAY_ALERTS_PATH", "data/streaming/velocity_alerts.jsonl")
)


def _import_pathway():
    try:
        import pathway as pw
        return pw
    except ImportError as e:
        raise ImportError(
            "Pathway is not installed. Install with:\n"
            "    pip install pathway\n\n"
            f"Original error: {e}"
        )


def _build_dataflow(pw):
    """Construct the Pathway dataflow graph.

    Returns the pw graph and the entry table — caller owns the run loop.
    """
    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic = os.getenv("KAFKA_TOPIC", "rudra.transactions")

    # Schema must match the JSON we're putting on the bus from src.streaming.kafka_producer
    class TxnSchema(pw.Schema):
        transaction_id: str
        sender_id: str
        receiver_id: str
        amount: float
        timestamp: str
        channel: str
        transaction_type: str
        currency: str

    rdkafka_settings = {
        "bootstrap.servers": bootstrap,
        "group.id": "rudra-pathway",
        "auto.offset.reset": "latest",
    }

    raw = pw.io.kafka.read(
        rdkafka_settings,
        topic=topic,
        schema=TxnSchema,
        format="json",
        autocommit_duration_ms=1000,
    )

    # Parse timestamp into a Pathway-compatible event time (ms since epoch).
    parsed = raw.select(
        sender_id=pw.this.sender_id,
        receiver_id=pw.this.receiver_id,
        amount=pw.this.amount,
        currency=pw.this.currency,
        # naive parse — assumes ISO-8601; in prod use pw.utils.col.parse_datetime
        ts_ms=(pw.this.timestamp.str.slice(0, 19) + "Z").dt.strptime("%Y-%m-%dT%H:%M:%SZ").dt.timestamp() * 1000,
    )

    # Sliding window per sender — 5-min window, 1-min hop
    windowed = parsed.windowby(
        parsed.ts_ms,
        window=pw.temporal.sliding(duration=WINDOW_DURATION_MS, hop=WINDOW_HOP_MS),
        instance=parsed.sender_id,
    ).reduce(
        sender_id=pw.this._pw_instance,
        window_start_ms=pw.this._pw_window_start,
        window_end_ms=pw.this._pw_window_end,
        total_amount=pw.reducers.sum(pw.this.amount),
        txn_count=pw.reducers.count(),
    )

    # Filter: emit only the windows where volume exceeds the velocity threshold.
    alerts = windowed.filter(pw.this.total_amount >= VELOCITY_THRESHOLD_INR)
    return alerts


def run(blocking: bool = True):
    """Start the Pathway dataflow.

    blocking=True (the default) blocks the current thread on `pw.run()`. The
    CLI uses this. The background-thread variant calls run(blocking=True) on
    its own thread.
    """
    pw = _import_pathway()
    alerts = _build_dataflow(pw)

    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write alerts as JSONL so the backend can tail the file.
    pw.io.jsonlines.write(alerts, str(ALERTS_PATH))

    publish_kafka = os.getenv("PATHWAY_PUBLISH_KAFKA", "").lower() in ("1", "true", "yes")
    if publish_kafka:
        out_topic = os.getenv("KAFKA_ALERT_TOPIC", "rudra.velocity_alerts")
        bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        pw.io.kafka.write(
            alerts,
            rdkafka_settings={"bootstrap.servers": bootstrap},
            topic_name=out_topic,
            format="json",
        )
        logger.info("Pathway also publishing velocity alerts to Kafka topic '%s'", out_topic)

    logger.info("Pathway dataflow started. Velocity threshold: ₹%.0f over %dm window",
                VELOCITY_THRESHOLD_INR, WINDOW_DURATION_MS // 60_000)
    pw.run()


def start_in_thread() -> threading.Thread:
    """Start the dataflow on a daemon thread. Returns the thread."""
    t = threading.Thread(target=lambda: run(blocking=True), daemon=True, name="pathway-engine")
    t.start()
    return t


def read_recent_alerts(limit: int = 50) -> list:
    """Tail the velocity_alerts.jsonl file the dataflow writes to.

    The backend exposes this via /api/stream/velocity_alerts so the UI can
    display velocity alerts without depending on Pathway being importable.
    """
    if not ALERTS_PATH.exists():
        return []
    out = []
    with open(ALERTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    out.reverse()
    return out[:limit]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [pathway] %(message)s")
    run(blocking=True)
