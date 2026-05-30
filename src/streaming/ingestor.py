"""
Stream ingestion — Kafka with in-process fallback.

The class shipped from here is `StreamIngestor`. It owns:
  - an asyncio Task running the consumer loop (Kafka or in-process)
  - an in-memory ring buffer of the most recent scored events
  - lifecycle: start() / stop() / status()

backend/main.py creates one StreamIngestor at startup and binds the
already-loaded graph + ML bundle so the consumer can score each event
through src.live_scoring.score_live_txn (which is the same function the
batch and live-inject endpoints use — no separate "fast path").

Selection of Kafka vs in-process is driven by:
  STREAM_BACKEND=kafka     → require Kafka, fail loud if unreachable
  STREAM_BACKEND=inproc    → asyncio.Queue only
  STREAM_BACKEND=auto      → probe Kafka once, fall back to inproc on TimeoutError
  (default = auto)

KAFKA_BOOTSTRAP_SERVERS env var sets the broker (default localhost:9092).
KAFKA_TOPIC env var sets the topic (default rudra.transactions).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Callable, Deque, Dict, Optional

import pandas as pd

from .stream_types import StreamTxn, ScoredTxn


logger = logging.getLogger(__name__)


DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC", "rudra.transactions")
DEFAULT_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
DEFAULT_BUFFER_SIZE = 500
KAFKA_PROBE_TIMEOUT = 2.0


# Approximate static FX → INR (RUDRA's home currency). Used ONLY to put a
# multi-currency dataset on one scale for large-value escalation; production
# swaps this for a live FX feed. Not exact — it just needs the right order of
# magnitude to decide "is this a very large transfer?".
_INR_FX = {
    "INR": 1.0, "USD": 83.0, "EUR": 90.0, "GBP": 105.0, "CHF": 92.0,
    "CNY": 11.5, "JPY": 0.55, "RUB": 0.9, "ILS": 22.0, "AUD": 55.0,
    "CAD": 61.0, "MXN": 4.8, "SAR": 22.0, "BRL": 16.0, "BTC": 5_000_000.0,
}
_FX_DEFAULT = 83.0                       # unknown currency → treat as USD-scale

# Large-value escalation tiers, in INR *after* FX-normalisation. A transfer this
# big is escalated regardless of the model score. On IBM AML this fires for
# ~6% (HIGH) / ~2.7% (CRITICAL) — genuine outliers, not the whole feed.
_AMOUNT_HIGH_INR = 1_00_00_000           # ₹1 crore
_AMOUNT_CRITICAL_INR = 5_00_00_000       # ₹5 crore

# ML-score severity bands, expressed as fractions of the model's OWN tuned
# operating threshold τ (read from the bundle). CRITICAL = the model's actual
# fraud decision; HIGH/MEDIUM are watchlist tiers below it. Single source of
# truth — never hardcode a raw 0.4/0.6/0.8 cutoff here.
_HIGH_FRAC = 0.85
_MEDIUM_FRAC = 0.70
_FALLBACK_TAU = 0.5                      # only if the bundle predates threshold saving

_SEVERITY_RANK = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def _amount_to_inr(amount: float, currency: str) -> float:
    return amount * _INR_FX.get((currency or "INR").upper(), _FX_DEFAULT)


def _severity_from_score(
    score: Optional[float],
    amount: float,
    currency: str = "INR",
    threshold: Optional[float] = None,
) -> Optional[str]:
    """Severity for one scored txn.

    ML-score severity is banded around the *model's own* tuned threshold τ, so
    what the UI flags tracks what the model actually calls fraud (≈3.4% at τ on
    IBM AML) rather than a hardcoded cutoff. A large-value (FX-normalised)
    escalation can only RAISE the severity, never lower it.
    """
    tau = threshold if (threshold and 0.0 < threshold < 1.0) else _FALLBACK_TAU

    sev = None
    if score is not None:
        if score >= tau:
            sev = "CRITICAL"             # model's fraud decision
        elif score >= _HIGH_FRAC * tau:
            sev = "HIGH"
        elif score >= _MEDIUM_FRAC * tau:
            sev = "MEDIUM"

    inr = _amount_to_inr(amount, currency)
    amt_sev = ("CRITICAL" if inr >= _AMOUNT_CRITICAL_INR
               else "HIGH" if inr >= _AMOUNT_HIGH_INR
               else None)
    if amt_sev and _SEVERITY_RANK[amt_sev] > _SEVERITY_RANK.get(sev, 0):
        sev = amt_sev                    # escalation raises only

    return sev


def _signals_from_features(features: Dict) -> list:
    """Derive honest, human-readable flags from the live feature row.

    Every flag maps to a feature score_live_txn actually computed for this
    transaction — no fabrication. These replace the old fake "pattern" label
    that the removed /api/live/inject endpoint used to invent.
    """
    if not features:
        return []
    signals = []
    if features.get("in_scc_3plus", 0) >= 1.0:
        signals.append("cycle")            # both endpoints already in a ≥3-node cycle
    if features.get("near_threshold_score", 0) >= 0.7:
        signals.append("near-threshold")   # amount hugs the reporting threshold (structuring)
    if features.get("sender_is_shell", 0) >= 1.0:
        signals.append("shell-sender")
    if features.get("receiver_is_shell", 0) >= 1.0:
        signals.append("shell-receiver")
    if features.get("high_value_rail_share", 0) >= 0.5:
        signals.append("high-value-rail")  # RTGS / wire heavy
    if features.get("night_ratio", 0) >= 1.0:
        signals.append("off-hours")
    return signals


class StreamIngestor:
    """Owns the consumer task + ring buffer + producer side of the bus.

    Constructor doesn't connect to anything — call start() (await-able) for that.
    """

    def __init__(
        self,
        score_fn: Callable,
        graph_provider: Callable,
        bundle_provider: Callable,
        topic: str = DEFAULT_TOPIC,
        bootstrap: str = DEFAULT_BOOTSTRAP,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
    ):
        # score_fn signature: (bundle, graph, sender, receiver, amount, channel, rail, timestamp, currency) -> dict
        self._score_fn = score_fn
        self._graph_provider = graph_provider
        self._bundle_provider = bundle_provider
        self.topic = topic
        self.bootstrap = bootstrap

        self._buffer: Deque[Dict] = deque(maxlen=buffer_size)
        self._task: Optional[asyncio.Task] = None
        self._producer = None                # aiokafka producer instance
        self._consumer = None                # aiokafka consumer instance
        self._inproc_queue: Optional[asyncio.Queue] = None
        self._mode = "stopped"               # "kafka" | "inproc" | "stopped"
        self._started_at: Optional[float] = None
        self._processed = 0
        self._errors = 0
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    async def start(self) -> Dict:
        async with self._lock:
            if self._task and not self._task.done():
                return self.status()

            requested = os.getenv("STREAM_BACKEND", "auto").lower()
            chosen = "inproc"
            if requested in ("kafka", "auto"):
                ok = await self._try_kafka_setup()
                if ok:
                    chosen = "kafka"
                elif requested == "kafka":
                    raise RuntimeError(
                        f"STREAM_BACKEND=kafka but no broker reachable at {self.bootstrap}"
                    )

            if chosen == "inproc":
                self._inproc_queue = asyncio.Queue(maxsize=1000)

            self._mode = chosen
            self._started_at = time.time()
            self._task = asyncio.create_task(self._consume_loop())
            logger.info("StreamIngestor started in %s mode (topic=%s, bootstrap=%s)",
                        chosen, self.topic, self.bootstrap)
            return self.status()

    async def stop(self) -> Dict:
        async with self._lock:
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
                self._task = None
            if self._producer:
                try:
                    await self._producer.stop()
                except Exception:
                    pass
                self._producer = None
            if self._consumer:
                try:
                    await self._consumer.stop()
                except Exception:
                    pass
                self._consumer = None
            self._inproc_queue = None
            mode_was = self._mode
            self._mode = "stopped"
            logger.info("StreamIngestor stopped (was: %s)", mode_was)
            return self.status()

    async def reset(self) -> Dict:
        """Hard reset: stop the consumer, then wipe the ring buffer + counters so
        the next start() begins from an empty feed at seq 0.

        Note: stop() must run *before* we take the lock (it acquires the lock
        itself, and asyncio.Lock isn't reentrant). Messages already sitting on a
        Kafka topic are not re-consumed — the consumer resubscribes at `latest`.
        """
        await self.stop()
        async with self._lock:
            self._buffer.clear()
            self._processed = 0
            self._errors = 0
            self._started_at = None
        logger.info("StreamIngestor hard-reset (buffer + counters cleared).")
        return self.status()

    def status(self) -> Dict:
        return {
            "mode": self._mode,
            "topic": self.topic,
            "bootstrap": self.bootstrap if self._mode == "kafka" else None,
            "running": self._task is not None and not self._task.done(),
            "uptime_s": round(time.time() - self._started_at, 2) if self._started_at else 0,
            "processed": self._processed,
            "errors": self._errors,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self._buffer.maxlen,
        }

    def recent(self, limit: int = 50) -> list:
        # deque returns oldest→newest; reverse so newest is first
        items = list(self._buffer)
        items.reverse()
        return items[:limit]

    async def publish(self, txn: StreamTxn) -> None:
        """Publish one transaction to the bus.

        In Kafka mode this hits the broker; in inproc mode it pushes to the
        asyncio queue.  Same call site either way.
        """
        if self._mode == "stopped":
            raise RuntimeError("StreamIngestor is not running. Call start() first.")
        payload = txn.to_dict()
        if self._mode == "kafka":
            data = json.dumps(payload).encode("utf-8")
            await self._producer.send_and_wait(self.topic, data)
        else:
            await self._inproc_queue.put(payload)

    # ── Kafka setup probe ─────────────────────────────────────────────────────

    async def _try_kafka_setup(self) -> bool:
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
        except ImportError:
            logger.info("aiokafka not installed — Kafka mode unavailable.")
            return False

        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap,
                # idempotent producer + light compression — production defaults.
                enable_idempotence=True,
                compression_type="gzip",
                request_timeout_ms=5000,
            )
            await asyncio.wait_for(self._producer.start(), timeout=KAFKA_PROBE_TIMEOUT)

            self._consumer = AIOKafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap,
                group_id="rudra-live-scorer",
                auto_offset_reset="latest",        # only stream live events; old ones already in batch model
                enable_auto_commit=True,
            )
            await asyncio.wait_for(self._consumer.start(), timeout=KAFKA_PROBE_TIMEOUT)
            return True
        except Exception as e:
            logger.info("Kafka probe failed (%s) — falling back to inproc.", e)
            if self._producer:
                try: await self._producer.stop()
                except Exception: pass
            if self._consumer:
                try: await self._consumer.stop()
                except Exception: pass
            self._producer = None
            self._consumer = None
            return False

    # ── Consume loop ──────────────────────────────────────────────────────────

    async def _consume_loop(self):
        try:
            if self._mode == "kafka":
                async for msg in self._consumer:
                    try:
                        payload = json.loads(msg.value.decode("utf-8"))
                        await self._handle(payload)
                    except Exception as e:
                        self._errors += 1
                        logger.warning("kafka consume error: %s", e)
            else:
                while True:
                    payload = await self._inproc_queue.get()
                    try:
                        await self._handle(payload)
                    except Exception as e:
                        self._errors += 1
                        logger.warning("inproc consume error: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Consumer loop crashed: %s", e)

    async def _handle(self, payload: Dict):
        """Score one payload and append to the ring buffer."""
        graph = self._graph_provider()
        bundle = self._bundle_provider()
        if graph is None or bundle is None:
            self._errors += 1
            return

        try:
            ts = pd.to_datetime(payload.get("timestamp")) if payload.get("timestamp") else pd.Timestamp.utcnow()
        except Exception:
            ts = pd.Timestamp.utcnow()

        try:
            res = self._score_fn(
                bundle,
                graph,
                payload["sender_id"],
                payload["receiver_id"],
                float(payload["amount"]),
                payload.get("channel", "NetBanking"),
                payload.get("transaction_type", "NEFT"),
                ts,
                payload.get("currency", "INR"),
            )
            score = res.get("ml_score")
            latency = res.get("latency_ms")
            signals = _signals_from_features(res.get("features") or {})
            err = None
        except Exception as e:
            score = None
            latency = None
            signals = []
            err = str(e)
            self._errors += 1

        scored = ScoredTxn(
            txn=StreamTxn(
                transaction_id=payload.get("transaction_id", f"stream-{int(time.time()*1000)}"),
                sender_id=payload["sender_id"],
                receiver_id=payload["receiver_id"],
                amount=float(payload["amount"]),
                timestamp=str(payload.get("timestamp", ts.isoformat())),
                channel=payload.get("channel", "NetBanking"),
                transaction_type=payload.get("transaction_type", "NEFT"),
                currency=payload.get("currency", "INR"),
            ),
            ml_score=score,
            latency_ms=latency,
            severity=_severity_from_score(
                score,
                float(payload["amount"]),
                payload.get("currency", "INR"),
                threshold=(bundle.get("threshold") if isinstance(bundle, dict) else None),
            ),
            error=err,
            seq=self._processed,
            signals=signals,
        )
        self._buffer.append(scored.to_dict())
        self._processed += 1


# ── Module-level singleton helper ─────────────────────────────────────────────
# Backend code uses get_ingestor() so a single instance is shared across the app.

_INGESTOR: Optional[StreamIngestor] = None


def get_ingestor(
    score_fn: Optional[Callable] = None,
    graph_provider: Optional[Callable] = None,
    bundle_provider: Optional[Callable] = None,
) -> StreamIngestor:
    """Lazy singleton. First call must supply the callbacks; later calls reuse."""
    global _INGESTOR
    if _INGESTOR is None:
        if not (score_fn and graph_provider and bundle_provider):
            raise RuntimeError("get_ingestor first call requires all three callbacks.")
        _INGESTOR = StreamIngestor(score_fn, graph_provider, bundle_provider)
    return _INGESTOR
