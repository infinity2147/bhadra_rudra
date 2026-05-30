"""
Kafka producer — replays a transactions CSV (or any DataFrame) onto the bus.

Used two ways:

  1. As a CLI:
        python -m src.streaming.kafka_producer \\
            --source data/real/ibm_aml/HI-Small_Trans_100k.csv \\
            --rate 10 --total 500
     Streams `total` transactions at `rate` txns/sec to KAFKA_TOPIC on
     KAFKA_BOOTSTRAP_SERVERS.  Use this to demo the real streaming path from
     a separate terminal — judges can SEE messages flow through Kafka.

  2. Programmatically (from /api/stream/replay):
        await replay_transactions(ingestor, df, rate=5, total=100)
     The backend uses this to fan one HTTP call out into a stream of events,
     which the ingestor's running consumer will pick up and score.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
import sys
from typing import Iterable, Optional

import pandas as pd

# Make src/ importable when invoked as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.stream_types import StreamTxn
from streaming.ingestor import StreamIngestor


logger = logging.getLogger(__name__)


def _row_to_txn(row, idx: int) -> StreamTxn:
    return StreamTxn(
        transaction_id=str(row.get("transaction_id", f"REPLAY{idx:08d}")),
        sender_id=str(row["sender_id"]),
        receiver_id=str(row["receiver_id"]),
        amount=float(row["amount"]),
        timestamp=str(row.get("timestamp", pd.Timestamp.utcnow().isoformat())),
        channel=str(row.get("channel", "NetBanking")),
        transaction_type=str(row.get("transaction_type", "NEFT")),
        currency=str(row.get("currency", "INR")),
    )


async def replay_transactions(
    ingestor: StreamIngestor,
    df: pd.DataFrame,
    rate: float = 5.0,
    total: Optional[int] = None,
    shuffle: bool = True,
) -> int:
    """Publish rows of `df` onto the ingestor at `rate` events/sec.

    Returns the number of events actually published.
    """
    needed = ["sender_id", "receiver_id", "amount"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if shuffle:
        df = df.sample(frac=1, random_state=random.randint(0, 1_000_000))
    if total is not None:
        df = df.head(total)

    interval = 1.0 / max(rate, 0.01)
    count = 0
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            await ingestor.publish(_row_to_txn(row, i))
            count += 1
        except Exception as e:
            logger.warning("publish failed at row %d: %s", i, e)
        await asyncio.sleep(interval)
    return count


# ── Standalone CLI mode ────────────────────────────────────────────────────────
# When invoked as `python -m src.streaming.kafka_producer`, we open a direct
# aiokafka producer to the broker, no FastAPI involved. Lets you demo Kafka
# from one terminal while the backend's consumer reads from another.

async def _cli_main(args):
    from aiokafka import AIOKafkaProducer
    import json

    df = pd.read_csv(args.source)
    if args.shuffle:
        df = df.sample(frac=1, random_state=42)
    if args.total:
        df = df.head(args.total)

    producer = AIOKafkaProducer(
        bootstrap_servers=args.bootstrap,
        enable_idempotence=True,
        compression_type="gzip",
    )
    await producer.start()
    print(f"[producer] connected to {args.bootstrap}, topic={args.topic}, "
          f"streaming {len(df)} txns at {args.rate}/s")
    try:
        interval = 1.0 / max(args.rate, 0.01)
        for i, (_, row) in enumerate(df.iterrows()):
            txn = _row_to_txn(row, i)
            await producer.send_and_wait(args.topic, json.dumps(txn.to_dict()).encode("utf-8"))
            if (i + 1) % 20 == 0:
                print(f"[producer] sent {i+1}/{len(df)}")
            await asyncio.sleep(interval)
        print(f"[producer] done — sent {len(df)} txns")
    finally:
        await producer.stop()


def _build_argparser():
    p = argparse.ArgumentParser(description="Replay a transactions CSV onto Kafka.")
    # Default to the active variant's transactions.csv (e.g. data/ibm_aml/transactions.csv).
    # Pre-sampled IBM AML CSV at data/real/ibm_aml/HI-Small_Trans_100k_sampled.csv also works.
    p.add_argument(
        "--source",
        default=f"data/{os.getenv('RUDRA_DATASET', 'ibm_aml')}/transactions.csv",
        help="CSV to replay (default: data/<RUDRA_DATASET>/transactions.csv).",
    )
    p.add_argument("--bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
                   help="Kafka bootstrap servers (default from KAFKA_BOOTSTRAP_SERVERS env).")
    p.add_argument("--topic", default=os.getenv("KAFKA_TOPIC", "rudra.transactions"),
                   help="Kafka topic (default rudra.transactions).")
    p.add_argument("--rate", type=float, default=5.0,
                   help="Events per second (default 5).")
    p.add_argument("--total", type=int, default=None,
                   help="Stop after this many events (default: replay whole file).")
    p.add_argument("--shuffle", action="store_true", default=True)
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    asyncio.run(_cli_main(args))
