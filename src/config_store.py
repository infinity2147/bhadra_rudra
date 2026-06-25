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
import threading
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    # Circular detection
    "circular_amount_tolerance": 0.15,    # max within-cycle amount deviation
    "circular_min_total_flow": 100_000,
    "circular_max_cycle_length": 8,
    "circular_max_alerts": 50,
    # Scale guards — AML rings are tight clusters of 3-20 shell accounts,
    # not whole interbank SCCs. Skip giant SCCs and time-box per-SCC enumeration
    # so dense graphs (IBM AML, real bank networks) stay tractable.
    "circular_scc_size_cap": 200,
    "circular_scc_time_budget_s": 8.0,
    # Risk scoring — exact betweenness is O(VE); Monte Carlo approximation
    # over k pivots stays usable on 100k+ node graphs (IBM AML).
    "centrality_sample_k": 500,
    # Layering — a tranche relayed quickly through a chain of intermediaries.
    # min_chain_length counts ENTITIES (nodes). Layering means MULTIPLE
    # obfuscation layers, so the floor is 4 (= source → ≥2 intermediaries →
    # destination, i.e. ≥3 hops). A single intermediary (A→B→C) is a plain
    # pass-through — that's the funnel detector's job, and on dense graphs
    # 2-hop "chains" are mostly coincidence, not laundering.
    "layering_min_chain_length": 4,
    "layering_max_chains": 200,           # budget; biggest-pipe starts processed first
    "layering_decrease_ratio": 0.85,      # each hop must carry >= ratio * previous hop
    "layering_amount_grow_tolerance": 0.15,  # ...and <= (1+tol) * previous hop (no merging-in)
    "layering_max_branching_per_node": 10,   # max valid continuations explored per node
    "layering_max_depth": 8,              # longest chain (in entities) to follow
    "layering_min_hop_amount": 50_000,    # ignore dust hops below this (defeats ₹1 decoys)
    "layering_time_window_hours": 48,     # a hop must forward funds within this of receiving them
    "layering_max_chain_span_hours": 72,  # whole chain must complete this fast — else it isn't "rapid"
    "layering_max_start_nodes": 5_000,    # scale guard: only seed from the N biggest-outflow nodes
    # Smurfing
    "smurfing_threshold": 200_000,        # ₹2L reporting threshold (USD adjusts via currency check)
    "smurfing_cluster_tolerance": 0.10,
    # Edge-cluster proximity band: avg/threshold must fall in [min,max]. The
    # lower bound used to be hardcoded at 0.7, so deliberate structuring well
    # below the threshold (e.g. at 0.5) was invisible. Low size-variance is the
    # real structuring signature; proximity just sets how aggressive we are.
    "smurfing_cluster_min_ratio": 0.5,
    "smurfing_cluster_max_ratio": 1.0,
    # Burst smurfing — N+ txns below threshold from one sender within M minutes,
    # regardless of which receiver they go to. Catches the structuring pattern
    # where one source fans out fast across many mules within a short window.
    "smurfing_burst_min_txns": 5,
    "smurfing_burst_window_minutes": 60,
    # Fan-out structuring — one sender spraying sub-threshold transfers once each
    # to many distinct receivers. Window-INDEPENDENT (complements the burst
    # window, which a launderer evades by spacing transfers past its edge).
    "smurfing_fanout_min_txns": 8,
    "smurfing_fanout_min_receivers": 5,
    "smurfing_fanout_max_txns_per_receiver": 1.5,   # ~one shot per mule
    # The structured transfers are isolated as the tightest amount-band cluster
    # within the sender's sub-threshold activity (so background noise mixed in
    # doesn't hide them). Amounts within this relative spread count as one band.
    "smurfing_fanout_band_tolerance": 0.15,
    # Recruiter / coordinator — one source funding a fleet of pass-through mules.
    # Names the orchestrator upstream of the mules (distinct from smurfing).
    "recruiter_min_fanout": 5,                 # min mule-like recipients to flag a coordinator
    "recruiter_pass_through_ratio": 0.6,       # recipient min(in,out)/max(in,out) above this = forwards funds
    "recruiter_min_seed_amount": 10_000,       # ignore dust seed transfers
    "recruiter_min_funding_share": 0.3,        # U must supply >= this share of each mule's inflow
                                               # (a coordinator FUNDS its fleet; not just one of many payers)
    # Shell funnel
    "funnel_imbalance_threshold": 0.7,
    "funnel_min_in_degree": 3,
    # Pass-through detection — flags balanced flows that the imbalance rule
    # misses (in ≈ out, money doesn't sit, classic mule behaviour).
    "funnel_pass_through_min_ratio": 0.9,           # min(in,out)/max(in,out) above this is "balanced"
    "funnel_max_holding_seconds": 3600,             # avg time between in and out < this triggers
    "funnel_pass_through_min_flow": 500_000,        # only flag pass-through when total flow >= this
    # Dormant activation
    "dormant_threshold_days": 30,
    "dormant_z_score_threshold": 2.5,
    # Post-activation window (active days) over which we take the PEAK daily
    # amount — peak, not mean, so a slow drip (₹1/day) followed by one huge day
    # can't average the spike away.
    "dormant_post_activation_days": 30,
    # When pre-activation history is perfectly regular (std == 0), fall back to
    # this fraction of the pre-mean instead of an absolute 1, so the z-score
    # stays scale-relative (a hardcoded 1 turned a 20% bump into z≈200000).
    "dormant_pre_std_floor_ratio": 0.25,
    # Profile mismatch
    # All thresholds below were hardcoded in advanced_detectors.ProfileMismatchDetector
    # until T1.1. INR amounts assume the IBM AML loader normalises currency to ISO codes
    # but the detector still applies these as raw thresholds; tune them per dataset.
    "profile_individual_max_avg_amount": 1_000_000,       # ₹10L — flagged if individual averages above this
    "profile_individual_max_total_volume": 50_000_000,    # ₹5Cr — total volume cap for individuals
    "profile_individual_max_import_payments": 2,           # how many Import Payment txns before flagging
    "profile_individual_max_vendor_payments": 3,
    "profile_business_min_received_with_no_sent": 5,       # business only receiving funds
    "profile_business_max_upi_ratio": 0.8,                 # UPI share above this is anomalous for business
    "profile_business_min_avg_amount": 10_000,             # below this with ≥ N txns is anomalous
    "profile_business_min_txns_for_low_avg_check": 20,
    "profile_shell_max_txns": 10,                          # shell shouldn't transact often
    # A shell moving large money is anomalous regardless of COUNT — count alone
    # let a launderer push 9 massive transfers through a shell and stay silent.
    "profile_shell_max_volume": 2_000_000,                 # ₹20L total through a shell is anomalous
    "profile_shell_max_single_txn": 1_000_000,             # any single ₹10L+ transfer through a shell
    "profile_max_branches": 4,                             # activity across this many branches is suspicious
    "profile_max_night_ratio": 0.4,                        # >40% nighttime txns is anomalous
    # Night window, inclusive of the start hour: a txn is "night" when
    # hour >= start OR hour < end. The boundary used to be hardcoded `> 22`,
    # which silently excluded the whole 22:00–22:59 hour.
    "profile_night_hour_start": 22,
    "profile_night_hour_end": 6,
    "profile_score_per_mismatch": 0.2,                     # confidence per mismatch found
    "profile_max_score": 0.95,                             # cap on combined score
    "profile_critical_score_threshold": 0.6,
    "profile_high_score_threshold": 0.4,
    # Minimum rule_score required to emit a Profile-Mismatch alert. At
    # 0.4 (= 2 × profile_score_per_mismatch) the entity must violate its
    # declared profile on at least TWO independent behavioural dimensions.
    # This stops the detector from mirroring the ML edge classifier and
    # flooding the alert stream on real datasets with no declared KYC profile
    # (e.g. IBM AML). The ML edge score still ESCALATES severity here, but a
    # high ML score alone surfaces in the ML / graph views, not as a
    # mislabelled "profile mismatch".
    "profile_min_rule_score": 0.4,
    # ML
    "ml_alert_threshold": 0.6,
    # Fund tracer annotation thresholds (kept in sync with detectors)
    "tracer_structuring_threshold": 200_000,
    "tracer_high_value_threshold": 1_000_000,
    "tracer_night_hour_start": 22,
    "tracer_night_hour_end": 6,
    "tracer_baseline_z_threshold": 3.0,
    "tracer_burst_window_minutes": 60,
    "tracer_burst_min_cluster": 3,
    "tracer_transit_window_hours": 1.0,
    "tracer_transit_ratio_threshold": 0.5,
}


class ConfigStore:
    """File-backed key/value config — uses the same SQLite as CaseStore."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # FastAPI's threadpool can dispatch two requests to the same store
        # concurrently. sqlite3 with `check_same_thread=False` allows shared
        # access but the caller must serialize — concurrent cursors on one
        # connection scramble each other's row descriptions (we saw
        # `IndexError: tuple index out of range` on `row["value_json"]`).
        self._lock = threading.RLock()
        self._init_schema()
        self._seed_defaults()

    def _init_schema(self) -> None:
        with self._lock:
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
        with self._lock:
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
        with self._lock:
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
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT key, value_json FROM config")
            rows = c.fetchall()
        out = dict(DEFAULT_CONFIG)
        for row in rows:
            try:
                out[row["key"]] = json.loads(row["value_json"])
            except json.JSONDecodeError:
                pass
        return out

    def set(self, key: str, value: Any) -> None:
        from datetime import datetime
        with self._lock:
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
        with self._lock:
            c = self.conn.cursor()
            c.execute("DELETE FROM config")
            self.conn.commit()
        self._seed_defaults()
        return self.get_all()
