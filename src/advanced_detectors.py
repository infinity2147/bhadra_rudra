"""
Advanced Fraud Detectors
1. Dormant Account Activation (Z-score spike detection)
2. Profile Mismatch Detection (KYC behavioral delta analysis)
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timedelta


class DormantActivationDetector:
    """Detect accounts that were dormant and suddenly activated with unusual activity."""

    def __init__(self, graph: nx.DiGraph, transactions: pd.DataFrame,
                 dormant_threshold_days: int = 30,
                 z_score_threshold: float = 2.5,
                 config: Optional[Dict] = None):
        self.graph = graph
        self.transactions = transactions
        cfg = config or {}
        self.dormant_threshold_days = cfg.get("dormant_threshold_days", dormant_threshold_days)
        self.z_score_threshold = cfg.get("dormant_z_score_threshold", z_score_threshold)

    def detect(self) -> List[Dict]:
        """Find dormant accounts that were suddenly activated with Z-score anomaly."""
        alerts = []
        txns = self.transactions.copy()
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])

        # Pre-index by sender and receiver so per-entity lookup is O(1) instead
        # of an O(n) DataFrame filter inside a per-node loop. On a 100k-txn /
        # 100k-entity graph (IBM AML) this turns ~30 minutes into seconds.
        by_sender = {
            sid: idxs for sid, idxs in txns.groupby("sender_id").indices.items()
        }
        by_receiver = {
            rid: idxs for rid, idxs in txns.groupby("receiver_id").indices.items()
        }

        for node in self.graph.nodes():
            send_idx = by_sender.get(node)
            recv_idx = by_receiver.get(node)
            if send_idx is None and recv_idx is None:
                continue
            if send_idx is not None and recv_idx is not None:
                idxs = np.unique(np.concatenate([send_idx, recv_idx]))
            else:
                idxs = send_idx if send_idx is not None else recv_idx
            if len(idxs) < 3:
                continue
            node_txns = txns.iloc[idxs]

            node_txns = node_txns.sort_values("timestamp")

            # Compute daily activity amounts
            daily = node_txns.groupby(node_txns["timestamp"].dt.date)["amount"].sum().reset_index()
            daily.columns = ["date", "amount"]
            daily["date"] = pd.to_datetime(daily["date"])

            if len(daily) < 3:
                continue

            # Look for gaps (dormant periods)
            daily["gap_days"] = daily["date"].diff().dt.total_seconds() / 86400

            # Find activation points: large gap followed by activity
            for i in range(1, len(daily)):
                gap = daily.iloc[i]["gap_days"]
                if pd.isna(gap):
                    continue

                if gap >= self.dormant_threshold_days:
                    # Compute Z-score of post-activation activity
                    pre_amounts = daily.iloc[:i]["amount"].values
                    post_amounts = daily.iloc[i:i+5]["amount"].values

                    if len(pre_amounts) < 2 or len(post_amounts) == 0:
                        continue

                    pre_mean = np.mean(pre_amounts)
                    pre_std = np.std(pre_amounts) if np.std(pre_amounts) > 0 else 1
                    post_mean = np.mean(post_amounts)

                    z_score = (post_mean - pre_mean) / pre_std

                    if z_score >= self.z_score_threshold:
                        node_data = self.graph.nodes[node]
                        node_name = node_data.get("name", node)

                        alert = {
                            "alert_id": f"ALERT_DORM_{len(alerts) + 1:04d}",
                            "pattern_type": "Dormant Activation",
                            "severity": "CRITICAL" if z_score > 4 else "HIGH",
                            "confidence": round(min(z_score / 5 * 100, 98), 1),
                            "entities": [node],
                            "entity_names": [node_name],
                            "z_score": round(z_score, 2),
                            "dormant_days": round(gap),
                            "pre_avg_amount": round(pre_mean, 2),
                            "post_avg_amount": round(post_mean, 2),
                            "activation_date": str(daily.iloc[i]["date"].date()),
                            "description": (
                                f"Dormant account '{node_name}' activated after {gap:.0f} days "
                                f"with Z-score spike of {z_score:.1f} "
                                f"(avg ₹{pre_mean:,.0f} → ₹{post_mean:,.0f})"
                            ),
                            "recommendation": (
                                "Verify recent KYC updates. Check for account takeover. "
                                "Compare activity with historical pattern. Flag for EDD."
                            ),
                        }
                        alerts.append(alert)

        return alerts


class ProfileMismatchDetector:
    """Detect behavioral mismatches between entity type and transaction patterns.

    Score composition (T2.9):
      * Rule mismatches (e.g. "individual averaging > ₹10L") generate a
        `rule_score = count(mismatches) × profile_score_per_mismatch`.
      * The XGBoost edge classifier scores every edge involving this entity;
        we take the *max* score across those edges as an `ml_score` —
        this lifts the alert when the model independently flags any of the
        entity's relationships as suspicious.
      * Final confidence = max(rule_score, ml_score) capped at profile_max_score.

    The reason for `max` instead of a learned mix: the rules and the ML model
    catch *different* failure modes — a rule-based "high night-time ratio"
    won't be in the ML feature set; conversely, the ML model can flag
    profile-consistent behaviour that violates learned patterns the rules
    don't encode. Either signal triggering should raise the alert. (A linear
    mix is the wrong inductive bias here — we'd just dilute strong signals.)

    Every threshold below reads from the ConfigStore-backed `config` dict
    (see src/config_store.py DEFAULT_CONFIG, keys prefixed `profile_`).
    """

    def __init__(self, graph: nx.DiGraph, transactions: pd.DataFrame,
                 risk_scores: List[Dict], config: Optional[Dict] = None,
                 edge_scores: Optional[Dict[str, float]] = None):
        """Construct.

        Args:
            edge_scores: optional {"u->v": ml_score} mapping. When provided,
                each entity's alert confidence is lifted toward the max ML
                score across its incoming and outgoing edges. When absent,
                rule_score alone determines confidence (backwards-compatible
                with code paths that don't load the ML bundle).
        """
        self.graph = graph
        self.transactions = transactions
        self.risk_scores = {r["entity_id"]: r for r in risk_scores}
        self.config = config or {}
        self.edge_scores = edge_scores or {}

    def _cfg(self, key: str, default):
        return self.config.get(key, default)

    def _ml_score_for_entity(self, node: str) -> Optional[float]:
        """Max ML score across all edges adjacent to this node, or None if no edges scored."""
        if not self.edge_scores:
            return None
        scores = []
        for u in self.graph.predecessors(node):
            s = self.edge_scores.get(f"{u}->{node}")
            if s is not None:
                scores.append(s)
        for v in self.graph.successors(node):
            s = self.edge_scores.get(f"{node}->{v}")
            if s is not None:
                scores.append(s)
        if not scores:
            return None
        return float(max(scores))

    def detect(self) -> List[Dict]:
        """Find entities whose transaction behavior mismatches their declared profile."""
        alerts = []
        txns = self.transactions.copy()
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])

        # Pull every threshold up-front so the hot loop stays tight.
        max_individual_avg     = self._cfg("profile_individual_max_avg_amount", 1_000_000)
        max_individual_volume  = self._cfg("profile_individual_max_total_volume", 50_000_000)
        max_import_payments    = self._cfg("profile_individual_max_import_payments", 2)
        max_vendor_payments    = self._cfg("profile_individual_max_vendor_payments", 3)
        biz_min_received       = self._cfg("profile_business_min_received_with_no_sent", 5)
        biz_max_upi_ratio      = self._cfg("profile_business_max_upi_ratio", 0.8)
        biz_min_avg            = self._cfg("profile_business_min_avg_amount", 10_000)
        biz_min_txns_low_avg   = self._cfg("profile_business_min_txns_for_low_avg_check", 20)
        shell_max_txns         = self._cfg("profile_shell_max_txns", 10)
        max_branches           = self._cfg("profile_max_branches", 4)
        max_night_ratio        = self._cfg("profile_max_night_ratio", 0.4)
        score_per_mismatch     = self._cfg("profile_score_per_mismatch", 0.2)
        max_score              = self._cfg("profile_max_score", 0.95)
        critical_threshold     = self._cfg("profile_critical_score_threshold", 0.6)
        high_threshold         = self._cfg("profile_high_score_threshold", 0.4)

        # Pre-index by sender + receiver so each per-node lookup is O(1)
        # instead of an O(n) DataFrame scan.
        by_sender = {
            sid: idxs for sid, idxs in txns.groupby("sender_id").indices.items()
        }
        by_receiver = {
            rid: idxs for rid, idxs in txns.groupby("receiver_id").indices.items()
        }

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            entity_type = node_data.get("type", "individual")
            node_name = node_data.get("name", node)

            send_idx = by_sender.get(node)
            recv_idx = by_receiver.get(node)
            if send_idx is None and recv_idx is None:
                continue
            if send_idx is not None and recv_idx is not None:
                idxs = np.unique(np.concatenate([send_idx, recv_idx]))
            else:
                idxs = send_idx if send_idx is not None else recv_idx
            if len(idxs) < 3:
                continue
            node_txns = txns.iloc[idxs]

            mismatches = []
            sent = node_txns[node_txns["sender_id"] == node]
            received = node_txns[node_txns["receiver_id"] == node]

            avg_amount = node_txns["amount"].mean()
            max_amount = node_txns["amount"].max()
            total_volume = node_txns["amount"].sum()

            # Type-specific checks
            if entity_type == "individual":
                if len(sent) > 0 and sent["amount"].mean() > max_individual_avg:
                    mismatches.append(f"Individual averaging > {max_individual_avg:,.0f} per transaction")
                purposes = sent["purpose_code"].value_counts().to_dict()
                if purposes.get("Import Payment", 0) > max_import_payments:
                    mismatches.append("Individual making import payments")
                if purposes.get("Vendor Payment", 0) > max_vendor_payments:
                    mismatches.append("Individual making frequent vendor payments")
                if total_volume > max_individual_volume:
                    mismatches.append(f"Individual with {total_volume:,.0f} total volume")

            elif entity_type == "business":
                if len(sent) == 0 and len(received) > biz_min_received:
                    mismatches.append("Business only receiving funds, no outflows")
                if len(sent) > 0:
                    txn_types = sent["transaction_type"].value_counts().to_dict()
                    if txn_types.get("UPI", 0) > len(sent) * biz_max_upi_ratio:
                        mismatches.append(f"Business using UPI for >{biz_max_upi_ratio:.0%} of transactions")
                if avg_amount < biz_min_avg and len(node_txns) > biz_min_txns_low_avg:
                    mismatches.append("Business with consistently low transaction amounts")

            elif entity_type == "shell_company":
                if len(node_txns) > shell_max_txns:
                    mismatches.append(f"Shell company with {len(node_txns)} transactions")

            # Cross-branch activity
            branches = set()
            if len(sent) > 0:
                branches.update(sent["sender_branch"].dropna().tolist())
            if len(received) > 0:
                branches.update(received["receiver_branch"].dropna().tolist())
            if len(branches) > max_branches:
                mismatches.append(f"Activity across {len(branches)} different branches")

            # Time-based anomalies (nighttime transactions)
            if len(node_txns) > 0:
                hours = pd.to_datetime(node_txns["timestamp"]).dt.hour
                night_ratio = ((hours < 6) | (hours > 22)).mean()
                if night_ratio > max_night_ratio:
                    mismatches.append(f"{night_ratio:.0%} transactions during nighttime (10PM-6AM)")

            # Compose the confidence: rule signal AND ML signal, take the max.
            # An alert fires if *either* signal is strong enough — different
            # failure modes (see class docstring).
            ml_score = self._ml_score_for_entity(node)
            rule_score = min(len(mismatches) * score_per_mismatch, max_score) if mismatches else 0.0

            # Combine. `max` not `mean` — strong signal on either side wins.
            combined_score = max(rule_score, ml_score or 0.0)

            # Alert if either side is meaningful. The original "needs mismatches
            # to alert" rule is preserved when no ML bundle is loaded; with ML,
            # high ml_score alone is enough to flag.
            should_alert = (
                (mismatches and rule_score > 0)
                or (ml_score is not None and ml_score >= high_threshold)
            )

            if should_alert:
                risk_info = self.risk_scores.get(node, {})
                base_risk = risk_info.get("risk_score", 0)

                if combined_score > critical_threshold:
                    severity = "CRITICAL"
                elif combined_score > high_threshold:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                # Generate a description that names which signal(s) fired.
                if mismatches and ml_score is not None:
                    desc = (
                        f"Entity '{node_name}' ({entity_type}) — {len(mismatches)} behavioural "
                        f"mismatches AND ML score {ml_score:.2f} on related edges. "
                        f"Mismatches: {'; '.join(mismatches[:3])}"
                    )
                elif mismatches:
                    desc = (
                        f"Entity '{node_name}' ({entity_type}) shows {len(mismatches)} behavioural "
                        f"mismatches: {'; '.join(mismatches[:3])}"
                    )
                else:
                    desc = (
                        f"Entity '{node_name}' ({entity_type}) has no rule-based mismatches but "
                        f"the ML model flagged its edges (max score {ml_score:.2f}). "
                        f"Profile may be consistent with type but transaction-level patterns are suspicious."
                    )

                alert = {
                    "alert_id": f"ALERT_PROF_{len(alerts) + 1:04d}",
                    "pattern_type": "Profile Mismatch",
                    "severity": severity,
                    "confidence": round(combined_score * 100, 1),
                    "rule_score": round(rule_score, 4),
                    "ml_score": round(ml_score, 4) if ml_score is not None else None,
                    "scoring_mode": (
                        "rule+ml" if (mismatches and ml_score is not None)
                        else "rule_only" if mismatches
                        else "ml_only"
                    ),
                    "entities": [node],
                    "entity_names": [node_name],
                    "entity_type": entity_type,
                    "mismatches": mismatches,
                    "mismatch_count": len(mismatches),
                    "base_risk_score": base_risk,
                    "behavioral_volume": round(total_volume, 2),
                    "description": desc,
                    "recommendation": (
                        "Review KYC documents. Verify declared business activity. "
                        "Compare with peer entities of same type. Consider re-classification."
                    ),
                }
                alerts.append(alert)

        return alerts
