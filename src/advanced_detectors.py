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

        for node in self.graph.nodes():
            node_txns = txns[(txns["sender_id"] == node) | (txns["receiver_id"] == node)]
            if len(node_txns) < 3:
                continue

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
    """Detect behavioral mismatches between entity type and transaction patterns."""

    def __init__(self, graph: nx.DiGraph, transactions: pd.DataFrame,
                 risk_scores: List[Dict]):
        self.graph = graph
        self.transactions = transactions
        self.risk_scores = {r["entity_id"]: r for r in risk_scores}

    def detect(self) -> List[Dict]:
        """Find entities whose transaction behavior mismatches their declared profile."""
        alerts = []
        txns = self.transactions.copy()
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])

        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            entity_type = node_data.get("type", "individual")
            node_name = node_data.get("name", node)

            node_txns = txns[(txns["sender_id"] == node) | (txns["receiver_id"] == node)]
            if len(node_txns) < 3:
                continue

            mismatches = []
            sent = node_txns[node_txns["sender_id"] == node]
            received = node_txns[node_txns["receiver_id"] == node]

            avg_amount = node_txns["amount"].mean()
            max_amount = node_txns["amount"].max()
            total_volume = node_txns["amount"].sum()

            # Type-specific checks
            if entity_type == "individual":
                # Individual sending very large amounts
                if len(sent) > 0 and sent["amount"].mean() > 1000000:
                    mismatches.append("Individual averaging >₹10L per transaction")
                # Individual using business payment channels
                purposes = sent["purpose_code"].value_counts().to_dict()
                if purposes.get("Import Payment", 0) > 2:
                    mismatches.append("Individual making import payments")
                if purposes.get("Vendor Payment", 0) > 3:
                    mismatches.append("Individual making frequent vendor payments")
                # High volume for individual
                if total_volume > 50000000:
                    mismatches.append(f"Individual with ₹{total_volume/1e7:.1f} Cr total volume")

            elif entity_type == "business":
                # Business only receiving, never sending
                if len(sent) == 0 and len(received) > 5:
                    mismatches.append("Business only receiving funds, no outflows")
                # Business using personal channels excessively
                if len(sent) > 0:
                    txn_types = sent["transaction_type"].value_counts().to_dict()
                    if txn_types.get("UPI", 0) > len(sent) * 0.8:
                        mismatches.append("Business using UPI for >80% of transactions")
                # Low average amount for business
                if avg_amount < 10000 and len(node_txns) > 20:
                    mismatches.append("Business with consistently low transaction amounts")

            elif entity_type == "shell_company":
                # Shell company with high activity (should be dormant)
                if len(node_txns) > 10:
                    mismatches.append(f"Shell company with {len(node_txns)} transactions")

            # Cross-branch activity
            branches = set()
            if len(sent) > 0:
                branches.update(sent["sender_branch"].dropna().tolist())
            if len(received) > 0:
                branches.update(received["receiver_branch"].dropna().tolist())
            if len(branches) > 4:
                mismatches.append(f"Activity across {len(branches)} different branches")

            # Time-based anomalies (nighttime transactions)
            if len(node_txns) > 0:
                hours = pd.to_datetime(node_txns["timestamp"]).dt.hour
                night_ratio = ((hours < 6) | (hours > 22)).mean()
                if night_ratio > 0.4:
                    mismatches.append(f"{night_ratio:.0%} transactions during nighttime (10PM-6AM)")

            if mismatches:
                score = min(len(mismatches) * 0.2, 0.95)
                risk_info = self.risk_scores.get(node, {})
                base_risk = risk_info.get("risk_score", 0)

                alert = {
                    "alert_id": f"ALERT_PROF_{len(alerts) + 1:04d}",
                    "pattern_type": "Profile Mismatch",
                    "severity": "CRITICAL" if score > 0.6 else "HIGH" if score > 0.4 else "MEDIUM",
                    "confidence": round(score * 100, 1),
                    "entities": [node],
                    "entity_names": [node_name],
                    "entity_type": entity_type,
                    "mismatches": mismatches,
                    "mismatch_count": len(mismatches),
                    "base_risk_score": base_risk,
                    "behavioral_volume": round(total_volume, 2),
                    "description": (
                        f"Entity '{node_name}' ({entity_type}) shows {len(mismatches)} behavioral "
                        f"mismatches: {'; '.join(mismatches[:3])}"
                    ),
                    "recommendation": (
                        "Review KYC documents. Verify declared business activity. "
                        "Compare with peer entities of same type. Consider re-classification."
                    ),
                }
                alerts.append(alert)

        return alerts
