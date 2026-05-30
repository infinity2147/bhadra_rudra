"""
Fraud Detection Engine
Implements multiple detection algorithms:
1. Circular Transaction Detection (cycle analysis)
2. Rapid Layering Detection (temporal chain analysis)
3. Smurfing/Structuring Detection (threshold clustering)
4. Shell Company Funnel Detection (centrality + flow analysis)
5. Anomaly Scoring (composite risk score)
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import json


class FraudDetector:
    def __init__(self, graph: nx.DiGraph, transactions: Optional[pd.DataFrame] = None,
                 config: Optional[Dict] = None,
                 risk_weights_bundle: Optional[Dict] = None):
        """Build a detector around a fund-flow graph.

        Args:
            graph: NetworkX DiGraph with edge attributes (total_amount, etc).
            transactions: required for burst-pattern detection (temporal
                smurfing) which can't be computed from edge aggregates
                alone. Optional so existing call sites that only pass the
                graph keep working — burst detection no-ops in that case.
            config: ConfigStore.get_all() result. Detector falls back to
                DEFAULT_CONFIG values when keys are missing.
            risk_weights_bundle: trained LR bundle from
                src.risk_score_learner.load_risk_weights(). When provided,
                compute_node_risk_scores uses the learned model; when None,
                falls back to the hand-tuned weighted sum.
        """
        self.graph = graph
        self.transactions = transactions
        self.config = config or {}
        self.risk_weights_bundle = risk_weights_bundle
        self.alerts: List[Dict] = []
        self.node_risk_scores: Dict[str, float] = {}
        self.detected_patterns: Dict[str, List] = defaultdict(list)
        # Centrality cache — betweenness is O(VE), expensive to recompute on
        # every call to compute_node_risk_scores (e.g. during re-detection).
        self._centrality_cache: Optional[Dict] = None

    def _cfg(self, key: str, default):
        return self.config.get(key, default)

    def detect_circular_transactions(self,
                                      amount_tolerance: Optional[float] = None,
                                      max_alerts: Optional[int] = None,
                                      min_total_flow: Optional[float] = None,
                                      max_cycle_length: Optional[int] = None) -> List[Dict]:
        amount_tolerance = amount_tolerance if amount_tolerance is not None else self._cfg("circular_amount_tolerance", 0.15)
        max_alerts = max_alerts if max_alerts is not None else self._cfg("circular_max_alerts", 50)
        min_total_flow = min_total_flow if min_total_flow is not None else self._cfg("circular_min_total_flow", 100_000)
        max_cycle_length = max_cycle_length if max_cycle_length is not None else self._cfg("circular_max_cycle_length", 8)
        """Detect circular / round-tripping patterns using Johnson's algorithm.

        This is the textbook approach for AML cycle detection. We:

          1. Decompose the graph into strongly-connected components (SCCs);
             only SCCs of size >= 3 can contain a cycle of length >= 3.
          2. Run nx.simple_cycles (which implements Johnson's algorithm) inside
             each non-trivial SCC, bounded by max_cycle_length.
          3. For each cycle, check the amount-variance and total-flow filters
             a round-trip would satisfy: amounts on every hop should be within
             `amount_tolerance` of their mean.

        Strongly-connected-component decomposition is what keeps this tractable
        on dense graphs — without it Johnson's would explode.
        """
        alerts = []
        seen_cycles = set()

        # Johnson's enumerates *every* simple cycle. On a dense graph that's
        # quickly into the millions — most of them random noise where the
        # edge amounts vary wildly. AML round-tripping cycles have the
        # opposite property: every edge in the cycle carries a similar
        # amount (the laundered tranche). So we filter edges into log-scale
        # amount buckets first and only enumerate cycles whose edges fall in
        # the same bucket. This combines Johnson's correctness with a
        # signal-aware pruning that matches the structure of the fraud we
        # actually want to detect.
        max_cycles_per_bucket = 5_000

        for scc in nx.strongly_connected_components(self.graph):
            if len(alerts) >= max_alerts:
                break
            if len(scc) < 3:
                continue
            sub = self.graph.subgraph(scc)

            # Group edges by log-amount bucket (0.2 = ±20% in log10 space).
            edges_by_bucket: Dict[int, List[Tuple[str, str]]] = defaultdict(list)
            for u, v in sub.edges():
                amt = self.graph[u][v]["total_amount"]
                if amt < min_total_flow / max(max_cycle_length, 1):
                    continue
                bucket = int(np.log10(max(amt, 1)) / 0.2)
                edges_by_bucket[bucket].append((u, v))

            for bucket_edges in edges_by_bucket.values():
                if len(alerts) >= max_alerts:
                    break
                if len(bucket_edges) < 3:
                    continue
                bucket_graph = nx.DiGraph()
                bucket_graph.add_edges_from(bucket_edges)
                if bucket_graph.number_of_nodes() < 3:
                    continue

                try:
                    cycles_iter = nx.simple_cycles(bucket_graph, length_bound=max_cycle_length)
                except TypeError:
                    def _bounded():
                        for c in nx.simple_cycles(bucket_graph):
                            if 3 <= len(c) <= max_cycle_length:
                                yield c
                    cycles_iter = _bounded()

                examined = 0
                for cycle in cycles_iter:
                    examined += 1
                    if examined >= max_cycles_per_bucket or len(alerts) >= max_alerts:
                        break
                    if len(cycle) < 3:
                        continue
                    cycle_key = frozenset(cycle)
                    if cycle_key in seen_cycles:
                        continue
                    seen_cycles.add(cycle_key)

                    # 3. Validate with the real edge amounts.
                    edge_amounts = []
                    valid = True
                    for i in range(len(cycle)):
                        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                        if self.graph.has_edge(u, v):
                            edge_amounts.append(self.graph[u][v]["total_amount"])
                        else:
                            valid = False
                            break
                    if not valid or not edge_amounts:
                        continue

                    total_flow = float(sum(edge_amounts))
                    if total_flow < min_total_flow:
                        continue
                    avg_amount = float(np.mean(edge_amounts))
                    max_dev = max(abs(a - avg_amount) / max(avg_amount, 1) for a in edge_amounts)
                    if max_dev > amount_tolerance:
                        continue

                    score = round(1.0 - max_dev, 4)
                    node_names = [self.graph.nodes[n].get("name", n) for n in cycle]
                    alerts.append({
                        "alert_id": f"ALERT_CIRC_{len(alerts) + 1:04d}",
                        "pattern_type": "Circular Transaction",
                        "severity": "HIGH" if total_flow > 5_000_000 else "MEDIUM",
                        "confidence": round(score * 100, 1),
                        "entities": list(cycle),
                        "entity_names": node_names,
                        "cycle_length": len(cycle),
                        "total_flow": round(total_flow, 2),
                        "avg_flow_per_edge": round(avg_amount, 2),
                        "amount_variance": round(max_dev * 100, 1),
                        "algorithm": "johnson_simple_cycles",
                        "description": (
                            f"Circular flow of ₹{total_flow:,.0f} detected through "
                            f"{len(cycle)} entities: {' → '.join(node_names)} → {node_names[0]}"
                        ),
                        "recommendation": "Flag accounts for enhanced due diligence. Verify business relationships between cycle participants.",
                    })
                    self.detected_patterns["circular"].append(alerts[-1])

        self.alerts.extend(alerts)
        return alerts

    def detect_rapid_layering(self,
                               time_window_hours: int = 48,
                               min_chain_length: Optional[int] = None) -> List[Dict]:
        """Detect rapid layering by finding chains of sequential transactions."""
        min_chain_length = min_chain_length if min_chain_length is not None else self._cfg("layering_min_chain_length", 3)
        decrease_ratio = self._cfg("layering_decrease_ratio", 0.85)
        max_chains_cfg = self._cfg("layering_max_chains", 200)
        # BFS branching factor — config-driven (was hardcoded to 5, missing many
        # chains on real graphs with high out-degree). Default raised to 10 so
        # production-scale graphs surface the same chains the demo does.
        max_branching = self._cfg("layering_max_branching_per_node", 10)
        alerts = []

        # Build adjacency with temporal info
        node_transactions = defaultdict(list)

        for u, v, data in self.graph.edges(data=True):
            try:
                first_seen = datetime.strptime(data["first_seen"], "%Y-%m-%d %H:%M:%S")
                last_seen = datetime.strptime(data["last_seen"], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue

            node_transactions[u].append({
                "target": v,
                "amount": data["total_amount"],
                "count": data["transaction_count"],
                "first_seen": first_seen,
                "last_seen": last_seen,
            })

        # Find chains where money moves rapidly through multiple accounts
        visited_chains = set()
        max_chains = max_chains_cfg

        for start_node in self.graph.nodes():
            if len(visited_chains) >= max_chains:
                break
            out_degree = self.graph.out_degree(start_node)

            # BFS to find chains, limit search to avoid explosion
            queue = [(start_node, [start_node], 0, None)]
            while queue and len(visited_chains) < max_chains:
                current, chain, depth, prev_time = queue.pop(0)

                if depth >= min_chain_length and len(chain) >= min_chain_length:
                    chain_key = tuple(chain)
                    if chain_key not in visited_chains:
                        visited_chains.add(chain_key)

                        # Calculate chain metrics
                        total_amount = 0
                        for i in range(len(chain) - 1):
                            if self.graph.has_edge(chain[i], chain[i + 1]):
                                total_amount += self.graph[chain[i]][chain[i + 1]]["total_amount"]

                        # Check for decreasing amounts (layering signature)
                        amounts = []
                        for i in range(len(chain) - 1):
                            if self.graph.has_edge(chain[i], chain[i + 1]):
                                amounts.append(self.graph[chain[i]][chain[i + 1]]["total_amount"])

                        is_decreasing = len(amounts) >= 2 and all(
                            amounts[i] >= amounts[i + 1] * decrease_ratio
                            for i in range(len(amounts) - 1)
                        )

                        # Check for shell companies in chain
                        has_shell = any(
                            self.graph.nodes[n].get("type") == "shell_company"
                            for n in chain
                        )

                        score = 0.7
                        if is_decreasing:
                            score += 0.15
                        if has_shell:
                            score += 0.1
                        if len(chain) >= 5:
                            score += 0.05

                        score = min(score, 1.0)

                        node_names = [
                            self.graph.nodes[n].get("name", n) for n in chain
                        ]

                        alert = {
                            "alert_id": f"ALERT_LAYER_{len(alerts) + 1:04d}",
                            "pattern_type": "Rapid Layering",
                            "severity": "CRITICAL" if total_amount > 10000000 else "HIGH",
                            "confidence": round(score * 100, 1),
                            "entities": chain,
                            "entity_names": node_names,
                            "chain_length": len(chain),
                            "total_flow": round(total_amount, 2),
                            "has_shell_company": has_shell,
                            "amounts_decreasing": is_decreasing,
                            "description": (
                                f"Layering chain of ₹{total_amount:,.0f} through "
                                f"{len(chain)} entities: {' → '.join(node_names)}"
                            ),
                            "recommendation": "Investigate source of funds. Verify legitimacy of intermediary entities. Check for shell company involvement.",
                        }
                        alerts.append(alert)
                        self.detected_patterns["layering"].append(alert)

                if depth >= 7:  # Max depth
                    continue

                for txn in node_transactions.get(current, [])[:max_branching]:
                    next_node = txn["target"]
                    if next_node not in chain:
                        queue.append((next_node, chain + [next_node], depth + 1, txn["last_seen"]))

        self.alerts.extend(alerts)
        return alerts

    def detect_smurfing(self,
                         threshold: Optional[float] = None,
                         cluster_tolerance: Optional[float] = None) -> List[Dict]:
        """Detect smurfing/structuring in two complementary ways:

          1. **Edge-level clustering** — multiple sender→receiver edges with
             low-variance amounts just below the reporting threshold.
             (The original detector — catches the "same mules, repeat
             transfers" pattern.)

          2. **Temporal bursts** — one sender firing N+ below-threshold txns
             to *any* receivers within an M-minute window. Catches the
             "fan-out fast" pattern the edge-level view can't see because
             each individual edge is small.

        Both share the same alert envelope (`pattern_type` =
        "Smurfing / Structuring") but carry different sub-fields
        (`detection_mode: "edge_cluster"|"temporal_burst"`).
        """
        threshold = threshold if threshold is not None else self._cfg("smurfing_threshold", 200_000)
        cluster_tolerance = cluster_tolerance if cluster_tolerance is not None else self._cfg("smurfing_cluster_tolerance", 0.10)
        burst_min_txns = self._cfg("smurfing_burst_min_txns", 5)
        burst_window_min = self._cfg("smurfing_burst_window_minutes", 60)
        alerts = []

        # ── Pattern 1: edge-level clustering (original) ──────────────────
        # Find edges with amounts just below reporting threshold
        suspicious_edges = []
        for u, v, data in self.graph.edges(data=True):
            avg = data["avg_amount"]
            if data["min_amount"] < threshold and data["transaction_count"] >= 2:
                # Check if amounts are clustered near threshold
                ratio = avg / threshold
                if 0.7 <= ratio <= 1.0:
                    variability = data["std_amount"] / max(data["avg_amount"], 1)
                    if variability < 0.15:  # Low variability = structured
                        suspicious_edges.append((u, v, data, variability))

        # Group by common sender or receiver
        sender_groups = defaultdict(list)
        receiver_groups = defaultdict(list)
        for u, v, data, var in suspicious_edges:
            sender_groups[u].append((v, data, var))
            receiver_groups[v].append((u, data, var))

        # Find senders sending to multiple receivers with similar amounts
        for sender, targets in sender_groups.items():
            if len(targets) < 2:
                continue

            total_amount = sum(d["total_amount"] for _, d, _ in targets)
            all_amounts = [d["avg_amount"] for _, d, _ in targets]
            avg_amount = np.mean(all_amounts)
            amount_spread = (max(all_amounts) - min(all_amounts)) / max(avg_amount, 1)

            if amount_spread < cluster_tolerance:
                score = min(0.95, 0.6 + len(targets) * 0.05)
                sender_name = self.graph.nodes[sender].get("name", sender)
                target_names = [self.graph.nodes[v].get("name", v) for v, _, _ in targets]

                alert = {
                    "alert_id": f"ALERT_SMURF_{len(alerts) + 1:04d}",
                    "pattern_type": "Smurfing / Structuring",
                    "detection_mode": "edge_cluster",
                    "severity": "HIGH" if total_amount > 5000000 else "MEDIUM",
                    "confidence": round(score * 100, 1),
                    "entities": [sender] + [v for v, _, _ in targets],
                    "entity_names": [sender_name] + target_names,
                    "n_splits": len(targets),
                    "total_flow": round(total_amount, 2),
                    "avg_split_amount": round(avg_amount, 2),
                    "amount_spread": round(amount_spread * 100, 1),
                    "description": (
                        f"Structured deposits of ₹{total_amount:,.0f} split into "
                        f"{len(targets)} transactions of ~₹{avg_amount:,.0f} each "
                        f"via intermediaries"
                    ),
                    "recommendation": "File STR (Suspicious Transaction Report). Verify source of funds and relationship between parties.",
                }
                alerts.append(alert)
                self.detected_patterns["smurfing"].append(alert)

        # ── Pattern 2: temporal bursts ───────────────────────────────────
        # Only computable from raw transactions; skip if not provided.
        if self.transactions is not None and len(self.transactions) > 0:
            alerts.extend(self._detect_smurfing_bursts(
                threshold=threshold,
                burst_min_txns=burst_min_txns,
                burst_window_min=burst_window_min,
                starting_seq=len(alerts),
            ))

        self.alerts.extend(alerts)
        return alerts

    def _detect_smurfing_bursts(self, threshold: float, burst_min_txns: int,
                                  burst_window_min: int, starting_seq: int) -> List[Dict]:
        """Find senders who fire >= burst_min_txns sub-threshold txns within a sliding window.

        For each sender, collect their outgoing txns sorted by time. Slide a
        `burst_window_min`-minute window over them; if any window contains
        `burst_min_txns` or more sub-threshold txns, that's a temporal burst.

        We don't require all txns to go to the same receiver — the whole point
        is to catch fan-out structuring where each mule receives only one.
        """
        burst_alerts = []
        txns = self.transactions
        below = txns[txns["amount"] < threshold].copy()
        if below.empty:
            return burst_alerts

        # Parse timestamps once
        below["_ts"] = pd.to_datetime(below["timestamp"], errors="coerce")
        below = below.dropna(subset=["_ts"])
        window = pd.Timedelta(minutes=burst_window_min)

        # Group by sender; sliding window over each sender's timeline.
        for sender, group in below.groupby("sender_id"):
            if len(group) < burst_min_txns:
                continue
            g = group.sort_values("_ts").reset_index(drop=True)
            n = len(g)
            l = 0
            best_burst = None   # (count, l, r, total)

            # Two-pointer sliding window
            for r in range(n):
                while g["_ts"].iat[r] - g["_ts"].iat[l] > window:
                    l += 1
                count = r - l + 1
                if count >= burst_min_txns:
                    total = float(g["amount"].iloc[l:r+1].sum())
                    if best_burst is None or count > best_burst[0]:
                        best_burst = (count, l, r, total)

            if best_burst is None:
                continue
            count, l, r, total = best_burst
            burst_rows = g.iloc[l:r+1]
            receivers = burst_rows["receiver_id"].unique().tolist()
            sender_name = self.graph.nodes[sender].get("name", sender) if self.graph.has_node(sender) else sender
            receiver_names = [
                self.graph.nodes[v].get("name", v) if self.graph.has_node(v) else v
                for v in receivers[:10]
            ]
            window_start = g["_ts"].iat[l].isoformat()
            window_end   = g["_ts"].iat[r].isoformat()
            window_seconds = (g["_ts"].iat[r] - g["_ts"].iat[l]).total_seconds()
            confidence = min(0.95, 0.5 + (count - burst_min_txns) * 0.05 + min(len(receivers), 20) * 0.02)

            alert = {
                "alert_id": f"ALERT_SMURF_{starting_seq + len(burst_alerts) + 1:04d}",
                "pattern_type": "Smurfing / Structuring",
                "detection_mode": "temporal_burst",
                "severity": "HIGH" if total > 1_000_000 or count >= burst_min_txns * 2 else "MEDIUM",
                "confidence": round(confidence * 100, 1),
                "entities": [sender] + receivers,
                "entity_names": [sender_name] + receiver_names,
                "n_txns_in_burst": int(count),
                "n_distinct_receivers": int(len(receivers)),
                "total_flow": round(total, 2),
                "burst_window_seconds": round(window_seconds, 0),
                "burst_window_start": window_start,
                "burst_window_end": window_end,
                "avg_amount": round(total / count, 2),
                "description": (
                    f"Sender '{sender_name}' fired {count} sub-threshold txns "
                    f"(<{threshold:,.0f}) totalling ₹{total:,.0f} to "
                    f"{len(receivers)} receivers in a {burst_window_min}-min window. "
                    f"Classic fan-out structuring."
                ),
                "recommendation": (
                    "File STR. Verify all receivers' KYC. Check if receivers are "
                    "mules (no return relationship with sender)."
                ),
            }
            burst_alerts.append(alert)
            self.detected_patterns["smurfing"].append(alert)

        return burst_alerts

    def detect_shell_funnels(self,
                              flow_imbalance_threshold: Optional[float] = None,
                              min_in_degree: Optional[int] = None) -> List[Dict]:
        """Detect shell-company funnels via two complementary patterns:

          1. **Flow imbalance** — much more inflow than outflow (classic
             collection funnel) or much more outflow than inflow (classic
             distribution funnel).
          2. **Pass-through** — inflow ≈ outflow AND money doesn't sit on the
             account (holding time < threshold). The current imbalance rule
             *misses* this because the flows look balanced. Pass-through is
             the dominant mule-account behaviour at PSB scale.

        Both attach `detection_mode` so a SAR-filing investigator knows which
        signature triggered the alert.
        """
        flow_imbalance_threshold = flow_imbalance_threshold if flow_imbalance_threshold is not None else self._cfg("funnel_imbalance_threshold", 0.7)
        min_in_degree = min_in_degree if min_in_degree is not None else self._cfg("funnel_min_in_degree", 3)
        pt_min_ratio = self._cfg("funnel_pass_through_min_ratio", 0.9)
        pt_max_holding = self._cfg("funnel_max_holding_seconds", 3600)
        pt_min_flow = self._cfg("funnel_pass_through_min_flow", 500_000)
        alerts = []

        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get("type", "")
            if node_type not in ("shell_company", "business"):
                continue

            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)

            if in_degree < min_in_degree:
                continue

            in_strength = sum(
                self.graph[u][node]["total_amount"]
                for u in self.graph.predecessors(node)
            )
            out_strength = sum(
                self.graph[node][v]["total_amount"]
                for v in self.graph.successors(node)
            )

            total_flow = in_strength + out_strength
            if total_flow == 0:
                continue

            imbalance = abs(in_strength - out_strength) / total_flow
            pass_through_ratio = min(in_strength, out_strength) / max(in_strength, out_strength, 1)
            avg_holding_seconds = self._avg_holding_time(node)

            triggers = []
            if imbalance >= flow_imbalance_threshold:
                triggers.append("flow_imbalance")
            if (pass_through_ratio >= pt_min_ratio
                    and avg_holding_seconds is not None
                    and avg_holding_seconds < pt_max_holding
                    and in_strength >= pt_min_flow and out_strength >= pt_min_flow):
                triggers.append("pass_through")

            if not triggers:
                continue

            in_partners = list(self.graph.predecessors(node))
            out_partners = list(self.graph.successors(node))
            branches = set()
            for p in in_partners:
                br = self.graph.nodes[p].get("branch", "")
                if br:
                    branches.add(br)

            # Score composition depends on which trigger(s) fired.
            score = 0.5
            if "flow_imbalance" in triggers:
                score += imbalance * 0.3
            if "pass_through" in triggers:
                # The shorter the holding time, the higher the score.
                score += 0.2 + (1.0 - min(avg_holding_seconds / pt_max_holding, 1.0)) * 0.15
            score += len(branches) * 0.05
            score = min(score, 0.95)

            node_name = self.graph.nodes[node].get("name", node)
            in_names = [self.graph.nodes[p].get("name", p) for p in in_partners]
            detection_mode = "+".join(triggers)

            if "flow_imbalance" in triggers:
                desc = (
                    f"₹{in_strength:,.0f} in / ₹{out_strength:,.0f} out at '{node_name}'. "
                    f"Imbalance {imbalance:.0%} across {len(in_partners)} sources / "
                    f"{len(branches)} branches."
                )
            else:
                # Pass-through only — different shape
                holding_str = f"{avg_holding_seconds/60:.1f}min" if avg_holding_seconds else "?"
                desc = (
                    f"'{node_name}' shows pass-through behaviour: ₹{in_strength:,.0f} in, "
                    f"₹{out_strength:,.0f} out (ratio {pass_through_ratio:.2f}), "
                    f"avg holding {holding_str}. Classic mule-account signature."
                )

            severity = "CRITICAL" if max(in_strength, out_strength) > 10_000_000 else "HIGH"

            alert = {
                "alert_id": f"ALERT_FUNNEL_{len(alerts) + 1:04d}",
                "pattern_type": "Shell Company Funnel",
                "detection_mode": detection_mode,
                "severity": severity,
                "confidence": round(score * 100, 1),
                "entities": [node] + in_partners,
                "entity_names": [node_name] + in_names,
                "funnel_entity": node,
                "funnel_name": node_name,
                "total_inflow": round(in_strength, 2),
                "total_outflow": round(out_strength, 2),
                "n_sources": len(in_partners),
                "branch_diversity": len(branches),
                "imbalance_ratio": round(imbalance, 4),
                "pass_through_ratio": round(pass_through_ratio, 4),
                "avg_holding_seconds": round(avg_holding_seconds, 1) if avg_holding_seconds is not None else None,
                "description": desc,
                "recommendation": "Verify business purpose. Check KYC of funnel entity. Investigate beneficial ownership.",
            }
            alerts.append(alert)
            self.detected_patterns["funnel"].append(alert)

        self.alerts.extend(alerts)
        return alerts

    def _avg_holding_time(self, node: str) -> Optional[float]:
        """Average time funds stay at `node` between arrival and departure, in seconds.

        Approximated via edge first_seen/last_seen on incoming and outgoing
        edges (we don't track per-txn matching). Returns None when either
        side has no temporal information.
        """
        in_times = []
        out_times = []
        for u in self.graph.predecessors(node):
            t = self.graph[u][node].get("last_seen")
            if t:
                try:
                    in_times.append(pd.to_datetime(t))
                except Exception:
                    pass
        for v in self.graph.successors(node):
            t = self.graph[node][v].get("first_seen")
            if t:
                try:
                    out_times.append(pd.to_datetime(t))
                except Exception:
                    pass
        if not in_times or not out_times:
            return None
        # Pair each outgoing txn with the most-recent prior incoming.
        in_sorted = sorted(in_times)
        deltas = []
        for ot in out_times:
            prior = [t for t in in_sorted if t <= ot]
            if not prior:
                continue
            deltas.append((ot - prior[-1]).total_seconds())
        if not deltas:
            return None
        return float(sum(deltas) / len(deltas))

    def compute_node_risk_scores(
        self,
        time_window_days: Optional[int] = None,
        risk_weights_bundle: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Compute composite risk score for each node.

        Two scoring modes:

        1. **Learned path (T2.10)** — when a trained risk-weights bundle is
           passed (from `src.risk_score_learner.load_risk_weights`) or stored
           on the instance via the constructor, uses the learned
           LogisticRegression instead of the hand-tuned weighted sum.

           The learned model uses a wider, leakage-free feature set (branch
           diversity, singleton-out signature, total transaction count) that
           the hand-tuned formula doesn't expose. See risk_score_learner.py
           for the full feature list and training procedure.

        2. **Hand-tuned fallback** — composite of entity type + degree
           centrality + betweenness centrality + flow imbalance.  If
           `time_window_days` is set, betweenness + degree centrality are
           computed on a subgraph containing only edges with `last_seen`
           within the window.  This is much closer to real AML practice:
           a shell node active in 2023 but quiet for two years shouldn't
           drive today's risk score.
        """
        # ── Learned path (T2.10) ──────────────────────────────────────────
        bundle = risk_weights_bundle or self.risk_weights_bundle
        if bundle:
            try:
                from risk_score_learner import score_nodes
                scores = score_nodes(self.graph, bundle)
                if scores:
                    self.node_risk_scores = scores
                    return scores
            except Exception:
                # Fall through to the hand-tuned path below if anything fails.
                pass

        # ── Hand-tuned fallback ───────────────────────────────────────────
        scores = {}

        # Cache key: include window so different windows don't collide.
        cache_key = time_window_days if time_window_days is not None else "all"
        if self._centrality_cache is None or self._centrality_cache.get("__key__") != cache_key:
            target = self.graph
            if time_window_days is not None and self.graph.number_of_edges() > 0:
                from datetime import datetime, timedelta
                last_seens = [
                    d.get("last_seen") for _, _, d in self.graph.edges(data=True)
                    if d.get("last_seen") is not None
                ]
                if last_seens:
                    max_ts = pd.to_datetime(max(last_seens))
                    cutoff = max_ts - timedelta(days=time_window_days)
                    keep_edges = [
                        (u, v) for u, v, d in self.graph.edges(data=True)
                        if d.get("last_seen") is not None
                        and pd.to_datetime(d["last_seen"]) >= cutoff
                    ]
                    target = self.graph.edge_subgraph(keep_edges).copy()
            try:
                betweenness = nx.betweenness_centrality(target, weight="total_amount")
            except Exception:
                betweenness = {n: 0 for n in self.graph.nodes()}
            try:
                degree_centrality = nx.degree_centrality(target)
            except Exception:
                degree_centrality = {n: 0 for n in self.graph.nodes()}
            self._centrality_cache = {
                "__key__": cache_key,
                "betweenness": betweenness,
                "degree": degree_centrality,
            }
        betweenness = self._centrality_cache["betweenness"]
        degree_centrality = self._centrality_cache["degree"]

        for node in self.graph.nodes():
            signals = []

            # Signal 1: Entity type
            entity_type = self.graph.nodes[node].get("type", "individual")
            if entity_type == "shell_company":
                signals.append(0.4)
            elif entity_type == "business":
                signals.append(0.1)
            else:
                signals.append(0.0)

            # Signal 2: Degree centrality (high connectivity = suspicious)
            signals.append(degree_centrality.get(node, 0) * 0.3)

            # Signal 3: Betweenness centrality (bridge between communities)
            signals.append(betweenness.get(node, 0) * 0.2)

            # Signal 4: Flow imbalance
            in_str = sum(
                self.graph[u][node]["total_amount"]
                for u in self.graph.predecessors(node)
            )
            out_str = sum(
                self.graph[node][v]["total_amount"]
                for v in self.graph.successors(node)
            )
            total = in_str + out_str
            if total > 0:
                imbalance = abs(in_str - out_str) / total
                signals.append(imbalance * 0.15)
            else:
                signals.append(0)

            # Signal 5: Fraud edge count
            fraud_edges = sum(
                1 for u, v in self.graph.in_edges(node)
                if self.graph[u][v].get("fraud_count", 0) > 0
            )
            fraud_edges += sum(
                1 for u, v in self.graph.out_edges(node)
                if self.graph[u][v].get("fraud_count", 0) > 0
            )
            signals.append(min(fraud_edges * 0.1, 0.3))

            scores[node] = round(min(sum(signals), 1.0), 4)

        self.node_risk_scores = scores
        return scores

    def run_all_detections(self) -> Dict[str, List[Dict]]:
        """Run all detection algorithms and return comprehensive results."""
        print("Running Circular Transaction Detection...")
        circular = self.detect_circular_transactions()

        print("Running Rapid Layering Detection...")
        layering = self.detect_rapid_layering()

        print("Running Smurfing Detection...")
        smurfing = self.detect_smurfing()

        print("Running Shell Company Funnel Detection...")
        funnels = self.detect_shell_funnels()

        print("Computing Node Risk Scores...")
        risk_scores = self.compute_node_risk_scores(risk_weights_bundle=self.risk_weights_bundle)

        results = {
            "circular_transactions": circular,
            "rapid_layering": layering,
            "smurfing": smurfing,
            "shell_funnels": funnels,
            "all_alerts": self.alerts,
            "node_risk_scores": risk_scores,
            "summary": {
                "total_alerts": len(self.alerts),
                "circular_count": len(circular),
                "layering_count": len(layering),
                "smurfing_count": len(smurfing),
                "funnel_count": len(funnels),
                "high_risk_nodes": sum(1 for s in risk_scores.values() if s >= 0.5),
                "critical_alerts": sum(1 for a in self.alerts if a["severity"] == "CRITICAL"),
                "high_alerts": sum(1 for a in self.alerts if a["severity"] == "HIGH"),
                "medium_alerts": sum(1 for a in self.alerts if a["severity"] == "MEDIUM"),
            },
        }

        print(f"\nDetection Summary:")
        print(f"  Total Alerts: {results['summary']['total_alerts']}")
        print(f"  Circular Transactions: {results['summary']['circular_count']}")
        print(f"  Rapid Layering: {results['summary']['layering_count']}")
        print(f"  Smurfing: {results['summary']['smurfing_count']}")
        print(f"  Shell Funnels: {results['summary']['funnel_count']}")
        print(f"  High-Risk Nodes: {results['summary']['high_risk_nodes']}")

        return results

    def save_results(self, results: Dict, output_dir: str = "data"):
        """Save detection results to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save alerts
        alerts_json = []
        for alert in results["all_alerts"]:
            alerts_json.append(alert)
        with open(os.path.join(output_dir, "fraud_alerts.json"), "w") as f:
            json.dump(alerts_json, f, indent=2, default=str)

        # Save risk scores
        risk_data = []
        for node_id, score in results["node_risk_scores"].items():
            node_data = dict(self.graph.nodes[node_id])
            risk_data.append({
                "entity_id": node_id,
                "name": node_data.get("name", ""),
                "type": node_data.get("type", ""),
                "risk_score": score,
                "risk_level": "CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.3 else "LOW",
            })
        risk_data.sort(key=lambda x: x["risk_score"], reverse=True)

        with open(os.path.join(output_dir, "risk_scores.json"), "w") as f:
            json.dump(risk_data, f, indent=2)

        # Save summary
        with open(os.path.join(output_dir, "detection_summary.json"), "w") as f:
            json.dump(results["summary"], f, indent=2)

        print(f"\nResults saved to {output_dir}/")
