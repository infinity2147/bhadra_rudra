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
        import time as _time
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
        # Cap SCC size + wall-clock per SCC. On IBM AML the bank-to-bank
        # graph has one giant SCC of 50k+ nodes with 10^9+ simple cycles —
        # AML rings are tight clusters of 3-20 shell accounts, not the
        # whole interbank topology. Skipping the giant SCC matches the
        # structure of the fraud and keeps the pipeline tractable.
        scc_size_cap = self._cfg("circular_scc_size_cap", 200)
        scc_time_budget = float(self._cfg("circular_scc_time_budget_s", 8.0))
        scc_skipped = 0

        for scc in nx.strongly_connected_components(self.graph):
            if len(alerts) >= max_alerts:
                break
            if len(scc) < 3:
                continue
            if len(scc) > scc_size_cap:
                scc_skipped += 1
                continue
            scc_start = _time.perf_counter()
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
                    # Per-SCC wall-clock budget so a single degenerate SCC
                    # can't lock the pipeline forever.
                    if _time.perf_counter() - scc_start > scc_time_budget:
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
                               time_window_hours: Optional[int] = None,
                               min_chain_length: Optional[int] = None) -> List[Dict]:
        """Detect rapid layering: a single tranche of funds relayed through a
        chain of intermediary accounts quickly enough — and with the amount
        preserved closely enough — that the chain reads as one laundering route
        rather than unrelated transfers that merely share endpoints.

        A path A→B→C→… qualifies only when EVERY consecutive hop satisfies all
        three foundational constraints:

          1. **Temporal causality + rapidity.** Money cannot leave an account
             before it arrives, and a layering relay forwards it *fast*. Each
             aggregated edge carries a [first_seen, last_seen] range, so we test
             interval feasibility — there must exist an arrival time on the
             incoming edge and a departure on the outgoing edge with
             ``arrival <= departure <= arrival + window``::

                 outgoing.last_seen  >= incoming.first_seen           (causal)
                 outgoing.first_seen <= incoming.last_seen + window    (rapid)

             A hop dated before its predecessor, or a month after it, fails.
             This is the gate that makes the detector about *layering* and not
             "any path of length ≥ 3". The previous implementation threaded a
             ``prev_time`` through the search and then never compared it, so the
             temporal signal — the whole point of *rapid* layering — was
             silently discarded.

          2. **Amount preservation.** A relay passes (nearly) the whole tranche
             on; it neither accumulates nor mixes in unrelated money. Each hop
             must land in ``[decrease_ratio, 1 + grow_tolerance] × prev_amount``.

          3. **Materiality.** Every hop must move at least ``min_hop_amount`` so
             dust can't form a chain.

        Search strategy (the two flaws the team reported):

          * Start nodes are processed **largest-outflow-first**, not in
            graph-insertion order, so the ``max_chains`` budget is spent on the
            biggest money movements rather than whichever nodes were inserted
            first.
          * At each node we follow outgoing edges **by amount, descending**, and
            keep the first ``max_branching`` that pass the gates. The old code
            sliced the adjacency list in insertion order, so a launderer could
            bury the real transfer behind a few tiny decoys and evade detection.
            Following the money by size — with the materiality and
            amount-preservation gates — defeats that: decoys sort to the bottom
            and fail the gates anyway.

        Discovery collects every **maximal** causal walk; emission then drops
        any chain whose entity set is contained in a longer one, so the alert
        stream isn't flooded with every prefix/suffix of one route (and the
        result is independent of the order chains were discovered in).

        Operates purely on the aggregated graph (edge amount + first/last_seen),
        so behaviour is identical whether or not raw ``transactions`` were
        supplied to the detector.
        """
        window_hours = time_window_hours if time_window_hours is not None else self._cfg("layering_time_window_hours", 48)
        window = timedelta(hours=float(window_hours))
        max_span = timedelta(hours=float(self._cfg("layering_max_chain_span_hours", 72)))
        min_chain_length = min_chain_length if min_chain_length is not None else self._cfg("layering_min_chain_length", 4)
        decrease_ratio = float(self._cfg("layering_decrease_ratio", 0.85))
        grow_tolerance = float(self._cfg("layering_amount_grow_tolerance", 0.15))
        max_chains = int(self._cfg("layering_max_chains", 200))
        max_branching = int(self._cfg("layering_max_branching_per_node", 10))
        max_depth = int(self._cfg("layering_max_depth", 8))
        min_hop_amount = float(self._cfg("layering_min_hop_amount", 50_000))
        max_start_nodes = int(self._cfg("layering_max_start_nodes", 5_000))
        # How far down a node's amount-ranked edge list to scan for valid
        # continuations before giving up — bounds work at high-degree hubs.
        scan_cap = max(max_branching * 8, 64)

        def _parse(ts):
            try:
                t = pd.to_datetime(ts)
                return None if pd.isna(t) else t
            except (ValueError, TypeError):
                return None

        # Build amount-ranked, time-stamped adjacency once. Edges below the dust
        # floor, or with unparseable timestamps, are dropped (we can't reason
        # about causality without time).
        adj: Dict[str, List[Tuple]] = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            amt = float(data.get("total_amount", 0.0) or 0.0)
            if amt < min_hop_amount:
                continue
            tf = _parse(data.get("first_seen"))
            tl = _parse(data.get("last_seen"))
            if tf is None or tl is None:
                continue
            adj[u].append((v, amt, tf, tl))
        for u in adj:
            adj[u].sort(key=lambda e: e[1], reverse=True)

        # Seed largest-outflow-first; cap the seed set on very large graphs.
        start_nodes = sorted(adj.keys(), key=lambda n: adj[n][0][1], reverse=True)[:max_start_nodes]

        def _hop_ok(prev_first, prev_amt, nxt_first, nxt_amt) -> bool:
            # Gate on each relationship's FIRST activity, not its [first,last]
            # range. Range-feasibility (next.first <= prev.last + window) is
            # silently disabled for any pair that transacts repeatedly: their
            # last_seen sits months out, so almost any next hop "fits the
            # window" and chains end up spanning weeks — the opposite of rapid.
            # Requiring consecutive first-activity times to be monotonic and
            # within `window` keeps the relay actually rapid.
            if not (prev_first <= nxt_first <= prev_first + window):
                return False
            return (prev_amt * decrease_ratio) <= nxt_amt <= (prev_amt * (1.0 + grow_tolerance))

        # ── Discovery: collect maximal causal walks ───────────────────────────
        candidates: List[Tuple] = []          # (chain, amounts, firsts, lasts)
        seen_paths = set()
        cand_cap = max(max_chains * 5, 1_000)
        for start in start_nodes:
            if len(candidates) >= cand_cap:
                break
            stack = [(start, [start], [], [], [])]   # node, chain, amounts, firsts, lasts
            while stack:
                node, chain, amounts, firsts, lasts = stack.pop()
                extended = False
                if len(chain) < max_depth:
                    prev_amt = amounts[-1] if amounts else None
                    prev_first = firsts[-1] if firsts else None
                    prev_last = lasts[-1] if lasts else None
                    chain_start = firsts[0] if firsts else None
                    pushed = 0
                    for (v, amt, tf, tl) in adj.get(node, [])[:scan_cap]:
                        if v in chain:
                            continue
                        if prev_amt is not None and not _hop_ok(prev_first, prev_amt, tf, amt):
                            continue
                        # Whole-chain span cap: a rapid relay completes fast; a
                        # chain that dribbles across many windows isn't "rapid".
                        if chain_start is not None and (tf - chain_start) > max_span:
                            continue
                        stack.append((v, chain + [v], amounts + [amt], firsts + [tf], lasts + [tl]))
                        extended = True
                        pushed += 1
                        if pushed >= max_branching:
                            break
                if not extended and len(chain) >= min_chain_length:
                    key = tuple(chain)
                    if key not in seen_paths:
                        seen_paths.add(key)
                        candidates.append((chain, amounts, firsts, lasts))
                        if len(candidates) >= cand_cap:
                            break

        # ── Dedup: drop chains contained in a longer one (longest-first) ──────
        candidates.sort(key=lambda c: len(c[0]), reverse=True)
        kept: List[Tuple] = []
        kept_sets: List[frozenset] = []
        for cand in candidates:
            ns = frozenset(cand[0])
            if any(ns <= ks for ks in kept_sets):
                continue
            kept.append(cand)
            kept_sets.append(ns)

        # Emit biggest-bottleneck-first so the budget keeps the largest schemes.
        kept.sort(key=lambda c: min(c[1]), reverse=True)

        alerts: List[Dict] = []
        window_minutes = window.total_seconds() / 60.0
        for chain, amounts, firsts, lasts in kept[:max_chains]:
            bottleneck = float(min(amounts))      # the tranche that traversed the whole chain
            gross = float(sum(amounts))            # sum of legs — same money counted once per hop
            gaps = [max(0.0, (firsts[i] - firsts[i - 1]).total_seconds()) / 60.0
                    for i in range(1, len(firsts))]
            mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
            span_minutes = max(0.0, (firsts[-1] - firsts[0]).total_seconds()) / 60.0
            is_decreasing = all(amounts[i] <= amounts[i - 1] for i in range(1, len(amounts)))
            preservation = (sum(min(amounts[i] / amounts[i - 1], 1.0) for i in range(1, len(amounts)))
                            / max(len(amounts) - 1, 1))
            tightness = (1.0 - min(mean_gap / window_minutes, 1.0)) if window_minutes else 0.0
            has_shell = any(self.graph.nodes[n].get("type") == "shell_company" for n in chain)

            score = 0.55
            score += 0.18 * tightness          # the faster the relay, the more suspicious
            score += 0.12 * preservation       # the cleaner the pass-through, the more suspicious
            score += min(len(chain) - min_chain_length, 3) * 0.03
            score += 0.06 if has_shell else 0.0
            score = min(max(score, 0.5), 0.98)

            node_names = [self.graph.nodes[n].get("name", n) for n in chain]
            alert = {
                "alert_id": f"ALERT_LAYER_{len(alerts) + 1:04d}",
                "pattern_type": "Rapid Layering",
                "severity": "CRITICAL" if bottleneck > 10_000_000 else "HIGH",
                "confidence": round(score * 100, 1),
                "entities": list(chain),
                "entity_names": node_names,
                "chain_length": len(chain),
                "total_flow": round(bottleneck, 2),        # the laundered tranche (bottleneck)
                "gross_chain_flow": round(gross, 2),       # sum of every hop (secondary)
                "bottleneck_amount": round(bottleneck, 2),
                "has_shell_company": has_shell,
                "amounts_decreasing": is_decreasing,
                "span_minutes": round(span_minutes, 1),
                "avg_hop_gap_minutes": round(mean_gap, 1),
                "algorithm": "temporal_causal_walk",
                "description": (
                    f"₹{bottleneck:,.0f} relayed through {len(chain)} entities in "
                    f"{span_minutes / 60:.1f}h: {' → '.join(node_names)}"
                ),
                "recommendation": "Investigate source of funds. Verify legitimacy of intermediary entities. Check for shell company involvement.",
            }
            alerts.append(alert)
            self.detected_patterns["layering"].append(alert)

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
        cluster_min_ratio = self._cfg("smurfing_cluster_min_ratio", 0.5)
        cluster_max_ratio = self._cfg("smurfing_cluster_max_ratio", 1.0)
        burst_min_txns = self._cfg("smurfing_burst_min_txns", 5)
        burst_window_min = self._cfg("smurfing_burst_window_minutes", 60)
        alerts = []

        # ── Pattern 1: edge-level clustering (original) ──────────────────
        # Find edges with amounts below reporting threshold. The proximity band
        # is config-driven (was hardcoded 0.7-1.0): structuring well below the
        # threshold is still structuring, and low size-variance is what actually
        # marks it as deliberate.
        suspicious_edges = []
        for u, v, data in self.graph.edges(data=True):
            avg = data["avg_amount"]
            if data["min_amount"] < threshold and data["transaction_count"] >= 2:
                ratio = avg / threshold
                if cluster_min_ratio <= ratio <= cluster_max_ratio:
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
            # ── Pattern 3: window-independent fan-out structuring ──────────
            alerts.extend(self._detect_smurfing_fanout(
                threshold=threshold,
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

    def _detect_smurfing_fanout(self, threshold: float, starting_seq: int) -> List[Dict]:
        """Window-INDEPENDENT fan-out structuring.

        Catches the "single-shot fan-out" the other two modes miss: one sender
        spraying sub-threshold transfers *once each* to many distinct mules.
        Mode 1 needs ≥ 2 transfers to the *same* receiver; Mode 2 needs them
        inside one short window. A launderer who sends ₹1.9L once to each of 50
        fresh accounts, spaced over days, evades both — but the signature
        (many distinct receivers, each hit ~once, all sub-threshold, all
        similarly sized) is unmistakable and timing-agnostic.

        Deliberately uses no time window: spacing transfers out is exactly the
        evasion, so the detector must not depend on how they're spaced.
        """
        fanout_alerts: List[Dict] = []
        min_txns = int(self._cfg("smurfing_fanout_min_txns", 8))
        min_receivers = int(self._cfg("smurfing_fanout_min_receivers", 5))
        max_per_receiver = float(self._cfg("smurfing_fanout_max_txns_per_receiver", 1.5))
        band_tol = float(self._cfg("smurfing_fanout_band_tolerance", 0.15))

        txns = self.transactions
        below = txns[txns["amount"] < threshold]
        if below.empty:
            return fanout_alerts

        from collections import Counter as _Counter
        for sender, group in below.groupby("sender_id"):
            if len(group) < min_txns:
                continue
            # Isolate the structured subset as the tightest amount-band cluster
            # of this sender's sub-threshold transfers — a launderer mixes
            # structuring into normal activity, so requiring the WHOLE set to be
            # uniform misses it. Sort by amount and slide a relative-width band;
            # the widest cluster (most distinct receivers) is the candidate.
            g2 = group.sort_values("amount").reset_index(drop=True)
            amts = g2["amount"].to_numpy(dtype=float)
            recv_arr = g2["receiver_id"].to_numpy()
            recv_counts: "_Counter" = _Counter()
            best = None        # (n_distinct, l, r, count)
            l = 0
            for r in range(len(amts)):
                recv_counts[recv_arr[r]] += 1
                while l < r and (amts[r] - amts[l]) > band_tol * max(amts[l], 1.0):
                    recv_counts[recv_arr[l]] -= 1
                    if recv_counts[recv_arr[l]] == 0:
                        del recv_counts[recv_arr[l]]
                    l += 1
                count = r - l + 1
                distinct = len(recv_counts)
                if count >= min_txns and distinct >= min_receivers and count / distinct <= max_per_receiver:
                    if best is None or distinct > best[0]:
                        best = (distinct, l, r, count)
            if best is None:
                continue
            _, bl, br, _ = best
            window = g2.iloc[bl:br + 1]
            receivers = window["receiver_id"].unique()
            n_recv = len(receivers)
            amounts = window["amount"].to_numpy(dtype=float)
            mean_amt = float(amounts.mean())
            total = float(amounts.sum())
            n = len(window)
            sender_name = self.graph.nodes[sender].get("name", sender) if self.graph.has_node(sender) else sender
            receiver_names = [
                self.graph.nodes[r].get("name", r) if self.graph.has_node(r) else r
                for r in receivers[:10]
            ]
            # Closer to the threshold + more mules + tighter sizing → higher confidence.
            cluster_cv = float(amounts.std() / mean_amt) if mean_amt > 0 else 1.0
            proximity = min(mean_amt / threshold, 1.0)
            confidence = min(
                0.95,
                0.45 + min(n_recv, 30) * 0.015 + proximity * 0.2 + max(0.0, band_tol - cluster_cv),
            )

            alert = {
                "alert_id": f"ALERT_SMURF_{starting_seq + len(fanout_alerts) + 1:04d}",
                "pattern_type": "Smurfing / Structuring",
                "detection_mode": "fan_out",
                "severity": "HIGH" if total > 1_000_000 or n_recv >= min_receivers * 2 else "MEDIUM",
                "confidence": round(confidence * 100, 1),
                "entities": [sender] + list(receivers),
                "entity_names": [sender_name] + receiver_names,
                "n_txns": int(n),
                "n_distinct_receivers": int(n_recv),
                "txns_per_receiver": round(n / n_recv, 2),
                "total_flow": round(total, 2),
                "avg_amount": round(mean_amt, 2),
                "amount_cv": round(cluster_cv, 4),
                "description": (
                    f"Sender '{sender_name}' sprayed {n} sub-threshold txns "
                    f"(<{threshold:,.0f}, ~₹{mean_amt:,.0f} each) once each to "
                    f"{n_recv} distinct receivers. Fan-out structuring "
                    f"(timing-agnostic)."
                ),
                "recommendation": (
                    "File STR. Verify KYC of all receivers. One-shot fan-out to "
                    "many fresh accounts is a classic mule-recruitment signature."
                ),
            }
            fanout_alerts.append(alert)
            self.detected_patterns["smurfing"].append(alert)

        return fanout_alerts

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
            # Account *type* is a scoring signal, not a gate. Compromised
            # individual retail accounts are the most common pass-through mules
            # globally; gating on shell_company/business blinded the detector to
            # them. The behavioural gates below (in-degree, flow imbalance,
            # pass-through holding time + min flow) are what separate a mule
            # from a normal account — and they apply to every account type.

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
            # Entity type is a signal, not a gate: a shell behaving like a funnel
            # is more suspicious than an individual doing the same, but both are
            # flagged on the behaviour above.
            score += {"shell_company": 0.1, "business": 0.05}.get(node_type, 0.0)
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

    def detect_recruiters(self,
                          min_fanout: Optional[int] = None,
                          pass_through_ratio: Optional[float] = None,
                          min_seed_amount: Optional[float] = None) -> List[Dict]:
        """Detect the **recruiter / coordinator** upstream of a mule fleet.

        Mule detection asks "is this account a pass-through?". This asks the more
        actionable question: *who is funding the pass-throughs?* A coordinator is
        a node that seeds money into many accounts which then forward it on. We
        flag the orchestrator — the highest-value target — not just the
        disposable mules.

        An account U is a coordinator when at least ``min_fanout`` of its direct
        recipients are "mule-like": they both receive and send, and forward most
        of what they receive (``min(in,out)/max(in,out) >= pass_through_ratio``).
        Only seed transfers of at least ``min_seed_amount`` count, so ordinary
        disbursement (salary, refunds) to consumers — who don't forward — is not
        mistaken for recruitment.
        """
        min_fanout = int(min_fanout if min_fanout is not None else self._cfg("recruiter_min_fanout", 5))
        pt = float(pass_through_ratio if pass_through_ratio is not None else self._cfg("recruiter_pass_through_ratio", 0.6))
        min_seed = float(min_seed_amount if min_seed_amount is not None else self._cfg("recruiter_min_seed_amount", 10_000))
        min_funding_share = float(self._cfg("recruiter_min_funding_share", 0.3))
        alerts: List[Dict] = []

        in_str: Dict[str, float] = {}
        out_str: Dict[str, float] = {}
        for n in self.graph.nodes():
            in_str[n] = sum(self.graph[u][n]["total_amount"] for u in self.graph.predecessors(n))
            out_str[n] = sum(self.graph[n][v]["total_amount"] for v in self.graph.successors(n))

        def _is_mule_like(r: str) -> bool:
            i, o = in_str.get(r, 0.0), out_str.get(r, 0.0)
            if i <= 0 or o <= 0:
                return False
            return min(i, o) / max(i, o) >= pt

        for u in self.graph.nodes():
            recruited = []
            seeded_total = 0.0
            for v in self.graph.successors(u):
                edge_amt = self.graph[u][v]["total_amount"]
                if edge_amt < min_seed:
                    continue
                # U must be a DOMINANT funder of v — supplying a real share of v's
                # inflow. This is what separates a coordinator funding its fleet
                # from a node that merely sends a little to a busy account.
                if edge_amt < min_funding_share * in_str.get(v, edge_amt):
                    continue
                if _is_mule_like(v):
                    recruited.append(v)
                    seeded_total += edge_amt
            if len(recruited) < min_fanout:
                continue

            downstream_outflow = float(sum(out_str.get(v, 0.0) for v in recruited))
            score = min(0.95, 0.5 + min(len(recruited), 20) * 0.03)
            u_name = self.graph.nodes[u].get("name", u)
            recruited_names = [self.graph.nodes[v].get("name", v) for v in recruited[:15]]
            severity = "CRITICAL" if (len(recruited) >= min_fanout * 2 or downstream_outflow > 10_000_000) else "HIGH"

            alert = {
                "alert_id": f"ALERT_RECRUIT_{len(alerts) + 1:04d}",
                "pattern_type": "Recruiter / Coordinator",
                "severity": severity,
                "confidence": round(score * 100, 1),
                "entities": [u] + recruited,
                "entity_names": [u_name] + recruited_names,
                "coordinator": u,
                "coordinator_name": u_name,
                "n_recruited": len(recruited),
                "total_flow": round(seeded_total, 2),
                "seeded_amount": round(seeded_total, 2),
                "downstream_outflow": round(downstream_outflow, 2),
                "description": (
                    f"'{u_name}' seeded ₹{seeded_total:,.0f} into {len(recruited)} accounts "
                    f"that forward funds onward — coordinator/recruiter signature. "
                    f"Downstream relay volume ₹{downstream_outflow:,.0f}."
                ),
                "recommendation": (
                    "Investigate the coordinator as the principal, not just the mules. "
                    "Freeze the fleet together; map shared KYC / device / funding signals."
                ),
            }
            alerts.append(alert)
            self.detected_patterns["recruiter"].append(alert)

        self.alerts.extend(alerts)
        return alerts

    def _avg_holding_time(self, node: str) -> Optional[float]:
        """Amount-weighted average time funds stay at `node`, in seconds.

        Funds are matched FIFO **by amount**: each outflow consumes the oldest
        still-available inflows that arrived before it, and each matched rupee
        contributes ``(departure - arrival)`` to an amount-weighted average.

        This replaces the previous "subtract the single most-recent inflow"
        rule, which a launderer could game (or which would mis-fire on an
        innocent long-hold account): a ₹10 decoy deposited five minutes before a
        ₹10M exit made holding time look like five minutes even though the ₹10M
        had been sitting for months. With FIFO-by-amount the ₹10M exit is
        matched to the ₹10M deposit and the true (long) holding time dominates;
        the ₹10 decoy is just a negligible-weight tranche.

        Times are approximated from edge first_seen (we don't track per-txn
        matching on aggregated edges). Returns None when either side is empty
        or nothing can be matched.
        """
        deposits = []     # (arrival_time, amount)
        for u in self.graph.predecessors(node):
            ed = self.graph[u][node]
            t = ed.get("first_seen")
            if t:
                try:
                    deposits.append((pd.to_datetime(t), float(ed.get("total_amount", 0.0) or 0.0)))
                except Exception:
                    pass
        withdrawals = []  # (departure_time, amount)
        for v in self.graph.successors(node):
            ed = self.graph[node][v]
            t = ed.get("first_seen")
            if t:
                try:
                    withdrawals.append((pd.to_datetime(t), float(ed.get("total_amount", 0.0) or 0.0)))
                except Exception:
                    pass
        if not deposits or not withdrawals:
            return None

        from collections import deque
        deposits.sort(key=lambda d: d[0])         # oldest first (FIFO)
        withdrawals.sort(key=lambda w: w[0])
        queue = deque([t, a] for t, a in deposits)  # mutable remaining balances

        weighted_holding = 0.0
        matched_amount = 0.0
        for wt, wa in withdrawals:
            remaining = wa
            while remaining > 1e-9 and queue:
                dt, da = queue[0]
                if dt > wt:                       # oldest deposit is after this exit → unfunded
                    break
                take = min(remaining, da)
                weighted_holding += max(0.0, (wt - dt).total_seconds()) * take
                matched_amount += take
                remaining -= take
                da -= take
                if da <= 1e-9:
                    queue.popleft()
                else:
                    queue[0][1] = da
        if matched_amount <= 0:
            return None
        return float(weighted_holding / matched_amount)

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
            # Exact betweenness is O(VE) — on 100k+ node graphs (IBM AML)
            # that's multi-hour. Monte Carlo with k pivots (BFS, unweighted)
            # gives a usable ranking in seconds at any scale. Variance scales
            # as O(1/sqrt(k)); k=500 is plenty for risk ordering.
            n_nodes = target.number_of_nodes()
            sample_k = self._cfg("centrality_sample_k", 500)
            try:
                if n_nodes > 2_000 and sample_k and sample_k < n_nodes:
                    # Scale pivots with graph size (config value is the floor,
                    # capped at 2000) — error ∝ 1/sqrt(k), so a flat 500 on a
                    # 70k-node graph was needlessly noisy. Seeded → deterministic.
                    k_eff = min(max(int(sample_k), int(0.05 * n_nodes)), 2000, n_nodes)
                    betweenness = nx.betweenness_centrality(target, k=k_eff, seed=42)
                else:
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

        print("Running Recruiter / Coordinator Detection...")
        recruiters = self.detect_recruiters()

        print("Computing Node Risk Scores...")
        risk_scores = self.compute_node_risk_scores(risk_weights_bundle=self.risk_weights_bundle)

        results = {
            "circular_transactions": circular,
            "rapid_layering": layering,
            "smurfing": smurfing,
            "shell_funnels": funnels,
            "recruiters": recruiters,
            "all_alerts": self.alerts,
            "node_risk_scores": risk_scores,
            "summary": {
                "total_alerts": len(self.alerts),
                "circular_count": len(circular),
                "layering_count": len(layering),
                "smurfing_count": len(smurfing),
                "funnel_count": len(funnels),
                "recruiter_count": len(recruiters),
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
        print(f"  Recruiters/Coordinators: {results['summary']['recruiter_count']}")
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
