"""
LLM Copilot — AI-powered investigation assistant.
Uses the Claude API (Haiku) with tool-calling to answer investigator queries
over the live fund flow graph. Set ANTHROPIC_API_KEY to enable; without it,
queries fall back to a deterministic quick-commands router.

Tools: trace_funds(), find_cycles(), explain_alert(), get_profile_delta()
"""

import json
import os
import re
from typing import Dict, List, Optional, Any
import pandas as pd
import networkx as nx


# ── Response-enforcement layer ────────────────────────────────────────────────
# The model's formatting instructions resist prompt-only control (it keeps
# appending option-menus and emojis). These post-processors guarantee the
# enterprise format regardless of what the model emits.

_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002300-\U000023FF\U00002B00-\U00002BFF️]",
    flags=re.UNICODE,
)

# Phrases that open a trailing "what would you like to do next" menu. Whenever
# one appears, everything from it onward is a closer-menu we strip — answers
# should end on the answer, not a list of options.
_MENU_TRIGGERS = (
    "what would you like",
    "what else would you like",
    "would you like me to",
    "is there anything else",
    "anything else i can",
    "how would you like to proceed",
    "i can help you with",
    "here are some things i can",
    "you can ask me",
    "let me know what",
    "what can i help",
    "how can i help you",
)


def _find_menu_cut(text: str) -> Optional[int]:
    """Index where a trailing option-menu begins, or None."""
    low = text.lower()
    cut = None
    for pat in _MENU_TRIGGERS:
        i = low.find(pat)
        if i != -1 and (cut is None or i < cut):
            cut = i
    return cut


def _strip_response(text: str) -> str:
    """Enforce enterprise format on a complete response: drop trailing menus,
    emojis, and dangling list-header lines."""
    cut = _find_menu_cut(text)
    # Only cut when real content precedes the trigger. A trigger at the very
    # start means the whole response is a standalone clarifying question
    # ("What would you like to investigate?") — keep it, don't blank it out.
    if cut is not None and text[:cut].strip():
        text = text[:cut]
    text = _EMOJI_RE.sub("", text)
    # Drop trailing blank lines and dangling "header:" lines left behind once
    # the menu under them was removed.
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or lines[-1].strip().endswith(":")):
        lines.pop()
    return "\n".join(lines).rstrip()


class LLMCopilot:
    """AI copilot that answers investigator queries using graph tools."""

    def __init__(self, graph: nx.DiGraph, transactions: pd.DataFrame,
                 alerts: List[Dict], risk_scores: List[Dict],
                 fraud_cases: List[Dict], api_key: Optional[str] = None,
                 model_bundle: Optional[Dict] = None):
        self.graph = graph
        self.transactions = transactions
        self.alerts = alerts
        self.risk_scores = risk_scores
        self.fraud_cases = fraud_cases
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # ML bundle (XGB + feature columns + SHAP background sample). When
        # present, explain_alert uses real SHAP attributions instead of the
        # legacy hardcoded reasoning_chain templates.
        self.model_bundle = model_bundle
        self.conversation_history: List[Dict] = []
        self.tool_results_log: List[Dict] = []

    # ── Tool Definitions ──────────────────────────────────────

    def trace_funds(self, entity_name: str, depth: int = 3, direction: str = "both") -> Dict:
        """Trace fund flows from/to a given entity.

        Args:
            entity_name: Name or partial name of the entity to trace.
            depth: How many hops to trace (1-5).
            direction: 'incoming', 'outgoing', or 'both'.
        """
        # Find matching node
        node_id = self._find_node(entity_name)
        if not node_id:
            return {"error": f"Entity '{entity_name}' not found. Try a different name or partial match."}

        node_data = dict(self.graph.nodes[node_id])
        result = {
            "entity": {"id": node_id, **node_data},
            "depth": depth,
            "direction": direction,
            "flows": {"incoming": [], "outgoing": []},
        }

        if direction in ("incoming", "both"):
            result["flows"]["incoming"] = self._trace_direction(node_id, "incoming", depth)
        if direction in ("outgoing", "both"):
            result["flows"]["outgoing"] = self._trace_direction(node_id, "outgoing", depth)

        # Summary stats
        in_total = sum(f["amount"] for f in result["flows"]["incoming"])
        out_total = sum(f["amount"] for f in result["flows"]["outgoing"])
        result["summary"] = {
            "total_inflow": round(in_total, 2),
            "total_outflow": round(out_total, 2),
            "net_flow": round(in_total - out_total, 2),
            "num_sources": len(set(f["from"] for f in result["flows"]["incoming"])),
            "num_destinations": len(set(f["to"] for f in result["flows"]["outgoing"])),
        }

        return result

    def find_cycles(self, entity_name: Optional[str] = None,
                    min_amount: float = 0, max_length: int = 8) -> Dict:
        """Find circular transaction patterns (round-tripping).

        This is a thin filter over alerts the FraudDetector has already
        produced — single source of truth lives in
        `src/fraud_detector.py:detect_circular_transactions`. Re-doing the
        DFS here would create drift (the real detector uses Johnson's with
        SCC pre-filtering + log-amount bucketing; a copilot-local DFS
        would miss the same fraud the alerts page does, or worse, *find
        cycles the alerts page didn't*).

        Args:
            entity_name: Optional — filter cycles containing this entity.
            min_amount: Minimum total cycle flow to include.
            max_length: Maximum cycle length.
        """
        filter_node = self._find_node(entity_name) if entity_name else None

        circ_alerts = [
            a for a in self.alerts
            if a.get("pattern_type") == "Circular Transaction"
        ]

        results = []
        for a in circ_alerts:
            cycle_length = a.get("cycle_length", len(a.get("entities", [])))
            total_flow = a.get("total_flow", 0)
            entities = a.get("entities", [])
            names = a.get("entity_names") or [
                self.graph.nodes[n].get("name", n) if self.graph.has_node(n) else n
                for n in entities
            ]

            if cycle_length > max_length:
                continue
            if total_flow < min_amount:
                continue
            if filter_node and filter_node not in entities:
                continue

            results.append({
                "alert_id": a.get("alert_id"),
                "entities": names,
                "cycle_length": cycle_length,
                "total_flow": round(total_flow, 2),
                "avg_edge_flow": round(a.get("avg_flow_per_edge", total_flow / max(cycle_length, 1)), 2),
                "amount_variance_pct": a.get("amount_variance"),
                "confidence": a.get("confidence"),
                "severity": a.get("severity"),
                "path": (" → ".join(names) + (" → " + names[0] if names else "")),
            })

        results.sort(key=lambda x: x["total_flow"], reverse=True)
        return {
            "total_cycles_found": len(results),
            "cycles": results[:10],
            "source": "fraud_detector.detect_circular_transactions",
        }

    def explain_alert(self, alert_id: str) -> Dict:
        """Explain a specific fraud alert with full reasoning chain.

        When the ML bundle is available, the reasoning chain is the
        per-instance SHAP narrative (real feature attributions for this
        specific alert's edge). Falls back to a pattern-templated chain
        only when SHAP isn't installed or the alert has no scoreable edge.

        Args:
            alert_id: The alert ID to explain (e.g., 'ALERT_CIRC_0001').
        """
        alert = next((a for a in self.alerts if a["alert_id"] == alert_id), None)
        if not alert:
            return {"error": f"Alert '{alert_id}' not found. Check the alert ID."}

        explanation = {
            "alert": alert,
            "reasoning_chain": [],
            "reasoning_source": "fallback",
            "entity_profiles": [],
            "related_cases": [],
        }

        # ── Real SHAP-driven reasoning ────────────────────────────────────
        if self.model_bundle is not None:
            try:
                from shap_explainer import explain_alert as shap_explain
                shap_res = shap_explain(self.model_bundle, self.graph, self.transactions, alert)
                if shap_res:
                    explanation["reasoning_chain"] = shap_res.get("narrative", [])
                    explanation["reasoning_source"] = "shap"
                    explanation["model_score"] = shap_res.get("predicted_proba")
                    explanation["top_features"] = shap_res.get("top_features", [])
                    explanation["explained_edge"] = shap_res.get("edge")
                    explanation["base_value"] = shap_res.get("base_value")
            except ImportError:
                # shap not installed — leave reasoning_chain empty; fallback fills it below
                pass
            except Exception as e:
                explanation["shap_error"] = str(e)

        # ── Pattern-templated fallback ────────────────────────────────────
        # Only used when SHAP is unavailable or the alert has no scoreable
        # edge (e.g., single-entity dormant alert with no outgoing edge).
        if not explanation["reasoning_chain"]:
            pattern = alert.get("pattern_type", "")
            if "Circular" in pattern:
                explanation["reasoning_chain"] = [
                    "Closed loop of transactions between 3+ entities",
                    "Transaction amounts within the loop show low variance",
                    "Funds return to origin (round-tripping / artificial volume creation)",
                    "Pattern inconsistent with normal business payment behaviour",
                ]
            elif "Layering" in pattern:
                explanation["reasoning_chain"] = [
                    "Sequential chain of rapid fund transfers",
                    "Each step shows decreasing amounts (skimming at each layer)",
                    "Chain involves shell companies or high-risk entity types",
                    "Time between transfers is abnormally short",
                ]
            elif "Smurfing" in pattern:
                explanation["reasoning_chain"] = [
                    "Multiple transactions clustered just below reporting threshold",
                    "Low amount variability (structured pattern)",
                    "Common sender distributing to multiple recipients",
                    "Designed to avoid mandatory reporting requirements",
                ]
            elif "Funnel" in pattern:
                explanation["reasoning_chain"] = [
                    "Multiple diverse sources funnelling funds into a single entity",
                    "High flow imbalance (much more inflow than outflow or vice versa)",
                    "Sources span multiple branches, suggesting coordinated activity",
                    "Target entity shows characteristics of a shell company",
                ]
            elif "Dormant" in pattern:
                explanation["reasoning_chain"] = [
                    "Account inactive for an extended period",
                    "Sudden activation with Z-score above the 2.5-sigma threshold",
                    "Post-activation transaction average exceeds historical baseline",
                ]
            elif "Profile" in pattern:
                explanation["reasoning_chain"] = [
                    f"Entity declared as {alert.get('entity_type', '?')} but behaviour mismatches the type",
                    *alert.get("mismatches", [])[:3],
                ]

        # Entity profiles
        for eid in alert.get("entities", []):
            if self.graph.has_node(eid):
                ndata = dict(self.graph.nodes[eid])
                in_deg = self.graph.in_degree(eid)
                out_deg = self.graph.out_degree(eid)
                in_str = sum(self.graph[u][eid]["total_amount"] for u in self.graph.predecessors(eid))
                out_str = sum(self.graph[eid][v]["total_amount"] for v in self.graph.successors(eid))
                explanation["entity_profiles"].append({
                    "id": eid, "name": ndata.get("name", eid),
                    "type": ndata.get("type", ""), "branch": ndata.get("branch", ""),
                    "in_degree": in_deg, "out_degree": out_deg,
                    "inflow": round(in_str, 2), "outflow": round(out_str, 2),
                })

        # Related fraud cases
        entity_ids = set(alert.get("entities", []))
        for case in self.fraud_cases:
            if set(case.get("entities", [])) & entity_ids:
                explanation["related_cases"].append(case)

        return explanation

    def get_profile_delta(self, entity_name: str) -> Dict:
        """Get the behavioural profile of an entity + any profile-mismatch deltas.

        Mismatch deltas are sourced from `ProfileMismatchDetector` alerts that
        the pipeline has already produced for this entity — single source of
        truth. The behavioural-profile summary (txn count, avg, branch span,
        fraud ratio) is computed locally because it's useful context the
        detector itself doesn't expose.

        Args:
            entity_name: Name or partial name of the entity.
        """
        node_id = self._find_node(entity_name)
        if not node_id:
            return {"error": f"Entity '{entity_name}' not found."}

        node_data = dict(self.graph.nodes[node_id])
        entity_type = node_data.get("type", "individual")

        # Behavioural profile — useful context (not a detector duplicate).
        sent_txns = self.transactions[self.transactions["sender_id"] == node_id]
        recv_txns = self.transactions[self.transactions["receiver_id"] == node_id]
        all_txns = pd.concat([sent_txns, recv_txns])

        if all_txns.empty:
            return {
                "entity": {"id": node_id, "name": node_data.get("name", node_id), "type": entity_type},
                "delta": "No transactions found",
            }

        avg_amount = all_txns["amount"].mean()
        max_amount = all_txns["amount"].max()
        tx_types = all_txns["transaction_type"].value_counts().to_dict()
        purposes = all_txns["purpose_code"].value_counts().to_dict()
        branches = set(all_txns["sender_branch"].tolist() + all_txns["receiver_branch"].tolist())
        fraud_ratio = float(all_txns["is_fraud"].mean()) if "is_fraud" in all_txns.columns else 0.0

        # Mismatch deltas — pulled from existing alerts, not re-derived here.
        pm_alerts = [
            a for a in self.alerts
            if a.get("pattern_type") == "Profile Mismatch"
            and node_id in a.get("entities", [])
        ]
        deltas = []
        worst_severity = None
        sev_rank = {"MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        for a in pm_alerts:
            sev = a.get("severity", "MEDIUM")
            if worst_severity is None or sev_rank.get(sev, 0) > sev_rank.get(worst_severity, 0):
                worst_severity = sev
            for m in a.get("mismatches", []):
                deltas.append({
                    "description": m,
                    "severity": sev,
                    "alert_id": a.get("alert_id"),
                    "confidence": a.get("confidence"),
                })

        risk_info = next((r for r in self.risk_scores if r.get("entity_id") == node_id), {})

        return {
            "entity": {"id": node_id, "name": node_data.get("name", node_id),
                       "type": entity_type, "branch": node_data.get("branch", "")},
            "behavioral_profile": {
                "total_transactions": len(all_txns),
                "avg_amount": round(avg_amount, 2),
                "max_amount": round(max_amount, 2),
                "transaction_types": tx_types,
                "top_purposes": {k: v for k, v in list(purposes.items())[:5]},
                "branches_active": len(branches),
                "fraud_ratio": round(fraud_ratio, 3),
            },
            "risk_score": risk_info.get("risk_score", "N/A"),
            "risk_level": risk_info.get("risk_level", "N/A"),
            "deltas": deltas,
            "worst_severity": worst_severity,
            "profile_mismatch_score": (
                round(max(a.get("confidence", 0) for a in pm_alerts) / 100, 2)
                if pm_alerts else 0.0
            ),
            "source": "advanced_detectors.ProfileMismatchDetector",
        }

    def get_graph_stats(self) -> Dict:
        """Get overall graph statistics."""
        degrees = [d for _, d in self.graph.degree()]
        return {
            "total_entities": self.graph.number_of_nodes(),
            "total_connections": self.graph.number_of_edges(),
            "total_transactions": len(self.transactions),
            "total_alerts": len(self.alerts),
            "avg_degree": round(sum(degrees) / len(degrees), 2) if degrees else 0,
        }

    # ── Claude Integration ────────────────────────────────────

    def query(self, user_message: str) -> Dict:
        """Process a user query.

        Two modes:
          - **claude**: real LLM with multi-round tool calling (requires ANTHROPIC_API_KEY).
          - **quick_commands**: deterministic keyword router with the same tools
            but no natural-language understanding. Used when the API key is
            absent or Claude rejects the call.

        Every response carries `mode_label` so the UI can show the operator
        which path served their query — calling rule-based routing "AI" would
        be theatre.
        """
        tools = self._get_tool_definitions()

        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})

        if not self.api_key:
            return self._fallback_response(user_message)

        try:
            return self._call_claude(user_message, tools)
        except Exception as e:
            return self._fallback_response(user_message, str(e))

    def _build_system_prompt(self) -> str:
        return (
            "You are RUDRA, an AI fraud investigation copilot for Indian public-sector banks.\n\n"

            "ABOUT RUDRA:\n"
            "RUDRA is a real-time AML system replacing T+1 batch detection. It scores live "
            "transactions off a Kafka stream in 0.56 ms mean latency. It uses a stacked ensemble: "
            "XGBoost + GraphSAGE (3-hop SAGEConv) + GAT, with a logistic meta-learner. "
            "Trained on IBM AML HI-Small benchmark (100k stratified sample, 87,772 edges, 5.6% fraud rate). "
            "Threshold chosen at F2 (β=2, recall-favouring) because a missed launderer costs more than a false positive. "
            "Key metrics: AUC-ROC 0.926, AUPRC 0.661, Recall 75.2%, F1 0.402 (lower by design — F2 operating point).\n\n"

            "SEVEN DETECTORS:\n"
            "1. Circular transactions — Johnson's algorithm + SCC decomposition for round-tripping loops.\n"
            "2. Rapid layering — temporal-causal BFS money-following chain (≥3 hops, amount preservation gate).\n"
            "3. Smurfing/structuring — transactions clustered below ₹2L RBI reporting threshold (sliding window + fan-out).\n"
            "4. Shell funnel — FIFO holding-time + inflow/outflow imbalance for shell company pass-throughs.\n"
            "5. Dormant activation — Z-score spike after months of inactivity (>2.5σ above historical baseline).\n"
            "6. Profile mismatch — KYC behavioural deviation (individual acting like a business, etc.).\n"
            "7. Recruiter/fan-out — single funder seeding a network of mule accounts.\n\n"

            "CONFIDENCE TIERS:\n"
            "T1 = ML + rule agree (highest priority). T2 = ML-only (novel patterns). T3 = rule-only typology.\n\n"

            "REGULATORY CONTEXT (India):\n"
            "- RBI Fraud Risk Management Master Direction: July 2024 — mandates real-time monitoring in core banking.\n"
            "- PMLA: banks must file STR with FIU-IND within 7 days of suspicion forming.\n"
            "- DPDP Act: live now, full compliance due May 2027 — PII minimisation, role-based access.\n"
            "RUDRA compliance: PII redacted by default, SHA-256 hash-chain audit log, maker-checker RBAC, "
            "one-click FIU evidence package (STR XML + SAR PDF + subgraph + audit trail).\n\n"

            "RESPONSE FORMAT (STRICT — this is an enterprise banking tool, not a chatbot):\n\n"

            "ANSWER-FIRST. The first line is always the answer or a headline. Never open with "
            "'I see…', 'Sure', 'Great question', 'Let me…', or a self-introduction. Lead with the substance.\n\n"

            "PICK ONE STRUCTURE based on the query:\n"
            "1. DIRECT FACT/METRIC (e.g. 'what is AUPRC') — State the value in the first line in **bold**. "
            "Add at most 1-2 lines of context. Stop.\n"
            "2. ENTITY / ALERT LOOKUP — Lead with a one-line headline, then a compact key-value block "
            "(see STAT BLOCKS), then a short reasoning paragraph if needed.\n"
            "3. LIST / DATA RESULT (cycles, alerts, high-risk entities) — One headline line stating the count, "
            "then a tight bulleted list sorted by importance (highest amount/severity first). No preamble.\n"
            "4. EXPLANATION ('how does X work') — One-sentence definition, then the mechanism in 2-4 lines. "
            "No history, no filler.\n\n"

            "STAT BLOCKS — when reporting multiple related numbers, use bold-label key-value bullets, "
            "value in backticks, NOT prose. Example:\n"
            "- **AUPRC** — `0.661`\n"
            "- **Recall** — `75.2%`\n"
            "Do NOT bury numbers inside sentences when there are several of them. Never use markdown tables.\n\n"

            "HARD RULES:\n"
            "- NEVER end with a menu of options or 'What would you like to investigate?'. Answer, then stop.\n"
            "- If you offer a follow-up, it is ONE short imperative question (e.g. 'Trace this entity's funds?') — never a list.\n"
            "- Match length to the question. A one-line question gets a short answer, not a data dump.\n"
            "- Report ONLY what the data/tools return. Never invent statistics (e.g. 'alert density', 'X% aligns with…') "
            "or qualitative judgments ('healthy graph', 'typical volume'). Every number on screen must be defensible.\n"
            "- Never use emojis. Tone is precise, calm, declarative. No exclamation marks.\n"
            "- For specific entities/alerts/fund-flows — call the appropriate tool. For RUDRA internals/metrics/"
            "regulations — answer directly, no tool needed.\n"
            "- If the input is unclear, gibberish, a URL, or out of scope — reply with ONE sentence asking what "
            "they want to investigate. Never fabricate an analysis.\n"
            "- Use ₹ for rupee amounts (₹X Cr / ₹X L for large values). Never fabricate alert IDs or entity names."
        )

    def _build_context(self, user_message: str) -> str:
        return (
            f"Graph snapshot: {self.graph.number_of_nodes()} entities, "
            f"{self.graph.number_of_edges()} connections, "
            f"{len(self.alerts)} active alerts, {len(self.fraud_cases)} known fraud cases.\n\n"
            f"Investigator query: {user_message}"
        )

    def _build_anthropic_tools(self) -> List[Dict]:
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in self._get_tool_definitions()
        ]

    def stream_query(self, user_message: str):
        """Generator yielding SSE chunks for real-time streaming to the frontend.

        Tool-calling rounds run synchronously (fast, usually <500 ms total).
        The final answer synthesis is streamed token-by-token via the Anthropic
        streaming API so the UI renders words as they arrive.
        """
        if not self.api_key:
            result = self._fallback_response(user_message)
            yield f"data: {json.dumps({'token': result['response']})}\n\n"
            yield f"data: {json.dumps({'done': True, 'mode': result['mode'], 'mode_label': result['mode_label']})}\n\n"
            return

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            system = self._build_system_prompt()
            anthropic_tools = self._build_anthropic_tools()
            messages = [{"role": "user", "content": self._build_context(user_message)}]

            # Non-streaming tool-calling rounds — execute tools, build up messages
            for _ in range(3):
                resp = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                    tools=anthropic_tools,
                )
                tool_use_blocks = [b for b in resp.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    break
                tool_results = []
                for block in tool_use_blocks:
                    args = dict(block.input) if block.input else {}
                    result = self._execute_tool(block.name, args)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": tool_results})

            # Stream the final synthesis answer token by token, through the
            # enforcement filter. `buf` holds an unflushed tail of up to HOLD
            # chars so a trailing option-menu (or an emoji split across tokens)
            # is detected and truncated before it reaches the browser.
            HOLD = 48
            buf = ""
            cut = False
            emitted_any = False
            menu_active = True
            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    buf += text
                    if menu_active:
                        menu_at = _find_menu_cut(buf)
                        if menu_at is not None:
                            head = _strip_response(buf[:menu_at])
                            if head or emitted_any:
                                # Real answer precedes the trigger → it's a trailing
                                # menu. Emit the clean head and stop.
                                if head:
                                    yield f"data: {json.dumps({'token': head})}\n\n"
                                cut = True
                                break
                            # Trigger at the very start with nothing before it →
                            # a standalone clarifying question. Keep streaming it.
                            menu_active = False
                    if len(buf) > HOLD:
                        safe = _EMOJI_RE.sub("", buf[:-HOLD])
                        buf = buf[-HOLD:]
                        if safe:
                            emitted_any = True
                            yield f"data: {json.dumps({'token': safe})}\n\n"

            # No menu hit — flush the remaining clean tail.
            if not cut and buf:
                tail = _EMOJI_RE.sub("", buf).rstrip()
                if tail:
                    yield f"data: {json.dumps({'token': tail})}\n\n"

            yield f"data: {json.dumps({'done': True, 'mode': 'ai_copilot', 'mode_label': 'AI Copilot (Claude)'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {e}'})}\n\n"
            yield f"data: {json.dumps({'done': True, 'mode': 'ai_copilot', 'mode_label': 'AI Copilot (Claude)'})}\n\n"

    def _call_claude(self, user_message: str, tools: List[Dict]) -> Dict:
        """Call Claude API with tool-calling, sending tool results back for synthesis.

        Protocol: send user message → Claude may emit tool_use blocks →
        execute them locally → send tool_result blocks back → Claude
        produces a final natural-language answer. Up to 3 rounds.
        """
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        system_prompt = self._build_system_prompt()

        # Anthropic uses input_schema instead of parameters
        anthropic_tools = self._build_anthropic_tools()

        context = (
            f"Graph snapshot: {self.graph.number_of_nodes()} entities, "
            f"{self.graph.number_of_edges()} connections, "
            f"{len(self.alerts)} active alerts, {len(self.fraud_cases)} known fraud cases.\n\n"
            f"Investigator query: {user_message}"
        )

        messages = [{"role": "user", "content": context}]
        tool_calls: List[Dict] = []
        response = None

        for _ in range(3):
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=anthropic_tools,
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            # Execute every tool requested this turn
            tool_results = []
            for block in tool_use_blocks:
                args = dict(block.input) if block.input else {}
                result = self._execute_tool(block.name, args)
                tool_calls.append({"tool": block.name, "args": args, "result": result})
                self.tool_results_log.append(result)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

            # Append assistant turn + tool results for next round
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        final_text = ""
        if response:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    final_text += block.text

        if not final_text and tool_calls:
            final_text = self._summarize_tool_results(tool_calls)
        elif not final_text:
            final_text = self._generate_local_response(user_message)
        else:
            # Same enterprise-format enforcement the streaming path applies.
            final_text = _strip_response(final_text)

        return {
            "response": final_text,
            "tool_calls": tool_calls,
            "source": "claude" if tool_calls else "claude_textonly",
            "mode": "ai_copilot",
            "mode_label": "AI Copilot (Claude)",
        }

    def _get_tool_definitions(self) -> List[Dict]:
        """Return tool definitions for Claude tool calling."""
        return [
            {
                "name": "trace_funds",
                "description": "Trace fund flows from/to a given entity. Shows incoming and outgoing transactions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string", "description": "Name or partial name of entity"},
                        "depth": {"type": "integer", "description": "Hops to trace (1-5)", "default": 3},
                        "direction": {"type": "string", "enum": ["incoming", "outgoing", "both"], "default": "both"},
                    },
                    "required": ["entity_name"],
                },
            },
            {
                "name": "find_cycles",
                "description": "Find circular transaction patterns (round-tripping) in the fund flow graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string", "description": "Optional entity to filter cycles"},
                        "min_amount": {"type": "number", "description": "Minimum cycle flow amount", "default": 0},
                        "max_length": {"type": "integer", "description": "Maximum cycle length", "default": 8},
                    },
                },
            },
            {
                "name": "explain_alert",
                "description": "Explain a specific fraud alert with full reasoning chain and entity profiles.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_id": {"type": "string", "description": "Alert ID (e.g., ALERT_CIRC_0001)"},
                    },
                    "required": ["alert_id"],
                },
            },
            {
                "name": "get_profile_delta",
                "description": "Detect KYC profile mismatches and behavioral anomalies for an entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_name": {"type": "string", "description": "Entity name to analyze"},
                    },
                    "required": ["entity_name"],
                },
            },
        ]

    def _execute_tool(self, tool_name: str, args: Dict) -> Any:
        """Execute a tool by name with given arguments."""
        if tool_name == "trace_funds":
            return self.trace_funds(**args)
        elif tool_name == "find_cycles":
            return self.find_cycles(**args)
        elif tool_name == "explain_alert":
            return self.explain_alert(**args)
        elif tool_name == "get_profile_delta":
            return self.get_profile_delta(**args)
        return {"error": f"Unknown tool: {tool_name}"}

    def _fallback_response(self, user_message: str, error: str = "") -> Dict:
        """Deterministic keyword-routing fallback when Claude is unavailable.

        We explicitly do NOT call this "AI" — `mode` is `quick_commands` and
        `mode_label` reads "Quick Commands". The UI surfaces this so an
        operator knows they're hitting rule-based routing, not an LLM.
        """
        response = self._generate_local_response(user_message)
        if error:
            reason = f"Claude unavailable: {error}"
        elif not self.api_key:
            reason = "ANTHROPIC_API_KEY not set"
        else:
            reason = "fallback"
        return {
            "response": response,
            "tool_calls": [],
            "source": f"quick_commands ({reason})",
            "mode": "quick_commands",
            "mode_label": "Quick Commands (no LLM)",
            "fallback_reason": reason,
        }

    def _generate_local_response(self, user_message: str) -> str:
        """Generate response using local tool execution and template-based answers."""
        msg = user_message.lower()

        # Route to appropriate tool based on intent
        if any(kw in msg for kw in ["trace", "flow", "follow", "track", "where did", "money go"]):
            entity = self._extract_entity_name(user_message)
            if entity:
                result = self.trace_funds(entity)
                return self._format_trace_response(entity, result)
            return "Please specify an entity name to trace fund flows. For example: 'Trace funds for Amit Sharma'"

        if any(kw in msg for kw in ["cycle", "circular", "round-trip", "round trip", "loop"]):
            result = self.find_cycles()
            return self._format_cycles_response(result)

        if any(kw in msg for kw in ["alert", "explain", "investigate", "suspicious"]):
            alert_id = self._extract_alert_id(user_message)
            if alert_id:
                result = self.explain_alert(alert_id)
                return self._format_explain_response(result)
            # List top alerts
            return self._format_alerts_summary()

        if any(kw in msg for kw in ["profile", "kyc", "mismatch", "anomaly", "behavior"]):
            entity = self._extract_entity_name(user_message)
            if entity:
                result = self.get_profile_delta(entity)
                return self._format_profile_response(entity, result)
            return "Please specify an entity name for profile analysis."

        if any(kw in msg for kw in ["summary", "overview", "status", "dashboard", "how many"]):
            return self._format_overview()

        if any(kw in msg for kw in ["risk", "risky", "dangerous", "high risk"]):
            return self._format_risk_summary()

        # Default: provide helpful guidance
        return (
            "I can help you investigate fund flows and fraud patterns. Try asking:\n\n"
            "- **Trace funds** for an entity: 'Trace funds for Amit Sharma'\n"
            "- **Find circular patterns**: 'Show me circular transactions'\n"
            "- **Explain an alert**: 'Explain alert ALERT_CIRC_0001'\n"
            "- **Profile analysis**: 'Check KYC profile for Apex Trading Co.'\n"
            "- **Risk overview**: 'Show me high-risk entities'\n"
            "- **Dashboard summary**: 'Give me an overview'\n"
        )

    # ── Helper Methods ────────────────────────────────────────

    def _find_node(self, name: str) -> Optional[str]:
        """Find a node by name (case-insensitive partial match)."""
        if not name:
            return None
        name_lower = name.lower().strip()
        for node in self.graph.nodes():
            node_name = self.graph.nodes[node].get("name", "")
            if name_lower == node_name.lower():
                return node
        for node in self.graph.nodes():
            node_name = self.graph.nodes[node].get("name", "")
            if name_lower in node_name.lower():
                return node
        return None

    def _trace_direction(self, node_id: str, direction: str, depth: int) -> List[Dict]:
        """Trace flows in a direction up to given depth."""
        flows = []
        visited = {node_id}
        current_level = [node_id]

        for d in range(depth):
            next_level = []
            for n in current_level:
                if direction == "incoming":
                    neighbors = list(self.graph.predecessors(n))
                else:
                    neighbors = list(self.graph.successors(n))

                for nb in neighbors:
                    if nb not in visited and len(flows) < 50:
                        visited.add(nb)
                        next_level.append(nb)
                        edge = self.graph[nb][n] if direction == "incoming" else self.graph[n][nb]
                        flows.append({
                            "from": self.graph.nodes[nb].get("name", nb) if direction == "incoming" else self.graph.nodes[n].get("name", n),
                            "to": self.graph.nodes[n].get("name", n) if direction == "incoming" else self.graph.nodes[nb].get("name", nb),
                            "amount": round(edge["total_amount"], 2),
                            "transaction_count": edge["transaction_count"],
                            "depth": d + 1,
                        })
            current_level = next_level
            if not current_level:
                break

        return flows

    def _extract_entity_name(self, message: str) -> Optional[str]:
        """Extract an entity name from user message."""
        # Try to find entity names mentioned in the message
        for node in self.graph.nodes():
            name = self.graph.nodes[node].get("name", "")
            if name.lower() in message.lower():
                return name
        # Try partial match
        words = message.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 5, len(words) + 1)):
                candidate = " ".join(words[i:j])
                found = self._find_node(candidate)
                if found:
                    return self.graph.nodes[found].get("name", found)
        return None

    def _extract_alert_id(self, message: str) -> Optional[str]:
        """Extract alert ID from message."""
        import re
        match = re.search(r"ALERT_\w+_\d+", message.upper())
        if match:
            return match.group(0)
        # Try by index
        for i, alert in enumerate(self.alerts):
            if str(i + 1) in message:
                return alert.get("alert_id")
        return None

    # ── Response Formatters ───────────────────────────────────

    def _format_trace_response(self, entity: str, result: Dict) -> str:
        if "error" in result:
            return result["error"]
        s = result.get("summary", {})
        flows = result.get("flows", {})
        lines = [
            f"## Fund Flow Trace: {entity}\n",
            f"**Net Flow:** ₹{s.get('net_flow', 0):,.0f}",
            f"**Total Inflow:** ₹{s.get('total_inflow', 0):,.0f} from {s.get('num_sources', 0)} sources",
            f"**Total Outflow:** ₹{s.get('total_outflow', 0):,.0f} to {s.get('num_destinations', 0)} destinations\n",
        ]
        if flows.get("outgoing"):
            lines.append("**Top Outgoing Flows:**")
            for f in sorted(flows["outgoing"], key=lambda x: x["amount"], reverse=True)[:5]:
                lines.append(f"  → {f['to']}: ₹{f['amount']:,.0f} ({f['transaction_count']} txns)")
        if flows.get("incoming"):
            lines.append("\n**Top Incoming Flows:**")
            for f in sorted(flows["incoming"], key=lambda x: x["amount"], reverse=True)[:5]:
                lines.append(f"  ← {f['from']}: ₹{f['amount']:,.0f} ({f['transaction_count']} txns)")
        return "\n".join(lines)

    def _format_cycles_response(self, result: Dict) -> str:
        cycles = result.get("cycles", [])
        if not cycles:
            return "No circular transaction patterns detected in the current dataset."
        lines = [f"## Circular Transaction Patterns Detected: {result.get('total_cycles_found', 0)}\n"]
        for i, c in enumerate(cycles[:5]):
            lines.append(f"**Cycle {i+1}:** {c['cycle_length']} entities, Total Flow: ₹{c['total_flow']:,.0f}")
            lines.append(f"  Path: {c['path']}\n")
        return "\n".join(lines)

    def _format_explain_response(self, result: Dict) -> str:
        if "error" in result:
            return result["error"]
        alert = result.get("alert", {})
        lines = [
            f"## Alert Analysis: {alert.get('alert_id', '')}\n",
            f"**Pattern:** {alert.get('pattern_type', '')}",
            f"**Severity:** {alert.get('severity', '')}",
            f"**Confidence:** {alert.get('confidence', 0)}%",
            f"**Total Flow:** ₹{alert.get('total_flow', 0):,.0f}\n",
            "**Reasoning Chain:**",
        ]
        for r in result.get("reasoning_chain", []):
            lines.append(f"  {r}")
        lines.append(f"\n**Recommendation:** {alert.get('recommendation', 'N/A')}")
        profiles = result.get("entity_profiles", [])
        if profiles:
            lines.append("\n**Entities Involved:**")
            for p in profiles:
                lines.append(f"  - {p['name']} ({p['type']}) | In: ₹{p['inflow']:,.0f} | Out: ₹{p['outflow']:,.0f}")
        return "\n".join(lines)

    def _format_profile_response(self, entity: str, result: Dict) -> str:
        if "error" in result:
            return result["error"]
        profile = result.get("behavioral_profile", {})
        deltas = result.get("deltas", [])
        lines = [
            f"## Profile Analysis: {entity}\n",
            f"**Type:** {result['entity'].get('type', 'N/A')}",
            f"**Risk Score:** {result.get('risk_score', 'N/A')} ({result.get('risk_level', 'N/A')})",
            f"**Profile Mismatch Score:** {result.get('profile_mismatch_score', 0):.2f}\n",
            "**Behavioral Profile:**",
            f"  - Total Transactions: {profile.get('total_transactions', 0)}",
            f"  - Average Amount: ₹{profile.get('avg_amount', 0):,.0f}",
            f"  - Max Amount: ₹{profile.get('max_amount', 0):,.0f}",
            f"  - Active Branches: {profile.get('branches_active', 0)}",
            f"  - Fraud Ratio: {profile.get('fraud_ratio', 0):.1%}\n",
        ]
        if deltas:
            lines.append("**Profile Deltas Detected (from ProfileMismatchDetector alerts):**")
            for d in deltas:
                aid = d.get("alert_id", "")
                conf = d.get("confidence")
                conf_str = f" — {conf}% confidence" if conf is not None else ""
                lines.append(f"  - [{d['severity']}] {d['description']} ({aid}{conf_str})")
        else:
            lines.append("**No profile-mismatch alerts for this entity.**")
        return "\n".join(lines)

    def _format_alerts_summary(self) -> str:
        if not self.alerts:
            return "No alerts currently active."
        lines = [f"## Active Fraud Alerts ({len(self.alerts)})\n"]
        crit = [a for a in self.alerts if a["severity"] == "CRITICAL"]
        high = [a for a in self.alerts if a["severity"] == "HIGH"]
        med = [a for a in self.alerts if a["severity"] == "MEDIUM"]
        lines.append(f"**CRITICAL:** {len(crit)} | **HIGH:** {len(high)} | **MEDIUM:** {len(med)}\n")
        for a in sorted(self.alerts, key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x["severity"], 3))[:8]:
            lines.append(f"- [{a['severity']}] {a['alert_id']}: {a['pattern_type']} (₹{a.get('total_flow', 0):,.0f}, {a['confidence']}% confidence)")
        lines.append("\nAsk me to explain any specific alert, e.g., 'Explain alert ALERT_CIRC_0001'")
        return "\n".join(lines)

    def _format_overview(self) -> str:
        stats = self.get_graph_stats()
        fraud_txns = self.transactions[self.transactions["is_fraud"]]
        total_vol = self.transactions["amount"].sum()
        fraud_vol = fraud_txns["amount"].sum()
        lines = [
            "## RUDRA Dashboard Overview\n",
            f"**Total Entities:** {stats['total_entities']}",
            f"**Total Connections:** {stats['total_connections']}",
            f"**Total Transactions:** {stats['total_transactions']:,}",
            f"**Active Alerts:** {stats['total_alerts']}\n",
            f"**Total Volume:** ₹{total_vol/1e7:.2f} Cr",
            f"**Fraud Volume:** ₹{fraud_vol/1e7:.2f} Cr ({len(fraud_txns)} transactions)\n",
            "**Alert Breakdown:**",
        ]
        for pattern in ["Circular Transaction", "Rapid Layering", "Smurfing / Structuring", "Shell Company Funnel"]:
            count = sum(1 for a in self.alerts if a["pattern_type"] == pattern)
            lines.append(f"  - {pattern}: {count}")
        lines.append("\nAsk me to investigate specific patterns or entities.")
        return "\n".join(lines)

    def _format_risk_summary(self) -> str:
        high_risk = [r for r in self.risk_scores if r.get("risk_score", 0) >= 0.5]
        lines = [f"## High-Risk Entities ({len(high_risk)})\n"]
        for r in sorted(high_risk, key=lambda x: x.get("risk_score", 0), reverse=True)[:10]:
            lines.append(f"- **{r.get('name', 'N/A')}** ({r.get('type', '')}) — Risk: {r.get('risk_score', 0):.2f} [{r.get('risk_level', '')}]")
        lines.append("\nAsk me to trace funds or check profiles for any entity.")
        return "\n".join(lines)

    def _summarize_tool_results(self, tool_results: List[Dict]) -> str:
        """Summarize tool results into a readable response."""
        parts = []
        for tr in tool_results:
            result = tr["result"]
            if "error" in result:
                parts.append(f"**{tr['tool']}:** {result['error']}")
            else:
                parts.append(f"**{tr['tool']}:** Analysis complete. {json.dumps(result, default=str)[:500]}")
        return "\n\n".join(parts) if parts else "Analysis complete."
