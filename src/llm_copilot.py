"""
LLM Copilot — AI-powered investigation assistant.
Uses Gemini API with tool-calling to answer investigator queries
over the live fund flow graph.

Tools: trace_funds(), find_cycles(), explain_alert(), get_profile_delta()
"""

import json
import os
from typing import Dict, List, Optional, Any
import pandas as pd
import networkx as nx


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
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
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

    # ── Gemini Integration ────────────────────────────────────

    def query(self, user_message: str) -> Dict:
        """Process a user query.

        Two modes:
          - **gemini**: real LLM with multi-round tool calling (requires GEMINI_API_KEY).
          - **quick_commands**: deterministic keyword router with the same tools
            but no natural-language understanding. Used when the API key is
            absent or Gemini rejects the call.

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
            return self._call_gemini(user_message, tools)
        except Exception as e:
            return self._fallback_response(user_message, str(e))

    def _call_gemini(self, user_message: str, tools: List[Dict]) -> Dict:
        """Call Gemini API with tool-calling, sending tool results back for synthesis.

        The protocol is: send user message → Gemini may emit function_call(s) →
        we execute them locally → send the function responses back → Gemini
        produces a final natural-language answer. Up to 3 rounds of tool calls
        to support multi-step investigations like "trace funds for X, then
        explain the alerts you find".
        """
        import google.generativeai as genai
        from google.generativeai.types import FunctionDeclaration, Tool

        genai.configure(api_key=self.api_key)

        system_prompt = (
            "You are RUDRA, an AI fraud investigation copilot for banking. "
            "You have access to a live fund flow graph with transaction data. "
            "When the investigator asks something you can answer by calling a tool, "
            "call the appropriate tool. After receiving tool results, synthesise "
            "a clear actionable answer. Use Indian Rupee (₹). Reference RBI / PMLA "
            "guidelines where appropriate. Be concise — investigators are busy."
        )

        # Wrap our tools into Gemini's typed format
        tool_decls = [
            FunctionDeclaration(name=t["name"], description=t["description"],
                                  parameters=t["parameters"])
            for t in tools
        ]
        gemini_tools = [Tool(function_declarations=tool_decls)]

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt,
            tools=gemini_tools,
        )
        chat = model.start_chat(history=[])
        context = (
            f"Graph snapshot: {self.graph.number_of_nodes()} entities, "
            f"{self.graph.number_of_edges()} connections, "
            f"{len(self.alerts)} active alerts, {len(self.fraud_cases)} known fraud cases.\n\n"
            f"Investigator query: {user_message}"
        )

        response = chat.send_message(context)
        tool_calls: List[Dict] = []
        max_rounds = 3
        for _ in range(max_rounds):
            if not (response.candidates and response.candidates[0].content.parts):
                break
            calls_this_turn = []
            for part in response.candidates[0].content.parts:
                fc = getattr(part, "function_call", None)
                if fc and fc.name:
                    calls_this_turn.append(fc)
            if not calls_this_turn:
                break
            # Execute every tool call requested this turn
            function_responses = []
            for fc in calls_this_turn:
                args = dict(fc.args) if fc.args else {}
                result = self._execute_tool(fc.name, args)
                tool_calls.append({"tool": fc.name, "args": args, "result": result})
                self.tool_results_log.append(result)
                function_responses.append({
                    "function_response": {
                        "name": fc.name,
                        "response": result,
                    },
                })
            # Send tool results back for synthesis
            response = chat.send_message(function_responses)

        final_text = ""
        try:
            final_text = response.text or ""
        except Exception:
            # When the response is purely a function call (no text part)
            final_text = ""
        if not final_text and tool_calls:
            final_text = self._summarize_tool_results(tool_calls)
        elif not final_text:
            final_text = self._generate_local_response(user_message)

        return {
            "response": final_text,
            "tool_calls": tool_calls,
            "source": "gemini" if tool_calls else "gemini_textonly",
            "mode": "ai_copilot",
            "mode_label": "AI Copilot (Gemini)",
        }

    def _get_tool_definitions(self) -> List[Dict]:
        """Return tool definitions for Gemini function calling."""
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
        """Deterministic keyword-routing fallback when Gemini is unavailable.

        We explicitly do NOT call this "AI" — `mode` is `quick_commands` and
        `mode_label` reads "Quick Commands". The UI surfaces this so an
        operator knows they're hitting rule-based routing, not an LLM.
        """
        response = self._generate_local_response(user_message)
        if error:
            reason = f"Gemini unavailable: {error}"
        elif not self.api_key:
            reason = "GEMINI_API_KEY not set"
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
