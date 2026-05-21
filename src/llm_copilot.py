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
                 fraud_cases: List[Dict], api_key: Optional[str] = None):
        self.graph = graph
        self.transactions = transactions
        self.alerts = alerts
        self.risk_scores = risk_scores
        self.fraud_cases = fraud_cases
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
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

        Args:
            entity_name: Optional — filter cycles containing this entity.
            min_amount: Minimum total cycle flow to include.
            max_length: Maximum cycle length.
        """
        import signal
        try:
            # Targeted DFS-based cycle search instead of nx.simple_cycles
            cycles = []
            cycles_found = set()
            for start_node in self.graph.nodes():
                if len(cycles) >= 50:
                    break
                stack = [(start_node, [start_node], {start_node})]
                while stack and len(cycles) < 50:
                    current, path, visited = stack.pop()
                    if len(path) > max_length:
                        continue
                    for neighbor in self.graph.successors(current):
                        if neighbor == start_node and len(path) >= 3:
                            cycle_key = tuple(sorted(path))
                            if cycle_key not in cycles_found:
                                cycles_found.add(cycle_key)
                                cycles.append(list(path))
                        elif neighbor not in visited and len(path) < max_length:
                            stack.append((neighbor, path + [neighbor], visited | {neighbor}))
        except Exception:
            return {"error": "Could not compute cycles on this graph.", "cycles": []}

        results = []
        filter_node = self._find_node(entity_name) if entity_name else None

        for cycle in cycles:
            if len(cycle) < 3 or len(cycle) > max_length:
                continue

            # Check edges exist and compute flow
            edge_amounts = []
            valid = True
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if self.graph.has_edge(u, v):
                    edge_amounts.append(self.graph[u][v]["total_amount"])
                else:
                    valid = False
                    break

            if not valid:
                continue

            total_flow = sum(edge_amounts)
            if total_flow < min_amount:
                continue

            if filter_node and filter_node not in cycle:
                continue

            names = [self.graph.nodes[n].get("name", n) for n in cycle]
            results.append({
                "entities": names,
                "cycle_length": len(cycle),
                "total_flow": round(total_flow, 2),
                "avg_edge_flow": round(total_flow / len(cycle), 2),
                "path": " → ".join(names) + " → " + names[0],
            })

        return {
            "total_cycles_found": len(results),
            "cycles": sorted(results, key=lambda x: x["total_flow"], reverse=True)[:10],
        }

    def explain_alert(self, alert_id: str) -> Dict:
        """Explain a specific fraud alert with full reasoning chain.

        Args:
            alert_id: The alert ID to explain (e.g., 'ALERT_CIRC_0001').
        """
        alert = next((a for a in self.alerts if a["alert_id"] == alert_id), None)
        if not alert:
            return {"error": f"Alert '{alert_id}' not found. Check the alert ID."}

        explanation = {
            "alert": alert,
            "reasoning_chain": [],
            "entity_profiles": [],
            "related_cases": [],
        }

        # Build reasoning chain
        pattern = alert.get("pattern_type", "")
        if "Circular" in pattern:
            explanation["reasoning_chain"] = [
                "1. Detected a closed loop of transactions between 3+ entities",
                "2. Transaction amounts within the loop show low variance (similar values)",
                "3. Funds return to origin, indicating round-tripping or artificial volume creation",
                "4. Pattern violates normal business payment behavior",
            ]
        elif "Layering" in pattern:
            explanation["reasoning_chain"] = [
                "1. Identified a sequential chain of rapid fund transfers",
                "2. Each step shows decreasing amounts (skimming/fees at each layer)",
                "3. Chain involves shell companies or high-risk entity types",
                "4. Time between transfers is abnormally short",
            ]
        elif "Smurfing" in pattern:
            explanation["reasoning_chain"] = [
                "1. Multiple transactions clustered just below reporting threshold (₹2,00,000)",
                "2. Transactions show low amount variability (structured pattern)",
                "3. Common sender distributing to multiple recipients",
                "4. Pattern designed to avoid mandatory reporting requirements",
            ]
        elif "Funnel" in pattern:
            explanation["reasoning_chain"] = [
                "1. Multiple diverse sources funneling funds into a single entity",
                "2. High flow imbalance ratio (much more inflow than outflow or vice versa)",
                "3. Sources span multiple branches, suggesting coordinated activity",
                "4. Target entity shows characteristics of shell company",
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
        """Detect KYC profile mismatches and behavioral anomalies.

        Args:
            entity_name: Name or partial name of the entity.
        """
        node_id = self._find_node(entity_name)
        if not node_id:
            return {"error": f"Entity '{entity_name}' not found."}

        node_data = dict(self.graph.nodes[node_id])
        entity_type = node_data.get("type", "individual")

        # Analyze transaction behavior
        sent_txns = self.transactions[self.transactions["sender_id"] == node_id]
        recv_txns = self.transactions[self.transactions["receiver_id"] == node_id]
        all_txns = pd.concat([sent_txns, recv_txns])

        if all_txns.empty:
            return {"entity": {"name": node_data.get("name", node_id)}, "delta": "No transactions found"}

        # Compute behavioral profile
        avg_amount = all_txns["amount"].mean()
        max_amount = all_txns["amount"].max()
        tx_types = all_txns["transaction_type"].value_counts().to_dict()
        purposes = all_txns["purpose_code"].value_counts().to_dict()
        branches = set(all_txns["sender_branch"].tolist() + all_txns["receiver_branch"].tolist())
        fraud_ratio = all_txns["is_fraud"].mean()

        # Compute expected vs actual for entity type
        deltas = []
        if entity_type == "individual":
            if avg_amount > 500000:
                deltas.append({"field": "avg_transaction_amount", "expected": "< ₹5,00,000",
                              "actual": f"₹{avg_amount:,.0f}", "severity": "HIGH",
                              "reason": "Individual averaging very high transaction amounts"})
            if max_amount > 2000000:
                deltas.append({"field": "max_transaction_amount", "expected": "< ₹20,00,000",
                              "actual": f"₹{max_amount:,.0f}", "severity": "MEDIUM",
                              "reason": "Single transaction unusually large for individual"})
        elif entity_type == "business":
            sent_types = sent_txns["transaction_type"].value_counts().to_dict()
            if "Wire Transfer" in sent_types and sent_types["Wire Transfer"] > 3:
                deltas.append({"field": "wire_transfer_frequency", "expected": "≤ 3",
                              "actual": str(sent_types["Wire Transfer"]), "severity": "MEDIUM",
                              "reason": "High frequency of wire transfers for domestic business"})

        if fraud_ratio > 0.3:
            deltas.append({"field": "fraud_transaction_ratio", "expected": "< 5%",
                          "actual": f"{fraud_ratio:.0%}", "severity": "CRITICAL",
                          "reason": "Significant proportion of transactions flagged as fraudulent"})

        if len(branches) > 5:
            deltas.append({"field": "branch_diversity", "expected": "≤ 5 unique branches",
                          "actual": f"{len(branches)} branches", "severity": "MEDIUM",
                          "reason": "Transactions spread across unusually many branches"})

        # Check for dormant activation
        if len(all_txns) > 0:
            all_txns_sorted = all_txns.sort_values("timestamp")
            time_diffs = pd.to_datetime(all_txns_sorted["timestamp"]).diff().dt.total_seconds() / 3600
            if len(time_diffs) > 1:
                max_gap = time_diffs.max()
                if max_gap > 720:  # 30 days
                    deltas.append({"field": "dormant_period", "expected": "No gaps > 30 days",
                                  "actual": f"{max_gap/24:.0f} days gap", "severity": "HIGH",
                                  "reason": "Account showed dormant behavior then sudden activation"})

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
            "profile_mismatch_score": round(min(len(deltas) * 0.25, 1.0), 2),
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
        """Process a user query using Gemini with tool-calling."""
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
        """Generate a local response when Gemini API is unavailable."""
        response = self._generate_local_response(user_message)
        return {
            "response": response,
            "tool_calls": [],
            "source": "local" + (f" (Gemini unavailable: {error})" if error else " (no API key)"),
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
            lines.append("**Profile Deltas Detected:**")
            for d in deltas:
                lines.append(f"  - [{d['severity']}] {d['field']}: Expected {d['expected']}, Actual {d['actual']}")
                lines.append(f"    Reason: {d['reason']}")
        else:
            lines.append("**No significant profile mismatches detected.**")
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
