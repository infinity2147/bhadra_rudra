"""
FIU Evidence Package Builder

Bundles everything an investigator needs to file a Suspicious Transaction Report
with FIU-IND (Financial Intelligence Unit, India) into a single downloadable zip:

  evidence_summary.md     — human-readable case summary + reasoning
  SAR_<id>.pdf            — the formal SAR document
  subgraph.json           — NetworkX-format export of the fraud subgraph
  transaction_chain.csv   — every transaction in the case, in order
  pmla_citations.txt      — PMLA / RBI sections relevant to the pattern
  cases_audit_log.json    — full audit trail for regulatory review

This package matches the structure of what a real STR submission to FIU
contains, minus the PII fields we intentionally do not store.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import networkx as nx


# Citations per pattern type — chosen because they are the sections most often
# cited in actual STR filings for these typologies.
PMLA_CITATIONS = {
    "Circular Transaction": [
        "PMLA 2002, Section 3 — Offence of money-laundering",
        "PMLA 2002, Section 12 — Obligation of reporting entities",
        "RBI Master Direction on KYC 2016, Chapter VII — Suspicious Transaction Reporting",
        "FIU-IND STR Filing Guidelines: Round-tripping typology",
    ],
    "Rapid Layering": [
        "PMLA 2002, Section 3 — Layering stage of money laundering",
        "PMLA 2002, Section 12",
        "RBI Master Direction on KYC 2016, Chapter VII",
        "FIU-IND STR Filing Guidelines: Layering typology",
    ],
    "Smurfing / Structuring": [
        "PMLA Rules 2005, Rule 3 — Maintenance of records of suspicious transactions",
        "RBI Master Direction on KYC 2016, Chapter VII",
        "Income-tax Act 1961, Section 269ST — Cash transaction limits",
        "FIU-IND STR Filing Guidelines: Structuring typology",
    ],
    "Shell Company Funnel": [
        "PMLA 2002, Section 3",
        "Companies Act 2013, Section 248 — Power of ROC to strike off shell entities",
        "RBI Master Direction on KYC 2016, Chapter VII",
        "FIU-IND STR Filing Guidelines: Shell company typology",
    ],
    "Dormant Activation": [
        "RBI Circular DBR.No.Leg.BC.59/09.07.005/2018-19 — Inoperative accounts",
        "PMLA 2002, Section 12",
        "RBI Master Direction on KYC 2016, Chapter VII",
        "FIU-IND STR Filing Guidelines: Dormant account reactivation typology",
    ],
    "Profile Mismatch": [
        "RBI Master Direction on KYC 2016, Chapter VI — Customer Due Diligence",
        "RBI Master Direction on KYC 2016, Chapter VII",
        "PMLA Rules 2005, Rule 9 — KYC norms",
    ],
}

DEFAULT_CITATIONS = [
    "PMLA 2002, Section 3 — Offence of money-laundering",
    "PMLA 2002, Section 12 — Obligation of reporting entities",
    "RBI Master Direction on KYC 2016, Chapter VII — Suspicious Transaction Reporting",
]


def _citations_for(pattern: str) -> List[str]:
    return PMLA_CITATIONS.get(pattern, DEFAULT_CITATIONS)


def _format_inr(n: float) -> str:
    return "INR {:,.0f}".format(float(n))


def _build_evidence_summary(
    alert: Dict,
    case: Optional[Dict],
    graph: nx.DiGraph,
    entity_txns: pd.DataFrame,
) -> str:
    """Human-readable case summary, written as markdown."""
    aid = alert.get("alert_id", "UNKNOWN")
    entities = alert.get("entities", [])
    entity_names = []
    for e in entities:
        if graph.has_node(e):
            entity_names.append(graph.nodes[e].get("name", e))

    lines = [
        f"# FIU Evidence Package — {aid}",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Reporting Institution: Public Sector Bank (Demo)",
        "",
        "## 1. Case Summary",
        "",
        f"- Alert ID: `{aid}`",
        f"- Pattern: **{alert.get('pattern_type', 'Unknown')}**",
        f"- Severity: **{alert.get('severity', 'N/A')}**",
        f"- Detection Confidence: {alert.get('confidence', 0)}%",
        f"- Total Flagged Flow: {_format_inr(alert.get('total_flow', 0))}",
        f"- Entities Involved: {len(entities)}",
    ]
    if case:
        lines += [
            f"- Case Status: **{case.get('status', 'OPEN')}**",
            f"- Assigned To: {case.get('assigned_to') or 'Unassigned'}",
            f"- Last Updated: {case.get('updated_at', 'N/A')}",
        ]
    lines += ["", "## 2. Description", "", alert.get("description", "No description recorded."), ""]
    lines += ["## 3. Entities", ""]
    for nid, name in zip(entities, entity_names):
        if graph.has_node(nid):
            nd = graph.nodes[nid]
            lines.append(
                f"- `{nid}` — **{name}** ({nd.get('type', 'N/A')}, branch: {nd.get('branch', 'N/A')}, "
                f"product: {nd.get('product', 'N/A')})"
            )
        else:
            lines.append(f"- `{nid}` — {name}")
    lines += ["", "## 4. Transaction Volume", ""]
    if not entity_txns.empty:
        fraud_txns = entity_txns[entity_txns["is_fraud"]]
        lines += [
            f"- Total transactions in scope: {len(entity_txns)}",
            f"- Flagged as fraudulent: {len(fraud_txns)}",
            f"- Total volume: {_format_inr(entity_txns['amount'].sum())}",
            f"- Fraud volume: {_format_inr(fraud_txns['amount'].sum())}",
        ]
    lines += ["", "## 5. Recommended Action", "", alert.get("recommendation", "Follow standard STR procedure.")]
    lines += [
        "",
        "## 6. Regulatory Basis",
        "",
        "The patterns above are reported under the following statutes and circulars:",
        "",
    ]
    for c in _citations_for(alert.get("pattern_type", "")):
        lines.append(f"- {c}")

    if case and case.get("audit_log"):
        lines += ["", "## 7. Investigator Audit Trail", ""]
        for entry in case["audit_log"]:
            lines.append(
                f"- {entry.get('timestamp', '')} | {entry.get('author', '')} | "
                f"{entry.get('action', '')} — {entry.get('note', '')}"
            )

    lines += [
        "",
        "---",
        "Disclaimer: This evidence package is generated by the RUDRA fraud detection system. ",
        "All findings must be reviewed by the bank's compliance officer before STR submission.",
    ]
    return "\n".join(lines)


def _build_subgraph_json(graph: nx.DiGraph, entities: List[str]) -> bytes:
    """Export the entities + their direct neighbors as NetworkX node-link JSON."""
    nodes = set(entities)
    for e in entities:
        if graph.has_node(e):
            nodes.update(graph.successors(e))
            nodes.update(graph.predecessors(e))
    sub = graph.subgraph(nodes).copy()

    # Convert numpy types in edge attributes to plain python before serialising
    data = nx.node_link_data(sub, edges="links")
    return json.dumps(data, indent=2, default=str).encode("utf-8")


def _build_transaction_chain_csv(transactions: pd.DataFrame, entities: List[str]) -> bytes:
    mask = transactions["sender_id"].isin(entities) | transactions["receiver_id"].isin(entities)
    sub = transactions.loc[mask].sort_values("timestamp")
    buf = io.StringIO()
    sub.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _build_citations_txt(pattern: str) -> bytes:
    return "\n".join(_citations_for(pattern)).encode("utf-8")


def build_package(
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    alert: Dict,
    sar_pdf_path: Optional[str],
    case: Optional[Dict] = None,
) -> bytes:
    """Build the zip in memory and return its raw bytes.

    sar_pdf_path is the path to a previously-generated SAR PDF if one exists;
    pass None and the zip will simply not contain the PDF (the markdown summary
    still captures every detail).
    """
    entities = alert.get("entities", [])
    mask = transactions["sender_id"].isin(entities) | transactions["receiver_id"].isin(entities)
    entity_txns = transactions.loc[mask]

    summary_md = _build_evidence_summary(alert, case, graph, entity_txns)
    subgraph_json = _build_subgraph_json(graph, entities)
    chain_csv = _build_transaction_chain_csv(transactions, entities)
    citations_txt = _build_citations_txt(alert.get("pattern_type", ""))
    audit_json = json.dumps(case or {}, indent=2, default=str).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("evidence_summary.md", summary_md)
        z.writestr("subgraph.json", subgraph_json)
        z.writestr("transaction_chain.csv", chain_csv)
        z.writestr("pmla_citations.txt", citations_txt)
        z.writestr("case_audit_log.json", audit_json)
        if sar_pdf_path and os.path.exists(sar_pdf_path):
            z.write(sar_pdf_path, arcname=f"SAR_{alert.get('alert_id', 'UNK')}.pdf")
    return buf.getvalue()
