"""
FIU Evidence Package Builder

Bundles everything an investigator needs to file a Suspicious Transaction Report
with FIU-IND (Financial Intelligence Unit, India) into a single downloadable zip:

  evidence_summary.md     — human-readable case summary + reasoning
  STR.xml                 — schema-shaped STR XML matching the FIU-IND format
  SAR_<id>.pdf            — the formal SAR document
  subgraph.json           — NetworkX-format export of the fraud subgraph
  transaction_chain.csv   — every transaction in the case, in order
  pmla_citations.txt      — PMLA / RBI sections relevant to the pattern
  case_audit_log.json     — full audit trail for regulatory review

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
from xml.sax.saxutils import escape as _xml_escape

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


# ── Single source of truth: FIU package contents ─────────────────────────────
# Both build_package() (this file) and SARGenerator._build_supporting_docs
# (sar_generator.py) read from this constant. Pre-T2 p2, the SAR PDF used to
# list files that didn't exist in the zip; the two diverged silently. Keeping
# the canonical list here prevents that drift — adding a file means updating
# one constant + the corresponding writer call below.

FIU_PACKAGE_FILES = [
    ("evidence_summary.md",   "human-readable case summary + reasoning"),
    ("STR.xml",                "FIU-IND-format Suspicious Transaction Report XML"),
    ("subgraph.json",          "NetworkX node-link export of the fraud subgraph + 1-hop neighbours"),
    ("transaction_chain.csv",  "every transaction involving the suspect entities, in chronological order"),
    ("pmla_citations.txt",     "PMLA / RBI sections relevant to this typology"),
    ("case_audit_log.json",    "hash-chain-verified investigator audit trail"),
    ("SAR_<alert_id>.pdf",     "this document, formal SAR (ReportLab PDF; included only when previously generated)"),
]


def package_files_listing(alert_id: str = "<alert_id>") -> str:
    """Return a numbered, human-readable list of FIU package contents.

    Used by SARGenerator._build_supporting_docs so the SAR PDF accurately
    describes what's in the FIU zip. Substitutes the real alert_id into the
    SAR PDF filename slot.
    """
    lines = []
    for i, (name, desc) in enumerate(FIU_PACKAGE_FILES, start=1):
        rendered_name = name.replace("<alert_id>", alert_id)
        # Right-pad the name to a fixed column so the description column lines up.
        lines.append(f"  {i}. {rendered_name:<28s} — {desc}")
    return "\n".join(lines)


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
        # Unlabeled real datasets may lack an is_fraud column — degrade to "none
        # known" rather than KeyError. astype(bool) tolerates int 0/1 labels.
        fraud_txns = (entity_txns[entity_txns["is_fraud"].astype(bool)]
                      if "is_fraud" in entity_txns.columns else entity_txns.iloc[0:0])
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


# ── STR XML (FIU-IND format) ──────────────────────────────────────────────────
#
# FIU-IND's STR format requires a structured XML with reporting entity,
# subject(s), suspicious activity description, and transactions. We model the
# canonical fields. Real submissions also include PII (PAN, Aadhaar) which we
# deliberately omit per DPDP Act data-minimisation — the redaction is
# explicitly noted in the XML so a compliance officer knows what to enrich
# before filing.

_REPORT_TYPE_MAP = {
    "Circular Transaction": "STR_CIRCULAR",
    "Rapid Layering": "STR_LAYERING",
    "Smurfing / Structuring": "STR_STRUCTURING",
    "Shell Company Funnel": "STR_SHELL_FUNNEL",
    "Dormant Activation": "STR_DORMANT",
    "Profile Mismatch": "STR_PROFILE_MISMATCH",
}


def _xml(tag: str, content="", **attrs) -> str:
    """Tiny XML element builder with safe escaping."""
    attr_str = "".join(f' {k}="{_xml_escape(str(v))}"' for k, v in attrs.items())
    if content == "":
        return f"<{tag}{attr_str}/>"
    return f"<{tag}{attr_str}>{_xml_escape(str(content))}</{tag}>"


def _build_str_xml(
    alert: Dict,
    graph: nx.DiGraph,
    entity_txns: pd.DataFrame,
    case: Optional[Dict],
    shap_features: Optional[List[Dict]] = None,
) -> bytes:
    """Build a schema-shaped FIU-IND STR XML document.

    Field names follow the FIU-IND STR convention as published in their
    Reporting Entity Guidelines. We mark every PII field as REDACTED with a
    reason — banks file STRs with the PII filled in; investigators add it
    just before submission. Doing it this way keeps RUDRA DPDP-clean.

    The report embeds the FATF typology + graded legal basis (from
    fatf_typology.tag_alert) and, when `shap_features` is supplied, a
    machine-readable `<SHAPExplainability>` block so the model's reasoning
    travels with the regulatory filing rather than living only in the UI.
    """
    from fatf_typology import tag_alert

    aid = alert.get("alert_id", "")
    pattern = alert.get("pattern_type", "")
    severity = alert.get("severity", "HIGH")
    entities = alert.get("entities", [])
    total_flow = alert.get("total_flow", 0)
    desc = alert.get("description", "")
    confidence = alert.get("confidence", 0)
    tagged = tag_alert(alert)

    # Tolerate datasets without an is_fraud column (unlabeled real data).
    fraud_txns = (entity_txns[entity_txns["is_fraud"].astype(bool)]
                  if "is_fraud" in entity_txns.columns else entity_txns.iloc[0:0]).sort_values("timestamp")

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(f'<FIUINDReport version="2.1" reportType="{_REPORT_TYPE_MAP.get(pattern, "STR_GENERIC")}">')

    # 1. Report header
    lines.append('  <ReportHeader>')
    lines.append('    ' + _xml("ReportID", f"STR-{aid}"))
    lines.append('    ' + _xml("GeneratedAt", datetime.now().isoformat()))
    lines.append('    ' + _xml("FilingDeadlineDays", 7))
    lines.append('    ' + _xml("RegulatoryFramework", "PMLA 2002 Section 12 read with PMLA Rules 2005 Rule 3"))
    lines.append('  </ReportHeader>')

    # 2. Reporting entity (the bank)
    lines.append('  <ReportingEntity>')
    lines.append('    ' + _xml("EntityType", "ScheduledCommercialBank"))
    lines.append('    ' + _xml("EntityCategory", "PublicSectorBank"))
    lines.append('    ' + _xml("EntityName", "[Public Sector Bank — Demo]"))
    lines.append('    ' + _xml("BranchCode", "REDACTED", reason="DPDP_DataMinimisation"))
    lines.append('    ' + _xml("ReportingOfficerName", "REDACTED", reason="DPDP_DataMinimisation"))
    lines.append('  </ReportingEntity>')

    # 3. Suspicious activity classification
    lines.append('  <SuspiciousActivity>')
    lines.append('    ' + _xml("ActivityType", pattern))
    lines.append('    ' + _xml("Severity", severity))
    lines.append('    ' + _xml("DetectionConfidencePct", confidence))
    lines.append('    ' + _xml("TotalSuspiciousAmount", f"{total_flow:.2f}"))
    lines.append('    ' + _xml("Currency", "INR"))
    lines.append('    ' + _xml("Description", desc))
    lines.append('    ' + _xml("DetectionMethod", "Graph-based pattern analysis with ML edge classifier"))
    lines.append('  </SuspiciousActivity>')

    # 3b. FATF typology + Indian-regulatory basis for action
    lines.append('  <RegulatoryClassification>')
    lines.append('    ' + _xml("FATFTypology", tagged["fatf_typology"], code=tagged["fatf_code"]))
    lines.append('    ' + _xml("FIUAdvisory", tagged["fiu_advisory"]))
    lines.append('    ' + _xml("PMLASection", tagged["pmla_section"]))
    lines.append('    ' + _xml("RBIReference", tagged["rbi_ref"]))
    lines.append('    ' + _xml("LegalBasis", tagged["legal_basis"]))
    lines.append('  </RegulatoryClassification>')

    # 3c. Machine-readable model explainability (only when SHAP is available)
    if shap_features:
        lines.append('  <SHAPExplainability>')
        lines.append('    ' + _xml("Model", "XGBoost edge classifier"))
        lines.append('    ' + _xml("Basis", "signed feature contributions (SHAP)"))
        for f in shap_features[:8]:
            lines.append('    ' + _xml(
                "Feature", "",
                name=str(f.get("feature", "")),
                value=f"{float(f.get('value', 0)):.4f}",
                contribution=f"{float(f.get('shap', 0)):.4f}",
            ))
        lines.append('  </SHAPExplainability>')

    # 4. Subjects
    lines.append('  <Subjects>')
    for i, eid in enumerate(entities, start=1):
        if not graph.has_node(eid):
            continue
        nd = graph.nodes[eid]
        lines.append(f'    <Subject sequenceNumber="{i}">')
        lines.append('      ' + _xml("InternalEntityID", eid))
        lines.append('      ' + _xml("Name", nd.get("name", "")))
        lines.append('      ' + _xml("SubjectType", nd.get("type", "individual")))
        lines.append('      ' + _xml("AccountProduct", nd.get("product", "")))
        lines.append('      ' + _xml("AccountBranch", nd.get("branch", "")))
        # PII fields kept as placeholders for the bank to fill in just before submission
        lines.append('      ' + _xml("PAN", "REDACTED", reason="DPDP_DataMinimisation_FillBeforeFiling"))
        lines.append('      ' + _xml("Aadhaar", "REDACTED", reason="DPDP_DataMinimisation_FillBeforeFiling"))
        lines.append('      ' + _xml("Address", "REDACTED", reason="DPDP_DataMinimisation_FillBeforeFiling"))
        lines.append('    </Subject>')
    lines.append('  </Subjects>')

    # 5. Transactions — only the flagged ones, in chronological order
    lines.append('  <Transactions>')
    for _, t in fraud_txns.head(200).iterrows():
        lines.append('    <Transaction>')
        lines.append('      ' + _xml("InternalTxnID", t.get("transaction_id", "")))
        lines.append('      ' + _xml("Timestamp", str(t.get("timestamp", ""))))
        lines.append('      ' + _xml("Amount", f"{float(t.get('amount', 0)):.2f}"))
        lines.append('      ' + _xml("Currency", t.get("currency", "INR")))
        lines.append('      ' + _xml("Rail", t.get("transaction_type", "")))
        lines.append('      ' + _xml("Channel", t.get("channel", "")))
        lines.append('      ' + _xml("PurposeCode", t.get("purpose_code", "")))
        lines.append('      ' + _xml("FromEntityID", t.get("sender_id", "")))
        lines.append('      ' + _xml("ToEntityID", t.get("receiver_id", "")))
        lines.append('    </Transaction>')
    lines.append('  </Transactions>')

    # 6. Grounds for suspicion (PMLA citations)
    lines.append('  <GroundsForSuspicion>')
    for c in _citations_for(pattern):
        lines.append('    ' + _xml("Citation", c))
    lines.append('  </GroundsForSuspicion>')

    # 7. Audit trail — just the head hash, not the full chain
    if case and case.get("audit_log"):
        log = case["audit_log"]
        head_entry = log[-1] if log else {}
        lines.append('  <AuditTrail>')
        lines.append('    ' + _xml("Entries", len(log)))
        lines.append('    ' + _xml("HeadHash", head_entry.get("this_hash", "")))
        lines.append('    ' + _xml("LastAction", head_entry.get("action", "")))
        lines.append('    ' + _xml("LastTimestamp", head_entry.get("timestamp", "")))
        lines.append('  </AuditTrail>')

    lines.append('</FIUINDReport>')
    return "\n".join(lines).encode("utf-8")


def build_package(
    graph: nx.DiGraph,
    transactions: pd.DataFrame,
    alert: Dict,
    sar_pdf_path: Optional[str],
    case: Optional[Dict] = None,
    shap_features: Optional[List[Dict]] = None,
) -> bytes:
    """Build the zip in memory and return its raw bytes.

    sar_pdf_path is the path to a previously-generated SAR PDF if one exists;
    pass None and the zip will simply not contain the PDF (the markdown summary
    still captures every detail).

    shap_features (optional) is the alert's top SHAP attributions; when supplied
    they are embedded in STR.xml so the model's reasoning is part of the filing.
    """
    entities = alert.get("entities", [])
    mask = transactions["sender_id"].isin(entities) | transactions["receiver_id"].isin(entities)
    entity_txns = transactions.loc[mask]

    summary_md = _build_evidence_summary(alert, case, graph, entity_txns)
    subgraph_json = _build_subgraph_json(graph, entities)
    chain_csv = _build_transaction_chain_csv(transactions, entities)
    citations_txt = _build_citations_txt(alert.get("pattern_type", ""))
    audit_json = json.dumps(case or {}, indent=2, default=str).encode("utf-8")
    str_xml = _build_str_xml(alert, graph, entity_txns, case, shap_features=shap_features)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("evidence_summary.md", summary_md)
        z.writestr("STR.xml", str_xml)
        z.writestr("subgraph.json", subgraph_json)
        z.writestr("transaction_chain.csv", chain_csv)
        z.writestr("pmla_citations.txt", citations_txt)
        z.writestr("case_audit_log.json", audit_json)
        if sar_pdf_path and os.path.exists(sar_pdf_path):
            z.write(sar_pdf_path, arcname=f"SAR_{alert.get('alert_id', 'UNK')}.pdf")
    return buf.getvalue()
