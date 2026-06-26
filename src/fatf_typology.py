"""
FATF typology + Indian-regulatory tagging for alerts.

Every detector pattern is mapped to a named FATF money-laundering typology and
the specific Indian-regulatory hooks an FIU-IND filer cites: the relevant PMLA
2002 section, the RBI Master Direction reference, and the FIU-IND red-flag
indicator family. We also derive a graded **legal basis for action** so the UI
and STR can state *what the bank is empowered to do now*:

  * RBI KYC Master Direction (2016, as amended) ¶38 lets a bank impose
    enhanced monitoring / partial operating restrictions on a suspicious
    account *before* a predicate offence is established — a pre-emptive lever.
  * PMLA 2002 §12 is the mandatory Suspicious-Transaction-Report obligation
    once suspicion crosses the reporting bar.

This is the regulator-legible framing rivals narrated in their decks but never
wired into code. Here every tag is attached to real, running detectors.

NOTE: FATF publishes typologies as descriptive families, not numbered codes;
the short `fatf_code` values here are our stable internal handles for the UI,
with the human-readable `fatf_typology` carrying the actual FATF wording.
"""

from __future__ import annotations

from typing import Dict, List

# RBI KYC Master Direction master reference reused across entries.
_RBI_KYC = "RBI Master Direction – KYC, 2016 (as amended)"

TYPOLOGY: Dict[str, Dict] = {
    "Circular Transaction": {
        "fatf_code": "ML-ROUNDTRIP",
        "fatf_typology": "Round-tripping / circular fund movement to obscure origin",
        "fiu_advisory": "FIU-IND red-flag: funds returning to originator through intermediaries",
        "pmla_section": "PMLA 2002 §3 (offence) r/w §12 (STR)",
        "rbi_ref": _RBI_KYC,
    },
    "Rapid Layering": {
        "fatf_code": "ML-LAYERING",
        "fatf_typology": "Layering — rapid movement through multiple accounts to break the audit trail",
        "fiu_advisory": "FIU-IND red-flag: rapid pass-through with no economic rationale",
        "pmla_section": "PMLA 2002 §3 r/w §12",
        "rbi_ref": _RBI_KYC,
    },
    "Smurfing / Structuring": {
        "fatf_code": "ML-STRUCTURING",
        "fatf_typology": "Structuring (smurfing) — splitting value to stay below reporting thresholds",
        "fiu_advisory": "FIU-IND red-flag: multiple sub-threshold transactions to/from many parties",
        "pmla_section": "PMLA 2002 §12 r/w Rule 3 (cash/threshold reporting)",
        "rbi_ref": _RBI_KYC + "; PMLA Maintenance of Records Rules, 2005",
    },
    "Shell Company Funnel": {
        "fatf_code": "ML-SHELL",
        "fatf_typology": "Use of shell / front companies and funnel accounts",
        "fiu_advisory": "FIU-IND red-flag: high-velocity pass-through entity with no genuine business",
        "pmla_section": "PMLA 2002 §3 r/w §12",
        "rbi_ref": _RBI_KYC + " (beneficial-ownership identification)",
    },
    "Dormant Activation": {
        "fatf_code": "ML-DORMANT",
        "fatf_typology": "Sudden reactivation of dormant accounts (possible account takeover / mule reuse)",
        "fiu_advisory": "FIU-IND red-flag: dormant account with abrupt high-value activity",
        "pmla_section": "PMLA 2002 §12",
        "rbi_ref": _RBI_KYC + " (ongoing due diligence)",
    },
    "Profile Mismatch": {
        "fatf_code": "ML-PROFILE",
        "fatf_typology": "Transactions inconsistent with the customer's declared profile",
        "fiu_advisory": "FIU-IND red-flag: activity disproportionate to KYC profile",
        "pmla_section": "PMLA 2002 §12",
        "rbi_ref": _RBI_KYC + " (risk categorisation & periodic review)",
    },
    "Recruiter / Coordinator": {
        "fatf_code": "ML-MULE-HERDER",
        "fatf_typology": "Money-mule network — a coordinator funding and herding a fleet of pass-through accounts",
        "fiu_advisory": "FIU-IND / I4C mule-account advisory: single funder seeding many forwarding accounts",
        "pmla_section": "PMLA 2002 §3 r/w §12",
        "rbi_ref": _RBI_KYC + "; RBI mule-account & money-mule risk guidance",
    },
}

# Stable iteration order for tests/UIs.
PATTERN_TYPES: List[str] = list(TYPOLOGY.keys())

_GENERIC = {
    "fatf_code": "ML-GENERIC",
    "fatf_typology": "Suspicious money-laundering activity (unclassified typology)",
    "fiu_advisory": "FIU-IND general suspicious-transaction indicators",
    "pmla_section": "PMLA 2002 §12",
    "rbi_ref": _RBI_KYC,
}


def _legal_basis(severity: str, confidence: float) -> str:
    """Graded action basis: pre-emptive RBI restriction vs mandatory PMLA STR."""
    sev = (severity or "").upper()
    conf = float(confidence or 0.0)
    if sev == "CRITICAL" or conf >= 85:
        return "PMLA §12 — file Suspicious Transaction Report (STR) — mandatory"
    if sev == "HIGH":
        return ("RBI KYC MD ¶38 — enhanced monitoring / pre-emptive account "
                "restriction pending STR determination")
    return "Internal Enhanced Due Diligence — monitor and re-score"


def tag_alert(alert: Dict) -> Dict:
    """Return `alert` enriched with FATF typology + Indian-regulatory refs +
    a graded legal basis. Additive — never drops existing fields. Unknown
    pattern types fall back to a generic typology rather than crashing."""
    entry = TYPOLOGY.get(alert.get("pattern_type", ""), _GENERIC)
    out = dict(alert)
    out["fatf_code"] = entry["fatf_code"]
    out["fatf_typology"] = entry["fatf_typology"]
    out["fiu_advisory"] = entry["fiu_advisory"]
    out["pmla_section"] = entry["pmla_section"]
    out["rbi_ref"] = entry["rbi_ref"]
    out["regulatory_refs"] = [entry["pmla_section"], entry["rbi_ref"], entry["fiu_advisory"]]
    out["legal_basis"] = _legal_basis(alert.get("severity", ""), alert.get("confidence", 0))
    return out
