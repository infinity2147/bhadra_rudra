"""FATF typology + legal-basis tagging: every alert maps to a named FATF
typology and the relevant FIU-IND/PMLA/RBI references, plus a graded legal basis
for action (the regulator-legible framing competitors narrated but didn't wire)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fatf_typology import TYPOLOGY, tag_alert, PATTERN_TYPES  # noqa: E402


def test_every_pattern_has_a_typology():
    for pt in PATTERN_TYPES:
        entry = TYPOLOGY[pt]
        assert entry["fatf_typology"] and entry["pmla_section"] and entry["rbi_ref"], \
            f"incomplete typology for {pt}"


def test_tag_alert_adds_fields_without_dropping_existing():
    alert = {"alert_id": "A1", "pattern_type": "Rapid Layering", "severity": "HIGH",
             "confidence": 70.0, "entities": ["X"]}
    tagged = tag_alert(alert)
    assert tagged["alert_id"] == "A1"           # original fields preserved
    assert tagged["fatf_code"]
    assert tagged["fatf_typology"]
    assert isinstance(tagged["regulatory_refs"], list) and tagged["regulatory_refs"]
    assert tagged["legal_basis"]


def test_legal_basis_tiers_by_severity():
    crit = tag_alert({"pattern_type": "Circular Transaction", "severity": "CRITICAL", "confidence": 95})
    high = tag_alert({"pattern_type": "Circular Transaction", "severity": "HIGH", "confidence": 70})
    med = tag_alert({"pattern_type": "Profile Mismatch", "severity": "MEDIUM", "confidence": 45})
    assert "STR" in crit["legal_basis"]                 # mandatory reporting
    assert "38" in high["legal_basis"] or "restrict" in high["legal_basis"].lower()
    assert med["legal_basis"] and med["legal_basis"] != crit["legal_basis"]


def test_unknown_pattern_falls_back_gracefully():
    tagged = tag_alert({"pattern_type": "Something New", "severity": "HIGH", "confidence": 60})
    assert tagged["fatf_typology"]          # generic fallback, not a crash
