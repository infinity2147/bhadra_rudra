import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fatf_typology import TYPOLOGY, _GENERIC, PATTERN_TYPES


def test_every_typology_has_control_gap_and_remediation():
    for pattern in PATTERN_TYPES:
        entry = TYPOLOGY[pattern]
        assert entry.get("control_gap"), f"{pattern} missing control_gap"
        assert entry.get("remediation"), f"{pattern} missing remediation"
    assert _GENERIC.get("control_gap")
    assert _GENERIC.get("remediation")
