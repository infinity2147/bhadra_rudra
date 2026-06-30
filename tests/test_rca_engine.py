import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fatf_typology import TYPOLOGY, _GENERIC, PATTERN_TYPES
import rca_engine


def test_every_typology_has_control_gap_and_remediation():
    for pattern in PATTERN_TYPES:
        entry = TYPOLOGY[pattern]
        assert entry.get("control_gap"), f"{pattern} missing control_gap"
        assert entry.get("remediation"), f"{pattern} missing remediation"
    assert _GENERIC.get("control_gap")
    assert _GENERIC.get("remediation")


def test_reconstruct_shape(synthetic_pipeline):
    alerts = synthetic_pipeline["alerts"]
    alert = next(a for a in alerts if a.get("entities"))
    out = rca_engine.reconstruct(alert, synthetic_pipeline["graph"],
                                 synthetic_pipeline["df"], risk_scores=[])
    assert "error" not in out
    assert set(out["signals"]) == {
        "in_scc", "shell_count", "dormant_count", "subthreshold_deposits",
        "max_fan_in", "n_txns", "total_amount",
    }
    assert out["method"]["pattern"] == alert.get("pattern_type")
    assert isinstance(out["origin"], list) and isinstance(out["cashout"], list)
    assert out["signals"]["n_txns"] == len(out["trace"]["timeline"])
    assert isinstance(out["signals"]["in_scc"], bool)
    assert out["signals"]["total_amount"] >= 0
