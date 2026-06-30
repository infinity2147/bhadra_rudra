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


def test_diagnose_rule_typed_path():
    incident = {"primary_pattern": "Smurfing / Structuring", "patterns": []}
    recon = {"signals": {"subthreshold_deposits": 7, "n_txns": 7, "total_amount": 1330000.0,
                         "in_scc": False, "shell_count": 0, "dormant_count": 0, "max_fan_in": 1}}
    d = rca_engine.diagnose_root_cause(incident, recon)
    assert d["basis"] == "rule"
    assert d["pattern_resolved"] == "Smurfing / Structuring"
    assert "threshold" in d["control_gap"].lower()
    assert "7 transfers" in d["evidence"]


def test_diagnose_ml_anomaly_infers_from_signals():
    incident = {"primary_pattern": "ML-Detected Anomaly", "patterns": []}
    recon = {"signals": {"in_scc": False, "shell_count": 2, "max_fan_in": 5,
                         "dormant_count": 0, "subthreshold_deposits": 0,
                         "n_txns": 12, "total_amount": 5000000.0}}
    d = rca_engine.diagnose_root_cause(incident, recon)
    assert d["basis"] == "inferred"
    assert d["pattern_resolved"] == "Shell Company Funnel"


def test_diagnose_generic_when_no_signal():
    incident = {"primary_pattern": "ML-Detected Anomaly", "patterns": []}
    recon = {"signals": {"in_scc": False, "shell_count": 0, "max_fan_in": 1,
                         "dormant_count": 0, "subthreshold_deposits": 0,
                         "n_txns": 3, "total_amount": 90000.0}}
    d = rca_engine.diagnose_root_cause(incident, recon)
    assert d["basis"] == "generic"
    assert d["control_gap"] == _GENERIC["control_gap"]
