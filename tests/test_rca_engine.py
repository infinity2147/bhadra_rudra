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
    node_set = {n["id"] for n in out["trace"]["nodes"]}
    df = synthetic_pipeline["df"]
    m = df["sender_id"].isin(node_set) & df["receiver_id"].isin(node_set)
    assert out["signals"]["n_txns"] == int(m.sum())
    assert out["signals"]["total_amount"] == round(float(df.loc[m, "amount"].sum()), 2)
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
    assert "2 shell entity" in d["evidence"]


def test_diagnose_generic_when_no_signal():
    incident = {"primary_pattern": "ML-Detected Anomaly", "patterns": []}
    recon = {"signals": {"in_scc": False, "shell_count": 0, "max_fan_in": 1,
                         "dormant_count": 0, "subthreshold_deposits": 0,
                         "n_txns": 3, "total_amount": 90000.0}}
    d = rca_engine.diagnose_root_cause(incident, recon)
    assert d["basis"] == "generic"
    assert d["control_gap"] == _GENERIC["control_gap"]
    assert d["pattern_resolved"] == "Unclassified anomaly"


def test_recommend_account_and_policy():
    diag = {"control_gap": "gap text", "remediation": "fix text"}
    incident = {"entities": ["E1", "E2"], "entity_names": ["Acme", "Bravo"]}
    recs = rca_engine.recommend(diag, incident)
    assert recs["account_level"][0] == {
        "entity_id": "E1", "name": "Acme",
        "action": "Enhanced Due Diligence (EDD) + transaction hold pending review",
    }
    assert len(recs["account_level"]) == 2
    assert recs["account_level"][1] == {
        "entity_id": "E2", "name": "Bravo",
        "action": "Enhanced Due Diligence (EDD) + transaction hold pending review",
    }
    assert recs["policy_level"] == [{"recommendation": "fix text", "closes_gap": "gap text"}]


def test_recommend_name_falls_back_to_entity_id():
    diag = {"control_gap": "g", "remediation": "f"}
    incident = {"entities": ["E1", "E2"], "entity_names": ["Acme"]}  # shorter than entities
    recs = rca_engine.recommend(diag, incident)
    assert recs["account_level"][1]["name"] == "E2"


def test_recommend_caps_account_level():
    diag = {"control_gap": "g", "remediation": "f"}
    incident = {"entities": [f"E{i}" for i in range(6)], "entity_names": []}
    recs = rca_engine.recommend(diag, incident, max_accounts=3)
    assert len(recs["account_level"]) == 3


def test_build_rca_full_dossier(synthetic_pipeline):
    alerts = synthetic_pipeline["alerts"]
    alert = next((a for a in alerts if a.get("entities")), None)
    assert alert is not None, "No alert with entities found in synthetic_pipeline"
    incident = {
        "incident_id": "INC-TEST",
        "primary_pattern": alert.get("pattern_type"),
        "patterns": [alert.get("pattern_type")],
        "entities": alert.get("entities"),
        "entity_names": alert.get("entities"),
    }
    dossier = rca_engine.build_rca(incident, alert, synthetic_pipeline["graph"],
                                   synthetic_pipeline["df"], risk_scores=[])
    assert dossier["incident_id"] == "INC-TEST"
    assert {"reconstruction", "diagnosis", "recommendations", "narrative"} <= set(dossier)
    assert dossier["diagnosis"]["control_gap"] in dossier["narrative"]
