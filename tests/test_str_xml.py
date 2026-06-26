"""The FIU-IND STR XML must carry the FATF typology, the graded legal basis,
and (when available) machine-readable SHAP explainability embedded directly in
the report — not just a separate PDF. (PROTOTYPE embedded SHAP in its FINnet
XML; we match that and add the FATF + legal-basis framing.)"""

import os
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fiu_package import _build_str_xml  # noqa: E402


def _fixture():
    g = nx.DiGraph()
    g.add_node("X", name="Acme Co", type="business", product="Current", branch="BR1")
    g.add_node("Y", name="Mule One", type="individual", product="Savings", branch="BR2")
    g.add_edge("X", "Y", total_amount=1_000_000.0)
    df = pd.DataFrame([{
        "transaction_id": "T1", "timestamp": "2025-03-01 10:00:00", "amount": 1_000_000.0,
        "currency": "INR", "transaction_type": "RTGS", "channel": "NetBanking",
        "purpose_code": "Business", "sender_id": "X", "receiver_id": "Y", "is_fraud": True,
    }])
    alert = {"alert_id": "ALERT_LAYER_0001", "pattern_type": "Rapid Layering",
             "severity": "CRITICAL", "confidence": 92.0, "entities": ["X", "Y"],
             "total_flow": 1_000_000, "description": "Layering chain"}
    return alert, g, df


def test_str_xml_embeds_fatf_and_legal_basis():
    alert, g, df = _fixture()
    xml = _build_str_xml(alert, g, df, case=None).decode("utf-8")
    assert "<FATFTypology" in xml
    assert "ML-LAYERING" in xml
    assert "<LegalBasis>" in xml
    assert "STR" in xml          # CRITICAL → mandatory STR basis


def test_str_xml_embeds_shap_when_provided():
    alert, g, df = _fixture()
    shap_features = [
        {"feature": "log_total_amount", "value": 13.8, "shap": 0.42},
        {"feature": "in_scc_3plus", "value": 1, "shap": 0.20},
    ]
    xml = _build_str_xml(alert, g, df, case=None, shap_features=shap_features).decode("utf-8")
    assert "<SHAPExplainability>" in xml
    assert "log_total_amount" in xml
    assert xml.count("<Feature") >= 2


def test_str_xml_omits_shap_block_when_absent():
    alert, g, df = _fixture()
    xml = _build_str_xml(alert, g, df, case=None).decode("utf-8")
    assert "<SHAPExplainability>" not in xml   # graceful: no SHAP, no empty block
