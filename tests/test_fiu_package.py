"""FIU package zip must contain all required files including STR XML."""

import io
import zipfile


def test_package_contains_required_files(synthetic_pipeline):
    from fiu_package import build_package
    alerts = synthetic_pipeline["alerts"]
    alert = next((a for a in alerts if a["severity"] in {"CRITICAL", "HIGH"}), alerts[0])
    zip_bytes = build_package(
        synthetic_pipeline["graph"],
        synthetic_pipeline["df"],
        alert,
        sar_pdf_path=None,
        case=None,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = set(z.namelist())
    required = {
        "evidence_summary.md",
        "STR.xml",
        "subgraph.json",
        "transaction_chain.csv",
        "pmla_citations.txt",
        "case_audit_log.json",
    }
    missing = required - names
    assert not missing, f"FIU package missing files: {missing}"


def test_str_xml_is_well_formed(synthetic_pipeline):
    from fiu_package import build_package
    import xml.etree.ElementTree as ET
    alert = synthetic_pipeline["alerts"][0]
    zip_bytes = build_package(
        synthetic_pipeline["graph"], synthetic_pipeline["df"],
        alert, sar_pdf_path=None, case=None,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        xml_bytes = z.read("STR.xml")
    root = ET.fromstring(xml_bytes)
    assert root.tag == "FIUINDReport"
    assert root.find("ReportHeader") is not None
    assert root.find("ReportingEntity") is not None
    assert root.find("SuspiciousActivity") is not None
    assert root.find("Subjects") is not None


def test_subgraph_json_contains_alert_entities(synthetic_pipeline):
    from fiu_package import build_package
    import json
    alert = synthetic_pipeline["alerts"][0]
    zip_bytes = build_package(
        synthetic_pipeline["graph"], synthetic_pipeline["df"],
        alert, sar_pdf_path=None, case=None,
    )
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        sg = json.loads(z.read("subgraph.json"))
    node_ids = {n["id"] for n in sg.get("nodes", [])}
    for e in alert["entities"]:
        assert e in node_ids
