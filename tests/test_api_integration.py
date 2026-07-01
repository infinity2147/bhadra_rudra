"""HTTP-integration tests — exercise the real FastAPI app through TestClient.

The rest of the suite is unit-level (it imports engine functions directly), so
the entire HTTP layer — request coercion, the Depends(get_role) -> require()
RBAC chain, and response serialization — had ZERO coverage. Every bug found in
the June 2026 adversarial audit lived in exactly that layer:

  * /simulate/score 500'd on a non-numeric amount   (request coercion)
  * a junk X-User-Role was silently elevated to 200  (auth integration)
  * a string / Infinity threshold was accepted        (validation + serialize)

A unit test of `_coerce_value` cannot catch the Infinity case — it only fails
at response-serialize time. So these tests drive the app end-to-end and assert
exact status codes. They are the regression net for those fixes.

The TestClient boots the app once per module (startup loads the active dataset,
which is heavy), so keep this module's tests fast and side-effect-free; any test
that writes config resets it before returning.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

ADMIN = {"X-User-Role": "ADMIN"}
INV = {"X-User-Role": "INVESTIGATOR"}
JSON = {"Content-Type": "application/json"}


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import main
    with TestClient(main.app) as c:
        yield c


# ── Regression: /simulate/score amount coercion (was 500) ────────────────────

def test_simulate_rejects_non_numeric_amount(client):
    r = client.post("/api/simulate/score", headers=ADMIN,
                    json={"sender": "E001", "receiver": "E002",
                          "amount": "not_a_number", "timestamp": "2024-01-01T10:00:00"})
    assert r.status_code == 400, r.text


def test_simulate_rejects_negative_amount(client):
    r = client.post("/api/simulate/score", headers=ADMIN,
                    json={"sender": "E001", "receiver": "E002",
                          "amount": -5000, "timestamp": "2024-01-01T10:00:00"})
    assert r.status_code == 400, r.text


def test_simulate_accepts_valid_amount(client):
    r = client.post("/api/simulate/score", headers=ADMIN,
                    json={"sender": "E001", "receiver": "E002",
                          "amount": 250000, "timestamp": "2024-01-01T10:00:00"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "severity" in body or "probability" in body or "score" in body


# ── Regression: RBAC over HTTP (junk role was silently elevated to 200) ──────

def test_junk_role_rejected_on_gated_endpoint(client):
    # get_role raises 403 only where it's actually a dependency. /api/me gates;
    # ungated read-only endpoints (e.g. /api/dashboard) never inspect the role.
    r = client.get("/api/me", headers={"X-User-Role": "GUEST_ROLE"})
    assert r.status_code == 403, r.text


def test_no_role_header_defaults_to_investigator(client):
    # Demo convenience: absent header is allowed (read-only default).
    r = client.get("/api/me")
    assert r.status_code == 200


def test_investigator_blocked_from_config_write(client):
    r = client.post("/api/config/thresholds", headers={**INV, **JSON},
                    json={"circular_amount_tolerance": 0.2})
    assert r.status_code == 403, r.text


def test_audit_verify_gate_fires_before_existence(client):
    # INVESTIGATOR hits the RBAC gate (403) before the 404 existence check;
    # SUPERVISOR passes the gate and gets 404 for a non-existent case.
    assert client.get("/api/cases/FAKE_NOPE/verify", headers=INV).status_code == 403
    assert client.get("/api/cases/FAKE_NOPE/verify",
                      headers={"X-User-Role": "SUPERVISOR"}).status_code == 404


# ── Regression: config value validation over HTTP (string/inf were 200/500) ──

@pytest.mark.parametrize("payload", [
    '{"circular_amount_tolerance": "not_a_number"}',
    '{"circular_amount_tolerance": -50}',
    '{"circular_max_cycle_length": true}',
    '{"circular_amount_tolerance": Infinity}',
    '{"circular_amount_tolerance": NaN}',
    '{"this_is_not_a_real_key": 1}',
])
def test_config_write_rejects_bad_values(client, payload):
    r = client.post("/api/config/thresholds", headers={**ADMIN, **JSON}, content=payload)
    assert r.status_code == 400, f"{payload} -> {r.status_code} {r.text}"


def test_config_get_healthy_after_rejected_writes(client):
    # The bad writes above must NOT have poisoned the store (Infinity once
    # made every subsequent GET 500 on serialize).
    r = client.get("/api/config/thresholds", headers=ADMIN)
    assert r.status_code == 200, r.text
    assert isinstance(r.json()["current"]["circular_amount_tolerance"], (int, float))


def test_config_write_accepts_valid_then_resets(client):
    r = client.post("/api/config/thresholds", headers={**ADMIN, **JSON},
                    json={"circular_amount_tolerance": 0.2})
    assert r.status_code == 200, r.text
    # Leave the store as we found it.
    assert client.post("/api/config/thresholds/reset", headers=ADMIN).status_code == 200


# ── Smoke: no param-free GET endpoint returns a 5xx ──────────────────────────

PARAM_FREE_GETS = [
    "/", "/api/health", "/api/me", "/api/dashboard", "/api/alerts", "/api/cases",
    "/api/incidents", "/api/entities", "/api/graph", "/api/integrations/status",
    "/api/analytics/branches", "/api/analytics/channels",
    "/api/ml/metrics", "/api/ml/variants", "/api/config/thresholds", "/api/geo/flows",
    "/api/taint", "/api/simulate/scenarios", "/api/stream/status", "/api/stream/recent",
]


@pytest.mark.parametrize("path", PARAM_FREE_GETS)
def test_get_endpoint_no_server_error(client, path):
    r = client.get(path, headers=ADMIN)
    assert r.status_code < 500, f"{path} -> {r.status_code}: {r.text[:200]}"


# ── Smoke: path-param endpoints with real ids never 5xx ──────────────────────

@pytest.fixture(scope="module")
def real_ids(client):
    """Fetch genuine ids from the running app so the path-param smoke uses live
    data, not guesses. A missing list just leaves that id None (its paths skip)."""
    def first(path, key, sub="id"):
        try:
            d = client.get(path, headers=ADMIN).json()
            items = d.get(key, d) if isinstance(d, dict) else d
            if items:
                it = items[0]
                return it.get(sub) if isinstance(it, dict) else it
        except Exception:
            pass
        return None

    alerts = client.get("/api/alerts", headers=ADMIN).json().get("alerts", [])
    alert_id = alerts[0]["alert_id"] if alerts else None
    pattern = alerts[0].get("pattern_type") if alerts else None
    # an alert with >=2 entities gives us a real (u, v) edge to probe
    edge = next(((a["entities"][0], a["entities"][1])
                 for a in alerts if len(a.get("entities", [])) >= 2), (None, None))
    incident_id = first("/api/incidents", "incidents", "incident_id") or \
        first("/api/incidents", "incidents", "id")
    entity_id = first("/api/entities", "entities", "entity_id") or \
        first("/api/entities", "entities", "id")
    scenarios = client.get("/api/simulate/scenarios", headers=ADMIN).json()
    scen = scenarios.get("scenarios", scenarios) if isinstance(scenarios, dict) else scenarios
    scen_name = (scen[0].get("name") if isinstance(scen, list) and scen and isinstance(scen[0], dict)
                 else (list(scen.keys())[0] if isinstance(scen, dict) and scen else None))
    return {"alert": alert_id, "pattern": pattern, "edge": edge,
            "incident": incident_id, "entity": entity_id, "scenario": scen_name}


def test_path_param_gets_no_server_error(client, real_ids):
    a, e, inc, ent = real_ids["alert"], real_ids["edge"], real_ids["incident"], real_ids["entity"]
    paths = []
    if a:
        paths += [f"/api/alerts/{a}", f"/api/alerts/{a}/explain", f"/api/cases/{a}",
                  f"/api/journey/alert/{a}", f"/api/fiu/package/{a}"]  # /api/sar/generate is POST
    if inc:
        paths.append(f"/api/incidents/{inc}")
    if ent:
        paths += [f"/api/entities/{ent}", f"/api/graph/{ent}", f"/api/journey/{ent}"]
    if real_ids["pattern"]:
        paths.append(f"/api/patterns/{real_ids['pattern']}")
    if real_ids["scenario"]:
        paths.append(f"/api/simulate/scenario/{real_ids['scenario']}")
    if e[0] and e[1]:
        paths.append(f"/api/ml/ensemble/edge/{e[0]}/{e[1]}")

    assert paths, "no real ids were available to probe"
    errors = {p: client.get(p, headers=ADMIN).status_code for p in paths}
    server_errs = {p: c for p, c in errors.items() if c >= 500}
    assert not server_errs, f"5xx from path-param endpoints: {server_errs}"


# ── Mutations: isolated to a temp DB so the demo store is never polluted ─────

@pytest.fixture
def isolated_stores(tmp_path):
    """Swap the case + taint stores for fresh temp-DB instances for the duration
    of a mutation test, then restore. Keeps the real rudra.db pristine."""
    import main
    from case_manager import CaseStore
    from taint_store import TaintStore
    orig_cases, orig_taint = main.state.get("cases"), main.state.get("taint")
    main.state["cases"] = CaseStore(str(tmp_path))
    main.state["taint"] = TaintStore(str(tmp_path / "rudra.db"))
    main._invalidate_derived_caches()   # cached views must not reflect the old store
    try:
        yield
    finally:
        main.state["cases"] = orig_cases
        main.state["taint"] = orig_taint
        main._invalidate_derived_caches()   # restore the real-store view for later tests


def test_add_note_then_verify_chain(client, isolated_stores, real_ids):
    a = real_ids["alert"]
    r = client.post(f"/api/cases/{a}/note", headers={**INV, **JSON}, json={"note": "integration note"})
    assert r.status_code == 200, r.text
    assert any("integration note" in (e.get("note") or "")
               for e in r.json().get("audit_log", [])), r.text
    v = client.get(f"/api/cases/{a}/verify", headers={"X-User-Role": "SUPERVISOR"})
    assert v.status_code == 200 and v.json()["verified"] is True, v.text


def test_add_note_requires_text(client, isolated_stores, real_ids):
    r = client.post(f"/api/cases/{real_ids['alert']}/note", headers={**INV, **JSON}, json={"note": ""})
    assert r.status_code == 400, r.text


def test_note_on_unknown_alert_404(client, isolated_stores):
    r = client.post("/api/cases/NOPE_999/note", headers={**INV, **JSON}, json={"note": "x"})
    assert r.status_code == 404, r.text


def test_dispose_rbac_and_validation(client, isolated_stores, real_ids):
    a = real_ids["alert"]
    # INVESTIGATOR may move to INVESTIGATING ...
    assert client.post(f"/api/cases/{a}/dispose", headers={**INV, **JSON},
                       json={"status": "INVESTIGATING"}).status_code == 200
    # ... but NOT file a SAR (SUPERVISOR+ only)
    assert client.post(f"/api/cases/{a}/dispose", headers={**INV, **JSON},
                       json={"status": "SAR_FILED"}).status_code == 403
    assert client.post(f"/api/cases/{a}/dispose", headers={"X-User-Role": "SUPERVISOR", **JSON},
                       json={"status": "SAR_FILED"}).status_code == 200
    # garbage status is a 400, not a 500
    assert client.post(f"/api/cases/{a}/dispose", headers={**ADMIN, **JSON},
                       json={"status": "BOGUS"}).status_code == 400


def test_taint_seed(client, isolated_stores, real_ids):
    r = client.post(f"/api/taint/seed/{real_ids['alert']}", headers=INV)
    assert r.status_code == 200 and r.json()["seeded"] >= 0, r.text


def test_sar_generate_is_post_not_get(client, real_ids):
    """SAR generation is an action — POST only; GET must be rejected (405)."""
    a = real_ids["alert"]
    assert client.get(f"/api/sar/generate/{a}").status_code == 405
    r = client.post(f"/api/sar/generate/{a}")
    assert r.status_code == 200, r.text
    assert "report_text" in r.json()


def test_get_case_does_not_persist(client, isolated_stores, real_ids):
    """A GET on a case must be read-only — no junk case/audit rows on a refresh."""
    import main
    a = real_ids["alert"]
    assert main.state["cases"].get(a) is None          # fresh isolated store
    r = client.get(f"/api/cases/{a}")
    assert r.status_code == 200 and r.json()["status"] == "OPEN", r.text  # transient view
    assert main.state["cases"].get(a) is None, "GET persisted a case (side-effect on read)"
    assert main.state["cases"].list() == []


def test_alerts_cache_invalidated_on_dispose(client, isolated_stores, real_ids):
    """The cached alert list must reflect a dispose immediately — no stale status."""
    a = real_ids["alert"]
    before = client.get("/api/alerts").json()["alerts"]
    assert next(x["case_status"] for x in before if x["alert_id"] == a) == "OPEN"
    r = client.post(f"/api/cases/{a}/dispose", headers={**INV, **JSON}, json={"status": "INVESTIGATING"})
    assert r.status_code == 200, r.text
    after = client.get("/api/alerts").json()["alerts"]
    assert next(x["case_status"] for x in after if x["alert_id"] == a) == "INVESTIGATING", \
        "cache served stale case status after dispose"


def test_dashboard_cache_invalidated_on_dispose(client, isolated_stores, real_ids):
    """The cached dashboard must reflect a dispose (case_status_counts is live)."""
    a = real_ids["alert"]
    d0 = client.get("/api/dashboard").json()["case_status_counts"]
    r = client.post(f"/api/cases/{a}/dispose", headers={**INV, **JSON}, json={"status": "INVESTIGATING"})
    assert r.status_code == 200, r.text
    d1 = client.get("/api/dashboard").json()["case_status_counts"]
    assert d1.get("INVESTIGATING", 0) == d0.get("INVESTIGATING", 0) + 1, \
        f"dashboard served stale counts: {d0} -> {d1}"
    assert client.post("/api/taint/seed/NOPE_999", headers=INV).status_code == 404


def test_dashboard_total_alerts_is_the_real_alert_count(client):
    """Dashboard total_alerts must be the actual tiered alert set (what Cases /
    Incidents show), not the pre-fuse rule count from the detection summary."""
    import main
    kpis = client.get("/api/dashboard").json()["kpis"]
    assert kpis["total_alerts"] == len(main.state["alerts"])
    assert kpis["total_alerts"] == client.get("/api/alerts").json()["total"]
    assert kpis["critical_alerts"] == sum(
        1 for a in main.state["alerts"] if a.get("severity") == "CRITICAL"
    )


# ── Stream lifecycle (in-process, reversible) ────────────────────────────────

def test_stream_lifecycle(client):
    assert client.post("/api/stream/start", headers=ADMIN).status_code == 200
    assert client.get("/api/stream/status", headers=ADMIN).status_code == 200
    assert client.get("/api/stream/recent", headers=ADMIN).status_code == 200
    assert client.post("/api/stream/stop", headers=ADMIN).status_code == 200
    assert client.post("/api/stream/reset", headers=ADMIN).status_code == 200


# ── External-integration endpoints (real adapter -> mock fallback) ───────────

def test_kyc_screen(client):
    r = client.get("/api/kyc/screen", headers=ADMIN, params={"name": "John Doe"})
    assert r.status_code == 200, r.text
    assert "_real" in r.json()  # every adapter response carries its mode flag


def test_kyc_screen_requires_name(client):
    # `name` is a required query param -> FastAPI validation 422, never a 500
    assert client.get("/api/kyc/screen", headers=ADMIN).status_code == 422


def test_aa_consent_roundtrip(client):
    r = client.post("/api/aa/consent", headers={**ADMIN, **JSON}, json={"customer_id": "CUST-TEST"})
    assert r.status_code == 200 and "_real" in r.json(), r.text
    assert client.get("/api/aa/consents", headers=ADMIN).status_code == 200


def test_incident_rca_returns_dossier(client):
    inc = client.get("/api/incidents").json()["incidents"]
    if not inc:
        import pytest
        pytest.skip("no incidents in active dataset")
    iid = inc[0]["incident_id"]
    r = client.get(f"/api/incidents/{iid}/rca", headers=INV)
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"reconstruction", "diagnosis", "recommendations", "narrative"} <= set(body)
    assert body["diagnosis"]["basis"] in ("rule", "inferred", "generic")


def test_incident_rca_404_for_unknown(client):
    r = client.get("/api/incidents/NOPE-999/rca", headers=INV)
    assert r.status_code == 404, r.text


def test_tgn_endpoints_shape(client):
    # metrics endpoint always responds; tgn key present (may be None if untrained)
    r = client.get("/api/ml/metrics", headers=INV)
    assert r.status_code == 200, r.text
    assert "tgn" in r.json()
    # predictions endpoint returns a well-formed payload either way
    p = client.get("/api/tgn/predictions", headers=INV)
    assert p.status_code == 200, p.text
    body = p.json()
    assert "trained" in body and "predictions" in body
    assert "variant" in body
# ── Collusion rings (synthetic identity lane, variant-independent) ────────────

def test_collusion_rings_endpoint(client):
    r = client.get("/api/collusion/rings", headers=INV)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == "synthetic_identity"
    assert body["total"] >= 3  # planted: forged-KYC, device-farm, both-share rings
    ring = body["rings"][0]
    assert {"ring_id", "account_ids", "size", "shared_identifiers"} <= set(ring)
    assert ring["size"] >= 3
