# Forensic RCA Dossier (MVP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-incident **Forensic RCA dossier** — forensic reconstruction, root-cause control-gap diagnosis, and prescriptive recommendations — served at `GET /api/incidents/{id}/rca` and rendered as a panel on the Incidents page.

**Architecture:** A new pure-function engine module `src/rca_engine.py` orchestrates `reconstruct → diagnose_root_cause → recommend` over an incident and its primary alert. Reconstruction wraps the existing `fund_tracer.trace_for_alert`. Diagnosis extends `fatf_typology.TYPOLOGY` with a control-gap taxonomy and — because ~97% of incidents are ML-only (`primary_pattern == "ML-Detected Anomaly"`, not a TYPOLOGY key) — infers the control gap from the reconstruction's behavioural signals when no rule typology matches. The backend endpoint wires `state` objects into `build_rca`; the frontend renders the returned dossier.

**Tech Stack:** Python 3 / FastAPI / NetworkX / pandas (backend); React 19 + Vite + Tailwind 4 (frontend); pytest + FastAPI `TestClient` (tests).

## Global Constraints

- Detector thresholds are never hardcoded — read from config with a default; `rca_engine` uses `structuring_threshold` default `200000` (mirrors `fund_tracer.STRUCTURING_THRESHOLD`).
- This is a post-detection layer: it adds **no** alerts and suppresses none. Detection recall/precision is untouched.
- **No new ML model** and no fabricated "forecast". MVP is forensic + root-cause + recommendation only; foresight is a separate later plan.
- `rca_engine` functions are pure (no file/network I/O) so they unit-test without booting the app.
- Read endpoints in this app carry **no** `require()` gate (see `get_incident`) — keep `GET .../rca` parity (read-open).
- Commit after each task. Co-author trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

- `src/fatf_typology.py` — **modify**: add `control_gap` + `remediation` to each `TYPOLOGY` entry and `_GENERIC`.
- `src/rca_engine.py` — **create**: `reconstruct`, `diagnose_root_cause`, `recommend`, `build_rca`, `rca_narrative` + private helpers.
- `backend/main.py` — **modify**: import `build_rca`; add `GET /api/incidents/{incident_id}/rca`.
- `frontend/src/api.js` — **modify**: add `getIncidentRca(id)`.
- `frontend/src/components/RcaReport.jsx` — **create**: renders a dossier.
- `frontend/src/pages/Incidents.jsx` — **modify**: wire in the RCA panel.
- `tests/test_rca_engine.py` — **create**: unit tests for the engine + taxonomy.
- `tests/test_api_integration.py` — **modify**: endpoint test.

---

### Task 1: Control-gap taxonomy on TYPOLOGY

**Files:**
- Modify: `src/fatf_typology.py` (the `TYPOLOGY` dict + `_GENERIC`)
- Test: `tests/test_rca_engine.py`

**Interfaces:**
- Produces: every `TYPOLOGY[pattern]` and `_GENERIC` gains string keys `control_gap` and `remediation`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rca_engine.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rca_engine.py::test_every_typology_has_control_gap_and_remediation -v`
Expected: FAIL with `KeyError`/`AssertionError` (`control_gap` missing).

- [ ] **Step 3: Add the two keys to each entry**

Add `"control_gap"` and `"remediation"` to each of the seven `TYPOLOGY` entries and to `_GENERIC`, using exactly these values:

```python
# "Circular Transaction"
"control_gap": "Round-tripping not detected because velocity/closure is evaluated per-edge, not over the closed loop.",
"remediation": "Add closed-loop (cycle) detection to real-time monitoring; flag funds returning to origin within N hops.",
# "Rapid Layering"
"control_gap": "Cross-branch / cross-bank velocity is not correlated in real time, so rapid multi-hop layering clears before aggregation.",
"remediation": "Enable real-time per-entity velocity rollup across branches; alert on more than N hops within a short window.",
# "Smurfing / Structuring"
"control_gap": "The reporting (CTR) threshold is evaluated per transaction, not aggregated across a rolling window and across branches.",
"remediation": "Aggregate deposits per beneficiary over a rolling window and across branches; lower the effective structuring threshold.",
# "Shell Company Funnel"
"control_gap": "Weak account-opening due diligence; beneficial ownership unverified for high-velocity new entities.",
"remediation": "Add a beneficial-ownership verification gate and a fan-in flag for accounts receiving from many unrelated payers within N days of opening.",
# "Dormant Activation"
"control_gap": "No re-KYC / step-up authentication trigger when a dormant account reactivates.",
"remediation": "Auto-trigger Enhanced Due Diligence when an account dormant more than N days reactivates with volume above X.",
# "Profile Mismatch"
"control_gap": "The KYC profile is static and is not re-scored against observed behaviour.",
"remediation": "Schedule periodic behavioural KYC refresh; re-score risk when transaction behaviour deviates from the declared profile.",
# "Recruiter / Coordinator"
"control_gap": "Mule-network linkage is not surfaced at onboarding or at transaction time.",
"remediation": "Run a graph-proximity check to known mules/coordinators at onboarding and on the first high-value transfer.",
# _GENERIC
"control_gap": "An anomalous flow inconsistent with the entity profile was not caught by existing rule thresholds.",
"remediation": "Route to behavioural re-scoring and analyst review; tune detector thresholds against this incident's signature.",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rca_engine.py::test_every_typology_has_control_gap_and_remediation -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fatf_typology.py tests/test_rca_engine.py
git commit -m "feat(rca): control-gap taxonomy on FATF typology"
```

---

### Task 2: `rca_engine.reconstruct`

**Files:**
- Create: `src/rca_engine.py`
- Test: `tests/test_rca_engine.py`

**Interfaces:**
- Consumes: `fund_tracer.trace_for_alert(graph, transactions, risk_scores, alert, edge_ml_scores=, config=, **caches) -> dict` with keys `nodes` (each `{id, flags, ...}`), `timeline` (each `{sender_id, receiver_id, amount, ...}`), `summary.red_flags` (list[str]). `TYPOLOGY` / `_GENERIC` from Task 1.
- Produces: `reconstruct(primary_alert, graph, transactions, risk_scores, *, edge_ml_scores=None, config=None, **tracer_caches) -> dict` with keys `method{pattern,fatf_code,fatf_typology}`, `origin` (list[str]), `cashout` (list[str]), `signals{in_scc,shell_count,dormant_count,subthreshold_deposits,max_fan_in,n_txns,total_amount}`, `red_flags`, `trace`. On a bad alert: `{"error": str}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_rca_engine.py
import rca_engine


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rca_engine.py::test_reconstruct_shape -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rca_engine'`.

- [ ] **Step 3: Create `src/rca_engine.py` with `reconstruct` + helpers**

```python
"""rca_engine.py — assemble a Forensic RCA dossier for a clustered incident.

build_rca() is the public entrypoint; it orchestrates
reconstruct -> diagnose_root_cause -> recommend. All functions are pure
(no I/O) so they unit-test without booting the app.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fund_tracer import trace_for_alert
from fatf_typology import TYPOLOGY, _GENERIC

STRUCTURING_THRESHOLD = 200000  # ₹2L — mirrors fund_tracer.STRUCTURING_THRESHOLD


def _structuring_threshold(config: Optional[Dict]) -> float:
    if config:
        return float(config.get("structuring_threshold", STRUCTURING_THRESHOLD))
    return STRUCTURING_THRESHOLD


def reconstruct(primary_alert, graph, transactions, risk_scores,
                *, edge_ml_scores=None, config=None, **tracer_caches) -> Dict:
    trace = trace_for_alert(
        graph, transactions, risk_scores, alert=primary_alert,
        edge_ml_scores=edge_ml_scores, config=config, **tracer_caches,
    )
    if "error" in trace:
        return {"error": trace["error"]}

    nodes = trace.get("nodes", [])
    timeline = trace.get("timeline", [])
    red_flags = trace.get("summary", {}).get("red_flags", [])
    thr = _structuring_threshold(config)

    inflow: Dict[str, float] = {}
    outflow: Dict[str, float] = {}
    fan_in: Dict[str, set] = {}
    for t in timeline:
        s, r = t.get("sender_id"), t.get("receiver_id")
        amt = float(t.get("amount", 0) or 0)
        outflow[s] = outflow.get(s, 0.0) + amt
        inflow[r] = inflow.get(r, 0.0) + amt
        fan_in.setdefault(r, set()).add(s)

    def _net(e):
        return inflow.get(e, 0.0) - outflow.get(e, 0.0)

    ents = [n["id"] for n in nodes]
    origin = sorted([e for e in ents if _net(e) < 0], key=_net)[:3]
    cashout = sorted([e for e in ents if _net(e) > 0], key=_net, reverse=True)[:3]

    signals = {
        "in_scc": any(str(f).startswith("Cycle") for f in red_flags),
        "shell_count": sum(1 for n in nodes if "shell_company" in n.get("flags", [])),
        "dormant_count": sum(1 for n in nodes if "dormant_then_active" in n.get("flags", [])),
        "subthreshold_deposits": sum(
            1 for t in timeline
            if 0.5 * thr <= float(t.get("amount", 0) or 0) < thr
        ),
        "max_fan_in": max((len(v) for v in fan_in.values()), default=0),
        "n_txns": len(timeline),
        "total_amount": round(sum(float(t.get("amount", 0) or 0) for t in timeline), 2),
    }

    entry = TYPOLOGY.get(primary_alert.get("pattern_type", ""), _GENERIC)
    method = {
        "pattern": primary_alert.get("pattern_type"),
        "fatf_code": entry["fatf_code"],
        "fatf_typology": entry["fatf_typology"],
    }
    return {
        "method": method,
        "origin": origin,
        "cashout": cashout,
        "signals": signals,
        "red_flags": red_flags,
        "trace": trace,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rca_engine.py::test_reconstruct_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rca_engine.py tests/test_rca_engine.py
git commit -m "feat(rca): forensic reconstruction wrapper over fund_tracer"
```

---

### Task 3: `rca_engine.diagnose_root_cause`

**Files:**
- Modify: `src/rca_engine.py`
- Test: `tests/test_rca_engine.py`

**Interfaces:**
- Consumes: a `reconstruction` dict from Task 2 (`signals` block); an `incident` dict with `primary_pattern` (str) and `patterns` (list[str]). `TYPOLOGY` / `_GENERIC`.
- Produces: `diagnose_root_cause(incident, reconstruction, config=None) -> dict` with keys `pattern_resolved` (str), `basis` (`"rule"|"inferred"|"generic"`), `control_gap` (str), `remediation` (str), `evidence` (str). Also `_infer_pattern(signals, thr_count=3) -> str` and `_evidence_for(pattern, signals, thr) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_rca_engine.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_rca_engine.py -k diagnose -v`
Expected: FAIL with `AttributeError: module 'rca_engine' has no attribute 'diagnose_root_cause'`.

- [ ] **Step 3: Implement diagnosis + inference + evidence**

```python
# append to src/rca_engine.py
def _infer_pattern(signals: Dict, thr_count: int = 3) -> str:
    if signals.get("in_scc"):
        return "Rapid Layering"
    if signals.get("shell_count", 0) > 0 and signals.get("max_fan_in", 0) >= 3:
        return "Shell Company Funnel"
    if signals.get("dormant_count", 0) > 0:
        return "Dormant Activation"
    if signals.get("subthreshold_deposits", 0) >= thr_count:
        return "Smurfing / Structuring"
    return ""


def _evidence_for(pattern: str, signals: Dict, thr: float) -> str:
    if pattern == "Smurfing / Structuring":
        return (f"{signals.get('subthreshold_deposits', 0)} transfers between "
                f"₹{0.5 * thr:,.0f} and ₹{thr:,.0f}, each under the ₹{thr:,.0f} "
                f"reporting threshold")
    if pattern == "Shell Company Funnel":
        return (f"{signals.get('shell_count', 0)} shell entity(ies) with fan-in up to "
                f"{signals.get('max_fan_in', 0)} counterparties")
    if pattern == "Dormant Activation":
        return f"{signals.get('dormant_count', 0)} dormant-then-active account(s) in the flow"
    if pattern in ("Rapid Layering", "Circular Transaction"):
        return (f"funds cycled through a closed loop spanning "
                f"{signals.get('n_txns', 0)} transactions")
    return (f"anomalous flow of ₹{signals.get('total_amount', 0):,.0f} across "
            f"{signals.get('n_txns', 0)} transactions inconsistent with entity profiles")


def diagnose_root_cause(incident, reconstruction, config=None) -> Dict:
    signals = reconstruction.get("signals", {})
    candidates = [incident.get("primary_pattern")] + list(incident.get("patterns", []))
    matched = next((p for p in candidates if p in TYPOLOGY), "")
    basis = "rule"
    if not matched:
        matched = _infer_pattern(signals)
        basis = "inferred" if matched else "generic"
    entry = TYPOLOGY.get(matched, _GENERIC)
    thr = _structuring_threshold(config)
    return {
        "pattern_resolved": matched or "Unclassified anomaly",
        "basis": basis,
        "control_gap": entry["control_gap"],
        "remediation": entry["remediation"],
        "evidence": _evidence_for(matched, signals, thr),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_rca_engine.py -k diagnose -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/rca_engine.py tests/test_rca_engine.py
git commit -m "feat(rca): root-cause diagnosis with ML-anomaly signal inference"
```

---

### Task 4: `rca_engine.recommend`

**Files:**
- Modify: `src/rca_engine.py`
- Test: `tests/test_rca_engine.py`

**Interfaces:**
- Consumes: a `diagnosis` dict from Task 3 (`control_gap`, `remediation`); an `incident` dict with `entities` (list[str]) and `entity_names` (list[str]).
- Produces: `recommend(diagnosis, incident, *, max_accounts=5) -> dict` with `account_level` (list of `{entity_id, name, action}`) and `policy_level` (list of `{recommendation, closes_gap}`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_rca_engine.py
def test_recommend_account_and_policy():
    diag = {"control_gap": "gap text", "remediation": "fix text"}
    incident = {"entities": ["E1", "E2"], "entity_names": ["Acme", "Bravo"]}
    recs = rca_engine.recommend(diag, incident)
    assert recs["account_level"][0] == {
        "entity_id": "E1", "name": "Acme",
        "action": "Enhanced Due Diligence (EDD) + transaction hold pending review",
    }
    assert recs["policy_level"] == [{"recommendation": "fix text", "closes_gap": "gap text"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rca_engine.py::test_recommend_account_and_policy -v`
Expected: FAIL with `AttributeError: ... has no attribute 'recommend'`.

- [ ] **Step 3: Implement `recommend`**

```python
# append to src/rca_engine.py
def recommend(diagnosis, incident, *, max_accounts: int = 5) -> Dict:
    ents = list(incident.get("entities", []))[:max_accounts]
    names = list(incident.get("entity_names", []))
    account_level = [
        {
            "entity_id": e,
            "name": names[i] if i < len(names) else e,
            "action": "Enhanced Due Diligence (EDD) + transaction hold pending review",
        }
        for i, e in enumerate(ents)
    ]
    policy_level = [{
        "recommendation": diagnosis["remediation"],
        "closes_gap": diagnosis["control_gap"],
    }]
    return {"account_level": account_level, "policy_level": policy_level}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rca_engine.py::test_recommend_account_and_policy -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rca_engine.py tests/test_rca_engine.py
git commit -m "feat(rca): prescriptive recommendations (account + policy tiers)"
```

---

### Task 5: `rca_engine.build_rca` + `rca_narrative`

**Files:**
- Modify: `src/rca_engine.py`
- Test: `tests/test_rca_engine.py`

**Interfaces:**
- Consumes: `reconstruct`, `diagnose_root_cause`, `recommend` (Tasks 2–4).
- Produces: `build_rca(incident, primary_alert, graph, transactions, risk_scores, *, edge_ml_scores=None, config=None, **tracer_caches) -> dict` with keys `incident_id`, `reconstruction`, `diagnosis`, `recommendations`, `narrative` (str). On bad alert: `{"incident_id", "error"}`. Also `rca_narrative(dossier) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_rca_engine.py
def test_build_rca_full_dossier(synthetic_pipeline):
    alerts = synthetic_pipeline["alerts"]
    alert = next(a for a in alerts if a.get("entities"))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rca_engine.py::test_build_rca_full_dossier -v`
Expected: FAIL with `AttributeError: ... has no attribute 'build_rca'`.

- [ ] **Step 3: Implement `build_rca` + `rca_narrative`**

```python
# append to src/rca_engine.py
def rca_narrative(dossier: Dict) -> str:
    d = dossier["diagnosis"]
    r = dossier["reconstruction"]
    return (
        f"Method: {r['method']['fatf_typology']} ({r['method']['fatf_code']}). "
        f"Root cause: {d['control_gap']} Evidenced by {d['evidence']}. "
        f"Recommended fix: {d['remediation']}"
    )


def build_rca(incident, primary_alert, graph, transactions, risk_scores,
              *, edge_ml_scores=None, config=None, **tracer_caches) -> Dict:
    recon = reconstruct(primary_alert, graph, transactions, risk_scores,
                        edge_ml_scores=edge_ml_scores, config=config, **tracer_caches)
    if "error" in recon:
        return {"incident_id": incident.get("incident_id"), "error": recon["error"]}
    diag = diagnose_root_cause(incident, recon, config=config)
    recs = recommend(diag, incident)
    dossier = {
        "incident_id": incident.get("incident_id"),
        "reconstruction": recon,
        "diagnosis": diag,
        "recommendations": recs,
    }
    dossier["narrative"] = rca_narrative(dossier)
    return dossier
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rca_engine.py::test_build_rca_full_dossier -v`
Expected: PASS.

- [ ] **Step 5: Run the whole engine module + commit**

Run: `python -m pytest tests/test_rca_engine.py -v`
Expected: all PASS.

```bash
git add src/rca_engine.py tests/test_rca_engine.py
git commit -m "feat(rca): build_rca orchestrator + deterministic narrative"
```

---

### Task 6: Backend endpoint `GET /api/incidents/{id}/rca`

**Files:**
- Modify: `backend/main.py` (import near the other `fund_tracer` import at line 48; add the route after `get_incident` at line 774)
- Test: `tests/test_api_integration.py`

**Interfaces:**
- Consumes: `build_rca` (Task 5); `state["incidents"]`, `state["graph"]`, `state["transactions"]`, `state["risk_scores"]`, `state["edge_scores"]`, `_alerts_with_case_status()`, `_tracer_caches()` (all existing in `main.py`).
- Produces: `GET /api/incidents/{incident_id}/rca` → the dossier dict (200); 404 if no such incident; 422 if the incident has no resolvable alert.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_api_integration.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_integration.py -k incident_rca -v`
Expected: FAIL with 404 on the valid id (route not defined → FastAPI 404, but the unknown-id test may pass coincidentally; the dossier test fails on missing keys / 404).

- [ ] **Step 3: Add the import**

At `backend/main.py:48`, alongside `from fund_tracer import trace_journey, trace_for_alert`, add:

```python
from rca_engine import build_rca
```

- [ ] **Step 4: Add the route after `get_incident` (after line 774)**

```python
@app.get("/api/incidents/{incident_id}/rca")
def get_incident_rca(incident_id: str):
    inc = next((i for i in (state["incidents"] or []) if i.get("incident_id") == incident_id), None)
    if not inc:
        raise HTTPException(404, "Incident not found")
    alerts = _alerts_with_case_status()
    pid = inc.get("primary_alert_id")
    primary = next((a for a in alerts if a.get("alert_id") == pid), None)
    if primary is None:
        ids = set(inc.get("alert_ids", []))
        primary = next((a for a in alerts if a.get("alert_id") in ids), None)
    if primary is None:
        raise HTTPException(422, "Incident has no resolvable alert")
    return build_rca(
        inc, primary, state["graph"], state["transactions"], state["risk_scores"],
        edge_ml_scores=state["edge_scores"], **_tracer_caches(),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_integration.py -k incident_rca -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py tests/test_api_integration.py
git commit -m "feat(rca): GET /api/incidents/{id}/rca endpoint"
```

---

### Task 7: Frontend RCA panel

**Files:**
- Modify: `frontend/src/api.js` (add fetch helper next to the other incident calls)
- Create: `frontend/src/components/RcaReport.jsx`
- Modify: `frontend/src/pages/Incidents.jsx` (wire in the panel — read the file first to match its incident-selection state)
- Verify: browser (this app has no component test harness; verification is manual per project convention)

**Interfaces:**
- Consumes: `GET /api/incidents/{id}/rca` (Task 6) returning `{reconstruction, diagnosis, recommendations, narrative}`.
- Produces: `getIncidentRca(id)` in `api.js`; `<RcaReport dossier={...} />` component.

- [ ] **Step 1: Add the API helper**

In `frontend/src/api.js`, add (use the same `apiFetch`/role-header wrapper the other calls use — copy an adjacent incident call's style):

```js
export const getIncidentRca = (id) => apiFetch(`/api/incidents/${id}/rca`);
```

- [ ] **Step 2: Create `frontend/src/components/RcaReport.jsx`**

```jsx
export default function RcaReport({ dossier }) {
  if (!dossier) return null;
  if (dossier.error) return <div className="text-red-500 text-sm">{dossier.error}</div>;
  const { reconstruction: r, diagnosis: d, recommendations: rec, narrative } = dossier;
  return (
    <div className="space-y-4 text-sm">
      <p className="italic text-gray-300">{narrative}</p>

      <section>
        <h4 className="font-semibold text-gray-100">1. How it happened (forensic)</h4>
        <p>Method: {r.method.fatf_typology} <span className="text-gray-400">({r.method.fatf_code})</span></p>
        <p>Origin: {r.origin.join(", ") || "—"} → Cash-out: {r.cashout.join(", ") || "—"}</p>
        <p className="text-gray-400">{r.signals.n_txns} transactions, ₹{r.signals.total_amount.toLocaleString()}</p>
      </section>

      <section>
        <h4 className="font-semibold text-gray-100">2. Why it succeeded (root cause)</h4>
        <p>{d.control_gap}</p>
        <p className="text-gray-400">Basis: {d.basis} · Evidence: {d.evidence}</p>
      </section>

      <section>
        <h4 className="font-semibold text-gray-100">3. What to fix (recommendations)</h4>
        <ul className="list-disc ml-5">
          {rec.policy_level.map((p, i) => <li key={i}>{p.recommendation}</li>)}
        </ul>
        <p className="text-gray-400 mt-1">Immediate: EDD + hold on {rec.account_level.map(a => a.name).join(", ")}</p>
      </section>
    </div>
  );
}
```

- [ ] **Step 3: Wire into `Incidents.jsx`**

Read `frontend/src/pages/Incidents.jsx` to find how an incident is selected/expanded. When an incident is selected, fetch its RCA and render the panel:

```jsx
import { getIncidentRca } from "../api";
import RcaReport from "../components/RcaReport";
// ...inside the component:
const [rca, setRca] = useState(null);
// when an incident is opened (match the file's existing handler):
getIncidentRca(incident.incident_id).then(setRca).catch(() => setRca(null));
// in the incident detail JSX:
<RcaReport dossier={rca} />
```

- [ ] **Step 4: Lint + build**

Run: `cd frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 5: Manual verification (project convention: verify in the running app)**

Start the stack (`docker compose up` or the local dev commands), open the Incidents page, select an incident, and confirm the RCA panel shows all three sections with a non-empty narrative. Try one rule-typed incident (e.g. a Shell Company Funnel) and one `ML-Detected Anomaly` incident — the latter must still produce a non-generic root cause when its signals warrant (basis `"inferred"`).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.js frontend/src/components/RcaReport.jsx frontend/src/pages/Incidents.jsx
git commit -m "feat(rca): incident RCA panel (frontend)"
```

---

### Task 8: Full-suite regression

- [ ] **Step 1: Run the whole backend suite**

Run: `python -m pytest tests/ -v`
Expected: all pass (existing 136 + the new RCA tests).

- [ ] **Step 2: Frontend lint + build (final)**

Run: `cd frontend && npm run lint && npm run build`
Expected: clean.

---

## Self-Review

**Spec coverage:**
- Forensic reconstruction (§ dossier 1) → Task 2. ✓
- Root-cause diagnosis incl. ML-anomaly inference (§ dossier 2 + the "critical" note) → Tasks 1, 3. ✓
- Prescriptive recommendations (§ dossier 4) → Task 4. ✓
- Endpoint (§4.7) → Task 6. ✓
- Frontend (§4.8) → Task 7. ✓
- **Deferred to later plans (documented, not silently dropped):** Foresight / next-target ranking + exposure (§ dossier 3, §4.4) and the SimulationStudio "prove-the-fix" loop (§4.6). MVP recommendations' account-level tier therefore targets the incident's **own** entities, not predicted-next entities; Phase 3 will extend it.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code step is literal. ✓

**Type consistency:** `signals` keys are produced in Task 2 and consumed verbatim in Task 3 (`in_scc`, `shell_count`, `dormant_count`, `subthreshold_deposits`, `max_fan_in`, `n_txns`, `total_amount`). `build_rca` return keys (`reconstruction`, `diagnosis`, `recommendations`, `narrative`) match the endpoint test (Task 6) and the frontend component (Task 7). `diagnose_root_cause.basis` values (`rule`/`inferred`/`generic`) match the endpoint assertion. ✓
