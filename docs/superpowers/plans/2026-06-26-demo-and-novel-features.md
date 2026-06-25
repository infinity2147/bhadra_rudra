# RUDRA — Demo-Impact + Novel-Capability Implementation Plan

> **For agentic workers:** execute task-by-task with TDD. Backend/algorithm tasks are pytest-TDD'd; frontend tasks verified via `npm run lint && npm run build`; docs are prose. Checkbox (`- [ ]`) tracking.

**Goal:** Implement the 10 improvements from the competitor analysis so RUDRA's real depth becomes visible, interactive, and regulator-legible — plus ship the *real* versions of features rivals only faked.

**Architecture:** Additive only. New `src/` modules (taint, recruiter, fatf, geo) + new FastAPI endpoints in `backend/main.py` + new React pages/components. No rewrites of working code. Everything credential/data-real — no theater.

**Tech Stack:** FastAPI + NetworkX + pandas + sklearn/xgboost (backend); React 19 + Vite + Tailwind 4 + Recharts + react-force-graph-2d (frontend). Persistence in the existing `data/rudra.db` SQLite.

## Global Constraints
- No new heavyweight deps unless trivial/pure-JS-CDN-free. Geo-map uses hand-rolled SVG projection (no map lib), matching the codebase's custom-viz pattern.
- Every detector threshold reads from `ConfigStore` (`self._cfg(key, default)`); add defaults to `src/config_store.py::DEFAULT_CONFIG`.
- Alerts keep their existing dict shape; new fields are additive.
- Frontend pages register in `frontend/src/App.jsx` `NAV_GROUPS` + routes; API calls go through `frontend/src/api.js` helpers (`fetchAPI`, `postAPI`).
- XGBoost `FEATURE_COLUMNS` stays 30 (don't break persisted `model.pkl`).
- Backend serves precomputed scores from JSON; ML arch changes need a pipeline retrain (already noted).

---

## Execution order (value-first, so truncation leaves the best done)

**Phase A — Backend algorithm/data (pytest-TDD, isolated):**
1. Persistent taint memory (#4) — `src/taint_store.py`
2. Recruiter/coordinator detector (#5) — `src/fraud_detector.py`
3. FATF typology + legal basis tagging (#6, #8) — `src/fatf_typology.py`
4. SHAP-in-STR-XML + FATF in STR (#6) — `src/fiu_package.py`
5. Geo aggregation (#10 backend) — `src/geo.py`

**Phase B — Backend endpoints (`backend/main.py`):**
6. `/api/health` (#9); `/api/simulate/score` + scenarios (#1); `/api/ml/ensemble/edge/{u}/{v}` (#2); `/api/geo/flows` (#10); `/api/taint*` (#4); wire FATF tags into `_alerts_with_case_status`.

**Phase C — Frontend:**
7. Simulation Studio page (#1)
8. Ensemble-consensus component, surfaced in Journey + Studio (#2)
9. Temporal replay component in Journey (#3)
10. Geo-map page (#10)
11. FATF + legal-basis + taint badges across Incidents/Journey/SAR/Entities

**Phase D — Infra + docs:**
12. docker-compose healthchecks + `/api/health` wiring + `scripts/quickstart.sh` (#9)
13. `docs/REGULATORY_MAPPING.md` (#7) + legal-jurisdiction framing (#8)

**Phase E — Verify:** full `pytest tests/ -q`; `cd frontend && npm run lint && npm run build`; integrated smoke; summary.

---

## Phase A details

### #4 Persistent taint memory — `src/taint_store.py` (+ test `tests/test_taint.py`)
**Concept (PRISM described it, never built it):** when an entity is confirmed fraudulent, decaying taint propagates to graph neighbors, PERSISTS in SQLite, and FLOORS future risk scores across runs.
- `TaintStore(db_path)`: table `taint(entity_id PK, taint REAL, source TEXT, hops INT, updated_at TEXT)`. Methods: `seed(graph, seed_entities, source, decay=0.7, max_hops=4)`, `get(entity_id)->float`, `get_all()->dict`, `apply_floor(risk_scores: dict)->dict` (returns `max(risk, taint)` per node), `clear()`.
- Propagation: BFS from seeds over the *undirected* view; `taint(hop) = decay**hop` (hop0 seed=1.0). Persist `max` taint per entity across calls (cross-time accumulation).
- **Tests:** propagation decays by hop; persistence survives a new `TaintStore` on same db; `apply_floor` raises a clean node's score to its taint; reseeding accumulates (max).

### #5 Recruiter/coordinator detector — `src/fraud_detector.py::detect_recruiters` (+ `tests/test_recruiter.py`)
**Concept:** name the *orchestrator* funding a fleet of mules, not the mules. Distinct from smurfing (structuring) — this is one source → many recipients that then pass funds through.
- For each node U: recipients = successors; count `mule_like` recipients (recipient has out-degree≥1 and pass-through ratio `min(in,out)/max(in,out) ≥ pt` and forwards within window — reuse `_avg_holding_time`). If `n_seeded_mules ≥ recruiter_min_fanout` (default 5) → emit alert `pattern_type="Recruiter / Coordinator"`, entities `[U]+mules`, naming U as coordinator, severity by fleet size + downstream volume.
- Config: `recruiter_min_fanout=5`, `recruiter_pass_through_ratio=0.6`, `recruiter_min_seed_amount=10_000`.
- Wire into `run_all_detections` + `run_pipeline.py` + summary counts.
- **Tests:** one source → 6 pass-through mules ⇒ U flagged as coordinator; a normal hub (recipients don't forward) ⇒ not flagged.

### #6/#8 FATF typology + legal basis — `src/fatf_typology.py` (+ `tests/test_fatf.py`)
- `TYPOLOGY` map: pattern_type → `{fatf_code, fatf_label, fiu_advisory, pmla_section, rbi_ref}`. Covers all 7 patterns + recruiter.
- `tag_alert(alert)->alert`: adds `fatf_code`, `fatf_typology`, `regulatory_refs` (list), and `legal_basis` derived from severity/confidence: CRITICAL/≥85 → "PMLA §12 STR (mandatory)"; HIGH → "RBI KYC MD §38 enhanced monitoring / pre-STR restriction"; else "Internal EDD".
- **Tests:** every pattern_type maps to a non-empty typology; legal_basis tiers by severity.

### #6 SHAP + FATF in STR XML — `src/fiu_package.py::_build_str_xml`
- Add `<FATFTypology code=.. >label</FATFTypology>` and `<LegalBasis>..</LegalBasis>` from the tagged alert.
- Add `<SHAPExplainability>` block: top-5 features for the alert's primary edge (compute via `shap_explainer.shap_explain_alert` when ml bundle available; skip gracefully if not). Each `<Feature name=.. value=.. contribution=../>`.
- **Tests:** STR XML for a tagged alert contains `<FATFTypology>` and (when bundle present) `<SHAPExplainability>`.

### #10 Geo aggregation — `src/geo.py` (+ `tests/test_geo.py`)
- `INDIA_CITIES`: dict of ~20 Indian metro/PSB-hub cities → (lat, lng). `branch_to_city(branch)`: deterministic hash-map of branch string → city (stable). `city_flows(transactions)`: aggregate sender_branch-city → receiver_branch-city flows with `{amount, txn_count, fraud_count}`; per-city `{inflow, outflow, fraud_volume, risk}`.
- **Tests:** flows aggregate correctly on a tiny df; every branch maps to a known city with coords.

## Phase B details — `backend/main.py`
- `GET /api/health`: probe graph loaded, ml bundle, db (cases.count), ingestor.status; return `{status:"ok"|"degraded", checks:{...}}`.
- `POST /api/simulate/score` body `{sender,receiver,amount,channel,rail,timestamp?}`: call `score_live_txn`; compute SHAP top-features on the live vector; look up ensemble per-model scores if edge exists; derive rule hints (near-threshold, night, high-value rail, cross-branch) from features; publish a `ScoredTxn` into `state["ingestor"]` ring buffer so the Live feed shows it. Return full breakdown.
- `GET /api/simulate/scenarios` + `POST /api/simulate/scenario/{name}`: predefined layering/smurfing/cycle templates injected into the stream.
- `GET /api/ml/ensemble/edge/{u}/{v}`: per-model scores for one edge (from ensemble edge_scores).
- `GET /api/geo/flows`: `geo.city_flows(state["transactions"])`.
- `GET /api/taint`, `POST /api/taint/seed/{alert_id}` (INVESTIGATOR+): seed taint from an alert's entities; auto-seed on `dispose` to ESCALATED/SAR_FILED. Apply `taint.apply_floor` when serving `/api/entities`, `/api/graph` risk, dashboard risk tiers.
- Decorate alerts with FATF tags in `_alerts_with_case_status`.

## Phase C details — frontend
- `pages/SimulationStudio.jsx` (`/simulate`, nav "Investigate"): txn-builder form + Score → ml gauge, SHAP bars (reuse Recharts), EnsembleConsensus, rule-hint chips, severity; scenario buttons; mini live-feed (poll `/api/stream/recent`).
- `components/EnsembleConsensus.jsx`: given `{xgb,sage,gat,ensemble}`, render 3 model gauges converging on ensemble + agreement indicator. Use in Journey (primary edge) + Studio.
- `components/TemporalReplay.jsx`: play/pause/seek scrubber over Journey `timeline`; progressively reveal nodes/edges by first-txn time on an SVG canvas. Embed in Journey.
- `pages/GeoMap.jsx` (`/map`, nav "Analyse"): SVG India bounding-box projection of `/api/geo/flows`; city markers sized by volume, colored by fraud rate; flow arcs; click city → detail.
- Badges: FATF code + legal_basis + taint shown on Incidents detail, Journey nodes, SAR page, Entities.

## Phase D details
- `docker-compose.yml`: add `healthcheck` to backend (curl `/api/health`) + `depends_on: condition: service_healthy`; frontend waits on backend healthy.
- `scripts/quickstart.sh`: one command — generate data if missing (`run_pipeline.py`), then `docker compose up` (or local uvicorn+vite). Idempotent.
- `docs/REGULATORY_MAPPING.md`: table mapping each detector → FATF typology → FIU-IND advisory → PMLA section → RBI circular, each row citing the implementing `src/` file:line. Opening section on the KYC-MD-§38-pre-crime → PMLA-§12-post-crime legal framing, and the meta-line that every claim is backed by running code.
