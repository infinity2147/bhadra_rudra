# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RUDRA is a real-time fund-flow intelligence system for Indian public-sector banks (PSBs Hackathon Series 2026, PS3). It replaces T+1 batch fraud detection with sub-second graph-based analysis, ML scoring, and a full FIU compliance workflow.

## Commands

### Run the full stack (recommended for evaluators)
```bash
docker compose up
# Backend: http://localhost:8000  |  Frontend: http://localhost:5173
```

### Local development

```bash
# 1. Generate data + train all models (must run before starting the backend)
python src/run_pipeline.py

# 2. Backend (port 8000) — from repo root
cd backend && uvicorn main:app --reload --port 8000

# 3. Frontend (port 5173) — from repo root
cd frontend && npm install && npm run dev
```

### Tests
```bash
python -m pytest tests/ -v            # run all 43 tests
python -m pytest tests/test_case_store.py -v  # run a single module
```

### Frontend lint + build
```bash
cd frontend && npm run lint
cd frontend && npm run build
```

### Environment variables
```bash
export ANTHROPIC_API_KEY="..."   # Enables Claude (Haiku) copilot; omit for local quick-commands fallback
```

### Regenerate data in running Docker container
```bash
docker compose exec backend python src/run_pipeline.py
```

## Architecture

Three layers, wired together at startup:

### 1. Frontend (`frontend/src/`)
React 19 + Vite + Tailwind 4. Fourteen pages under `src/pages/`. API calls live in `src/api.js` — every `fetch` sends an `X-User-Role` header for RBAC. The Vite dev server proxies `/api/*` → `http://localhost:8000`.

Key non-trivial components:
- `components/Sankey.jsx` — custom SVG Sankey (edge thickness ∝ log-amount), used in the Journey page for non-cyclic fund flows
- `pages/Graph.jsx` — `react-force-graph-2d` for the interactive network view
- `pages/Journey.jsx` — combines Sankey + force-graph depending on whether the path is cyclic

### 2. Backend (`backend/main.py`)
Single FastAPI ASGI app, ~50 endpoints. All state is loaded once at startup into a `state` dict (transactions DataFrame, alerts list, NetworkX graph, ML bundles, AA + DiliSense clients, Kafka stream ingestor, etc.). `src/` is inserted into `sys.path` so all engine modules are importable directly. Two startup hooks: a sync one for data + ML, an async one to bring up the Kafka consumer. A shutdown hook stops the consumer cleanly.

RBAC pattern: `Depends(get_role)` extracts `X-User-Role` header → `require(action, role)` raises HTTP 403 on violations. Three roles: `INVESTIGATOR`, `SUPERVISOR`, `ADMIN`.

### 3. Python engine (`src/`)

The pipeline is orchestrated by `src/run_pipeline.py` (synthetic) and `src/train_ibm_aml.py` (real IBM AML 100k → XGB + SAGE + ensemble).

| Module | Responsibility |
|---|---|
| `data_generator.py` | Synthetic transactions (2.7k txns, 80 entities, 6 fraud pattern types) |
| `graph_engine.py` | Build NetworkX `DiGraph` from the transaction DataFrame |
| `fraud_detector.py` | 4 core heuristic detectors: cycle (Johnson's algorithm), layering (BFS), smurfing (threshold clustering), funnel (flow imbalance) |
| `advanced_detectors.py` | 2 more detectors: dormant activation (Z-score on daily aggregates), profile mismatch (KYC behavioural rules) |
| `ml_model.py` | XGBoost edge classifier — 30 features, stratified 80/20 split, persisted as pickle with SHAP background sample |
| `gnn_model.py` | GraphSAGE (PyTorch Geometric) — two SAGEConv layers + edge-classification MLP head; skipped if PyG not installed |
| `ensemble_model.py` | **Stacked ensemble**: XGBoost + GraphSAGE + GAT base models, LR meta-learner trained on 3-fold OOF predictions. Persisted under `data/ml/{variant}/ensemble/` |
| `train_ibm_aml.py` | Real-data trainer: stratified 100k IBM AML sample → XGB + SAGE + ensemble in `data/ml/ibm_aml/` |
| `shap_explainer.py` | `TreeExplainer` for per-alert SHAP attributions |
| `incident_clustering.py` | Union-find to collapse raw alerts into clustered incidents |
| `case_manager.py` | SQLite (`data/rudra.db`) case workflow + SHA-256 hash-chain audit log |
| `config_store.py` | Detector thresholds persisted in SQLite; every detector reads from here |
| `fund_tracer.py` | BFS forward/backward fund journey with red-flag annotation |
| `fiu_package.py` | Zip download: STR XML + SAR PDF + subgraph.json + transaction_chain.csv + case_audit_log.json |
| `sar_generator.py` | SAR text generation + PDF export via ReportLab |
| `live_scoring.py` | Per-transaction ML scoring with latency benchmark |
| `llm_copilot.py` | Claude (Haiku) tool-calling + local intent-routing fallback; 4 tool functions |
| `aa_kyc_mock.py` | Mock-fallback implementation for AA + DiliSense (called from the adapter when real creds are absent) |
| `integrations/aa_client.py` | **Real Sahamati AA adapter** — HTTPS calls when `SAHAMATI_CLIENT_ID/SECRET/FIU_ID` set, falls back to `aa_kyc_mock` |
| `integrations/dilisense_client.py` | **Real DiliSense adapter** — calls `https://api.dilisense.com/v1/checkIndividual` when `DILISENSE_API_KEY` set |
| `streaming/ingestor.py` | **Real Kafka stream ingestor** (aiokafka) + in-process fallback. Owns the consumer task + ring buffer of scored events |
| `streaming/kafka_producer.py` | CLI + library to replay any transactions CSV onto the Kafka topic |
| `streaming/pathway_engine.py` | Optional Pathway windowed-analytics layer (5-min sliding window per-entity velocity alerts). Skipped if `pathway` not installed |
| `real_data_loader.py` | Loaders for IBM AML, PaySim, IEEE-CIS (if CSVs are present under `data/real/`) |
| `rbac.py` | Role → permitted actions matrix |

## Key data files (generated, not committed)

| Path | Contents |
|---|---|
| `data/transactions.csv` | Synthetic transaction DataFrame |
| `data/fraud_alerts.json` | Raw detector output (~290 alerts) |
| `data/incidents.json` | Clustered incidents (~6) |
| `data/ml/synthetic/` | XGBoost `model.pkl`, `metrics.json`, `edge_scores.json` |
| `data/ml/synthetic/gnn/` | GraphSAGE weights + metrics |
| `data/<variant>/sar_reports/*.pdf` | SAR PDFs — written on-request by the backend (SAR view / FIU package), not pre-generated by the pipeline |
| `data/rudra.db` | SQLite — cases, audit log, config thresholds |

Run `python src/run_pipeline.py` to generate all of these.

## Important patterns and constraints

**XGBoost feature set (`ml_model.py:FEATURE_COLUMNS`)** is a fixed-order list of 30 columns. The order matters at inference time — any new feature must be appended, and inference code must use the same column order as training.

**`neighbor_fraud_density` is intentionally excluded** from the feature set. It was removed in v3 because it creates circular reasoning on fraud clusters. Do not re-add it.

**Hash-chain audit log**: each `audit_log` row stores `prev_hash` and `this_hash = SHA-256(prev_hash || canonical_entry_json)`. Any edit/insert/delete to the SQLite table breaks the chain, detected by `GET /api/cases/{id}/verify`. The test `tests/test_case_store.py::test_hash_chain_detects_tampering` exercises this.

**Detector thresholds** are all read from `ConfigStore` (SQLite-backed). Never hardcode a threshold value in a detector — use `self._cfg("key", default)`.

**GNN is optional** — `gnn_model.py` and `ensemble_model.py` require `torch` and `torch_geometric`. The pipeline catches `ImportError` and skips gracefully. These are listed as optional in `requirements.txt`.

**Real datasets** go in `data/real/{ibm_aml,paysim,ieee_cis}/` (hundreds of MB, not committed). `python src/train_ibm_aml.py` runs the full IBM AML 100k pipeline (XGB + SAGE + ensemble). The synthetic pipeline (`src/run_pipeline.py`) auto-trains extra variants when real datasets are present.

**Adapter pattern for AA + DiliSense** — never call `aa_kyc_mock` directly from new code. Always go through `state["aa_client"]` / `state["dilisense_client"]` so real creds (when present) flip the behaviour with zero code change. Both clients carry `_real: true|false` in every response and gracefully fall back to mock on transient HTTP failure (with `_fallback_reason` populated).

**Streaming has two backends** — the `StreamIngestor` probes Kafka at startup and falls back to an in-process `asyncio.Queue` if no broker is reachable. `STREAM_BACKEND=kafka` forces Kafka and fails loud if unreachable; `STREAM_BACKEND=inproc` skips the probe. Both backends call the same `score_live_txn` — no separate "fast path" exists, so what the batch endpoint scores is what the stream scores.

## Test fixtures

`tests/conftest.py` provides two session-scoped fixtures:
- `synthetic_pipeline` — runs the full data generation + graph build + detection once and shares the result across all tests (session scope = fast)
- `temp_data_dir` — creates a temporary directory and cleans it up after each test that writes to disk
