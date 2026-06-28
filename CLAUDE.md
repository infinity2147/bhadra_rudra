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
python -m pytest tests/ -v            # run all 136 tests
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
| `fraud_detector.py` | 4 core heuristic detectors: cycle (Johnson's algorithm), layering (temporal-causal money-following walk — amount-ranked search, per-hop causality+rapidity window, amount-preservation gate, ≥3 hops, bottleneck flow), smurfing (3 modes: edge-cluster + temporal burst + window-independent fan-out cluster), funnel (flow-imbalance OR pass-through; evaluates ALL account types, amount-weighted FIFO holding time) |
| `advanced_detectors.py` | 2 more detectors: dormant activation (peak-daily Z-score over post-gap window, scale-relative std floor), profile mismatch (KYC behavioural rules + shell volume/large-transfer rules, config-driven night window) |
| `ml_model.py` | XGBoost edge classifier — 30 features, stratified 80/20 split, persisted as pickle with SHAP background sample. Decision threshold chosen by `fbeta_optimal_threshold` at **F2** (recall-favouring), not F1 — shared by XGB, GNN, and ensemble so they agree on "fraud" |
| `gnn_model.py` | GraphSAGE (PyTorch Geometric) — 3 SAGEConv layers (3-hop receptive field) + max aggregation (resists neighbour-dilution camouflage). The edge head **fuses the per-edge XGB-style features** alongside the two endpoint embeddings (`train_gnn(transactions=...)`), model selection on **val AUPRC**, seeded for reproducibility — together these lifted AUPRC ~0.13→0.62. Skipped if PyG not installed |
| `ensemble_model.py` | **Stacked ensemble**: XGBoost + GraphSAGE + GAT base models (GNNs also use edge-feature fusion + AUPRC selection), LR meta-learner trained on 3-fold OOF predictions. Persisted under `data/ml/{variant}/ensemble/` |
| `ml_alert_generator.py` | **ML-led detection**: `generate_ml_alerts` emits a first-class alert per edge ≥ F2 threshold; `apply_tiers` combines with rule alerts as a confidence tier (T1 = ML+rule agree, T2 = ML only, T3 = clean rule typology; noisy rule-only suppressed); `fuse_ml_alerts` is the pipeline step. SHAP reason stays lazy (served by `/api/alerts/{id}/explain`) |
| `train_ibm_aml.py` | Real-data trainer: stratified 100k IBM AML sample → XGB + SAGE + ensemble in `data/ml/ibm_aml/` |
| `shap_explainer.py` | `TreeExplainer` for per-alert SHAP attributions |
| `incident_clustering.py` | Union-find to collapse raw alerts into clustered incidents; high-degree utility/hub entities (gateways, exchanges) are excluded as linking bridges so independent rings don't snowball into one super-incident |
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
| `data/<variant>/fraud_alerts.json` | Tiered alert set after the ML/rule fuse (ML-Detected Anomaly + corroborated/kept rule alerts). On IBM AML at β=2 this is ~9k alerts; each carries `tier`, `source`, `corroborated_by` |
| `data/<variant>/incidents.json` | Clustered incidents over the tiered alert set (~5k on IBM AML at β=2) |
| `data/ml/synthetic/` | XGBoost `model.pkl`, `metrics.json`, `edge_scores.json` |
| `data/ml/synthetic/gnn/` | GraphSAGE weights + metrics |
| `data/<variant>/sar_reports/*.pdf` | SAR PDFs — written on-request by the backend (SAR view / FIU package), not pre-generated by the pipeline |
| `data/rudra.db` | SQLite — cases, audit log, config thresholds |

Run `python src/run_pipeline.py` to generate all of these.

## Important patterns and constraints

**XGBoost feature set (`ml_model.py:FEATURE_COLUMNS`)** is a fixed-order list of 30 columns. The order matters at inference time — any new feature must be appended, and inference code must use the same column order as training.

**`neighbor_fraud_density` is intentionally excluded** from the feature set. It was removed in v3 because it creates circular reasoning on fraud clusters. Do not re-add it.

**Transit-ratio / velocity-burst features were trialled and reverted (v4).** They gave no AUPRC gain on XGBoost (0.661→0.662, a clean test since trees are scale-invariant) — too sparse on real data. Don't re-add them without evidence of a held-out AUPRC lift. See the note left in `ml_model.py:FEATURE_COLUMNS`.

**Decision threshold is F2, not F1.** All three trainers select the operating threshold via `ml_model.fbeta_optimal_threshold(..., beta=2.0)` — recall-favouring, because a missed launderer costs more than an analyst review. This is the dominant recall lever and the reason the reported F1 is lower than the F1-optimal point by design. Keep XGB / GNN / ensemble on the same beta so "fraud" means the same thing across models.

**Detection is ML-led, rules corroborate (tiered).** `ml_alert_generator` makes the ML model a first-class alert generator; the rule engine is the corroboration + explanation lane. Alerts combine as a confidence tier (T1 ML+rule / T2 ML-only / T3 clean rule typology), never an averaged score. The pipeline's fuse step re-writes `fraud_alerts.json` + re-clusters `incidents.json` after ML training. ML alerts carry `tier`, `corroborated_by`, `source: "ml"`, and their own ensemble `ml_score` (backend `_alerts_with_case_status` must not overwrite it).

**Hash-chain audit log**: each `audit_log` row stores `prev_hash` and `this_hash = SHA-256(prev_hash || canonical_entry_json)`. Any edit/insert/delete to the SQLite table breaks the chain, detected by `GET /api/cases/{id}/verify`. The test `tests/test_case_store.py::test_hash_chain_detects_tampering` exercises this.

**Detector thresholds** are all read from `ConfigStore` (SQLite-backed). Never hardcode a threshold value in a detector — use `self._cfg("key", default)`.

**GNN is optional** — `gnn_model.py` and `ensemble_model.py` require `torch` and `torch_geometric`. The pipeline catches `ImportError` and skips gracefully. These are listed as optional in `requirements.txt`. The backend serves GNN/ensemble scores from the persisted `edge_scores.json`, never by re-loading the `.pt` weights at runtime — so changing the SAGE architecture (depth/aggregation/edge-fusion) is runtime-safe but requires a `python src/run_pipeline.py` retrain to regenerate scores. Edge-feature fusion needs `transactions` passed to `train_gnn`/`train_ensemble` (pipeline callers already do this); without it the GNN falls back to node-embeddings-only. **Train ordering matters**: XGBoost must be fit before `torch` is imported in a process (the two segfault if torch loads first); the pipeline already does XGB→GNN, and the GNN test runs in an isolated subprocess for the same reason.

**Real datasets** go in `data/real/{ibm_aml,paysim,ieee_cis}/` (hundreds of MB, not committed). **`python src/run_pipeline.py` is the one command you need** — it trains XGB + SAGE + ensemble *and* generates the tiered alerts + incidents (defaults to the `ibm_aml` variant). `src/train_ibm_aml.py` is a standalone *model-only* trainer (no alerts/tiering) — a subset of what `run_pipeline.py` already does; you rarely need it directly.

**Adapter pattern for AA + DiliSense** — never call `aa_kyc_mock` directly from new code. Always go through `state["aa_client"]` / `state["dilisense_client"]` so real creds (when present) flip the behaviour with zero code change. Both clients carry `_real: true|false` in every response and gracefully fall back to mock on transient HTTP failure (with `_fallback_reason` populated).

**Streaming has two backends** — the `StreamIngestor` probes Kafka at startup and falls back to an in-process `asyncio.Queue` if no broker is reachable. `STREAM_BACKEND=kafka` forces Kafka and fails loud if unreachable; `STREAM_BACKEND=inproc` skips the probe. Both backends call the same `score_live_txn` — no separate "fast path" exists, so what the batch endpoint scores is what the stream scores.

## Test fixtures

`tests/conftest.py` provides two session-scoped fixtures:
- `synthetic_pipeline` — runs the full data generation + graph build + detection once and shares the result across all tests (session scope = fast)
- `temp_data_dir` — creates a temporary directory and cleans it up after each test that writes to disk
