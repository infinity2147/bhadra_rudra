# RUDRA — Shield Against Deception

> Real-time fund-flow intelligence for Indian public-sector banks.
> Built for **PSBs Hackathon Series 2026** — Problem Statement 3 (Fund Flow Tracking).
> By **Team Bhadra**.

[![tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)](#tests) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#setup) [![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

## TL;DR

Indian PSBs detect fraud the day after it happens. By then, layered funds have already left the bank. RUDRA replaces T+1 batch detection with:

- **Live fund-flow graph** (NetworkX) updated per transaction
- **6 specialised detectors** + **stacked ensemble** (XGBoost + GraphSAGE + GAT + LR meta-learner) trained on the **IBM AML 100k** public benchmark
- **Real Kafka streaming** (aiokafka producer + consumer, KRaft single-broker in docker-compose) + optional **Pathway** windowed-analytics layer
- **Real Sahamati Account Aggregator + DiliSense KYC** adapters — drop in env-driven credentials and they hit the real APIs; transparent mock fallback otherwise
- **SHAP explanations** for every ML decision
- **Case workflow** with **tamper-evident audit log** (SHA-256 hash chain)
- **One-click FIU evidence package** (STR XML + SAR PDF + subgraph + transaction chain + audit log)

**Detection latency: ~1 s end-to-end · sub-ms per-transaction ML scoring.**

> ⚠️ **Trained models, graph artefacts, and SAR PDFs are not committed to the repo** (license + size). First-time setup downloads one CSV from Kaggle and runs `python src/run_pipeline.py` once (~10–15 min). After that, `docker compose up` or local dev both light up the full stack.

---

## Table of contents

1. [Setup](#setup)
2. [Running RUDRA](#run)
3. [What you'll see](#what-youll-see)
4. [Why this matters](#why-this-matters)
5. [Architecture](#architecture)
6. [The 14 pages](#pages)
7. [API surface](#api)
8. [ML stack on IBM AML 100k](#ml-stack)
9. [Tamper-evident audit log](#audit)
10. [RBAC](#rbac)
11. [Tests](#tests)
12. [Integrations](#integrations)
13. [Regulatory alignment](#regulatory)
14. [Team](#team)

---

<a id="setup"></a>
## 1. Setup

### 1.1 Prerequisites

| | Version |
|---|---|
| Python | 3.10+ |
| Node | 20+ |
| Docker + Docker Compose | 24+ (only if you want the Docker path) |
| Disk | ~1.5 GB free (raw IBM AML CSV is 475 MB; trained models + SAR PDFs add ~500 MB) |
| Time | ~15 min on first pipeline run (graph build + detectors + XGB + GraphSAGE + ensemble + SAR generation) |

### 1.2 Clone and install Python deps

```bash
git clone https://github.com/infinity2147/bhadra_rudra.git
cd bhadra_rudra
pip install -r requirements.txt
```

### 1.3 Download the IBM AML benchmark (required)

The repo does **not** ship the dataset. RUDRA uses a stratified 100k sample of HI-Small for honest, production-grade ML metrics — you have to fetch it once.

1. Create your Kaggle account if you don't have one.
2. Download `HI-Small_Trans.csv` from [Kaggle — IBM Transactions for AML](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) (475 MB).
3. Drop it at:

```
data/real/ibm_aml/HI-Small_Trans.csv
```

```bash
mkdir -p data/real/ibm_aml
# Then move/copy your downloaded file into that folder.
```

### 1.4 Run the pipeline (once)

```bash
python src/run_pipeline.py --dataset ibm_aml
```

This is the slow step. On first run it:

1. Loads `HI-Small_Trans.csv`, takes a stratified 100k sample, caches it as `HI-Small_Trans_100k_sampled.csv` so future runs skip resampling.
2. Builds the fund-flow graph (118k nodes / 87k edges).
3. Runs all 6 detectors → ~7,700 alerts.
4. Clusters alerts into ~7,000 incidents.
5. Trains XGBoost edge classifier → `data/ml/ibm_aml/model.pkl`.
6. Trains GraphSAGE GNN → `data/ml/ibm_aml/gnn/`.
7. Trains the stacked ensemble (3-fold OOF: XGB + SAGE + GAT + LR meta) → `data/ml/ibm_aml/ensemble/`.
8. Writes ~5,100 SAR PDFs for HIGH+ alerts → `data/ibm_aml/sar_reports/`.

Re-running is idempotent and **skips GNN + ensemble re-training** when their metrics files already exist. Pass `--force-retrain-ml` to redo them.

### 1.5 Optional — credentials for real integrations

Every external integration falls back to a schema-accurate mock when its key is absent. Drop any of these in your shell (or in `.env` for Docker) to flip them to real mode:

```bash
export GEMINI_API_KEY="..."           # Gemini 2.0 Flash copilot (Google's free tier works)
export SAHAMATI_CLIENT_ID="..."       # Sahamati AA sandbox
export SAHAMATI_CLIENT_SECRET="..."
export SAHAMATI_FIU_ID="..."
export DILISENSE_API_KEY="..."        # KYC / sanctions screening
```

`GET /api/integrations/status` reports which providers are live vs mocked.

### 1.6 Optional — PaySim secondary benchmark

```bash
mkdir -p data/real/paysim
# Download paysim.csv from https://www.kaggle.com/datasets/ealtman2019/paysim1
# Place it under data/real/paysim/
python src/run_pipeline.py --dataset paysim
```

Then `RUDRA_DATASET=paysim` switches the entire stack onto it.

---

<a id="run"></a>
## 2. Running RUDRA

### Option A — Local dev (most common for evaluators)

Two terminals:

```bash
# Terminal 1 — backend on :8000
cd backend && RUDRA_DATASET=ibm_aml uvicorn main:app --reload --port 8000
```

```bash
# Terminal 2 — frontend on :5173 (proxies /api → :8000)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**.

The backend **fails loud** if `data/ibm_aml/` is missing — it does *not* silently fall back to fake data. If you see a startup error pointing at `data/ibm_aml/transactions.csv`, you skipped step 1.4.

### Option B — Docker (after the pipeline has been run)

```bash
docker compose up
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend  | http://localhost:8000 |
| Kafka    | `kafka:9092` (inside the network) / `localhost:29092` (from the host) |

The Dockerfile attempts a one-time pipeline run during image build, but it will skip cleanly if the Kaggle CSV isn't present in the build context. To regenerate inside the container after starting up:

```bash
docker compose exec backend python src/run_pipeline.py --dataset ibm_aml
```

### Option C — Stream the benchmark over Kafka

```bash
docker compose up

# Replay IBM AML onto the topic at 20 events/sec
docker compose exec backend python -m streaming.kafka_producer --rate 20 --total 1000
```

Open the **Live Stream** page to watch events flow through the consumer → ML scoring → ring buffer.

---

<a id="what-youll-see"></a>
## 3. What you'll see

An investigator opens RUDRA in the morning with a queue of fraud alerts from the overnight ingest. Within minutes:

- The **Incidents** page clusters thousands of raw alerts into a handful of actionable cases (union-find on entity overlap).
- They click a CRITICAL incident → see the underlying alerts and involved entities.
- They click **Trace Journey** → a Sankey or force-directed view of money flowing through the network, with red-flagged edges, transaction timeline, and a SHAP panel explaining *why* the ML model scored this 95%.
- They write an investigation note → case auto-moves to INVESTIGATING.
- They click **File SAR** → blocked, 403 (*"requires SUPERVISOR role"*). Switch role → file SAR → status moves.
- **Download FIU Package** → zip lands locally: `STR.xml + SAR_xxx.pdf + subgraph.json + transaction_chain.csv + case_audit_log.json + evidence_summary.md + pmla_citations.txt`.
- They click **Verify chain** → *"Chain intact, head hash 1c3ce68a…"*.

Total time saved vs the old T+1 + manual STR drafting workflow: roughly **a day per case**.

---

<a id="why-this-matters"></a>
## 4. Why this matters

The RBI's 2023 Framework for Real-time Fraud Risk Monitoring (FRM) mandates sub-second fraud detection. Most PSBs still run nightly batch jobs — by which time the layered funds are gone.

Beyond latency, three practical problems break the current AML stack at PSBs:

1. **Alert fatigue** — investigators see hundreds of overlapping alerts from the same incident. We solve this with incident clustering.
2. **No explainability** — *ML score = 0.94* means nothing without *"here are the 5 features that drove it"*. We surface SHAP on every alert.
3. **Manual STR drafting** — compliance teams hand-type FIU-IND submissions, copy-pasting from Excel. We generate the full STR XML + PDF + supporting documents in one click.

Plus the foundations PS3 explicitly asks for: graph visualisation of fund flows, ML-flagged subgraphs, NetworkX + visual graph. All present.

---

<a id="architecture"></a>
## 5. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ React 19 + Vite + Tailwind 4  (frontend)                             │
│   Dashboard · Incidents · Cases · Journey · Graph · Analytics ·      │
│   Live · Patterns · Entities · Model · Copilot · SAR · Settings · AA │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ REST (X-User-Role header for RBAC)
┌────────────────────────────▼─────────────────────────────────────────┐
│ FastAPI                                                              │
│   ~50 endpoints, dependency-injected RBAC, SQLite-backed state       │
│   Active dataset = RUDRA_DATASET env (default: ibm_aml)              │
└─────┬──────────────────────┬─────────────────────────┬───────────────┘
      │                      │                         │
      │ Real Kafka           │ Adapter layer           │ Detection +
      │ (aiokafka            │ (real creds →           │ ML engine
      │  producer +          │  real Sahamati AA /     │ (Python)
      │  consumer +          │  DiliSense; mock        │
      │  KRaft broker)       │  fallback)              │
      │                      │                         │
      │ + Pathway            │ src.integrations.{      │
      │  (optional           │   AAClient,             │
      │   5-min windowed     │   DilisenseClient }     │
      │   velocity alerts)   │                         │
      ▼                      ▼                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  • NetworkX directed weighted graph                                  │
│  • 6 heuristic detectors (Johnson's cycle, layering, smurfing,       │
│    funnel, dormant, profile-mismatch)                                │
│  • Stacked ensemble: XGBoost + GraphSAGE + GAT + LR meta-learner     │
│    (3-fold OOF stacking, trained on IBM AML 100k)                    │
│  • SHAP TreeExplainer per alert                                      │
│  • Incident clustering via union-find                                │
│  • Hash-chain audit log on SQLite                                    │
│  • FIU package (STR XML + SAR PDF + subgraph + chain)                │
│  • Live per-txn scoring + latency benchmark                          │
│  • Gemini 2.0 Flash copilot with proper function calling             │
└──────────────────────────────────────────────────────────────────────┘
```

### Data layout (after the pipeline has run)

```
data/
├── real/                          # Raw downloaded datasets (you bring these)
│   ├── ibm_aml/HI-Small_Trans.csv          # full IBM AML  ← step 1.3
│   ├── ibm_aml/HI-Small_Trans_100k_sampled.csv  # cached stratified sample
│   ├── paysim/*.csv                         # optional, secondary benchmark
│   └── ieee_cis/*.csv                       # optional, tabular baseline
├── ibm_aml/                       # Per-variant operational artefacts
│   ├── transactions.csv                     # mapped to internal schema
│   ├── fraud_alerts.json                    # detector output
│   ├── incidents.json                       # clustered alerts
│   ├── risk_scores.json                     # per-entity risk
│   ├── fraud_cases.json
│   ├── detection_summary.json
│   ├── rudra.db                             # SQLite: cases + audit log + config
│   └── sar_reports/*.pdf
└── ml/                            # Per-variant model artefacts
    └── ibm_aml/
        ├── model.pkl                        # XGBoost + SHAP background
        ├── metrics.json
        ├── edge_scores.json
        ├── gnn/                             # GraphSAGE weights + metrics
        └── ensemble/                        # XGB + SAGE + GAT base models + LR meta
```

Everything except `data/real/*/README.md` is git-ignored. The pipeline rebuilds the whole tree deterministically (seed=42).

---

<a id="pages"></a>
## 6. The 14 pages

| Route | What it shows |
|---|---|
| `/` Dashboard | KPIs, daily trends, time-travel slider, latency benchmark, risk + case breakdowns |
| `/incidents` Incidents | Alerts clustered by entity overlap |
| `/cases` Case Workbench | Triage queue, SHAP per alert, hash-verified audit log, role-gated actions |
| `/journey` Fund Journey | Sankey for small flows / force-graph for large; red-flag annotation, timeline below |
| `/graph` Network Graph | Top-N by risk with filters (fraud-only, high-risk-only, min-amount, custom limit) and three layouts (Matrix / Arc / Force) |
| `/analytics` Channel/Branch | Volume + fraud-rate by channel, rail, hour, branch, product |
| `/patterns` Pattern Library | Per-pattern alert list + raw transactions |
| `/entities` Entity Explorer | Search entities, view risk score, transaction history |
| `/model` ML Models | XGBoost + GraphSAGE + ensemble metrics (F1/AUC/precision/recall/CM/feature-importance) |
| `/live` Live Stream | Per-txn ML scoring with mean+p95 latency tile; start/stop |
| `/copilot` AI Copilot | Natural-language investigation; Gemini-powered with tool calling (falls back to quick-commands if no key) |
| `/sar` SAR Reports | Generate the formal SAR text per alert |
| `/settings` Detector Settings | Threshold sliders per detector (ADMIN only), re-run detection on save |
| `/aa` Account Aggregator | AA consent issue/pull/revoke + DiliSense KYC screening |

---

<a id="api"></a>
## 7. API surface

Every endpoint accepts an optional `X-User-Role` header for RBAC.

### Dashboard / overview
- `GET  /` — health + active variant + alert/incident counts
- `GET  /api/dashboard?until=YYYY-MM-DDTHH:MM` — KPIs + charts (time-travel via `until`)
- `GET  /api/me` — current role + permission matrix

### Alerts, cases, incidents
- `GET  /api/alerts?severity=&pattern=` — alerts decorated with case status + ML score + incident_id
- `GET  /api/alerts/{id}` — single alert
- `GET  /api/alerts/{id}/explain` — SHAP local explanation
- `GET  /api/incidents` / `GET /api/incidents/{id}`
- `GET  /api/cases` / `GET /api/cases/{id}` — case + audit log
- `POST /api/cases/{id}/dispose` — change status (RBAC-gated per target state)
- `POST /api/cases/{id}/note` — append audit-log note
- `GET  /api/cases/{id}/verify` — verify the hash chain (Supervisor+)

### Graph + journey
- `GET  /api/graph?fraud_only=&high_risk_only=&min_amount=&limit=` — capped to top-N by risk by default so the response stays usable on 100k+ node graphs. `limit=0` removes the cap. Response includes `total_nodes` / `showing_nodes`.
- `GET  /api/graph/{entity}?hops=N` — entity subgraph
- `GET  /api/journey/{entity}?direction=&hops=&min_amount=` — forward/backward fund journey
- `GET  /api/journey/alert/{id}?include_neighbors=` — journey scoped to an alert

### ML
- `GET  /api/ml/variants` — every trained model variant + summary metrics
- `GET  /api/ml/metrics?variant=` — full metrics (defaults to active variant)
- `GET  /api/ml/ensemble?variant=ibm_aml` — stacked-ensemble metrics + meta-learner coefficients
- `GET  /api/ml/ensemble/edge_scores?variant=ibm_aml` — per-edge XGB / SAGE / GAT / ensemble breakdown
- `POST /api/ml/retrain` — retrain on current graph (Admin)

### FIU + SAR
- `GET  /api/sar/generate/{id}` — SAR text
- `GET  /api/fiu/package/{id}` — zip download (STR XML, SAR PDF, subgraph, chain, citations, audit log)

### Config + analytics + live
- `GET  /api/config/thresholds` (anyone) / `POST` (Admin)
- `POST /api/config/rerun` — re-run all detectors with current config (Admin)
- `GET  /api/analytics/{channels,branches,products}`
- `GET  /api/benchmark/latency` — full pipeline timing + speedup vs T+1

### Real Kafka streaming
- `GET  /api/stream/status` — consumer mode (kafka|inproc), buffer size, throughput
- `POST /api/stream/start` / `POST /api/stream/stop` — toggle the local consumer (any role)
- `POST /api/stream/reset` — hard reset: stop + wipe ring buffer & counters (seq restarts at 0)
- `GET  /api/stream/recent?limit=N` — newest scored events (per-txn ML score, latency, signals)
- `POST /api/stream/replay` — replay loaded txns onto the bus (any role); this is what the Live page calls
- `GET  /api/stream/velocity_alerts` — Pathway windowed alerts (if installed)

### Real Account Aggregator + DiliSense
- `POST /api/aa/consent` / `GET /api/aa/pull/{handle}` / `POST /api/aa/revoke/{handle}` / `GET /api/aa/consents`
- `GET  /api/kyc/screen?name=&entity_type=`
- `GET  /api/integrations/status` — which providers are live vs mocked

### Copilot
- `POST /api/copilot/query` — natural-language investigation (Gemini + tool calling)

---

<a id="ml-stack"></a>
## 8. ML stack on IBM AML 100k

| Component | Notes |
|---|---|
| 6 heuristic detectors | Johnson's cycle (with SCC size cap + time budget), BFS layering, threshold smurfing, flow-imbalance funnel, Z-score dormant, behavioural profile mismatch |
| XGBoost edge classifier | 30 engineered features, stratified 80/20, scale_pos_weight, persisted with SHAP background sample |
| GraphSAGE GNN | PyTorch Geometric, two SAGEConv layers, edge-classification MLP head, 200 epochs with early stopping |
| GAT (Graph Attention Network) | GATv2Conv 4-head attention; complementary to SAGE in the ensemble |
| **Stacked ensemble** | 3-fold OOF stacking → LR meta-learner. No in-fold leakage |
| SHAP local explanations | TreeExplainer, exact for the tree base |

### Honest numbers (IBM AML 100k, 88k edges)

| Model | F1 | AUC | Precision | Recall |
|---|---|---|---|---|
| XGBoost (base) | **0.617** | 0.927 | 0.851 | 0.484 |
| GraphSAGE (base) | 0.500 | 0.794 | 0.793 | 0.369 |
| GAT (base) | 0.495 | 0.770 | 0.691 | 0.386 |
| **Stacked ensemble** | **0.629** | 0.927 | 0.831 | **0.507** |

Meta-learner coefficients: **XGB +6.31** • SAGE +1.34 • GAT +0.37. The tree-based features dominate; the GNNs lift recall by catching a slice of fraud the tabular signals miss.

`neighbor_fraud_density` was deliberately *removed* before training — it acted as a shortcut on clustered fraud rings (*"are my neighbours flagged?"*), which is circular reasoning in production.

### Scale guards built into the detectors

IBM AML has 118k nodes / 87k edges. Out-of-the-box Johnson's cycle and exact betweenness centrality don't finish in reasonable time on graphs that size. The detectors ship with:

- **Cycle**: SCC size cap (200) + per-SCC wall-clock budget (8s). AML rings are tight clusters of shell accounts, not interbank topology.
- **Betweenness centrality**: Monte Carlo sampled (k=500, BFS-based) above 2k nodes. Cached on the graph object.
- **Dormant + profile-mismatch detectors**: transactions pre-indexed by sender/receiver, O(1) per-entity lookup instead of O(n) DataFrame scans.

All caps are configurable via `ConfigStore` (Settings page) — `circular_scc_size_cap`, `circular_scc_time_budget_s`, `centrality_sample_k`.

---

<a id="audit"></a>
## 9. Tamper-evident audit log

Every case has a chain of audit entries. Each entry stores:

```
prev_hash  = hash of the previous entry in this case's chain
this_hash  = SHA-256(prev_hash || canonical_entry_json)
```

`canonical_entry_json` is the entry's `(timestamp, author, action, from_status, to_status, note)` serialised with sorted keys. Any edit/insert/delete to the SQLite table breaks the chain. `GET /api/cases/{id}/verify` walks the chain and reports tampering. `tests/test_case_store.py::test_hash_chain_detects_tampering` proves this end-to-end.

---

<a id="rbac"></a>
## 10. RBAC

Three roles, gated by the `X-User-Role` header (frontend role-switcher in the sidebar):

| Action | INVESTIGATOR | SUPERVISOR | ADMIN |
|---|:-:|:-:|:-:|
| Open / note / view cases | ✓ | ✓ | ✓ |
| Move case to INVESTIGATING / ESCALATED | ✓ | ✓ | ✓ |
| File SAR / Dismiss | — | ✓ | ✓ |
| Download FIU evidence package | ✓ | ✓ | ✓ |
| Verify audit chain | — | ✓ | ✓ |
| Read detector thresholds | ✓ | ✓ | ✓ |
| Write thresholds / Re-run detection | — | — | ✓ |
| Retrain ML | — | — | ✓ |

In production this is replaced by the bank's IDP. The current implementation is a clearly-marked demo gate — see `src/rbac.py`.

---

<a id="tests"></a>
## 11. Tests

```bash
python -m pytest tests/ -v
# → 43 passed
```

The tests are **hermetic** — they use a small in-memory synthetic graph (`src/data_generator.py` → `TransactionGenerator`), so they run without the IBM AML download.

Coverage includes: every detector, ML feature matrix, train/save/load cycle, predict_one, case state machine, hash-chain integrity, hash-chain tampering detection, journey tracer, FIU package contents, STR XML well-formedness, incident clustering, RBAC permission matrix, AA consent flow, DiliSense determinism, live scoring, latency benchmark, config store.

---

<a id="integrations"></a>
## 12. Integrations

Every external integration follows the same contract: real when credentials are set, schema-accurate mock otherwise. The response always carries `_real: true|false` so operators know which mode is active.

| Integration | Real path | Fallback |
|---|---|---|
| Sahamati Account Aggregator | HTTPS to `api.sahamati.org.in/sandbox/v2` when `SAHAMATI_CLIENT_ID/SECRET/FIU_ID` set | `src/aa_kyc_mock.py` |
| DiliSense KYC / sanctions | HTTPS to `api.dilisense.com/v1/checkIndividual` when `DILISENSE_API_KEY` set | `src/aa_kyc_mock.py` |
| Kafka streaming | `aiokafka` consumer + producer when broker reachable | in-process `asyncio.Queue` |
| Pathway analytics | 5-min sliding-window per-entity velocity | skipped if `pathway` not installed |
| Gemini 2.0 Flash copilot | Real API when `GEMINI_API_KEY` set | quick-commands intent router |

Production AA HSM signing: real Sahamati requires every request body to be signed by the FIU's ECC-256 private key via HSM. The adapter ships with bearer-token auth (which the sandbox accepts); production deployment plugs the HSM signer into the `_headers()` method.

---

<a id="regulatory"></a>
## 13. Regulatory alignment

- **PMLA 2002** §3 (money-laundering offence) and §12 (obligations of reporting entities) — cited in every STR XML
- **PMLA Rules 2005** Rule 3 — STR record-keeping
- **RBI Master Direction on KYC 2016**, Chapter VII — suspicious transaction reporting
- **RBI FRM Framework 2023** — real-time fraud risk monitoring mandate
- **DPDP Act 2023** — STR XML marks every PII field as `REDACTED` with reason `DPDP_DataMinimisation`
- **Account Aggregator Framework** (RBI / Sahamati 2016, amended 2021) — consent flow modelled in the AA adapter

---

<a id="team"></a>
## 14. Team Bhadra

| Member | Role | Areas |
|---|---|---|
| Satyadev Suvesh | Agentic AI | LLM Copilot, RAG, FIU evidence pipeline |
| Anant Asati | Agentic AI | KYC delta reasoning, integration adapters |
| Prashant Gautam | Graphs & ML | Detection engine, XGBoost + GraphSAGE + ensemble |
| Yash Kumar Maru | Web Development | React dashboard, FastAPI backend |

---

## License

MIT — see [`LICENSE`](LICENSE).
