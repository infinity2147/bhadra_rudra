# RUDRA — Shield Against Deception

> Real-time fund-flow intelligence for Indian public-sector banks.
> Built for **PSBs Hackathon Series 2026** — Problem Statement 3 (Fund Flow Tracking).
> By **Team Bhadra**.

[![tests](https://img.shields.io/badge/tests-66%20passing-brightgreen)](#tests) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#install) [![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

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

**Detection latency: ~1.1 s end-to-end · 0.56 ms per transaction · 78,000× faster than T+1.**

Demo: `docker compose up` → http://localhost:5173.

---

## Table of contents

1. [What you'll see](#what-youll-see)
2. [Why this matters](#why-this-matters)
3. [Architecture](#architecture)
4. [Quick start](#quick-start)
5. [The 14 pages](#the-14-pages)
6. [API reference](#api-reference)
7. [ML stack — what's real vs synthetic](#ml-stack)
8. [Real benchmark datasets](#real-datasets)
9. [Tamper-evident audit log](#audit)
10. [RBAC](#rbac)
11. [Tests](#tests)
12. [What's planned but not built](#planned)
13. [Regulatory alignment](#regulatory)
14. [Team](#team)

---

<a id="what-youll-see"></a>
## 1. What you'll see

An investigator opens RUDRA at 09:00 with a queue of 290 fraud alerts from the overnight batch. By 09:05:

- The **Incidents** page has clustered those 290 alerts into 6 actionable cases (union-find on entity overlap).
- They click the CRITICAL incident → see the 237 underlying alerts + 55 involved entities.
- They click **Trace Journey** → a Sankey diagram showing money flowing through the cycle, with red-flagged edges, transaction timeline, and a SHAP panel explaining *why* the ML model scored this 95%.
- They write an investigation note ("verified KYC mismatch on Apex Trading"), case auto-moves to INVESTIGATING.
- They click **File SAR** → blocked, 403 ("requires SUPERVISOR role"). Switch role → file SAR → status moves.
- **Download FIU Package** → zip lands in Downloads: STR.xml + SAR_xxx.pdf + subgraph.json + transaction_chain.csv + case_audit_log.json.
- They click **Verify chain** → "Chain intact, head hash 1c3ce68a…".

Total time saved vs the old T+1 + manual STR drafting workflow: roughly **a day per case**.

---

<a id="why-this-matters"></a>
## 2. Why this matters

The RBI's 2023 Framework for Real-time Fraud Risk Monitoring (FRM) mandates sub-second fraud detection. Most PSBs run nightly batch jobs that surface alerts the next morning — by which time the layered funds are already gone.

Beyond latency, three other practical problems break the current AML stack at PSBs:

1. **Alert fatigue** — investigators see hundreds of overlapping alerts and can't tell which ones are the same incident from different angles. We solve this with incident clustering.
2. **No explainability** — ML score = 0.94 means nothing without "*here are the 5 features that drove it*". We surface SHAP on every alert.
3. **Manual STR drafting** — compliance teams hand-type FIU-IND submissions, copy-pasting transaction lists from Excel. We generate the full STR XML + PDF + supporting documents in one click.

Plus the foundations PS3 explicitly asks for: graph visualisation of fund flows, ML-flagged subgraphs, NetworkX + visual graph. All present.

---

<a id="architecture"></a>
## 3. Architecture

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
│    (3-fold out-of-fold stacking, trained on IBM AML 100k)            │
│  • IEEE-CIS tabular baseline (Kaggle real data)                      │
│  • SHAP TreeExplainer per alert                                      │
│  • Incident clustering via union-find                                │
│  • Hash-chain audit log on SQLite                                    │
│  • FIU package (STR XML + SAR PDF + subgraph + chain)                │
│  • Live per-txn scoring + latency benchmark                          │
│  • Gemini 2.0 Flash copilot with proper function calling             │
└──────────────────────────────────────────────────────────────────────┘
```

Full architecture in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

<a id="quick-start"></a>
## 4. Quick start

### Option A — Docker (recommended for evaluators)

```bash
docker compose up
```

That brings up:
- Backend at http://localhost:8000 (FastAPI + pipeline pre-built into the image)
- Frontend at http://localhost:5173 (nginx serving the built bundle, proxying `/api/*` to the backend)

The image pre-runs `src/run_pipeline.py` during build, so the first request lands on a fully populated dataset.

### Option B — Local dev

```bash
# 1. Install Python deps (3.10+)
pip install -r requirements.txt

# 2. Generate data + train all models + write SAR PDFs
python src/run_pipeline.py
# → data/transactions.csv          (~2.7k txns, 6 fraud pattern types)
# → data/fraud_alerts.json         (~290 alerts)
# → data/incidents.json            (~6 clustered incidents)
# → data/ml/synthetic/             (XGBoost model.pkl + metrics + edge_scores)
# → data/ml/synthetic/gnn/         (GraphSAGE weights + metrics)
# → data/sar_reports/*.pdf         (200+ SAR PDFs for HIGH+ alerts)
# → data/rudra.db                  (SQLite — cases, audit log, config)

# 3. Start the backend (port 8000)
cd backend && uvicorn main:app --reload --port 8000

# 4. Start the frontend (port 5173)
cd frontend && npm install && npm run dev

# 5. Open http://localhost:5173
```

Optional:
```bash
export GEMINI_API_KEY="..."   # Enables the real Gemini-powered copilot. Without it the
                              # copilot uses a local intent-routing fallback (also real,
                              # just less natural-sounding).
```

### Tests

```bash
pip install pytest
python -m pytest tests/ -v
# → 66 passed in ~12s
```

### Streaming the real benchmark over Kafka

```bash
# 1. Train the ensemble on IBM AML 100k (one-time, ~3-4 min):
python src/train_ibm_aml.py

# 2. Start the stack (backend auto-starts the Kafka consumer):
docker compose up

# 3. From a separate terminal, replay IBM AML over the real Kafka topic:
docker compose exec backend python -m streaming.kafka_producer \
    --source data/real/ibm_aml/HI-Small_Trans_100k_sampled.csv \
    --rate 20 --total 1000

# 4. Watch /api/stream/recent or the Live page — every event flows through
#    the broker → consumer → score_live_txn → ring buffer.
```

### Real Account Aggregator / DiliSense

Drop credentials into the environment (no code change):
```bash
export SAHAMATI_CLIENT_ID="..."
export SAHAMATI_CLIENT_SECRET="..."
export SAHAMATI_FIU_ID="..."
export DILISENSE_API_KEY="..."
```
`GET /api/integrations/status` reports which provider is live.

---

<a id="the-14-pages"></a>
## 5. The 14 pages

| Route | What it shows |
|---|---|
| `/` Dashboard | KPIs, daily trends, time-travel slider (replay the dataset up to any moment), latency benchmark, risk + case breakdowns |
| `/incidents` Incidents | Alerts clustered by entity overlap; ~290 alerts → ~6 actionable incidents |
| `/cases` Case Workbench | Triage queue, SHAP explanation per alert, audit-log with hash verification, role-gated SAR/Dismiss buttons |
| `/journey` Fund Journey | Sankey + force-graph trace forward/backward from any entity or alert, red-flag annotation, timeline below |
| `/graph` Network Graph | Full bank graph with filters: fraud-only, high-risk-only, min-amount |
| `/analytics` Channel/Branch | Volume + fraud-rate by channel, rail, hour, branch, product |
| `/patterns` Pattern Library | Per-pattern alert list + raw transactions |
| `/entities` Entity Explorer | Search entities, view risk score, transaction history |
| `/model` ML Models | XGBoost + GraphSAGE + IEEE-CIS metrics (F1/AUC/precision/recall/CM/feature-importance) |
| `/live` Live Stream | Per-txn ML scoring with mean+p95 latency tile; start/stop |
| `/copilot` AI Copilot | Natural-language investigation; Gemini-powered with proper tool calling |
| `/sar` SAR Reports | Generate the formal SAR text per alert |
| `/settings` Detector Settings | Threshold sliders per detector (ADMIN only), re-run detection on save |
| `/aa` Account Aggregator | AA consent issue/pull/revoke + DiliSense KYC screening (both mocked, DPI-shaped) |

---

<a id="api-reference"></a>
## 6. API reference

The full surface (`backend/main.py`) — every endpoint accepts an optional `X-User-Role` header for RBAC:

### Dashboard / overview
- `GET  /api/dashboard?until=YYYY-MM-DDTHH:MM` — KPIs + charts (time-travel via `until`)
- `GET  /api/me` — current role + permission matrix

### Alerts, cases, incidents
- `GET  /api/alerts?severity=&pattern=` — alerts decorated with case status + ML score + incident_id
- `GET  /api/alerts/{id}` — single alert
- `GET  /api/alerts/{id}/explain` — SHAP local explanation
- `GET  /api/incidents` — clustered incidents
- `GET  /api/incidents/{id}` — incident + its underlying alerts
- `GET  /api/cases` / `GET /api/cases/{id}` — case + audit log
- `POST /api/cases/{id}/dispose` — change status (RBAC-gated per target state)
- `POST /api/cases/{id}/note` — add audit-log note
- `GET  /api/cases/{id}/verify` — verify the hash chain (Supervisor+)

### Graph + journey
- `GET  /api/graph?fraud_only=&high_risk_only=&min_amount=` — full graph
- `GET  /api/graph/{entity}?hops=N` — entity subgraph
- `GET  /api/journey/{entity}?direction=&hops=&min_amount=` — forward/backward fund journey
- `GET  /api/journey/alert/{id}?include_neighbors=` — journey scoped to an alert

### ML
- `GET  /api/ml/variants` — every trained model variant + summary metrics
- `GET  /api/ml/metrics?variant=` — full metrics for a variant
- `GET  /api/ml/ensemble?variant=ibm_aml` — stacked-ensemble metrics (per-base + ensemble F1/AUC, meta-learner coefficients)
- `GET  /api/ml/ensemble/edge_scores?variant=ibm_aml` — per-edge XGB / SAGE / GAT / ensemble breakdown
- `GET  /api/ml/tabular` — IEEE-CIS tabular baseline metrics
- `POST /api/ml/retrain` — retrain on current graph (Admin)

### FIU + SAR
- `GET  /api/sar/generate/{id}` — SAR text
- `GET  /api/fiu/package/{id}` — zip download with STR XML, SAR PDF, subgraph, chain, citations

### Config + analytics + live
- `GET  /api/config/thresholds` (anyone) / `POST` (Admin)
- `POST /api/config/rerun` — re-run all detectors with current config (Admin)
- `GET  /api/analytics/{channels,branches,products}`
- `GET  /api/live/inject?count=N` — N simulated txns with per-txn ML score + latency
- `GET  /api/benchmark/latency` — full pipeline timing + speedup vs T+1

### Real Kafka streaming
- `GET  /api/stream/status` — consumer mode (kafka|inproc), buffer size, throughput
- `POST /api/stream/start` / `POST /api/stream/stop` (Admin)
- `GET  /api/stream/recent?limit=N` — newest scored events from the consumer's ring buffer
- `POST /api/stream/replay` — replay loaded txns onto the bus at a chosen rate (Admin)
- `GET  /api/stream/velocity_alerts` — Pathway sliding-window velocity alerts (if Pathway is running)

### Real Account Aggregator + DiliSense
- `POST /api/aa/consent` / `GET /api/aa/pull/{handle}` / `POST /api/aa/revoke/{handle}` / `GET /api/aa/consents`
- `GET  /api/kyc/screen?name=&entity_type=` — DiliSense sanctions/PEP screen
- `GET  /api/integrations/status` — which providers are live (real creds) vs. mocked

### Copilot
- `POST /api/copilot/query` — natural-language investigation (Gemini + tool calling)

---

<a id="ml-stack"></a>
## 7. ML stack — what's real vs synthetic

| Component | Status | Notes |
|---|---|---|
| Synthetic data generator | **Synthetic** | 2.7k txns, 80 entities, 6 fraud pattern types, channels + products + KYC fields |
| 6 heuristic detectors | **Real** | Johnson's algorithm cycle detection, BFS layering, threshold clustering smurfing, flow imbalance funnel, Z-score dormant, behavioural profile mismatch |
| XGBoost edge classifier | **Real** | 30 engineered features, stratified 80/20, scale_pos_weight, persisted with SHAP background sample |
| GraphSAGE GNN | **Real** | PyTorch Geometric, two SAGEConv layers, edge-classification MLP head |
| GAT (Graph Attention Network) | **Real** | GATv2Conv 4-head attention; complementary to SAGE in the ensemble |
| **Stacked ensemble** (XGB + SAGE + GAT + LR meta) | **Real** | 3-fold OOF stacking on IBM AML 100k. Honest stack — meta-learner trained on out-of-fold base predictions, not in-fold leakage |
| Real Kafka streaming | **Real** | aiokafka producer + consumer, single-node KRaft broker in docker-compose. In-process fallback when no broker reachable |
| Pathway windowed analytics | **Real (optional)** | 5-min sliding-window velocity alerts on top of the Kafka topic. `pip install pathway` to enable |
| Sahamati AA integration | **Real adapter** | `src.integrations.AAClient` makes real Sahamati sandbox calls when `SAHAMATI_CLIENT_ID/SECRET/FIU_ID` are set. Transparent mock fallback otherwise |
| DiliSense KYC | **Real adapter** | `src.integrations.DilisenseClient` calls real DiliSense API when `DILISENSE_API_KEY` is set. Same fallback contract |
| IEEE-CIS tabular baseline | **Real** (if dataset downloaded) | XGBoost on the Kaggle dataset the evaluators recommend |
| SHAP local explanations | **Real** | TreeExplainer, exact for tree models |
| Risk scoring (per entity) | **Real** | multi-signal composite |
| Incident clustering | **Real** | union-find on entity overlap |
| Hash-chain audit log | **Real** | SHA-256, verified by walking the chain |
| FIU evidence package | **Real** | STR XML format matches FIU-IND schema |
| Per-transaction live scoring | **Real** | model.predict_proba() on freshly computed features |
| Latency benchmark | **Real** | time.perf_counter() over actual pipeline runs |

### Honest numbers — synthetic vs IBM AML 100k

| Metric | Synthetic (2.7k edges) | IBM AML 100k (88k edges, real fraud labels) |
|---|---|---|
| XGBoost F1 / AUC | 0.99 / 0.99 | **0.626 / 0.927** |
| GraphSAGE F1 / AUC | 0.94 / 0.96 | **0.504 / 0.735** |
| GAT F1 / AUC | — | **0.495 / 0.770** |
| Stacked ensemble F1 / AUC | — | **0.626 / 0.926** |
| Ensemble Recall (vs XGB-alone 0.489) | — | **0.507** (catches more fraud) |
| Ensemble meta-learner weights | — | XGB +6.24, SAGE +1.27, GAT +1.13 |

The synthetic numbers are inflated by construction — our generator's fraud patterns have tight statistical signatures a tree-based model finds trivially. **The IBM AML numbers above are the metric you should hold us to**.

The stack is honest: XGBoost is the strongest signal, the meta-learner correctly weights it heavily, and the GNNs lift the ensemble's recall (+2.3pp) by catching fraud cases the tabular features alone miss. T2 raised SAGE from F1=0.315→0.504 and GAT from 0.323→0.495 by training 200 epochs with early stopping on a stratified val set (was 60 epochs full-batch); the ensemble F1 plateaus near XGB because tabular signals already dominate AML at this dataset size.

The `neighbor_fraud_density` feature was deliberately *removed* before training — it acted as a shortcut on clustered fraud rings ("are my neighbours flagged?"), which would be circular reasoning in production.

---

<a id="real-datasets"></a>
## 8. Real benchmark datasets

The pipeline auto-trains a separate model variant for each real dataset it finds on disk:

| Dataset | Place files in | Download |
|---|---|---|
| IBM AML — HI-Small | `data/real/ibm_aml/HI-Small_Trans.csv` | [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) |
| PaySim mobile-money | `data/real/paysim/*.csv` | [Kaggle](https://www.kaggle.com/datasets/ealtman2019/paysim1) |
| IEEE-CIS Fraud Detection | `data/real/ieee_cis/train_transaction.csv` | [Kaggle competition](https://www.kaggle.com/c/ieee-fraud-detection/data) |

The evaluators' resource list suggests the Kaggle Credit Card Fraud and IEEE-CIS datasets. **Both are tabular credit-card fraud with no sender→receiver structure.** They don't fit our fund-flow graph problem. We use them for the tabular ML baseline page (where they belong) and use IBM AML / PaySim — which have real sender→receiver edges — for the graph models. This is the most honest dataset choice for the PS.

After downloading, re-run `python src/run_pipeline.py` and the new variants show up under `/api/ml/variants` and the **ML Models** page.

---

<a id="audit"></a>
## 9. Tamper-evident audit log

Every case has a chain of audit entries. Each entry stores:

```
prev_hash  = hash of the previous entry in this case's chain
this_hash  = SHA-256(prev_hash || canonical_entry_json)
```

`canonical_entry_json` is the entry's `(timestamp, author, action, from_status, to_status, note)` serialised with sorted keys. If anyone edits, inserts, or deletes a row, the chain breaks. `GET /api/cases/{id}/verify` walks the chain and reports tampering.

`tests/test_case_store.py::test_hash_chain_detects_tampering` proves this end-to-end: it directly UPDATEs a note in SQLite and verifies the chain detects it.

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

In production this is replaced by the bank's existing IDP. The current implementation is a clearly-marked demo gate — see `src/rbac.py`.

---

<a id="tests"></a>
## 11. Tests

```bash
python -m pytest tests/ -v
```

66 tests covering: data generator schema, every detector, ML feature matrix, no-leakage assertion, train/save/load cycle, predict_one, case state machine, hash-chain integrity, hash-chain tampering detection, status counts, journey tracer, FIU package contents, STR XML well-formedness, incident clustering, RBAC permission matrix, AA consent flow, DiliSense determinism, live scoring, latency benchmark, config store.

```
================== 66 passed ==================
```

---

<a id="planned"></a>
## 12. What's planned but not built

We are deliberately honest about scope. Three things were "planned but not built" in earlier drafts and are now shipped:

- ✅ **Kafka / Pathway streaming ingestion** — real `aiokafka` producer + consumer, single-node KRaft broker in docker-compose, in-process fallback for local dev. Optional Pathway windowed-analytics layer on top.
- ✅ **Real ensemble (replaces FraudGT / BDH framing)** — stacked XGBoost + GraphSAGE + GAT with LR meta-learner trained on IBM AML 100k via 3-fold OOF stacking. FraudGT / BDH themselves are research models needing multi-GPU pre-training; this ensemble is what production fraud teams actually deploy.
- ✅ **Account Aggregator + DiliSense adapters** — real Sahamati sandbox and DiliSense API calls when env-driven creds are present. Schema-accurate mock fallback when not. The response always carries `_real: true|false` so the operator knows which mode is active.

Still out of scope for this POC:

- **Multi-tenant / federated learning across banks** — the architecture supports it but is not implemented.
- **Hindi/regional voice queries via Bhashini** — would be a nice DPI add-on; not currently wired.
- **Production AA HSM signing** — real Sahamati requires every request body to be signed by the FIU's ECC-256 private key via HSM. The adapter ships with bearer-token auth, which the sandbox accepts; production deployment plugs the HSM signer into the `_headers()` method.

---

<a id="regulatory"></a>
## 13. Regulatory alignment

- **PMLA 2002** Sections 3 (money-laundering offence) and 12 (obligation of reporting entities) — cited in every STR XML
- **PMLA Rules 2005** Rule 3 — STR record-keeping
- **RBI Master Direction on KYC 2016**, Chapter VII — suspicious transaction reporting
- **RBI FRM Framework 2023** — real-time fraud risk monitoring mandate (we deliver 78,000× faster than T+1)
- **DPDP Act 2023** — STR XML marks every PII field as `REDACTED` with reason `DPDP_DataMinimisation`
- **Account Aggregator Framework** (RBI / Sahamati 2016, amended 2021) — consent flow modelled in the AA mock

---

<a id="team"></a>
## 14. Team Bhadra

| Member | Role | Areas |
|---|---|---|
| Satyadev Suvesh | Agentic AI | LLM Copilot, RAG, FIU evidence pipeline |
| Anant Asati | Agentic AI | KYC delta reasoning, integration mocks |
| Prashant Gautam | Graphs & ML | Detection engine, XGBoost + GraphSAGE |
| Yash Kumar Maru | Web Development | React dashboard, FastAPI backend |

---

## License

MIT. No PII is ingested; every customer-identifiable field in generated STR XMLs is explicitly marked `REDACTED` per DPDP Act data-minimisation principles.
