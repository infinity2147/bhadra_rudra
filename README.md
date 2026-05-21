# RUDRA — Shield Against Deception

> Real-time fund-flow intelligence for Indian public-sector banks.
> Built for **PSBs Hackathon Series 2026** — Problem Statement 3 (Fund Flow Tracking).
> By **Team Bhadra**.

[![43 tests passing](https://img.shields.io/badge/tests-43%20passing-brightgreen)](#tests) [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#install) [![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

---

## TL;DR

Indian PSBs detect fraud the day after it happens. By then, layered funds have already left the bank. RUDRA replaces T+1 batch detection with:

- **Live fund-flow graph** (NetworkX) updated per transaction
- **6 specialised detectors** + **XGBoost edge classifier** + **GraphSAGE GNN** running in parallel
- **SHAP explanations** for every ML decision
- **Case workflow** with **tamper-evident audit log** (SHA-256 hash chain)
- **One-click FIU evidence package** (STR XML + SAR PDF + subgraph + transaction chain + audit log)
- **Account Aggregator + DiliSense KYC** integration mocks for DPI alignment

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
│   30+ endpoints, dependency-injected RBAC, SQLite-backed state       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────┐
│ Detection + ML engine (Python)                                       │
│   • NetworkX directed weighted graph                                 │
│   • 6 heuristic detectors (Johnson's cycle, layering, smurfing,      │
│     funnel, dormant, profile-mismatch)                               │
│   • XGBoost edge classifier (30 features, no leakage)                │
│   • GraphSAGE GNN baseline (PyTorch Geometric)                       │
│   • IEEE-CIS tabular baseline (Kaggle real data)                     │
│   • SHAP TreeExplainer per alert                                     │
│   • Incident clustering via union-find                               │
│   • Hash-chain audit log on SQLite                                   │
│   • FIU package (STR XML + SAR PDF + subgraph + chain)               │
│   • Live per-txn scoring + latency benchmark                         │
│   • Gemini 2.0 Flash copilot with proper function calling            │
│   • AA + DiliSense KYC mocks for DPI                                 │
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
# → 43 passed in ~10s
```

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

### DPI mocks
- `POST /api/aa/consent` / `GET /api/aa/pull/{handle}` / `POST /api/aa/revoke/{handle}` / `GET /api/aa/consents`
- `GET  /api/kyc/screen?name=&entity_type=` — DiliSense-style sanctions/PEP screen

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
| GraphSAGE GNN | **Real** | PyTorch Geometric, two SAGEConv layers, edge-classification MLP head, trains in seconds on CPU |
| IEEE-CIS tabular baseline | **Real** (if dataset downloaded) | XGBoost on the Kaggle dataset the evaluators recommend |
| SHAP local explanations | **Real** | TreeExplainer, exact for tree models |
| Risk scoring (per entity) | **Real** | multi-signal composite |
| Incident clustering | **Real** | union-find on entity overlap |
| Hash-chain audit log | **Real** | SHA-256, verified by walking the chain |
| FIU evidence package | **Real** | STR XML format matches FIU-IND schema |
| Per-transaction live scoring | **Real** | model.predict_proba() on freshly computed features |
| Latency benchmark | **Real** | time.perf_counter() over actual pipeline runs |
| Real-time streaming ingestion (Kafka/Pathway) | **Planned** | Currently `/api/live/inject` rotates random pairs |
| Account Aggregator integration | **Mocked** (schema-accurate) | Honest mock with `_mock_disclaimer` field on every response |
| DiliSense KYC | **Mocked** (deterministic) | Same name → same risk score, every time |
| FraudGT / BDH ensemble | **Planned** | Out of scope for POC; GraphSAGE is the GNN baseline today |

### Honest caveat on the F1

XGBoost trains to F1 = 0.99 on our synthetic data. The patterns we generate are clean by construction — circular rings have ≤15% amount variance, layering chains have monotonically decreasing flow, smurfing transactions cluster below ₹2L. A tree-based model finds them almost trivially.

On the IBM AML-HI-Large public benchmark, the state-of-the-art (FraudGT + BDH ensemble) reports F1 = 0.72 against an XGBoost baseline at ~0.42. That is the metric a judge should hold us to when we go to production.

The `neighbor_fraud_density` feature was deliberately *removed* before final training — it acted as a shortcut on heavily-clustered synthetic fraud rings ("are my neighbours flagged?"), which would create circular reasoning in production.

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

43 tests covering: data generator schema, every detector, ML feature matrix, no-leakage assertion, train/save/load cycle, predict_one, case state machine, hash-chain integrity, hash-chain tampering detection, status counts, journey tracer, FIU package contents, STR XML well-formedness, incident clustering, RBAC permission matrix, AA consent flow, DiliSense determinism, live scoring, latency benchmark, config store.

```
================== 43 passed in 9.42s ==================
```

---

<a id="planned"></a>
## 12. What's planned but not built

We are deliberately honest about scope. The PoA describes these; the demo does not yet ship them:

- **Kafka / Pathway streaming ingestion** — current `/api/live/inject` simulates the feed but doesn't consume from a real broker. Migrating is a single connector class.
- **FraudGT + BDH ensemble** — GraphSAGE is the GNN baseline we deliver. FraudGT is a graph-transformer model from ICAIF 2024; training it requires multi-GPU and days of compute, out of scope for a hackathon POC.
- **Production Account Aggregator / DiliSense integration** — both are honest mocks with schema-accurate responses. Real production would call Sahamati-licensed AAs (Setu / OneMoney / Anumati) and DiliSense / Refinitiv directly.
- **Multi-tenant / federated learning across banks** — the architecture supports it but is out of scope here.
- **Hindi/regional voice queries via Bhashini** — would be a nice DPI add-on; not currently wired.

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
