# RUDRA — Shield Against Deception

> Real-time fund flow intelligence for Indian public sector banks.
> Built for **PSBs Hackathon Series 2026** — Problem Statement 3 (Fund Flow Tracking) — by **Team Bhadra**.

---

## What it does

Banks currently rely on T+1 batch analysis: a suspicious round-trip is only flagged the morning after the funds have already been layered. RUDRA replaces that with a live fund-flow graph and a real ML pipeline, so a compliance investigator can:

1. **Triage** a queue of fraud alerts ranked by severity + ML score.
2. **Trace** the complete journey of suspect funds — forward and backward through the network, with a timeline.
3. **Analyse** where fraud is entering the bank by channel, branch, product, and hour.
4. **Decide** — Investigating / SAR Filed / Escalated / Dismissed — with a full audit log per case.
5. **File** an FIU-ready evidence package (SAR PDF + subgraph + transaction chain + PMLA citations) in one click.
6. **Ask** a natural-language copilot to investigate any entity, alert, or pattern.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  React / Vite / Tailwind / Recharts / react-force-graph-2d    │
│  Dashboard │ Cases │ Journey │ Graph │ Analytics │ Live │ ML  │
└──────────────────────────────┬─────────────────────────────────┘
                               │ REST (vite proxy)
┌──────────────────────────────▼─────────────────────────────────┐
│  FastAPI                                                       │
│  /api/dashboard /alerts /graph /journey /cases /fiu/package    │
│  /api/analytics/{channels,branches,products} /ml/{metrics}     │
└──────────────────────────────┬─────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────┐
│  Detection + ML engine (Python)                                │
│  • 6 heuristic detectors (circular, layering, smurfing,        │
│    funnel, dormant, profile)                                   │
│  • XGBoost edge-classifier on 31 engineered graph features     │
│  • Fund-journey tracer with red-flag annotation                │
│  • Case state machine + audit log                              │
│  • FIU evidence-package zip builder                            │
│  • LLM copilot (Gemini + local fallback) with 4 tools          │
└────────────────────────────────────────────────────────────────┘
```

---

## Detection coverage

| Pattern | Method | Maps to PS |
|---|---|---|
| Circular Transaction | bucket-based cycle search + amount-variance filter | round-tripping |
| Rapid Layering | BFS chain detection with decreasing-flow heuristic | rapid layering through multiple accounts |
| Smurfing / Structuring | sliding-window threshold + variability check | structuring below reporting thresholds |
| Shell Company Funnel | flow-imbalance + branch-diversity analysis | shell-company patterns |
| Dormant Activation | Z-score spike on per-account activity timelines | sudden activation of dormant accounts |
| Profile Mismatch | behavioural-vs-declared (entity-type) delta | mismatches between KYC profile and behaviour |
| **Edge classifier** (ML) | XGBoost on 31 engineered features | overarching probabilistic scoring |

---

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React 19, Vite, Tailwind 4, Recharts, react-force-graph-2d |
| Backend  | Python 3.10+, FastAPI, Uvicorn |
| Graph    | NetworkX (directed weighted) |
| ML       | XGBoost (with scikit-learn GradientBoosting fallback) |
| Data     | Pandas, Parquet, JSON |
| Reports  | ReportLab (SAR PDF), Python zipfile (FIU package) |
| AI       | Google Gemini (optional) + local rule-based fallback |

---

## Quick start

```bash
# 1. Install Python deps
pip install -r requirements.txt

# 2. Generate data + train model + write SAR PDFs
python src/run_pipeline.py
# → data/transactions.csv (2.7k txns)
# → data/fraud_alerts.json (~290 alerts)
# → data/ml/{model.pkl, metrics.json, edge_scores.json}
# → data/sar_reports/*.pdf (200+ reports)

# 3. Start backend (port 8000)
cd backend && uvicorn main:app --reload --port 8000

# 4. Start frontend (port 5173)
cd frontend && npm install && npm run dev

# 5. Open http://localhost:5173
```

Optional: `export GEMINI_API_KEY=...` for the enhanced copilot. Without it, the copilot uses local rule-based tool execution.

---

## Pages

| Route | What it shows |
|---|---|
| `/`           | KPI dashboard, trends, risk + case-status distributions |
| `/cases`      | Investigator workbench — triage, dispose, audit log per case |
| `/journey`    | **Fund Journey Tracer** — pick an alert or entity, walk the chain forward/backward, see the timeline + red flags |
| `/graph`      | Full network graph with fraud / high-risk / amount filters |
| `/analytics`  | Channel, branch, hour-of-day and product analytics |
| `/patterns`   | Per-pattern alert list + raw transactions |
| `/entities`   | Search entities, view risk score + transaction history |
| `/model`      | ML metrics: F1, AUC, confusion matrix, feature importance |
| `/live`       | Live transaction stream with per-txn ML scoring |
| `/copilot`    | Natural-language fraud investigation copilot |
| `/sar`        | Generate raw SAR text per alert |

---

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET  | `/api/dashboard` | KPIs, charts, case-status breakdown |
| GET  | `/api/alerts?severity=&pattern=` | Alerts decorated with case status + ML score |
| GET  | `/api/alerts/{id}` | Single alert |
| GET  | `/api/graph?fraud_only=&high_risk_only=&min_amount=` | Full graph |
| GET  | `/api/graph/{entity}?hops=N` | Subgraph |
| GET  | `/api/journey/{entity}?direction=&hops=&min_amount=` | Forward/backward fund journey |
| GET  | `/api/journey/alert/{id}?include_neighbors=` | Journey scoped to alert |
| GET  | `/api/cases?status=` | Case list |
| GET  | `/api/cases/{id}` | Case detail (with audit log) |
| POST | `/api/cases/{id}/dispose` | Change case status (Investigating / SAR Filed / Escalated / Dismissed) |
| POST | `/api/cases/{id}/note` | Add audit-log note without status change |
| GET  | `/api/fiu/package/{id}` | Download FIU evidence package (zip) |
| GET  | `/api/ml/metrics` | Trained model metrics + feature importance |
| POST | `/api/ml/retrain` | Retrain on current graph |
| GET  | `/api/analytics/channels` | Volume + fraud rate by channel & hour |
| GET  | `/api/analytics/branches` | Volume + fraud rate by branch |
| GET  | `/api/analytics/products` | Volume by product (savings/current/loan/credit) |
| GET  | `/api/live/inject?count=N` | Stream N simulated txns with detection results |
| POST | `/api/copilot/query` | Natural-language investigation |
| GET  | `/api/sar/generate/{id}` | Generate SAR text |
| POST | `/api/pipeline/run` | Re-run full pipeline |

---

## Honest scope — what's real vs simulated

| Component | Status |
|---|---|
| Transaction data | **Simulated** — synthetic generator (2.7k txns, 82 entities, 6 pattern types) |
| Pattern detection | **Real** — graph-based algorithms, runs on the actual graph |
| ML model | **Real** — XGBoost trained on 31 features from real graph state |
| Risk scoring | **Real** — multi-signal composite (centrality + flow imbalance + type + fraud-edge density) |
| Fund journey tracing | **Real** — actual BFS over the graph with timeline reconstruction |
| Case workflow + audit log | **Real** — file-backed CaseStore, transitions persisted |
| SAR PDF generation | **Real** — ReportLab, template-based with regulatory citations |
| FIU evidence zip | **Real** — bundles SAR PDF + subgraph JSON + chain CSV + PMLA citations + audit log |
| LLM copilot | **Real** — works locally; enhanced with Gemini API key |
| Live streaming | **Simulated** — `/api/live/inject` emits realistic feed with on-the-fly ML scoring |
| FraudGT / BDH ensemble | **Planned** (PoA Phase 2/3); current ML baseline is XGBoost |
| Pathway / Kafka ingestion | **Planned** (PoA Phase 1); current is batch-on-startup |
| Account Aggregator | **Planned** (PoA Phase 1); current is synthetic data only |

We deliberately use synthetic data so the demo is reproducible on a laptop. The structure of every output (alerts, cases, evidence packages, SAR PDFs) matches what a real PSB compliance team would file with FIU-IND.

---

## ML model — what XGBoost learns

31 engineered features per edge, including:

- Per-edge: log(total/avg/max/min amount), txn count, coefficient of variation, time span
- Endpoint structure: in/out degree, log-strength, type one-hots, same-branch
- Channel/rail mix: diversity, RTGS+Wire share, UPI share
- Pattern hints: near-reporting-threshold score, night-time ratio, weekend ratio, SCC membership
- Context: neighbor-fraud density (one-hop)

Trained on a stratified 80/20 split, with `scale_pos_weight` for class imbalance. Metrics on the held-out test set are exposed at `/api/ml/metrics` and visualised at `/model`.

> Caveat: F1 on our synthetic data is very high because the embedded patterns are clean by construction. The real benchmark is IBM AML-HI-Large (2.1M nodes), where we target F1 = 0.72 with the planned FraudGT+BDH ensemble vs XGBoost ~0.42.

---

## Regulatory alignment

- **PMLA 2002** Sections 3 & 12 — money-laundering offence + reporting obligation
- **PMLA Rules 2005** Rule 3 — STR record-keeping
- **RBI Master Direction on KYC 2016** Chapter VII — suspicious transaction reporting
- **RBI FRM Framework 2023** — real-time fraud risk monitoring mandate
- **DPDP Act 2023** — data minimisation (no PII in graph edges)
- **Account Aggregator Framework** — consent-based data sharing (planned ingestion path)

Every FIU evidence package includes the relevant PMLA / RBI citations for its detected pattern.

---

## Team Bhadra

| Member | Role | Areas |
|---|---|---|
| Satyadev Suvesh | Agentic AI | LLM Copilot, RAG, FIU Evidence |
| Anant Asati    | Agentic AI | KYC delta, profile reasoning |
| Prashant Gautam | Graphs & ML | Detection engine, FraudGT (planned) |
| Yash Kumar Maru | Web Development | React dashboard, FastAPI backend |
