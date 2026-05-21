# RUDRA — Shield Against Deception | Project Context

## Hackathon
- **Event:** PSBs Hackathon Series 2026 (iDEA 2.0)
- **Problem Statement:** PS3 — Fund Flow Tracking
- **Stage:** Phase 2 — POC submission
- **Organizer:** Government of India, Ministry of Finance, Department of Financial Services

## Team Bhadra
| Member | Role | Responsibility |
|--------|------|---------------|
| Satyadev Suvesh | Agentic AI | LLM Copilot, RAG pipeline, FIU Evidence generation |
| Anant Asati | Agentic AI | LLM integration, KYC delta reasoning, local explainer |
| Prashant Gautam | Graphs & ML | Detection engine, FraudGT (planned), XGBoost classifier |
| Yash Kumar Maru | Web Development | React dashboard, FastAPI backend |

## Problem
Banks rely on T+1 batch fraud analysis. By the time a suspicious round-trip is flagged the morning after, funds are already layered. The PS asks for an end-to-end fund-flow tracking system that maps movement across **accounts, products, branches, channels** and lets investigators trace the complete journey of funds + generate FIU evidence packages.

## Solution (current build)
Real-time AML/fraud platform with six concrete deliverables:

1. **Live fund-flow graph** built per transaction (NetworkX directed weighted).
2. **6 heuristic detectors** running in parallel — Circular / Layering / Smurfing / Shell Funnel / Dormant Activation / Profile Mismatch.
3. **XGBoost edge classifier** trained on 31 engineered features (F1, AUC, confusion matrix exposed at `/model`).
4. **Fund Journey Tracer** (`/journey`) — pick any alert or entity, walk forward/backward through the graph with timeline and red-flag annotation. The killer demo feature.
5. **Case Workbench** (`/cases`) — investigator triage queue with state machine (Open / Investigating / SAR Filed / Escalated / Dismissed) and full audit log per case.
6. **FIU Evidence Package** — one-click zip containing SAR PDF + subgraph JSON + transaction chain CSV + PMLA citations + case audit log. This is the actual artefact a PSB compliance officer would file with FIU-IND.

Plus: Channel/Branch/Product analytics, live transaction stream with per-txn ML scoring, LLM copilot with tool-calling.

## Architecture
```
rudra/
├── backend/main.py                     # FastAPI with 30+ endpoints
├── frontend/                           # React 19 + Vite + Tailwind 4
│   └── src/pages/                      # 11 pages
├── src/                                # Detection + ML engine
│   ├── data_generator.py               # Synthetic txns with channel/product/branch metadata
│   ├── graph_engine.py                 # NetworkX graph builder (rail_mix/channel_mix per edge)
│   ├── fraud_detector.py               # 4 core detectors
│   ├── advanced_detectors.py           # Dormant + Profile mismatch
│   ├── ml_model.py                     # XGBoost on 31 features + metrics + edge-scoring
│   ├── fund_tracer.py                  # Journey tracer (entity-mode + alert-mode)
│   ├── case_manager.py                 # File-backed case store + state machine
│   ├── fiu_package.py                  # FIU evidence zip builder
│   ├── sar_generator.py                # SAR text + PDF
│   ├── llm_copilot.py                  # Gemini + local copilot
│   └── run_pipeline.py                 # End-to-end pipeline runner
└── data/                               # Generated outputs
    ├── transactions.csv                # 2.7k synthetic txns
    ├── fraud_alerts.json               # ~290 alerts
    ├── cases.json                      # case state + audit log
    ├── ml/                             # model.pkl, metrics.json, edge_scores.json
    └── sar_reports/*.pdf               # 200+ SAR PDFs
```

## Data dimensions (what's in each transaction)
- `sender_id` / `receiver_id` / `_name` / `_type` (individual / business / shell_company)
- `sender_branch` / `receiver_branch` — 10 branches across India
- `sender_product` / `receiver_product` — SavingsAccount / CurrentAccount / LoanAccount / etc.
- `amount` (INR), `transaction_type` (NEFT / RTGS / IMPS / UPI / Wire Transfer)
- `channel` — Branch / NetBanking / MobileApp / ATM / UPI / ThirdPartyAPI
- `purpose_code`, `is_fraud`, `fraud_pattern`, `fraud_case_id`

## Pipeline
1. Generate 2.7k synthetic txns (2k normal + 670 fraud across 6 pattern types).
2. Build NetworkX graph (82 nodes, ~1.8k edges).
3. Run 6 detectors in parallel → ~290 alerts.
4. Train XGBoost on 31 edge features → metrics persisted.
5. Score every edge with the trained model → `edge_scores.json`.
6. Generate SAR PDFs for HIGH+ alerts.
7. Backend boots, exposes 30+ REST endpoints, opens a case row for every alert.

## Tech stack
- **Backend**: Python 3.10+, FastAPI, NetworkX, Pandas, XGBoost, ReportLab
- **Frontend**: React 19, Vite, Tailwind 4, Recharts, react-force-graph-2d
- **AI**: Google Gemini (optional) + local rule-based fallback
- **Data**: Synthetic CSV/Parquet, JSON case store

## Design decisions
- **No database for POC** — file-backed (JSON / parquet) keeps the repo self-contained for evaluators.
- **Synthetic data only** — no real PII or bank data, but with the same schema a real PSB would use.
- **XGBoost over FraudGT for POC** — XGBoost is honest, runnable on a laptop, and gives a real baseline; FraudGT/BDH is in the PoA roadmap.
- **Light theme** — banking-appropriate.

## What's planned but not built (be honest in demo)
- **Pathway / Kafka streaming ingestion** — PoA Phase 1, currently batch-on-startup.
- **FraudGT + BDH ensemble** — PoA Phase 2, currently XGBoost baseline.
- **Account Aggregator integration** — PoA Phase 1, currently synthetic.
- **PostgreSQL + SQLite audit DB** — PoA Phase 1, currently JSON.
- **ChromaDB RAG + Gemini tool-calling agent** — PoA Phase 3, currently local fallback with hand-coded tools.

## Demo flow (suggested order)
1. `/` — Dashboard. Open Cases card → click.
2. `/cases` — Pick a CRITICAL circular alert. Look at audit log. Click "Trace Journey".
3. `/journey` — See the cycle laid out, click an entity to see its red flags. Show the timeline below.
4. Back to `/cases` — click "FIU Package" to download the zip. Open it on desktop to show the contents.
5. `/analytics` — Show channel + branch breakdown. Point at the "fraud rate by channel" chart.
6. `/model` — Show F1, AUC, confusion matrix, feature importance.
7. `/live` — Start the stream. Show transactions arriving with ML scores.
8. `/copilot` — Ask "Show me high-risk entities" and "Trace funds for Apex Trading Co."

Total demo time: 4–5 minutes.
