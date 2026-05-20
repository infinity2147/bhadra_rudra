# RUDRA — Shield Against Deception

> Real-time fund flow intelligence system for detecting money laundering patterns in banking transactions. Built for **PSBs Hackathon Series 2026** by **Team Bhadra**.

## Problem

Banks rely on T+1 batch analysis — fraud is discovered the morning after it occurs. By the time a batch job flags a suspicious round-trip, funds have already been layered across multiple accounts. The RBI's 2023 Framework for Real-time Fraud Risk Monitoring (FRM) mandates sub-second detection, but most PSBs lack the infrastructure.

## Solution

RUDRA replaces batch analysis with a **live fund flow graph** that updates per transaction. Five parallel ML detectors scan the graph continuously, flagging circular transactions, rapid layering, structuring, dormant account activation, and KYC profile mismatches. An LLM copilot with tool-calling lets investigators query the graph in plain language and auto-generates compliance-ready SAR reports.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  React Dashboard (Vite + TailwindCSS + Recharts)     │
│  Dashboard | Graph | Alerts | Patterns | Copilot     │
└──────────────────────┬───────────────────────────────┘
                       │ REST API
┌──────────────────────▼───────────────────────────────┐
│  FastAPI Backend                                      │
│  Serves graph data, alerts, copilot queries, SARs     │
└──────────────────────┬───────────────────────────────┘
                       │ imports
┌──────────────────────▼───────────────────────────────┐
│  Python Detection Engine                              │
│  NetworkX graph | 5 pattern detectors | SAR generator │
│  LLM copilot (Gemini + local fallback)               │
└──────────────────────────────────────────────────────┘
```

## Detection Patterns

| Pattern | Method | What It Detects |
|---------|--------|----------------|
| Circular Transaction | Bucket-based cycle search with amount tolerance | Round-tripping: funds loop back to origin through 3+ accounts |
| Rapid Layering | BFS chain detection with decreasing amount check | Money moves through multiple accounts quickly to obscure origin |
| Smurfing/Structuring | Threshold clustering with amount variability analysis | Large amounts split into small transactions below reporting limits |
| Shell Company Funnel | Flow imbalance + centrality analysis | Multiple sources funnel into a single shell entity |
| Dormant Activation | Z-score spike detection on activity timelines | Accounts dormant for 30+ days suddenly show high-value activity |
| Profile Mismatch | Behavioral vs declared type comparison | Entity transaction patterns don't match their KYC profile |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, TailwindCSS, Recharts, react-force-graph-2d |
| Backend | Python, FastAPI, Uvicorn |
| Graph | NetworkX (directed weighted graph) |
| Data | Pandas, NumPy |
| Reports | ReportLab (PDF generation) |
| AI | Google Gemini (optional), local rule-based fallback |
| Charts | Recharts (React charting) |

## Quick Start

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (generates data)
```bash
python src/run_pipeline.py
```

### 3. Start the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Start the frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
```

### 5. Open the dashboard
Navigate to **http://localhost:5173**

## Project Structure

```
rudra/
├── backend/
│   └── main.py              # FastAPI REST API (all endpoints)
├── frontend/
│   └── src/
│       ├── pages/            # 7 page components
│       ├── components/       # Shared UI (MetricCard, SeverityBadge)
│       ├── api.js            # API client
│       └── App.jsx           # Router + sidebar layout
├── src/                      # Python ML engine
│   ├── data_generator.py     # Synthetic transaction generator
│   ├── graph_engine.py       # NetworkX graph builder
│   ├── fraud_detector.py     # 4 core detection algorithms
│   ├── advanced_detectors.py # Dormant + Profile detectors
│   ├── sar_generator.py      # SAR report generator + PDF export
│   └── llm_copilot.py       # AI copilot with tool-calling
├── data/                     # Generated pipeline output
│   ├── transactions.csv
│   ├── fraud_alerts.json
│   ├── risk_scores.json
│   └── sar_reports/          # 206 auto-generated PDF reports
├── context.md                # Full project context
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/dashboard` | KPIs, chart data, summary stats |
| GET | `/api/graph` | Full network graph (with filters) |
| GET | `/api/graph/{id}` | Entity subgraph (2 hops) |
| GET | `/api/alerts` | All fraud alerts |
| GET | `/api/patterns/{type}` | Pattern-specific data |
| GET | `/api/entities` | Entity list with risk scores |
| GET | `/api/entities/{id}` | Entity detail + transaction history |
| POST | `/api/copilot/query` | Chat with AI copilot |
| GET | `/api/sar/generate/{id}` | Generate SAR report |
| GET | `/api/live/inject` | Simulate live transactions |

## What's Simulated vs Real

| Component | Status |
|-----------|--------|
| Transaction data | Simulated (synthetic generator) |
| Fraud detection algorithms | Real (graph-based pattern detection) |
| Network graph | Real (NetworkX, actual graph algorithms) |
| Risk scoring | Real (multi-signal composite scoring) |
| SAR report generation | Real (template-based with regulatory citations) |
| AI Copilot | Real (works locally, enhanced with Gemini API key) |
| Real-time streaming | Simulated (inject endpoint simulates live feed) |
| FraudGT/BDH models | Planned (heuristic detectors in POC) |
| Account Aggregator integration | Planned (simulated data only) |

## Optional: Gemini API for Enhanced Copilot

Set the environment variable for enhanced AI responses:
```bash
export GEMINI_API_KEY="your-key-here"
```
Without it, the copilot uses local rule-based responses with full tool-calling capability.

## Team Bhadra

| Member | Role |
|--------|------|
| Satyadev Suvesh | Agentic AI — LLM Copilot, RAG, FIU Evidence |
| Anant Asati | Agentic AI — LLM Integration, KYC Delta |
| Prashant Gautam | Graphs & ML — Detection Engine, FraudGT |
| Yash Kumar Maru | Web Development — React Dashboard, FastAPI |

## Regulatory Alignment

- **PMLA 2002** — SAR citations auto-generated per section
- **RBI Master Direction on KYC** — Chapter VII suspicious transaction reporting
- **RBI FRM Framework 2023** — Real-time detection mandate
- **DPDP Act 2023** — Data minimisation, no PII in graph
- **Account Aggregator Framework** — Consent-based data ingestion (planned)
