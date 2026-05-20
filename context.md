# RUDRA — Shield Against Deception | Project Context

## Hackathon Info
- **Event:** PSBs Hackathon Series 2026 (iDEA 2.0)
- **Problem Statement:** PS3 — Fund Flow Tracking
- **Stage:** Phase 2 — POC (Proof of Concept) submission
- **Organizer:** Government of India, Ministry of Finance, Department of Financial Services

## Team Bhadra
| Member | Role | Responsibility |
|--------|------|---------------|
| Satyadev Suvesh | Agentic AI | LLM Copilot, RAG pipeline, Auto FIU Evidence |
| Anant Asati | Agentic AI | LLM integration, KYC delta reasoning, Qwen local explainer |
| Prashant Gautam | Graphs & ML | PyTorch Geometric, FraudGT, BDH Model, XGBoost, 5-pattern detector |
| Yash Kumar Maru | Web Development | React/Vite dashboard, FastAPI backend, DB management |

## Problem
Banks rely on T+1 batch analysis — fraud discovered the morning after it occurs. By then funds are already layered. Need real-time fund flow tracking with instant pattern detection.

## Solution
Real-time AML/fraud detection platform:
1. Live fund flow graph updated per transaction
2. 5 parallel detectors: Layering, Round-tripping, Structuring, Dormant Activation, KYC Profile Mismatch
3. LLM copilot with tool-calling over live graph (trace_funds, find_cycles, explain_alert, get_profile_delta)
4. Auto SAR/FIU evidence package generation
5. DPI-native: Account Aggregator framework, RBI FRM mandate, DPDP Act 2023

## POC Requirements (from evaluator email)
- Working ML model/pipeline on sample/synthetic data
- Functional dashboard showing output
- End-to-end flow: input → processing → output
- Clear what is simulated vs real
- Code that runs on laptop
- NOT accepted: Figma mockups, slide decks, ChatGPT wrappers, manual-only demos

### Evaluation Rubric
1. **Technical Functionality** — Does the core system work and produce meaningful output?
2. **Problem Fit** — Tailored to Union Bank problem, not generic AI?
3. **Innovation & Depth** — Genuine technical work, not just API calls?
4. **Code Quality & Docs** — Clear README, evaluator can run it, readable code
5. **Demo Clarity** — Clear demo video, focused pitch deck
6. **Team & Execution** — Skills to take forward, honest about built vs planned

### Minimum Deliverable (from PS3 description)
"Show graph visualisation of fund flows. Demo a flagged pattern (circular transactions, rapid layering). Minimum: NetworkX/Neo4j + visual graph + ML-flagged suspicious subgraphs."

## Architecture (POC)
```
rudra/
├── backend/          # FastAPI Python backend (REST API)
│   └── main.py       # All API endpoints
├── frontend/         # React + Vite + TailwindCSS
│   └── src/          # Pages, components, API client
├── src/              # Python ML engine (data gen, graph, detection, SAR, copilot)
│   ├── data_generator.py    # Synthetic banking transactions with embedded fraud
│   ├── graph_engine.py      # NetworkX directed weighted graph builder
│   ├── fraud_detector.py    # 4 core detectors (circular, layering, smurfing, funnel)
│   ├── advanced_detectors.py # Dormant activation + profile mismatch
│   ├── sar_generator.py     # SAR report generation with PDF export
│   └── llm_copilot.py      # Gemini + local fallback copilot with tools
├── data/             # Generated data (transactions.csv, alerts, risk scores, SAR PDFs)
└── README.md
```

## Data
- 2,742 synthetic transactions (2,000 normal + 742 fraud)
- 82 entities (24 businesses, 50 individuals, 8 shell companies)
- 16 embedded fraud cases across 4 pattern types
- 10 bank branches across India
- Fraud patterns: circular_transaction, rapid_layering, smurfing, shell_funnel

## Detection Pipeline
1. Generate data (or load existing)
2. Build NetworkX directed weighted graph (82 nodes, 2027 edges)
3. Run 5 detectors in parallel → 313 alerts (50 circular, 200 layering, 18 smurfing, 5 funnels, 40 profile mismatch)
4. Generate SAR reports for HIGH+ severity alerts (206 PDFs)
5. Copilot available for natural language queries

## Tech Stack
- **Backend:** Python, FastAPI, NetworkX, Pandas, NumPy, ReportLab
- **Frontend:** React, Vite, TailwindCSS, react-force-graph-2d, Recharts
- **ML/AI:** Graph-based pattern detection, Gemini API (optional), local rule-based fallback
- **Data:** Synthetic CSV/Parquet (no real bank data)

## Key Design Decisions
- **No database for POC** — all data in memory, loaded from generated files
- **Synthetic data only** — no real PII or bank data
- **Gemini optional** — copilot works locally without API key
- **Graph visualization** — filtered views (fraud-only, high-risk, subgraph explorer) to avoid hairball effect
- **Light theme** — clean, professional, banking-appropriate

## Files That Matter
- `rudra.pptx` — Original pitch deck with solution outline
- `iDEA_2_0_PoA.pdf` — Plan of Action document with phase-wise implementation plan
- `src/fraud_detector.py` — Core detection engine (circular uses bucket-based cycle search, layering uses BFS with limits)
- `src/llm_copilot.py` — Gemini + local copilot (find_cycles uses targeted DFS)
- `backend/main.py` — FastAPI REST API
- `frontend/src/` — React pages and components

## Known Limitations (be honest in demo)
1. Graph is dense (30% density) due to random normal transactions — real banking data would be sparser
2. Cycle detection uses amount-bucket heuristic rather than Johnson's algorithm (too slow on dense graph)
3. No real-time streaming (Pathway/Kafka planned, not in POC)
4. No FraudGT/BDH models yet (graph ML detectors are heuristic-based for POC)
5. No Account Aggregator integration (simulated data only)
6. No PostgreSQL — in-memory for POC
