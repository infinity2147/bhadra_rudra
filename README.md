# RUDRA: Shield Against Deception

> Real-time fund-flow intelligence for Indian public sector banks.
> Built by **Team Bhadra** for the **PSBs Hackathon Series 2026**, Problem Statement 3 Fund Flow Tracking.

[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)](#how-to-run-locally) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#libraries-and-dependencies) [![Node](https://img.shields.io/badge/node-20%2B-blue)](#libraries-and-dependencies) [![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

For the formal submission documents, see [`docs/brief.tex`](docs/brief.tex) (two-page problem and solution brief), [`docs/ARCHITECTURE_BRIEF.tex`](docs/ARCHITECTURE_BRIEF.tex) (two-page technical architecture), and [`docs/architecture.mmd`](docs/architecture.mmd) (architecture diagram source).

---

## (a) Problem being solved

Indian public sector banks detect financial-crime patterns the day after they happen, by which point layered funds have already left the bank. The Reserve Bank's 2023 Framework for Real-time Fraud Risk Monitoring mandates sub-second detection, and the December 2024 Master Direction on Fraud Risk Management makes near-real-time monitoring, tamper-evident audit logs, and Digital Personal Data Protection Act compliance a hard requirement by March 2026. Most public sector banks still rely on nightly batch jobs, manual Suspicious Transaction Report drafting that takes around six hours per case, and vendor systems whose machine-learning scores cannot be defended before the Financial Intelligence Unit because the Prevention of Money Laundering Act Rules require the reasoning to be on record. The result: thousands of low-quality reports filed every year, investigators burning out on alert deduplication, and fraud detection that catches only single-digit percentages of actual losses.

**RUDRA replaces this with a real-time fund-flow intelligence platform that an investigator opens like email.** It scores every transaction in well under a millisecond, automatically clusters related alerts into a single investigable case, explains every decision with feature attributions, and generates the complete Financial Intelligence Unit filing package in one click. Every action is recorded in a cryptographically-chained audit log that Reserve Bank inspection can verify on demand.

The full problem-and-solution narrative is in [`docs/brief.tex`](docs/brief.tex). The technical architecture is in [`docs/ARCHITECTURE_BRIEF.tex`](docs/ARCHITECTURE_BRIEF.tex) with the diagram in [`docs/architecture.mmd`](docs/architecture.mmd).

---

## (b) How to run locally

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10 or newer |
| Node.js | 20 or newer |
| Docker and Docker Compose | 24+ (only required for the optional real Kafka path) |
| Disk space | About 1.5 GB total (the IBM AML CSV is 475 MB, trained models and reports add 500 MB) |
| Time | About 15 minutes on the first pipeline run |

### Step 1: Clone and install Python dependencies

```bash
git clone https://github.com/infinity2147/bhadra_rudra.git
cd bhadra_rudra
pip install -r requirements.txt
```

### Step 2: Provide a dataset

See Section (d). The fastest path is to download the IBM Anti-Money-Laundering benchmark from Kaggle. RUDRA refuses to start without a dataset by design, so it never silently serves fake data.

### Step 3: Run the pipeline once

```bash
python src/run_pipeline.py --dataset ibm_aml
```

This step takes around fifteen minutes on the first run. It loads the dataset, samples 100,000 transactions, builds the fund-flow graph, runs all six detectors, clusters the alerts into incidents, trains the XGBoost edge classifier, trains the GraphSAGE graph neural network if PyTorch is installed, trains the stacked ensemble, and writes the Suspicious Activity Report templates. Re-running is idempotent and skips already-trained models unless you pass `--force-retrain-ml`.

### Step 4: Start the backend (terminal one)

On Linux or macOS:

```bash
cd backend
RUDRA_DATASET=ibm_aml uvicorn main:app --port 8000
```

On Windows PowerShell:

```powershell
cd backend
$env:RUDRA_DATASET = "ibm_aml"
uvicorn main:app --port 8000
```

### Step 5: Start the frontend (terminal two)

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### Optional: real integration credentials

Drop any of these in the environment to flip the corresponding integrations from mock mode to live mode:

```bash
export ANTHROPIC_API_KEY="..."          # Claude Haiku investigation copilot
export SAHAMATI_CLIENT_ID="..."         # Sahamati Account Aggregator sandbox
export SAHAMATI_CLIENT_SECRET="..."
export SAHAMATI_FIU_ID="..."
export DILISENSE_API_KEY="..."          # Know Your Customer and sanctions screening
```

`GET /api/integrations/status` reports which providers are live and which are mocked.

### Optional: real Kafka streaming

```bash
docker compose up
```

This brings up a single-broker Kafka instance, the backend container, and the frontend container. The Live Stream page will flip the mode indicator from `inproc` (in-process queue) to `kafka` (real broker). Replay any subset of the loaded dataset onto the topic with:

```bash
docker compose exec backend python -m streaming.kafka_producer --rate 20 --total 1000
```

### Run the tests

```bash
python -m pytest tests/ -v
```

Forty-three tests pass in about ten seconds. They cover every detector, the machine-learning pipeline (training, persistence, prediction), the case workflow, hash-chain integrity and tampering detection, the FIU evidence package contents, incident clustering, role-based access enforcement, the Account Aggregator and DiliSense mocks, live scoring, the latency benchmark, and the configuration store.

---

## (c) Libraries and dependencies

### Backend, Python 3.10 or newer

| Library | Purpose |
|---|---|
| `fastapi` and `uvicorn[standard]` | Asynchronous web framework and ASGI server |
| `pandas`, `numpy`, `pyarrow` | Tabular data processing and Parquet support |
| `networkx` | In-memory directed graph for fund-flow modelling |
| `scikit-learn` | Logistic regression risk weights, train and test splits, evaluation metrics |
| `xgboost` | Gradient-boosted edge classifier, the primary machine-learning model |
| `reportlab` | Suspicious Activity Report PDF generation |
| `anthropic` | Claude Haiku 4.5 client for the investigation copilot |
| `httpx` | Asynchronous client for the Sahamati Account Aggregator and DiliSense KYC adapters |
| `aiokafka` | Asynchronous Kafka producer and consumer for the live streaming path |

Optional Python dependencies, auto-detected and falling back gracefully when missing:

- `torch` and `torch-geometric` for the GraphSAGE graph neural network
- `pathway` for the optional windowed-analytics streaming layer

### Frontend, Node.js 20 or newer

| Library | Purpose |
|---|---|
| `react@19`, `react-dom`, `react-router-dom` | Component framework and routing |
| `vite`, `@vitejs/plugin-react` | Development server with hot module reload, production build |
| `tailwindcss`, `@tailwindcss/vite` | Utility-first styling |
| `recharts` | Charting (dashboards, analytics, model metrics pages) |
| `react-force-graph-2d` | Force-directed network rendering on the Network Graph page |
| `react-markdown` | Renders Claude responses on the Copilot page |

### Infrastructure (optional)

Docker and Docker Compose for the Kafka broker, the backend container, and the frontend container. Single command: `docker compose up`.

The full list with versions is in [`requirements.txt`](requirements.txt) and [`frontend/package.json`](frontend/package.json).

---

## (d) Dataset

RUDRA is built and benchmarked against the **IBM Anti-Money-Laundering Transactions dataset**, the public benchmark Indian regulators benchmark against. The repository does not ship the dataset itself due to size and licensing. You have to fetch it once.

### Primary dataset: IBM Anti-Money-Laundering 100k

1. Create a Kaggle account if you do not have one.
2. Download `HI-Small_Trans.csv` (about 475 MB) from [IBM Transactions for Anti-Money-Laundering on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml).
3. Place it at:

```
data/real/ibm_aml/HI-Small_Trans.csv
```

4. Run the pipeline as shown in Section (b). It will take a stratified 100,000-row sample, cache it as `HI-Small_Trans_100k_sampled.csv` so future runs skip the resampling, build everything downstream, and write the artefacts under `data/ibm_aml/` and `data/ml/ibm_aml/`.

### Secondary dataset: PaySim (optional)

PaySim is supported as a second benchmark. Download `paysim.csv` from [PaySim on Kaggle](https://www.kaggle.com/datasets/ealtman2019/paysim1), place it under `data/real/paysim/`, and run:

```bash
python src/run_pipeline.py --dataset paysim
```

Then setting `RUDRA_DATASET=paysim` switches the entire backend onto it.

### Synthetic data generator (no download, for quick demos)

If you cannot use the Kaggle datasets, the repository includes a deterministic synthetic generator at `src/data_generator.py`. It produces approximately 2,700 transactions across 80 entities with six embedded fraud pattern types (circular, layering, smurfing, shell funnel, dormant reactivation, profile mismatch). To use it, place its output under the `ibm_aml` namespace so the dataset-aware backend can serve it without code changes:

```bash
python -c "
from src.data_generator import TransactionGenerator, save_data
import os
gen = TransactionGenerator(seed=42)
df, fraud_cases = gen.generate_all_data()
os.makedirs('data/ibm_aml', exist_ok=True)
save_data(df, fraud_cases, 'data/ibm_aml', entities=gen.entities)
"
python src/run_pipeline.py --dataset ibm_aml
```

The synthetic dataset produces an artificially high F1 of approximately 0.99 because the embedded fraud patterns are clean by construction. The IBM AML benchmark produces the honest production number of F1 = 0.625, which is what the platform is actually built to deliver.

---

## (e) Known limitations

- **Resource intensity on first run.** The first pipeline run on the IBM AML benchmark takes around fifteen minutes and uses roughly two gigabytes of memory at peak. Subsequent runs are idempotent and much faster.
- **GraphSAGE and the stacked ensemble require optional dependencies.** GraphSAGE needs PyTorch and PyTorch Geometric; the stacked ensemble depends on both. If they are not installed, the pipeline emits a clear notice and the backend falls back to the XGBoost classifier alone. Behaviour is correct in both cases, only the model variety differs.
- **Real Kafka is optional.** The default in-process consumer mode produces identical analytic results. Real Kafka is only required when you want to demonstrate the multi-broker production topology; bring it up with `docker compose up`.
- **Account Aggregator and DiliSense integrations default to mock mode.** Both have schema-accurate mocks marked clearly with a `_mock_disclaimer` field on every response. Set the relevant environment variables shown in Section (b) to flip them to the live REST adapters.
- **The Claude copilot requires an Anthropic API key.** Without `ANTHROPIC_API_KEY`, the copilot falls back to a deterministic intent-routing regex matcher that still dispatches the correct backend tool but writes terser, rule-based responses rather than multi-round natural-language synthesis. The fallback never crashes the page.
- **Browser compatibility.** Tested on Chrome 122 or newer, Edge 122 or newer, and Firefox 124 or newer. Older browsers may not render the Network Graph page's matrix view at the full grid resolution.
- **Single-user demonstration mode.** The role switcher in the sidebar is a clearly-marked demonstration gate using the `X-User-Role` header. A production deployment would replace it with the bank's existing identity provider (for example Okta, Keycloak, or Active Directory).
- **F1 = 0.625 is honest, not state-of-the-art.** The published industry leader on this benchmark (FraudGT plus BDH ensemble) reports F1 = 0.72 and needs multi-GPU training over several days. Our XGBoost result is what the same benchmark records as the strong-baseline number with single-CPU training, which is the most operationally realistic baseline a public sector bank can deploy on its own hardware.
- **Suspicious Activity Report PDFs are generated on demand.** Earlier versions pre-rendered all PDFs at pipeline time, which slowed setup considerably. The current pipeline writes a PDF only when an investigator clicks the Download FIU Package button, keeping startup fast and storage low.

---

## Project structure

```
bhadra_rudra/
├── backend/                FastAPI backend, 51 endpoints
│   └── main.py             ASGI entry point
├── frontend/               React 19 frontend (Vite + Tailwind 4)
│   └── src/
│       ├── api.js          Backend client with X-User-Role header
│       ├── App.jsx         Routes and sidebar
│       ├── components/     Reusable UI components
│       └── pages/          One file per route (14 pages)
├── src/                    Python engine, importable from backend
│   ├── data_generator.py        Synthetic dataset generator
│   ├── real_data_loader.py      IBM AML and PaySim loaders
│   ├── graph_engine.py          NetworkX fund-flow graph
│   ├── fraud_detector.py        Core four detectors
│   ├── advanced_detectors.py    Dormant and profile-mismatch detectors
│   ├── fund_tracer.py           Journey tracing engine
│   ├── ml_model.py              XGBoost training and inference
│   ├── gnn_model.py             GraphSAGE training and inference
│   ├── ensemble_model.py        Stacked ensemble (XGBoost + GraphSAGE + GAT + LR meta)
│   ├── risk_score_learner.py    Logistic-regression risk weights
│   ├── shap_explainer.py        SHAP TreeExplainer wrapper
│   ├── sar_generator.py         Suspicious Activity Report text and PDF
│   ├── fiu_package.py           One-click FIU evidence zip builder
│   ├── case_manager.py          SQLite case store with SHA-256 hash chain
│   ├── incident_clustering.py   Union-find on entity overlap
│   ├── live_scoring.py          Sub-millisecond per-transaction scoring
│   ├── config_store.py          Detector and tracer threshold persistence
│   ├── rbac.py                  Role-based access control matrix
│   ├── llm_copilot.py           Claude-powered copilot with tool calling
│   ├── dataset_config.py        Active-variant resolver
│   ├── run_pipeline.py          End-to-end pipeline runner
│   ├── integrations/            Sahamati AA and DiliSense KYC adapters
│   └── streaming/               Kafka producer, consumer, and ingestor
├── tests/                   Pytest suite, 43 tests
├── docs/
│   ├── ARCHITECTURE.md           Long-form technical architecture
│   ├── ARCHITECTURE_BRIEF.tex    Two-page technical brief
│   ├── architecture.mmd          Mermaid diagram source
│   └── brief.tex                 Two-page problem and solution brief
├── data/                    Generated artefacts (gitignored, regenerated by the pipeline)
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── requirements.txt
└── README.md
```

---

## License

MIT. See [`LICENSE`](LICENSE).

No personally identifiable information is ingested at runtime. Every customer-identifiable field in generated Suspicious Transaction Report XML and Suspicious Activity Report PDF output is explicitly marked as `REDACTED` with the reason `DPDP_DataMinimisation` in accordance with the Digital Personal Data Protection Act 2023 Section 11 minimisation principle.

---

## Team Bhadra

| Member | Areas |
|---|---|
| Satyadev Suvesh | Agentic AI, LLM Copilot, FIU evidence pipeline |
| Anant Asati | Agentic AI, KYC delta reasoning, integration adapters |
| Prashant Gautam | Graph algorithms, XGBoost, GraphSAGE, stacked ensemble |
| Yash Kumar Maru | React frontend, FastAPI backend, infrastructure |
