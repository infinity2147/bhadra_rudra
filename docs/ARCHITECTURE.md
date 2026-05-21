# RUDRA — Technical Architecture

**PSBs Hackathon Series 2026 · Problem Statement 3: Fund Flow Tracking**
**Team Bhadra · Document Version 3.0**

---

## 1. System overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│ React 19 + Vite + Tailwind 4 (frontend, http://localhost:5173)             │
│                                                                            │
│  Dashboard | Incidents | Cases | Journey | Graph | Analytics | Live |     │
│  Patterns  | Entities  | Model | Copilot | SAR   | Settings   | AA + KYC  │
│                                                                            │
│  - react-force-graph-2d for the network view                               │
│  - Custom SVG Sankey for fund-flow direction                               │
│  - Recharts for KPI / trend visualisations                                 │
│  - Role switcher in nav (sends X-User-Role on every fetch)                 │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │ REST (vite proxy in dev, nginx in docker)
┌────────────────────────────────▼───────────────────────────────────────────┐
│ FastAPI (backend, http://localhost:8000)                                   │
│                                                                            │
│  /api/dashboard         /api/journey/{id}        /api/cases/{id}/dispose   │
│  /api/incidents         /api/alerts/{id}/explain /api/cases/{id}/verify    │
│  /api/ml/variants       /api/fiu/package/{id}    /api/config/thresholds    │
│  /api/ml/metrics        /api/benchmark/latency   /api/aa/consent           │
│  /api/live/inject       /api/copilot/query       /api/kyc/screen           │
│                                                                            │
│  RBAC: X-User-Role gate (INVESTIGATOR / SUPERVISOR / ADMIN)                │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────────────┐
│ Python engine (src/)                                                       │
│                                                                            │
│  Data        ┌─ data_generator.py    (synthetic txns, channel/product)     │
│              └─ real_data_loader.py  (IBM AML, PaySim, IEEE-CIS)           │
│                                                                            │
│  Graph       ┌─ graph_engine.py      (NetworkX DiGraph, rail+channel mix)  │
│              └─ fund_tracer.py       (BFS journey, red-flag annotation)    │
│                                                                            │
│  Detect      ┌─ fraud_detector.py    (Johnson's cycle, layering, smurf,    │
│              │                        funnel — config-driven thresholds)   │
│              └─ advanced_detectors.py(dormant z-score, profile mismatch)   │
│                                                                            │
│  ML          ┌─ ml_model.py          (XGBoost edge classifier, 30 feats)   │
│              ├─ gnn_model.py         (GraphSAGE on PyTorch Geometric)      │
│              ├─ tabular_baseline.py  (IEEE-CIS Kaggle baseline)            │
│              └─ shap_explainer.py    (per-alert SHAP attributions)         │
│                                                                            │
│  Workflow    ┌─ case_manager.py      (SQLite store + SHA-256 hash chain)   │
│              ├─ incident_clustering.py (union-find on alert overlap)       │
│              ├─ config_store.py      (detector thresholds, in SQLite)      │
│              └─ rbac.py              (role → action permission matrix)     │
│                                                                            │
│  Output      ┌─ sar_generator.py     (SAR text + PDF via ReportLab)        │
│              ├─ fiu_package.py       (zip + STR XML + subgraph + chain)    │
│              └─ live_scoring.py      (per-txn ML score + latency bench)    │
│                                                                            │
│  DPI mocks   └─ aa_kyc_mock.py       (AA consent flow + DiliSense KYC)     │
│                                                                            │
│  AI          └─ llm_copilot.py       (Gemini 2.0 Flash + local fallback)   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component breakdown

### 2.1 Frontend

- **React 19 + Vite** for component model and dev server hot-reload.
- **Tailwind 4** for utility-first styling — no bespoke CSS file.
- **react-force-graph-2d** for the network graph (used in `/graph` and `/journey` for cyclic alerts).
- **Custom SVG Sankey** (`components/Sankey.jsx`) for non-cyclic journeys — laid out by depth/side, edge thickness proportional to log(amount).
- **Recharts** for trend / distribution / bar charts on Dashboard, Analytics, Model pages.
- **Role context** in `api.js` — every fetch sends `X-User-Role` so the backend can enforce permissions.

### 2.2 Backend (FastAPI)

- Single ASGI app (`backend/main.py`) with 30+ endpoints.
- State loaded once at startup: synthetic data + graph + alerts + case store + config store + ML bundle.
- `Depends(get_role)` extracts the role header; `require(action, role)` raises HTTP 403 on disallowed actions.
- CORS open in dev; production would tighten to the bank's domain.

### 2.3 Detection engine

- **Cycle detection**: SCC decomposition → bucket edges by log-amount → run Johnson's `nx.simple_cycles(length_bound=N)` per bucket → validate cycle amount variance ≤ tolerance. This combines correctness (Johnson's enumerates *every* simple cycle) with a signal-aware filter that matches actual round-tripping (amounts on every hop are similar).
- **Layering**: BFS chain expansion with decreasing-amount heuristic. Capped at `layering_max_chains`.
- **Smurfing**: sliding-window threshold + amount-CV check on edges below `smurfing_threshold` (₹2L default).
- **Shell funnels**: flow imbalance ratio + branch diversity over candidate shell nodes.
- **Dormant activation**: per-account daily aggregate → find gaps ≥ `dormant_threshold_days` → Z-score post-gap activity.
- **Profile mismatch**: behavioural rules per entity type compared against KYC declared profile.

Every detector reads its thresholds from `ConfigStore`; nothing is hardcoded except defaults.

### 2.4 ML stack

- **XGBoost (30 features)** is the primary edge classifier. Features cover amount stats, endpoint structure, channel/rail mix, near-threshold proximity, night/weekend ratios, and SCC membership. Trained with `scale_pos_weight` for class imbalance; 80/20 stratified split.
  - The `neighbor_fraud_density` feature (1-hop neighbour fraud rate) was deliberately *removed* because it created circular reasoning on heavily-clustered synthetic fraud rings. Production banks cannot trust a model that flags B because A nearby was already flagged.
- **GraphSAGE GNN** is the structural baseline. Two-layer SAGEConv on the same graph, edge classification head concatenates sender/receiver embeddings. Hidden dim 64, ~60 epochs. Trains in seconds on CPU.
- **IEEE-CIS tabular XGBoost** is a separate baseline trained on the public Kaggle dataset the evaluators suggested. ~400 anonymised features, no graph structure. Surfaced under "ML → Baselines" tab. Honest because IEEE-CIS is credit-card fraud, not fund flow.
- **SHAP TreeExplainer** computes per-edge attributions on the XGBoost model. Cached background sample stashed in the model pickle so the explainer is cheap to instantiate at request time.

### 2.5 Workflow + persistence

- **SQLite** (`data/rudra.db`) holds cases, audit log, and config. Single file — no Postgres dependency.
- **Hash-chain audit log**: each audit entry stores `prev_hash` (previous entry's hash) and `this_hash = SHA-256(prev_hash || canonical_entry_json)`. Verification walks the chain and recomputes — any insert/edit/delete is detectable. Test `test_hash_chain_detects_tampering` proves this.
- **Incident clustering** (union-find on entity overlap) collapses ~290 raw alerts into ~6 actionable incidents.

### 2.6 RBAC

| Action | Investigator | Supervisor | Admin |
|---|:-:|:-:|:-:|
| Open / note case | ✓ | ✓ | ✓ |
| Move case to INVESTIGATING / ESCALATED | ✓ | ✓ | ✓ |
| File SAR / Dismiss case | — | ✓ | ✓ |
| Download FIU package | ✓ | ✓ | ✓ |
| Verify audit chain | — | ✓ | ✓ |
| Read config | ✓ | ✓ | ✓ |
| Write config / Re-run detection | — | — | ✓ |
| Retrain ML | — | — | ✓ |
| Re-run pipeline | — | — | ✓ |

Demo gate via `X-User-Role` header. Production replaces with the bank's IDP (Okta / Keycloak / AD).

### 2.7 FIU evidence package

One zip download contains everything a compliance team needs to file a Suspicious Transaction Report with FIU-IND:

- `evidence_summary.md` — human-readable case overview
- `STR.xml` — FIU-IND format STR XML, with PII fields explicitly marked `REDACTED` per DPDP Act
- `SAR_<alert_id>.pdf` — formal SAR document
- `subgraph.json` — NetworkX-format subgraph export
- `transaction_chain.csv` — every txn in the case, chronologically
- `pmla_citations.txt` — relevant PMLA / RBI sections per pattern
- `case_audit_log.json` — full audit trail with hash chain

### 2.8 DPI integrations (mocked, schema-accurate)

- **Account Aggregator**: `/aa/consent`, `/aa/pull/{handle}`, `/aa/revoke/{handle}` — issues + validates consent handles, returns transaction-shaped data. Schema matches what Setu / OneMoney / Anumati return. Every response carries a `_mock_disclaimer` field.
- **DiliSense KYC**: `/kyc/screen?name=&entity_type=` — sanctions/PEP/adverse-media check. Deterministic per name in mock mode (Thunder Bolt Exports → CRITICAL OFAC SDN hit).

### 2.9 LLM copilot

- **Gemini 2.0 Flash** with proper function-calling protocol. Four tools: `trace_funds`, `find_cycles`, `explain_alert`, `get_profile_delta`. Multi-turn (up to 3 rounds) so the model can compose tool calls.
- **Local fallback** kicks in when no API key is set or Gemini errors — intent-routing regex matches the query to a tool, runs the tool, formats the result. Honest about what it is.

---

## 3. Data flow

### Cold start
1. Backend boots → checks `data/transactions.csv` → if absent, runs `src/run_pipeline.py`.
2. Pipeline: generate synthetic txns → build graph → run detectors → cluster incidents → train XGBoost + GraphSAGE → generate SAR PDFs → persist.
3. Backend loads everything into memory; opens a case row in SQLite for every alert that doesn't have one yet.

### Investigator workflow (real session)
1. Investigator opens `/incidents` → sees 6 clustered incidents (down from 290 alerts).
2. Picks the CRITICAL incident, sees its 237 underlying alerts and 55 involved entities.
3. Clicks "Trace primary alert" → `/journey?alert=ALERT_CIRC_0001` → fund-flow Sankey + transaction timeline + red flags + SHAP "why this score".
4. Reviews the audit trail on the Cases page. Verifies the hash chain (Supervisor button) — gets back "chain intact, head hash 1c3ce68a…".
5. Adds an investigation note, moves the case to INVESTIGATING (allowed for any role), then asks a Supervisor to file SAR.
6. Supervisor clicks "File SAR" → status moves to SAR_FILED. Downloads FIU zip — gets STR.xml + SAR PDF + subgraph + chain CSV.

### Real-time monitoring
1. Each incoming transaction (simulated via `/api/live/inject`) is fed into `live_scoring.score_live_txn`.
2. Feature row is computed from the *current* graph state, model produces a probability, latency is recorded.
3. Frontend Live page polls every second; per-transaction latency tile shows mean + p95 over last 500 samples.

---

## 4. Performance

Measured on a 2024 M1 MacBook Air (Python 3.12, no GPU):

| Operation | Time |
|---|---|
| Full graph rebuild (80 nodes, 1,800 edges) | ~290 ms |
| 6 detectors (all in parallel) | ~250 ms |
| ML scoring (every edge in the graph) | ~400 ms |
| Risk scoring (betweenness + degree centrality) | ~35 ms |
| **End-to-end pipeline** | **~1.1 s** |
| Per-transaction ML score | mean 0.56 ms · p95 0.64 ms · p99 0.83 ms |

`/api/benchmark/latency` exposes this live. The Dashboard surfaces "78,081× faster than T+1" — that's `86,400,000 ms / 1,106 ms`.

---

## 5. Honest limitations

- F1 = 0.994 on synthetic is artificially high because the embedded patterns are clean by construction. Real-world AML F1 is ~0.72 on IBM AML-HI-Large (SOTA FraudGT) and ~0.42 with an XGBoost baseline.
- IBM AML / PaySim / IEEE-CIS variants are *trainable* but not bundled (datasets are 100s of MB). Place the CSVs under `data/real/<dataset>/` and re-run the pipeline.
- Streaming is simulated (`/api/live/inject` rotates random pairs). A production deployment would consume from Kafka + Pathway as outlined in the PoA.
- Account Aggregator + DiliSense are mocks with realistic shapes. Production calls Sahamati-licensed AAs and the DiliSense API directly.
- Single-user mode in the demo; RBAC is via header switcher, not real auth.

---

## 6. Tech choices — why each

| Choice | Reason |
|---|---|
| **NetworkX** over Neo4j | Pure Python, no daemon, sub-millisecond queries on a 1,800-edge graph. Migrating to Neo4j is a search-and-replace job if scale demands it. |
| **XGBoost** as primary | Industry-standard tabular fraud-detection model. Fast inference (<1 ms per edge), no GPU needed, exact SHAP explanations via TreeExplainer. |
| **GraphSAGE** as secondary | Lightest credible GNN. Gives the model "see your neighbours" capability without the training cost of FraudGT. Honest baseline for our PoA's GNN claim. |
| **SQLite** over Postgres | Single-file, zero-config, atomic per-statement. Right call for a hackathon POC. Schema is portable to Postgres in production. |
| **SHA-256 hash chain** | Tamper-evidence is what regulators ask for. Implementing it client-side is one function and one column; outsourcing to a real blockchain would be overkill. |
| **FastAPI** over Flask | Native async, automatic OpenAPI docs at `/docs`, dependency injection (used for RBAC). |
| **Gemini 2.0 Flash** over OpenAI | Free tier, fast, supports function calling. Easy to swap to OpenAI / Anthropic if the bank prefers. |
| **Tailwind 4** over component library | Faster iteration in a demo; no bundle bloat from MUI / shadcn. |
