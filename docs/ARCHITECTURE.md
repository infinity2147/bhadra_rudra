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
│  /api/stream/recent     /api/copilot/query       /api/kyc/screen           │
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
│  ML          ┌─ ml_model.py          (XGBoost edge classifier, F2 point)   │
│              ├─ gnn_model.py         (GraphSAGE: edge-fusion + AUPRC sel.)  │
│              ├─ ensemble_model.py    (XGB + SAGE + GAT, LR meta-learner)    │
│              ├─ ml_alert_generator.py(ML alerts + rule/ML confidence tiers) │
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
│  AI          └─ llm_copilot.py       (Claude Haiku + local fallback)       │
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

- Single ASGI app (`backend/main.py`) with 62 endpoints.
- State loaded once at startup for the active dataset (`RUDRA_DATASET`, default `ibm_aml`): transactions + graph + alerts + incidents + case store + config store + ML bundle. The backend refuses to start if the variant's artefacts are missing, printing the exact regenerate command — it never silently serves stale or fake data.
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
  - **Operating point: F2, not F1.** The decision threshold is chosen to maximise F2 (recall weighted 2× precision) via a shared `fbeta_optimal_threshold` helper, because a missed launderer costs far more than an analyst review. This is the single biggest recall lever (XGB recall 0.48 → 0.67).
  - A transit-ratio / velocity-burst feature set was trialled and **reverted** — it gave no AUPRC gain on the held-out set (0.661 → 0.662; a clean test since trees are scale-invariant) because the signals are too sparse on real data. Documented so it isn't re-attempted blind.
- **GraphSAGE GNN** is a full co-detector, not just a structural baseline. 3 SAGEConv layers (3-hop receptive field) + max aggregation (resists neighbour-dilution camouflage), seeded for reproducibility, with two changes that lifted its AUPRC from ~0.13 to **0.62**: (a) the edge head **fuses the per-edge transaction features** alongside the two endpoint embeddings — node embeddings alone were blind to amount/rail/temporal signal; (b) early-stopping / model selection on **validation AUPRC** rather than threshold-dependent F1.
- **Stacked ensemble** = XGBoost + GraphSAGE + GAT base models with a logistic-regression meta-learner trained on out-of-fold predictions. AUPRC 0.661, recall 0.703; the meta-learner now genuinely weights all three bases.
- **Temporal Graph Network (TGN)** adds the axis the static models lack — the *time-evolution* of each account. Per Rossi et al. (2020): a GRU per-node **memory** updated by every transaction, a temporal-attention embedding (`TransformerConv` over recent neighbours), and a single-logit **fraud** decoder. It is supervised on `is_fraud` (not link-existence), trained/evaluated on a **strict chronological split** (train on the past, test on the future — no look-ahead leakage), selected on val AUPRC at the shared F2 operating point. On IBM AML it reaches **AUPRC 0.615**, competitive with GraphSAGE and reported side-by-side — no claim it beats the static models; its value is the forward-looking "which account offends next" signal. Optional (needs torch/PyG), trained after XGBoost, served from persisted JSON (weights never reloaded at runtime).
- **IEEE-CIS tabular XGBoost** is a separate baseline trained on the public Kaggle dataset the evaluators suggested. ~400 anonymised features, no graph structure. Surfaced under "ML → Baselines" tab. Honest because IEEE-CIS is credit-card fraud, not fund flow.
- **SHAP TreeExplainer** computes per-edge attributions on the XGBoost model. Cached background sample stashed in the model pickle so the explainer is cheap to instantiate at request time.

### 2.4a Detection architecture — ML detects, rules corroborate (tiered)

The ML model is the **primary detector**, not a decoration. Every edge it scores above the F2 threshold becomes a first-class alert via `ml_alert_generator.py`; the rule engine is the **corroboration + explanation** lane. The two are combined as a **confidence tier, never an averaged score** (averaging a 67%-recall probability with a ~3%-precision rule-only signal helps nothing — measured):

- **Tier 1** — ML and a typology rule agree on an entity. Highest precision (~74% at the entity level), priority queue, carries the rule's typology narrative for the STR.
- **Tier 2** — ML only. The recall workhorse.
- **Tier 3** — a clean high-precision typology (circular / layering / recruiter) with no ML corroboration, kept for its narrative.
- **Suppressed** — noisy rule-only alerts (smurfing / profile-mismatch / shell-funnel firing without ML agreement), which alone are near-noise.

Why this and not pure rules: on the IBM AML benchmark ~84% of laundering is structureless single transfers (degree ≤ 2, not in any cycle/chain), so graph-pattern rules have a hard recall ceiling (~17%). The ML edge classifier catches ~67%. Fusing the two lifts end-to-end fraud-entity recall from ~17% to ~67% while every alert stays explainable: SHAP attributions for ML alerts (served lazily by `/api/alerts/{id}/explain`), typology narratives for rule alerts.

### 2.4b Post-detection: explain, anticipate, link

Three lanes sit *on top of* detection — they change no recall/precision and add no alerts:

- **Forensic RCA dossier** (`rca_engine.py`, `GET /api/incidents/{id}/rca`). For a clustered incident it reconstructs the fund journey (wrapping `fund_tracer`), diagnoses the **root-cause control gap** — either from the matched FATF typology or, for the ~97% of incidents that are `ML-Detected Anomaly` (no rule typology), **inferred** from behavioural signals (cycle → layering; shell + fan-in → funnel; dormant reactivation; sub-threshold structuring) — and emits **prescriptive recommendations** plus a plain-language narrative. A pure, read-only layer; signals are computed from the uncapped transaction set.
- **Prediction** — the TGN (§2.4) is the forward-looking lane: it ranks which accounts are likely to transact fraudulently next, surfaced as a watchlist on the Model Metrics page.
- **Collusion Rings** (`collusion_detector.py`, `GET /api/collusion/rings`). Union-find over accounts sharing a `device_id` / `ip` / `kyc_doc_hash` (transitive across identifier types) — it catches mule rings that never transact with each other, which the money-flow graph is structurally blind to. Because IBM AML has none of those identity fields, this runs on a **standalone synthetic identity dataset**, clearly labelled a synthetic-identity demo in API and UI; the detection logic is real, only the identifiers are synthetic, and the IBM AML pipeline is untouched. Missing identifier columns → zero rings (tested no-op).

### 2.5 Workflow + persistence

- **SQLite** (`data/rudra.db`) holds cases, audit log, and config. Single file — no Postgres dependency.
- **Hash-chain audit log**: each audit entry stores `prev_hash` (previous entry's hash) and `this_hash = SHA-256(prev_hash || canonical_entry_json)`. Verification walks the chain and recomputes — any insert/edit/delete is detectable. Test `test_hash_chain_detects_tampering` proves this.
- **Incident clustering** (union-find on entity overlap, with high-degree hub entities excluded as bridges so independent rings don't snowball) collapses the tiered alert set into actionable incidents. On the IBM AML 100k benchmark the ML-led alert set (~9.3k alerts at the recall-favouring threshold) collapses to ~5.3k incidents; the value is that corroborated rings merge into one investigable case rather than scattering across the queue. Tier ordering then floats the highest-confidence (ML + rule) cases to the top.

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

- **Claude (Haiku)** with proper tool-calling protocol. Four tools: `trace_funds`, `find_cycles`, `explain_alert`, `get_profile_delta`. Multi-turn (up to 3 rounds) so the model can compose tool calls. Enabled by `ANTHROPIC_API_KEY`.
- **Local fallback** kicks in when no API key is set or Claude errors — intent-routing regex matches the query to a tool, runs the tool, formats the result. Honest about what it is.

---

## 3. Data flow

### Cold start
1. Run `python src/run_pipeline.py --dataset ibm_aml` once — this single command does everything (no need to run `train_ibm_aml.py` separately). Pipeline: load the dataset (stratified 100k IBM AML sample, cached) → build graph → run all six rule detectors → train XGBoost (+ GraphSAGE + stacked ensemble when PyTorch is installed) → **generate ML alerts above the F2 threshold, fuse + tier them with the rule alerts (`ml_alert_generator.fuse_ml_alerts`), then cluster into incidents** → persist artefacts under `data/<variant>/` and `data/ml/<variant>/`. (SAR PDFs are generated on-request by the backend, not in the pipeline.)
2. Backend boots bound to `RUDRA_DATASET` (default `ibm_aml`); if the variant's required artefacts are missing it raises with the exact regenerate command rather than starting on empty/fake state.
3. Backend loads everything into memory; opens a case row in SQLite for every alert that doesn't have one yet.

### Investigator workflow (real session)
1. Investigator opens `/cases` → the alert queue is sorted **tier-first** (Tier 1 = ML + rule agreement at the top), filterable by tier and ML-score band, or grouped into clustered incidents on `/incidents`.
2. Picks a top Tier-1 case, sees the corroborating typology ("corroborated by: Circular Transaction, …") and the involved entities.
3. Clicks "Trace primary alert" → `/journey?alert=…` → fund-flow Sankey + transaction timeline + red flags + SHAP "why this score" (computed on demand for ML alerts).
4. Reviews the audit trail on the Cases page. Verifies the hash chain (Supervisor button) — gets back "chain intact, head hash 1c3ce68a…".
5. Adds an investigation note, moves the case to INVESTIGATING (allowed for any role), then asks a Supervisor to file SAR.
6. Supervisor clicks "File SAR" → status moves to SAR_FILED. Downloads FIU zip — gets STR.xml + SAR PDF + subgraph + chain CSV.

### Real-time monitoring
1. The Live page replays the loaded dataset onto the ingest bus via `POST /api/stream/replay`; the `StreamIngestor` consumer (Kafka or in-process) picks up each event.
2. Each event is scored through `live_scoring.score_live_txn` — feature row computed from the *current* graph state, model produces a probability, latency recorded, and honest signal flags (cycle / near-threshold / shell / off-hours) derived from the feature row.
3. Scored events land in a 500-deep ring buffer; the Live page polls `GET /api/stream/recent` every second and shows the per-transaction latency tile (mean + p95 over last 500 samples). No synthetic transactions are generated.

---

## 4. Performance

Two numbers matter, and they are different things:

**Per-transaction streaming latency** — the real-time claim. Every transaction is scored against the *current* graph state on the CPU, no GPU:

| Operation | Time |
|---|---|
| Per-transaction ML score | sub-millisecond (mean ~0.56 ms · p95 ~0.64 ms · p99 ~0.83 ms) |

`GET /api/benchmark/latency` exposes this live, and the Dashboard contrasts it with the 24-hour T+1 batch cycle the RBI 2023 FRM framework rules out. This is the number that lets an investigator freeze funds before they leave the bank.

**One-time batch pipeline** — building the world from cold. Scale depends on the dataset: the synthetic set is ~80 nodes / ~1,800 edges (end-to-end in ~1 s); the IBM AML 100k benchmark is **~119k nodes / 87,772 edges**, where the full rebuild + all six detectors + XGBoost training takes a few minutes, and training the GraphSAGE + GAT + ensemble adds a few more (CPU). The streaming path then keeps the in-memory graph current with incremental `add_transaction` updates, so the batch cost is paid once at startup, not per transaction.

---

## 5. Honest limitations

- The honest ranking quality is **AUC = 0.927 / Average-Precision = 0.661** on the IBM AML 100k benchmark with single-CPU XGBoost (GraphSAGE AUPRC 0.623, ensemble 0.661). At the F1-optimal threshold the model scores **F1 ≈ 0.62** — the strong-baseline number for single-CPU training; the published SOTA (FraudGT + BDH) is ~0.72 and needs multi-GPU training over days. We deliberately operate at the **recall-favouring F2 point** (recall 0.67, precision 0.41) rather than F1-optimal, so the reported F1 is lower *by design* while recall — the metric that matters for catching laundering — is materially higher. The synthetic F1 of ~0.99 is artificially high because the embedded patterns are clean by construction and is not a number we stand behind.
- **Rule heuristics alone have low recall (~17%) on real data** — most IBM AML laundering is structureless single transfers with no ring/chain structure for a graph rule to match. This is expected, and is exactly why detection is ML-led with rules as the corroboration/explanation lane (§2.4a); the fused system reaches ~67% recall.
- **ML auto-alert volume is high at β=2** — scoring every edge above the recall-favouring threshold yields ~9k alerts / ~5k incidents on the 100k sample. Tier sorting handles triage; the threshold is a single configurable knob to trade recall for fewer, higher-precision alerts.
- IBM AML / PaySim / IEEE-CIS variants are *trainable* but not bundled (datasets are 100s of MB). Place the CSVs under `data/real/<dataset>/` and re-run the pipeline.
- Streaming replays the loaded dataset onto the ingest bus (Kafka when a broker is reachable, in-process `asyncio.Queue` otherwise) — there is no live bank feed in the demo, but the transport, scoring, and ring buffer are the real production path. A production deployment swaps the replay source for the bank's live transaction bus + Pathway as outlined in the PoA.
- Account Aggregator + DiliSense are mocks with realistic shapes. Production calls Sahamati-licensed AAs and the DiliSense API directly.
- Single-user mode in the demo; RBAC is via header switcher, not real auth.
- The **TGN** is optional (needs torch/PyG) and served from persisted artefacts — no live A→B scoring this release; its metrics are reported side-by-side with the static models, not claimed to beat them (AUPRC ~0.615, competitive with GraphSAGE).
- **Collusion Rings** runs on a synthetic identity dataset because IBM AML has no device/IP/KYC fields — clearly labelled a synthetic-identity demo; the detection logic is real graph analysis, only the identifiers are synthetic, and the IBM AML pipeline is untouched.

---

## 6. Tech choices — why each

| Choice | Reason |
|---|---|
| **NetworkX** over Neo4j | Pure Python, no daemon, sub-millisecond queries on a 1,800-edge graph. Migrating to Neo4j is a search-and-replace job if scale demands it. |
| **XGBoost** as primary detector | Industry-standard tabular fraud-detection model. Fast inference (<1 ms per edge), no GPU needed, exact SHAP explanations via TreeExplainer. Generates first-class alerts at the F2 operating point. |
| **GraphSAGE + GAT** co-detectors | "See your neighbours" relational signal XGBoost lacks, without FraudGT's training cost. Edge-feature fusion + AUPRC selection make them competitive (AUPRC ~0.62), not just a baseline; they feed the stacked ensemble. |
| **TGN** for prediction | Adds the *temporal* axis the static models lack — a per-account memory of behaviour over time. Answers "which account offends next", evaluated leak-free on a chronological split. Optional and served from JSON, so it never complicates the runtime path. |
| **Confidence tiers** over a blended score | Averaging a high-recall ML probability with a low-precision rule signal helps nothing (measured). Tiering by ML/rule agreement keeps each lane's strength: ML recall, rule precision + auditability. |
| **SQLite** over Postgres | Single-file, zero-config, atomic per-statement. Right call for a hackathon POC. Schema is portable to Postgres in production. |
| **SHA-256 hash chain** | Tamper-evidence is what regulators ask for. Implementing it client-side is one function and one column; outsourcing to a real blockchain would be overkill. |
| **FastAPI** over Flask | Native async, automatic OpenAPI docs at `/docs`, dependency injection (used for RBAC). |
| **Claude Haiku** over OpenAI | Fast, low-cost, strong tool-calling. The copilot is one client class (`llm_copilot.py`) — easy to swap providers if the bank prefers. |
| **Tailwind 4** over component library | Faster iteration in a demo; no bundle bloat from MUI / shadcn. |
