# RUDRA — Feature Analysis & Improvement Plan

> A critical, feature-by-feature review of everything that ships in this repo.
> For each feature: what it does, how to exercise it, and an honest verdict —
> REAL / WEAK / THEATER / DUPLICATE / MISSING.
>
> The last section is a prioritised work list for the next 24+ hours.
>
> Generated: 2026-05-29. Targets repo at commit-tip on `main`.

---

## How to read this document

| Verdict | Meaning |
|---|---|
| **REAL** | Substantive implementation. Would work in production with creds/data. |
| **WEAK** | Functional but limited — hardcoded thresholds, undertrained, or signal-poor. Listed in the improvement plan. |
| **THEATER** | Looks impressive but doesn't carry information. Hardcoded text, fake file references, or label-leaking pre-sets. Must fix. |
| **DUPLICATE** | Two implementations of the same thing — drift risk, single-source-of-truth violation. |
| **MISSING** | Claimed in README/code comments but not actually present. |

Each subsection lists the **file path:line**, **endpoint(s)**, **UI page**, and **env vars** so anyone can exercise it directly.

---

## Table of contents

1. [Data layer](#1-data-layer)
2. [Graph engine](#2-graph-engine)
3. [Detection layer (6 detectors)](#3-detection-layer)
4. [ML stack](#4-ml-stack)
5. [Streaming (Kafka + Pathway)](#5-streaming)
6. [Case workflow + audit log](#6-case-workflow)
7. [FIU + SAR reporting](#7-fiu--sar-reporting)
8. [DPI integrations (AA + DiliSense)](#8-dpi-integrations)
9. [Investigation tools](#9-investigation-tools)
10. [Analytics](#10-analytics)
11. [AI Copilot](#11-ai-copilot)
12. [Config + ops](#12-config--ops)
13. [Frontend pages](#13-frontend-pages)
14. [Prioritised improvement plan](#14-prioritised-improvement-plan)

---

## 1. Data layer

### 1.1 Synthetic transaction generator
**Files**: `src/data_generator.py`
**What it does**: Generates ~2.7k transactions across 82 entities (24 businesses, 50 individuals, 8 shell companies) with 6 embedded fraud patterns (circular, layering, smurfing, shell-funnel, dormant, profile-mismatch).
**How to use**:
- CLI: `python src/run_pipeline.py`
- API: `POST /api/pipeline/run` (ADMIN role required)
**Output**: `data/transactions.csv`, `data/fraud_cases.json`, `data/entities.json`.
**Verdict**: **REAL but with label leakage**. Shell-company entities are pre-tagged with `type="shell_company"` AND a hardcoded `risk_score=0.6–0.95`. The detector then uses both as input signals. Explains why synthetic F1 reaches 0.99 — the labels are partially handed to the model.
**Fix**: Strip the `risk_score` pre-set on entities; let the detector compute it from graph behaviour only. Keep the `type` (shells must be discoverable somehow).

### 1.2 IBM AML real-dataset loader
**Files**: `src/real_data_loader.py:73`
**What it does**: Loads `data/real/ibm_aml/HI-Small_Trans.csv` (~4.2M rows when un-sampled, 100k stratified by default), normalises to our internal schema, infers entity types (shell/business/individual) from graph behaviour.
**How to use**:
1. `kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -f HI-Small_Trans.csv -p data/real/ibm_aml/ --unzip`
2. `python src/train_ibm_aml.py`
**Verdict**: **REAL**. Currency normalisation (long names → ISO codes) was fixed this session. Entity-type inference is heuristic (out-fan ≥ 5 and ≥ 3× in-fan → shell) but reasonable.

### 1.3 PaySim loader
**Files**: `src/real_data_loader.py:212`
**Status**: Stub-ready. Auto-trains if `data/real/paysim/*.csv` is present. **Not used by default** because the file isn't downloaded.
**Verdict**: REAL (the loader is correct) but **MISSING data** in the current shipping repo.

### 1.4 IEEE-CIS tabular loader
**Files**: `src/real_data_loader.py:286`, `src/tabular_baseline.py`
**What it does**: Loads the Kaggle IEEE-CIS Fraud Detection dataset (tabular only, no graph), trains an XGBoost baseline on ~400 anonymised features.
**Endpoint**: `GET /api/ml/tabular`
**UI page**: `/model` → "Tabular Baseline" tab.
**Verdict**: REAL but **MISSING data**. The page shows "dataset not present" until you download it.

---

## 2. Graph engine

### 2.1 NetworkX directed-weighted graph builder
**Files**: `src/graph_engine.py`
**What it does**: Aggregates per-transaction rows into `(sender → receiver)` edges with attributes: `total_amount`, `avg_amount`, `std_amount`, `min_amount`, `max_amount`, `transaction_count`, `fraud_count`, `first_seen`, `last_seen`, `rail_mix`, `channel_mix`.
**How to use**: Built on backend startup (`FundFlowGraph().build_graph(df)`); exposed via `/api/graph`.
**UI page**: `/graph` (full network), `/graph/{entity_id}` (N-hop subgraph).
**Verdict**: **REAL**. Solid. Edges carry the right aggregations for downstream detection + ML.

---

## 3. Detection layer

### 3.1 Circular transaction detector
**Files**: `src/fraud_detector.py:34`
**Algorithm**: Johnson's `simple_cycles` inside each SCC ≥ 3, pre-filtered by log10 amount bucket (only enumerate cycles whose edges fall in the same log-amount bin — matches AML round-trip signature where every hop carries a similar tranche).
**Endpoint**: `GET /api/patterns/circular_transaction`
**UI page**: `/patterns` → Circular tab.
**Verdict**: **REAL — possibly the cleverest piece of the codebase**. SCC pre-filter keeps Johnson's tractable on dense graphs; log-bucket pruning targets the actual fraud signature. Configurable via `circular_amount_tolerance`, `circular_min_total_flow`, `circular_max_cycle_length`, `circular_max_alerts` in ConfigStore.

### 3.2 Rapid layering detector
**Files**: `src/fraud_detector.py:165`
**Algorithm**: BFS from each node up to depth 7, looking for chains of length ≥ `layering_min_chain_length` (default 3) with monotonically decreasing amounts.
**Endpoint**: `GET /api/patterns/rapid_layering`
**Verdict**: **WEAK**.
- BFS branching factor hard-capped at 5 successors per node (line 271) → misses many chains on dense real graphs.
- "Decreasing amount" check (`amounts[i] >= amounts[i+1] * 0.85`) is brittle — production layering can be roughly equal.
- Score formula is hand-tuned (`0.7 base + 0.15 if decreasing + 0.1 if shell + 0.05 if long`).
**Fix**: replace BFS cap with config-driven breadth/depth; drop hand-tuned score and use the XGBoost score for the chain's max-amount edge.

### 3.3 Smurfing / structuring detector
**Files**: `src/fraud_detector.py:279`
**Algorithm**: Edges with avg between 0.7× and 1.0× the ₹2L (USD $9.5k) reporting threshold, low std/avg variability, grouped by common sender with `amount_spread < cluster_tolerance`.
**Endpoint**: `GET /api/patterns/smurfing`
**Verdict**: **WEAK**.
- Only checks amount clustering, not temporal clustering. Misses smurfing-by-bursts.
- Threshold is fixed at ₹2L; for IBM AML it's normalised to $9.5k via currency check, but everything else uses INR thresholds.
**Fix**: add temporal clustering — N txns in M minutes below threshold (configurable).

### 3.4 Shell-company funnel detector
**Files**: `src/fraud_detector.py:345`
**Algorithm**: For each shell/business node with `in_degree ≥ 3`, compute `|in_strength - out_strength| / total_flow`; if imbalance > 0.7, flag.
**Endpoint**: `GET /api/patterns/shell_funnel`
**Verdict**: **WEAK**.
- Misses pass-through funnels where `in_strength ≈ out_strength` but the entity holds nothing for long.
- Doesn't check out-edge lifetime (a real shell empties fast).
**Fix**: add `pass_through_ratio = min(in,out)/max(in,out)` and `avg_holding_seconds` features.

### 3.5 Dormant-account activation detector
**Files**: `src/advanced_detectors.py:15`
**Algorithm**: For each node with ≥ 3 transactions, find gaps ≥ 30 days, compute Z-score of post-gap activity vs pre-gap.
**Endpoint**: `GET /api/patterns/dormant_activation`
**Verdict**: **REAL but coarse**. Z-score logic is sound. Only flags accounts with ≥ 3 transactions — small datasets miss this. Reads `dormant_threshold_days` + `dormant_z_score_threshold` from ConfigStore (good).

### 3.6 Profile-mismatch detector
**Files**: `src/advanced_detectors.py:103`
**Algorithm**: For each entity, check declared `type` (individual/business/shell) against transaction behaviour: avg amount, purpose codes, branch diversity, night-hour ratio.
**Endpoint**: `GET /api/patterns/profile_mismatch`
**Verdict**: **WEAK + THEATER** — **highest-priority fix in the detection layer**.
- **Hardcoded thresholds that bypass ConfigStore**: `avg > ₹10L`, `total > ₹5Cr`, `night > 40%`, `branches > 4`, `score = mismatches × 0.2`. The audit report missed these.
- Score is just `count(mismatches) × 0.2` — pure rule-counting.
**Fix**:
1. Move every threshold to `ConfigStore` keys (`profile_max_individual_avg`, `profile_max_volume`, `profile_max_night_ratio`, `profile_max_branches`).
2. Replace `count × 0.2` with the average XGBoost score across this entity's edges.

### 3.7 Composite entity risk score
**Files**: `src/fraud_detector.py:425`
**Algorithm**: Per-node weighted sum: `0.4*(type=shell) + 0.3*degree_centrality + 0.2*betweenness + 0.15*flow_imbalance + 0.3*fraud_edge_count`.
**Endpoint**: surfaced via `/api/entities` and risk dist on `/api/dashboard`.
**Verdict**: **WEAK**.
- Weights are hand-guessed.
- Adds 0.4 for `type=shell` *and* uses fraud-edge count → information leakage on synthetic data.
- Sum can exceed 1.0 but is clipped — so different-magnitude signals collide.
**Fix**: train a small LogisticRegression on (per-node features → was-in-flagged-incident) and use its learned weights.

### 3.8 Incident clustering
**Files**: `src/incident_clustering.py`
**Algorithm**: Union-find over alerts that share ≥ 1 entity. 290 raw alerts → ~5–6 actionable incidents.
**Endpoint**: `GET /api/incidents`, `GET /api/incidents/{id}`
**UI page**: `/incidents`
**Verdict**: **REAL**. Straight union-find on entity-set overlap. The right call — investigators look at incidents, not raw alerts.

---

## 4. ML stack

### 4.1 XGBoost edge classifier
**Files**: `src/ml_model.py`
**What it does**: 30 engineered features per edge → fraud probability via XGBoost. Stratified 80/20 split, F1-optimal threshold from PR curve.
**Endpoint**: `GET /api/ml/metrics?variant=synthetic|ibm_aml`
**UI page**: `/model`
**Numbers**:
- Synthetic: F1 0.99, AUC 0.99 *(inflated — see leakage in §1.1)*
- IBM AML 100k: **F1 0.617, AUC 0.927, P 0.851, R 0.484**
**Verdict**: **REAL**. The IBM AML numbers are the credible ones.

### 4.2 GraphSAGE GNN baseline
**Files**: `src/gnn_model.py`
**What it does**: Two SAGEConv layers + edge-classification MLP head. Trained 60 epochs on the synthetic graph, 60 epochs on IBM AML.
**Numbers (IBM AML)**: F1 0.315, AUC 0.754. **Undertrained.**
**Verdict**: **WEAK on real data**. Architecture is right; training budget is too small for a 119k-node graph. Needs 200+ epochs or NeighborLoader-based mini-batching for honest performance.

### 4.3 GAT (Graph Attention Network)
**Files**: `src/ensemble_model.py:_build_gat_model`
**What it does**: 4-head GATv2Conv + edge-classification MLP. Bundled into the ensemble.
**Numbers (IBM AML)**: F1 0.323, AUC 0.745. Same undertraining issue.
**Verdict**: **WEAK**. Same fix as SAGE.

### 4.4 Stacked ensemble (XGB + SAGE + GAT + LR meta)
**Files**: `src/ensemble_model.py`
**What it does**: 3-fold out-of-fold stacking. Each base model produces an OOF prediction per edge, then LogisticRegression learns to weight them.
**Endpoint**: `GET /api/ml/ensemble?variant=ibm_aml`, `GET /api/ml/ensemble/edge_scores`
**UI page**: should be added to `/model`.
**Numbers (IBM AML)**: F1 0.624, AUC 0.927. Meta-learner weights: XGB +6.48, SAGE +1.71, GAT +0.23 — correctly down-weights the weak GNNs.
**Verdict**: **REAL**. Honest stacking — no in-fold leakage. The ensemble doesn't beat XGB-alone by much, *because* the GNNs are weak. Fixing GNN training (§4.2) is the highest-leverage move to make the ensemble actually earn its keep.

### 4.5 SHAP local explanations
**Files**: `src/shap_explainer.py`
**What it does**: TreeExplainer per-edge SHAP values, ranked by absolute contribution, with human-readable narrative bullets ("Total flow raised the score, SHAP +0.91").
**Endpoint**: `GET /api/alerts/{alert_id}/explain`
**UI**: Cases page → "Explain" panel.
**Verdict**: **REAL**. Exact and fast (tree models). Narrative labels in `_FEATURE_LABEL` are good.

### 4.6 Live per-transaction scoring
**Files**: `src/live_scoring.py`
**What it does**: Feature extraction + model.predict_proba on a single new transaction, with µs-level latency measurement (feature extraction + inference).
**Endpoint**: `GET /api/live/inject?count=N` (simulator), or via Kafka stream (real).
**UI page**: `/live`
**Verdict**: **REAL**, but `_build_live_features` (lines 30–156) is a **partial duplicate** of `ml_model.extract_features` (lines 114–224) — same 30 features, different code path. Drift risk.

### 4.7 Pipeline latency benchmark
**Files**: `src/live_scoring.py:195`
**What it does**: Times every pipeline stage (graph rebuild, detectors, dormant, ML scoring, risk scoring) over real data. Compares against T+1 (24h) baseline.
**Endpoint**: `GET /api/benchmark/latency`
**Verdict**: **REAL**. The "78,000× faster than T+1" claim comes from this. Honest measurement.

### 4.8 ML model retrain
**Endpoint**: `POST /api/ml/retrain` (ADMIN)
**Verdict**: REAL. Retrains XGBoost only — doesn't refresh the ensemble. Acceptable for hot-fix retraining; full ensemble refresh needs `python src/train_ibm_aml.py`.

### 4.9 Model variant listing
**Endpoint**: `GET /api/ml/variants`
**What it does**: Lists every trained model variant (`synthetic`, `ibm_aml`, future `paysim`) with summary metrics.
**Verdict**: REAL.

---

## 5. Streaming

### 5.1 Kafka broker (docker-compose)
**Files**: `docker-compose.yml` (kafka service)
**What it does**: Single-node Bitnami Kafka 3.7 in KRaft mode (no Zookeeper). Internal listener `kafka:9092` for compose services, external listener `localhost:29092` for host-side producers.
**How to use**: `docker compose up kafka` (or just `docker compose up` — backend depends on `kafka` being healthy).
**Verdict**: **REAL**. Production would swap this for a managed cluster (MSK/Confluent).

### 5.2 aiokafka producer
**Files**: `src/streaming/kafka_producer.py`
**What it does**:
- Library mode: `replay_transactions(ingestor, df, rate=5, total=100)` — replays any DataFrame onto the stream.
- CLI mode: `python -m streaming.kafka_producer --source data/real/ibm_aml/HI-Small_Trans_100k_sampled.csv --rate 20 --total 1000` — replays a CSV from a separate terminal so judges can watch real Kafka traffic.
**Verdict**: **REAL**. Idempotent producer, gzip compression.

### 5.3 aiokafka consumer + ring buffer
**Files**: `src/streaming/ingestor.py`
**What it does**: Async consumer running as a FastAPI background task. Each incoming message is deserialised, scored through `score_live_txn`, and pushed to an in-memory deque (default 500 events).
**Endpoint**: `GET /api/stream/status`, `POST /api/stream/start|stop` (ADMIN), `GET /api/stream/recent?limit=N`
**Verdict**: **REAL**. End-to-end verified this session: produce → consume → score → buffer.

### 5.4 In-process fallback queue
**Files**: `src/streaming/ingestor.py` (inproc mode)
**What it does**: When no Kafka broker is reachable, falls back to `asyncio.Queue`. Same scoring path, same buffer. Means local dev / CI works without Docker.
**How to use**: Set `STREAM_BACKEND=inproc`, or just leave `STREAM_BACKEND=auto` (default) and don't run Kafka.
**Verdict**: **REAL** and **the right design** — same `score_live_txn` code path either way, so what the batch endpoint scores is what the stream scores.

### 5.5 Stream replay endpoint
**Endpoint**: `POST /api/stream/replay` (ADMIN) — body `{"rate": 50, "total": 100, "shuffle": true}`
**What it does**: Fans out one HTTP call into a stream of N events the running consumer scores.
**Verdict**: REAL. Useful demo affordance — no separate terminal needed.

### 5.6 Pathway windowed analytics
**Files**: `src/streaming/pathway_engine.py`
**What it does**: 5-min sliding window (1-min hop) of per-entity volume + count. Emits velocity alerts when an entity's window-volume crosses `VELOCITY_THRESHOLD_INR` (default ₹20L).
**Output**: `data/streaming/velocity_alerts.jsonl` (file-tailed) or Kafka topic `rudra.velocity_alerts` (if `PATHWAY_PUBLISH_KAFKA=1`).
**Endpoint**: `GET /api/stream/velocity_alerts`
**How to use**:
- `pip install pathway` (optional dep)
- `python -m streaming.pathway_engine` from a separate terminal
**Verdict**: **REAL but UNVERIFIED**. Code written against current Pathway API (`pw.temporal.sliding`, `pw.io.kafka.read`); I haven't actually run it against a live broker. Pathway's API has been evolving — `pw.this.timestamp.str.slice().dt.strptime()` may need adjustment.
**Fix**: smoke-test with `pip install pathway` and adjust the timestamp parsing if Pathway rejects it.

### 5.7 Simulator (`/api/live/inject`)
**Endpoint**: `GET /api/live/inject?count=N`
**What it does**: Generates N random (sender, receiver, amount) tuples and scores each through the real ML model with µs latency. Demo affordance — no Kafka needed.
**UI page**: `/live` → uses this for the demo feed.
**Verdict**: REAL (the scoring is real; the txn data is synthesised in-process). README is honest about it.

---

## 6. Case workflow

### 6.1 Case state machine
**Files**: `src/case_manager.py`
**States**: `OPEN → INVESTIGATING → {SAR_FILED | DISMISSED | ESCALATED}`
**Storage**: SQLite `data/rudra.db` (auto-migrates from old `cases.json` if present).
**Endpoints**:
- `GET /api/cases?status=`
- `GET /api/cases/{alert_id}`
- `POST /api/cases/{alert_id}/dispose` (body: `{status, note, author, assigned_to}`)
- `POST /api/cases/{alert_id}/note`
**UI page**: `/cases`
**Verdict**: **REAL**. Proper RBAC gating per-target-state (`case.move.SAR_FILED` etc).

### 6.2 SHA-256 hash-chain audit log
**Files**: `src/case_manager.py:53` (canonical JSON), `:68` (hash), `:365` (verify)
**What it does**: Each `audit_log` row stores `prev_hash` + `this_hash = SHA-256(prev_hash || canonical_entry_json)`. Canonical JSON is sorted-keys, no whitespace, UTF-8. Genesis hash = literal `"GENESIS"`.
**Endpoint**: `GET /api/cases/{alert_id}/verify` (SUPERVISOR+)
**Verdict**: **REAL and excellent**. Detects insertion, edit, deletion. Tested by `tests/test_case_store.py::test_hash_chain_detects_tampering`. Production-grade.

### 6.3 Bulk case open from alerts
**Files**: `src/case_manager.py:353`
**What it does**: On backend startup, walks all alerts and opens a case row for each (if not already present). Idempotent.
**Verdict**: REAL. The "290 alerts → 290 open cases" experience.

### 6.4 Case notes
**Endpoint**: `POST /api/cases/{id}/note` — appends to audit_log without changing status.
**Verdict**: REAL.

---

## 7. FIU + SAR reporting

### 7.1 SAR text generation
**Files**: `src/sar_generator.py`
**What it does**: Template-fills a 9-section SAR document with alert/entity/transaction data.
**Endpoint**: `GET /api/sar/generate/{alert_id}`
**UI page**: `/sar`
**Verdict**: **MIXED — REAL skeleton, THEATER fillings**.
- **Real**: section structure, transaction listing, fund-flow analysis, regulatory citations.
- **Theater**:
  - `[Bank Name — PSB]`, `[Compliance Officer]` placeholders that never get filled (hardcoded square brackets).
  - `_build_recommendations` is the same 10 hardcoded bullets for *every* alert.
  - `_build_supporting_docs` references files that **don't exist** (`{report_id}_network.png`, `{report_id}_risk.json`).
  - "Reasoning chain" per pattern is 4 hardcoded sentences — same text for every Circular alert.
**Fix**:
1. Make placeholders env-driven (`BANK_NAME`, `COMPLIANCE_OFFICER`).
2. Vary recommendations by pattern (already have `pattern → citations` dict; do same for `pattern → recommendations`).
3. Generate the missing network PNG or remove the references.

### 7.2 SAR PDF export (ReportLab)
**Files**: `src/sar_generator.py:189`
**What it does**: Renders the SAR text to a styled A4 PDF with colour-coded sections.
**Output**: `data/sar_reports/SAR-2026-XXXX.pdf`
**Verdict**: REAL (technically). PDF generation works. The *content* is limited by the theater issues in §7.1.

### 7.3 STR XML (FIU-IND format)
**Files**: `src/fiu_package.py:224`
**What it does**: Builds a schema-shaped FIU-IND STR XML with `<ReportHeader>`, `<ReportingEntity>`, `<SuspiciousActivity>`, `<Subjects>`, `<Transactions>`, `<GroundsForSuspicion>`, `<AuditTrail>`. PII fields (PAN, Aadhaar, Address) are explicitly `REDACTED` with `reason="DPDP_DataMinimisation_FillBeforeFiling"`.
**Verdict**: **REAL but FORMAT-SHAPED, NOT FORMAT-VALIDATED**. The structure follows FIU-IND Reporting Entity Guidelines but hasn't been validated against the actual FIU-IND XSD. Production submission would need that.
**Fix**: download the FIU-IND STR XSD, validate output via `lxml.etree.XMLSchema`.

### 7.4 PMLA / RBI citations
**Files**: `src/fiu_package.py:35` (PMLA_CITATIONS dict)
**What it does**: Per-pattern hardcoded citations — every Circular alert cites the same four sections (PMLA Section 3, Section 12, RBI KYC Chapter VII, FIU-IND round-tripping guidelines).
**Verdict**: **REAL** (the citations are real legal sections) **but TEMPLATED**. The same pattern produces the same citation set every time. Acceptable as-is — different alerts of the same pattern legitimately have the same legal basis. **Don't fix**.

### 7.5 FIU evidence package (zip)
**Files**: `src/fiu_package.py:335`
**What it does**: Bundles into a zip:
- `evidence_summary.md` — markdown case summary
- `STR.xml` — FIU-IND-shaped XML
- `SAR_<id>.pdf` — formal SAR
- `subgraph.json` — NetworkX node-link export of the fraud subgraph + neighbours
- `transaction_chain.csv` — every transaction in the case, in time order
- `pmla_citations.txt` — relevant legal sections
- `case_audit_log.json` — full audit trail
**Endpoint**: `GET /api/fiu/package/{alert_id}` (RBAC `fiu.download`)
**Verdict**: **REAL**. The whole zip is genuinely useful.

---

## 8. DPI integrations

### 8.1 Sahamati Account Aggregator client
**Files**: `src/integrations/aa_client.py`
**Env vars**: `SAHAMATI_CLIENT_ID`, `SAHAMATI_CLIENT_SECRET`, `SAHAMATI_FIU_ID`, `SAHAMATI_BASE_URL` (default `https://api.sahamati.org.in/sandbox/v2`), `USE_REAL_AA` (default true).
**Endpoints**:
- `POST /api/aa/consent` — body `{customer_id, fip_ids, purpose_code, duration_days}`
- `GET /api/aa/consents`
- `GET /api/aa/pull/{consent_handle}?days_back=30`
- `POST /api/aa/revoke/{consent_handle}`
**What it does**: When creds are set, makes real HTTPS calls to Sahamati v2 endpoints (`POST /Consent`, `POST /FI/request`, `GET /FI/fetch/{sid}`, `POST /Consent/Status`). When creds are missing, falls back to the schema-accurate mock. Every response carries `_real: true|false`.
**UI page**: `/aa`
**Verdict**: **REAL adapter, UNVERIFIED at the wire**. Code matches Sahamati v2 spec. Never tested against real Sahamati sandbox (requires FIU registration). Production needs HSM-signed bodies for every request — adapter ships with bearer-token auth (correct for sandbox).
**Fix for production**: plug HSM signer into `AAClient._headers()`.

### 8.2 DiliSense KYC client
**Files**: `src/integrations/dilisense_client.py`
**Env vars**: `DILISENSE_API_KEY`, `DILISENSE_BASE_URL` (default `https://api.dilisense.com/v1`), `USE_REAL_DILISENSE` (default true).
**Endpoint**: `GET /api/kyc/screen?name=...&entity_type=individual|entity`
**What it does**: Real call to `GET /v1/checkIndividual` or `/checkEntity` when key is set; normalises DiliSense's response shape to our envelope (`{queried_name, risk, hits, checked_lists}`). Mock fallback otherwise. `_real` flag in every response.
**UI**: `/aa` page → "KYC Screen" panel.
**Verdict**: **REAL adapter, UNVERIFIED at the wire**. Code matches DiliSense API surface. Requires a paid DiliSense contract to test.

### 8.3 Integrations status diagnostic
**Endpoint**: `GET /api/integrations/status`
**What it does**: Reports which providers are live (real creds present) vs. mocked, plus the list of missing env vars per provider.
**Verdict**: REAL. The operator sees at a glance which mode is active.

### 8.4 Mock fallback layer
**Files**: `src/aa_kyc_mock.py`
**What it does**: Schema-accurate mocks behind both adapters. Deterministic per-name risk score (same name → same risk). PEP keyword matching ("holdings", "international", "global"). Hardcoded sanctions hits for three specific names ("Thunder Bolt Exports" etc).
**Verdict**: REAL as a mock. **Don't promote this to production logic** — it's a fallback for demo.

---

## 9. Investigation tools

### 9.1 Fund Journey Tracer
**Files**: `src/fund_tracer.py`
**What it does**: BFS forward and/or backward from an entity (or alert) up to N hops, with per-edge velocity (txns/hour) and per-node red-flag annotation (shell, high_risk, part_of_cycle, multi_branch, dormant_then_active).
**Endpoints**:
- `GET /api/journey/{entity_id}?direction=both|forward|backward&hops=3&min_amount=0`
- `GET /api/journey/alert/{alert_id}?include_neighbors=false`
**UI page**: `/journey` — combines Sankey diagram (acyclic) with force-graph (cyclic), timeline panel below.
**Verdict**: **REAL**. Pre-grouped txn index (O(1) per-node lookup) avoids O(N_txns) scans. Solid implementation.

### 9.2 Entity Explorer
**Files**: backend handlers `GET /api/entities`, `GET /api/entities/{id}`
**What it does**: Search entities by name, filter by risk level, view individual entity's flow stats + transaction history.
**UI page**: `/entities`
**Verdict**: REAL.

### 9.3 Graph viewer
**Files**: `frontend/src/pages/Graph.jsx` — uses `react-force-graph-2d`.
**Endpoints**: `GET /api/graph?fraud_only=&high_risk_only=&min_amount=`, `GET /api/graph/{entity_id}?hops=N`
**UI page**: `/graph`
**Verdict**: REAL. Interactive network view.

### 9.4 Pattern library
**Endpoint**: `GET /api/patterns/{circular|layering|smurfing|funnel|dormant|profile}`
**UI page**: `/patterns`
**Verdict**: REAL. Per-pattern alert list + underlying transactions.

### 9.5 Subgraph extraction (N-hop)
**Files**: `src/graph_engine.py:extract_subgraph`
**Endpoint**: `GET /api/graph/{entity_id}?hops=N`
**Verdict**: REAL.

---

## 10. Analytics

### 10.1 Dashboard KPIs + time-travel
**Endpoint**: `GET /api/dashboard?until=YYYY-MM-DDTHH:MM`
**UI page**: `/` (Dashboard).
**What it does**: KPIs (total/fraud txns, volume, alerts), daily series, pattern breakdown, risk distribution, amount bucketing, time-travel filter via `until` parameter.
**Verdict**: REAL. Time-travel is just pandas filter by timestamp — simple but useful for replay demos.

### 10.2 Channel / rail / hour analytics
**Endpoint**: `GET /api/analytics/channels`
**Verdict**: REAL. Pandas groupby; functional.

### 10.3 Branch analytics
**Endpoint**: `GET /api/analytics/branches`
**Verdict**: REAL. Inflow + outflow + fraud rate per branch.

### 10.4 Product analytics
**Endpoint**: `GET /api/analytics/products`
**Verdict**: REAL. Same shape as branch analytics.

---

## 11. AI Copilot

### 11.1 LLM Copilot — Gemini path
**Files**: `src/llm_copilot.py:334`
**What it does**: Proper Gemini 2.0 Flash function calling. Up to 3 rounds of tool calls per query (multi-step investigation). System prompt frames it as a banking investigator assistant.
**Endpoint**: `POST /api/copilot/query` body `{message}`
**UI page**: `/copilot`
**Env**: `GEMINI_API_KEY`
**Tools surfaced to Gemini**: `trace_funds`, `find_cycles`, `explain_alert`, `get_profile_delta`
**Verdict**: **REAL** when `GEMINI_API_KEY` is set. Multi-round tool calling is properly implemented.

### 11.2 LLM Copilot — local fallback
**Files**: `src/llm_copilot.py:497`
**What it does**: Keyword routing — matches words like "trace", "flow", "cycle", "alert", "profile", "risk" against 6 hardcoded routes; falls back to a "Try asking..." help message otherwise.
**Verdict**: **THEATER — must rename**. This is not AI. It's a keyword-driven menu wearing an AI label. When `GEMINI_API_KEY` is absent, the UI still says "Gemini-powered" — that's a credibility hit if a judge asks "is that actually AI?".
**Fix**: in the response object, set `source="rule_based_fallback"` (already done) AND surface this in the UI as **"RUDRA Quick Commands"** instead of "AI Copilot" when in fallback mode.

### 11.3 `trace_funds` tool
**Files**: `src/llm_copilot.py:33`
**What it does**: BFS forward and/or backward up to N hops, returns flows + summary.
**Verdict**: REAL. Properly named, functional.

### 11.4 `find_cycles` tool
**Files**: `src/llm_copilot.py:72`
**What it does**: **Re-implements DFS cycle search from scratch**, returns up to 50 cycles.
**Verdict**: **DUPLICATE** — duplicates `fraud_detector.detect_circular_transactions` with different filtering (no log-bucket pruning, no SCC pre-filter). Two sources of truth.
**Fix**: delete this implementation; have the tool call `state["detector"].detect_circular_transactions()`.

### 11.5 `explain_alert` tool
**Files**: `src/llm_copilot.py:147`
**What it does**: Returns alert + 4 hardcoded reasoning bullets per pattern + entity profiles + related cases.
**Verdict**: **THEATER**. The "reasoning chain" is the same 4 sentences for every Circular alert, every Layering alert, etc. Carries no per-instance information.
**Fix**: replace `reasoning_chain` with `shap_explainer.explain_alert(...)` output. We already have it.

### 11.6 `get_profile_delta` tool
**Files**: `src/llm_copilot.py:218`
**What it does**: Re-checks profile-vs-behaviour with **different hardcoded thresholds than `ProfileMismatchDetector`**.
**Verdict**: **DUPLICATE + INCONSISTENT**. Right now the copilot can say "no profile mismatch" while the Cases page shows a profile-mismatch alert for the same entity. Two sources of truth diverging.
**Fix**: delete this implementation; call `ProfileMismatchDetector(graph, txns, risk_scores).detect()` filtered to the queried entity.

---

## 12. Config + ops

### 12.1 ConfigStore (threshold persistence)
**Files**: `src/config_store.py`
**What it does**: SQLite-backed key-value store for detector thresholds. Defaults baked into `DEFAULT_CONFIG`. Every detector reads via `self._cfg(key, default)`.
**Endpoints**:
- `GET /api/config/thresholds` (any role)
- `POST /api/config/thresholds` (ADMIN)
- `POST /api/config/thresholds/reset` (ADMIN)
- `POST /api/config/rerun` (ADMIN) — re-runs all detectors with current config
**UI page**: `/settings`
**Verdict**: **REAL but UNDERUSED**. `FraudDetector` and `DormantActivationDetector` honour it; `ProfileMismatchDetector` **does not** (see §3.6). LLM Copilot's `get_profile_delta` has yet another set of hardcoded values.

### 12.2 RBAC
**Files**: `src/rbac.py`
**Roles**: `INVESTIGATOR`, `SUPERVISOR`, `ADMIN`. Action matrix:
- INVESTIGATOR: open/note/view cases, move to INVESTIGATING/ESCALATED, download FIU
- SUPERVISOR: + file SAR, dismiss, verify audit chain
- ADMIN: + write thresholds, re-run, retrain
**Header**: `X-User-Role`
**Endpoint**: `GET /api/me`
**Verdict**: REAL but simple (no JWT, no IDP integration). Production replaces this with the bank's IDP.

### 12.3 Pipeline run trigger
**Endpoint**: `POST /api/pipeline/run` (ADMIN)
**What it does**: Re-runs the data generator → graph build → detectors → ML training → SAR PDFs cycle inside the running backend.
**Verdict**: **REAL but INCOMPLETE**. `backend.main.run_pipeline()` only trains XGBoost synthetic; doesn't train GNN, doesn't train IBM AML variants. The full pipeline is in `src/run_pipeline.py` (CLI). Two sources of truth — same drift risk as §11.4.
**Fix**: have the backend handler import + call `src.run_pipeline.main()` rather than reimplementing.

### 12.4 Health endpoint
**Endpoint**: `GET /`
**Returns**: `{name, version, ml_trained, alerts, incidents}`
**Verdict**: REAL.

### 12.5 Whoami
**Endpoint**: `GET /api/me`
**Returns**: `{role, valid_roles, permissions}`
**Verdict**: REAL.

---

## 13. Frontend pages

| Route | Page | What it surfaces | Verdict |
|---|---|---|---|
| `/` | Dashboard | KPIs, daily trends, time-travel slider, latency tile, risk dist, case status | REAL |
| `/incidents` | Incidents | Clustered alerts (290 → ~6 incidents) | REAL |
| `/cases` | Case Workbench | Triage queue, SHAP explanation, audit log, role-gated buttons | REAL |
| `/journey` | Fund Journey | Sankey + force-graph trace forward/backward + timeline | REAL |
| `/graph` | Network Graph | Full bank graph with filters | REAL |
| `/analytics` | Analytics | Channel/rail/branch/product analytics | REAL |
| `/patterns` | Pattern Library | Per-pattern alerts + raw txns | REAL |
| `/entities` | Entity Explorer | Search, risk view, txn history | REAL |
| `/model` | ML Models | F1/AUC/CM/feature-importance for every variant | REAL |
| `/live` | Live Stream | Per-txn ML scoring with mean+p95 latency | REAL (simulator-fed) |
| `/copilot` | AI Copilot | Natural-language investigation | REAL (Gemini) / THEATER (fallback) |
| `/sar` | SAR Reports | Generate formal SAR per alert | MIXED — see §7.1 |
| `/settings` | Detector Settings | Threshold sliders (ADMIN) | REAL |
| `/aa` | Account Aggregator | AA consent flow + DiliSense KYC | REAL (with adapter switch) |

**Pages that DON'T exist but probably should**:
- **Stream Inspector** — viewing `/api/stream/recent` events live. Right now there's no UI; you have to curl. Highest-value new page given the new Kafka path.
- **Ensemble Comparison** — currently `/model` only shows variant metrics. A page that compares XGB vs SAGE vs GAT vs Ensemble *per edge* (using `/api/ml/ensemble/edge_scores`) would showcase the new ensemble.
- **Integrations Status** — surface `/api/integrations/status` so an operator visually sees "AA: mock" vs "AA: REAL".

**Frontend lint state**: 19 `react-hooks/set-state-in-effect` warnings across 13 pre-existing pages. Build succeeds (1.05MB bundle). Not introduced by this session's work.

---

## 14. Prioritised improvement plan

Time budget: 24+ hours. Items grouped by impact and effort.

### TIER 1 — Honesty-critical (must do)

These are items where the current code creates a credibility risk if an evaluator inspects it carefully.

| # | Item | Effort | Files | Why critical |
|---|---|---|---|---|
| 1 | **Move ProfileMismatchDetector hardcoded thresholds to ConfigStore** | 30 min | `src/advanced_detectors.py:103`, `src/config_store.py` | Currently bypasses ConfigStore — contradicts the README claim that all detector thresholds are configurable. Easy fix, high credibility win. |
| 2 | **Replace `explain_alert` reasoning chain with SHAP narrative** | 45 min | `src/llm_copilot.py:166` | Reusing the SHAP narrative we already compute means every alert gets a per-instance explanation, not 4 generic sentences. Huge demo improvement. |
| 3 | **Delete duplicate `find_cycles` + `get_profile_delta` in copilot** | 1 hr | `src/llm_copilot.py:72,218` | Single source of truth — copilot and Cases page must agree. Two impl = two bugs. |
| 4 | **Fix SAR `_build_supporting_docs` non-existent file references** | 20 min | `src/sar_generator.py:439` | Right now the SAR PDF lists files that aren't in the zip. Either generate them or remove the list. |
| 5 | **Strip synthetic generator's pre-set `risk_score`** | 30 min | `src/data_generator.py:139` | Removes the label leakage that makes synthetic F1=0.99. The detector should compute risk from graph behaviour only. |
| 6 | **Make SAR bank/officer placeholders env-driven** | 20 min | `src/sar_generator.py:27`, env | `BANK_NAME` and `COMPLIANCE_OFFICER` env vars. Production deployment fills these. |
| 7 | **Rename Copilot to "Quick Commands" when in fallback mode** | 30 min | `frontend/src/pages/Copilot.jsx` (existing) + `src/llm_copilot.py:494` | Stop calling keyword routing "AI". Surface the mode in the UI header. |

**Tier 1 total**: ~4 hours

### TIER 2 — Detector quality (real fraud-detection wins)

| # | Item | Effort | Files | Impact |
|---|---|---|---|---|
| 8 | **Train GraphSAGE/GAT properly (200+ epochs + NeighborLoader)** | 3 hrs | `src/gnn_model.py`, `src/ensemble_model.py` | Biggest honest-numbers win. Current F1 0.32 is undertrained; should hit 0.50+ with proper training. Then ensemble actually outperforms XGB-alone. |
| 9 | **Replace ProfileMismatch `count × 0.2` with ML-derived score** | 1 hr | `src/advanced_detectors.py:185` | Use the avg XGBoost score across the entity's edges as the mismatch confidence. |
| 10 | **Replace hand-tuned composite risk score with learned weights** | 2 hrs | `src/fraud_detector.py:425` | Train a small LogisticRegression on (per-node features → was-in-flagged-incident). Removes label leakage. |
| 11 | **Add temporal clustering to smurfing detector** | 1.5 hrs | `src/fraud_detector.py:279` | Flag N txns in M minutes below threshold, not just amount clustering. |
| 12 | **Add pass-through + lifetime features to shell-funnel detector** | 1 hr | `src/fraud_detector.py:345` | `pass_through_ratio = min(in,out)/max(in,out)`, `avg_holding_seconds`. Catches the "money in, money out fast" pattern current detector misses. |
| 13 | **Lift BFS branching cap in layering detector** | 30 min | `src/fraud_detector.py:271` | Currently hard-coded to 5 successors. Make it config-driven. |

**Tier 2 total**: ~9 hours

### TIER 3 — Code health (single source of truth)

| # | Item | Effort | Files | Why |
|---|---|---|---|---|
| 14 | **Unify `_build_live_features` and `extract_features`** | 1.5 hrs | `src/live_scoring.py:30` vs `src/ml_model.py:114` | Same 30 features, two code paths, drift risk. Refactor to one function with `existing_edge: bool` overload. |
| 15 | **Have `backend.main.run_pipeline` call `src/run_pipeline.main()`** | 30 min | `backend/main.py:100` | Two pipeline implementations diverging — backend skips GNN + real datasets. Same fix pattern as #14. |
| 16 | **Generate `subgraph.png` for FIU package** | 1 hr | `src/fiu_package.py`, matplotlib | Either generate the PNG referenced in `_build_supporting_docs` or remove the reference. Currently misleading. |
| 17 | **Delete `tabular_baseline.py` if IEEE-CIS won't be shipped** | 10 min | `src/tabular_baseline.py` | Dead path otherwise. Or commit a tiny IEEE-CIS slice. |
| 18 | **Wire dormant + profile alerts into `_REPORT_TYPE_MAP` flow** | 30 min | `src/fiu_package.py:206` | Currently `_REPORT_TYPE_MAP` has entries for these patterns but no SAR is generated for them. Either wire or remove. |

**Tier 3 total**: ~3.5 hours

### TIER 4 — Production-readiness gaps

| # | Item | Effort | Files | Why |
|---|---|---|---|---|
| 19 | **Validate STR XML against FIU-IND XSD** | 2 hrs | `src/fiu_package.py`, lxml | The XML is "format-shaped" not "format-validated". A real PSB needs schema validation pass. |
| 20 | **Verify Pathway dataflow end-to-end** | 2 hrs | `src/streaming/pathway_engine.py` | I wrote against current Pathway API but haven't smoke-tested. May need timestamp-parsing tweaks. |
| 21 | **Add Stream Inspector page in frontend** | 3 hrs | `frontend/src/pages/StreamInspector.jsx` (new) | Currently `/api/stream/recent` has no UI. Highest-value new page given the new Kafka path. |
| 22 | **Add Ensemble Comparison page** | 3 hrs | `frontend/src/pages/Ensemble.jsx` (new) | Surface `/api/ml/ensemble/edge_scores` — show XGB vs SAGE vs GAT vs Ensemble per edge. |
| 23 | **Add Integrations Status banner in `/aa` page** | 1 hr | `frontend/src/pages/AccountAggregator.jsx` (existing) | Surface `/api/integrations/status`. Operator visually sees mock vs real. |
| 24 | **Stub HSM signer in `AAClient._headers()`** | 1.5 hrs | `src/integrations/aa_client.py:88` | Production Sahamati needs ECC-256 detached JWS. Add a `_sign(body)` hook even if the default impl is bearer-token. |
| 25 | **Add Postgres support via SQLAlchemy** | 4 hrs | `src/case_manager.py`, `src/config_store.py` | If RUDRA is going to multi-tenant or scale beyond a single SQLite file, this is the gate. |

**Tier 4 total**: ~16.5 hours

### TIER 5 — Nice-to-haves (if budget allows)

- Bundle a 50k-row IEEE-CIS slice so the tabular baseline actually runs (~1 hr).
- Add a "regenerate ensemble" button on `/model` (~1 hr).
- Improve Live page to show stream events from Kafka, not just simulator-injected (~2 hrs).
- Add CSV/JSON export from every analytics endpoint (~1 hr each).
- Frontend lint cleanup — 19 `set-state-in-effect` warnings (~2 hrs).
- Add SHAP narrative to SAR PDF (~2 hrs).

### Suggested 24-hour schedule

| Hours | Focus | Items |
|---|---|---|
| 0–4 | Tier 1 honesty fixes | #1–7 |
| 4–7 | Detector ML improvements (light) | #9, #13 |
| 7–10 | GNN training overhaul | #8 (this alone is worth the whole tier) |
| 10–14 | Detector logic improvements | #10, #11, #12 |
| 14–17 | Code health refactors | #14, #15, #16 |
| 17–20 | Frontend new pages | #21 (Stream Inspector) or #22 (Ensemble) |
| 20–23 | Production-readiness | #19 (XSD validation), #20 (Pathway verify) |
| 23–24 | Buffer + verification | run full test suite, frontend build, smoke-test endpoints, update README |

### What you should NOT do

- **Don't add new fraud-pattern detectors.** The 6 existing ones cover the core typologies; depth-over-breadth.
- **Don't try to train FraudGT / BDH.** Multi-GPU + days of compute. Ship the real ensemble we have.
- **Don't migrate everything to a different framework.** FastAPI + NetworkX + XGBoost is the right stack for this scope.
- **Don't over-polish the LLM Copilot.** It's already differentiating; the honesty fixes (#2, #3, #7) matter more than capability extensions.

---

## Appendix A — File inventory by status

### REAL (production-ready or close)
- `src/graph_engine.py`
- `src/fraud_detector.py:detect_circular_transactions`
- `src/case_manager.py` (every part)
- `src/shap_explainer.py`
- `src/ml_model.py`
- `src/ensemble_model.py`
- `src/streaming/ingestor.py`
- `src/streaming/kafka_producer.py`
- `src/integrations/aa_client.py`
- `src/integrations/dilisense_client.py`
- `src/fund_tracer.py`
- `src/fiu_package.py` (modulo XSD validation)
- `src/incident_clustering.py`
- `src/rbac.py`
- `src/config_store.py`
- `src/live_scoring.py` (modulo duplicate features)
- `src/aa_kyc_mock.py` (as a mock)

### WEAK (functional, needs investment)
- `src/fraud_detector.py:detect_rapid_layering`
- `src/fraud_detector.py:detect_smurfing`
- `src/fraud_detector.py:detect_shell_funnels`
- `src/fraud_detector.py:compute_node_risk_scores`
- `src/advanced_detectors.py:ProfileMismatchDetector`
- `src/gnn_model.py` (architecture right, training budget too small)
- `src/streaming/pathway_engine.py` (untested at the wire)

### THEATER (hardcoded text where data should be)
- `src/sar_generator.py:_build_recommendations`
- `src/sar_generator.py:_build_supporting_docs`
- `src/sar_generator.py` placeholder text
- `src/llm_copilot.py:explain_alert` reasoning chains
- `src/llm_copilot.py:_generate_local_response` (keyword routing labelled as AI)
- `src/data_generator.py` pre-set risk scores

### DUPLICATE
- `src/llm_copilot.py:find_cycles` ↔ `src/fraud_detector.py:detect_circular_transactions`
- `src/llm_copilot.py:get_profile_delta` ↔ `src/advanced_detectors.py:ProfileMismatchDetector`
- `src/live_scoring.py:_build_live_features` ↔ `src/ml_model.py:extract_features`
- `backend/main.py:run_pipeline` ↔ `src/run_pipeline.py:main`

### DEAD (consider removing)
- `src/tabular_baseline.py` — unused without IEEE-CIS data
- `_REPORT_TYPE_MAP` entries for `Dormant Activation` / `Profile Mismatch` in `src/fiu_package.py:206`

### MISSING (claimed but not implemented)
- Network PNG referenced in `src/sar_generator.py:445`
- Postgres support (context.md mentioned, never implemented)
- HSM signing for AA production wire
- Multi-tenant / federated learning
- Bhashini Hindi voice
- FIU-IND XSD validation

---

## Appendix B — Endpoint quick reference

```
Health
  GET  /                               — health
  GET  /api/me                         — whoami + permissions

Dashboard
  GET  /api/dashboard?until=...        — KPIs + time-travel

Graph
  GET  /api/graph?fraud_only=&...      — full graph
  GET  /api/graph/{id}?hops=N          — entity subgraph

Alerts + cases
  GET  /api/alerts
  GET  /api/alerts/{id}
  GET  /api/alerts/{id}/explain        — SHAP
  GET  /api/incidents
  GET  /api/incidents/{id}
  GET  /api/cases?status=
  GET  /api/cases/{id}
  POST /api/cases/{id}/dispose
  POST /api/cases/{id}/note
  GET  /api/cases/{id}/verify          — hash-chain check
  GET  /api/patterns/{pattern}

Entities
  GET  /api/entities
  GET  /api/entities/{id}

Fund journey
  GET  /api/journey/{id}?direction=&hops=&min_amount=
  GET  /api/journey/alert/{id}?include_neighbors=

ML
  GET  /api/ml/variants
  GET  /api/ml/metrics?variant=
  GET  /api/ml/ensemble?variant=ibm_aml
  GET  /api/ml/ensemble/edge_scores?variant=ibm_aml&limit=
  GET  /api/ml/tabular
  POST /api/ml/retrain                 (ADMIN)

FIU + SAR
  GET  /api/sar/generate/{id}
  GET  /api/fiu/package/{id}           — zip

Config
  GET  /api/config/thresholds
  POST /api/config/thresholds          (ADMIN)
  POST /api/config/thresholds/reset    (ADMIN)
  POST /api/config/rerun               (ADMIN)

Analytics
  GET  /api/analytics/channels
  GET  /api/analytics/branches
  GET  /api/analytics/products

Streaming
  GET  /api/stream/status
  POST /api/stream/start               (ADMIN)
  POST /api/stream/stop                (ADMIN)
  GET  /api/stream/recent?limit=N
  POST /api/stream/replay              (ADMIN)
  GET  /api/stream/velocity_alerts

Live + benchmark
  GET  /api/live/inject?count=N
  GET  /api/benchmark/latency

DPI integrations
  POST /api/aa/consent
  GET  /api/aa/consents
  GET  /api/aa/pull/{handle}
  POST /api/aa/revoke/{handle}
  GET  /api/kyc/screen?name=&entity_type=
  GET  /api/integrations/status

Copilot
  POST /api/copilot/query              — natural-language

Pipeline
  POST /api/pipeline/run               (ADMIN)
```

---

## Appendix C — Environment variables

```
# AI Copilot
GEMINI_API_KEY=...           # Enables Gemini-powered copilot

# Streaming
STREAM_BACKEND=auto|kafka|inproc       # default: auto (probe Kafka, fall back to inproc)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=rudra.transactions
KAFKA_ALERT_TOPIC=rudra.velocity_alerts
VELOCITY_THRESHOLD_INR=2000000         # Pathway sliding-window threshold
VELOCITY_WINDOW_MS=300000              # 5 min
VELOCITY_HOP_MS=60000                  # 1 min
PATHWAY_ALERTS_PATH=data/streaming/velocity_alerts.jsonl
PATHWAY_PUBLISH_KAFKA=0|1
PATHWAY_INPROC=0|1                     # run Pathway in-process (off by default)

# AA (Sahamati)
SAHAMATI_CLIENT_ID=...
SAHAMATI_CLIENT_SECRET=...
SAHAMATI_FIU_ID=...
SAHAMATI_BASE_URL=https://api.sahamati.org.in/sandbox/v2
USE_REAL_AA=true|false                 # force mock even with creds

# DiliSense
DILISENSE_API_KEY=...
DILISENSE_BASE_URL=https://api.dilisense.com/v1
USE_REAL_DILISENSE=true|false

# Suggested additions (not currently wired)
BANK_NAME=                              # SAR header — currently hardcoded
COMPLIANCE_OFFICER=                     # SAR header — currently hardcoded
```

---

*End of analysis. For the implementation work, start with Tier 1 (#1–7); they are the highest credibility-to-effort ratio.*
