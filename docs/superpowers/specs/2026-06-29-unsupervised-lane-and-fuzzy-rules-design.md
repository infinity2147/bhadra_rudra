# Design: Unsupervised Anomaly Lane + Fuzzy Rule Thresholds

**Date:** 2026-06-29
**Status:** Approved design (pre-implementation)
**Scope:** Two independent, additive detection upgrades. Temporal/TGN graph modelling
is explicitly **out of scope** for this spec (parked for a later, separate track).

---

## 1. Why

Two gaps in the current ML-led, rule-corroborated detection stack:

1. **No novel-typology coverage.** XGBoost, GraphSAGE and GAT are all supervised —
   they only catch fraud that resembles the labelled training fraud. A laundering
   pattern never seen in training is invisible. We want a lane that flags
   statistically bizarre edges *regardless of labels*.

2. **Hard rule gates drop near-miss structurers.** The rule detectors already emit a
   *graded* `confidence`, but the **fire/no-fire decision is a hard cut** (e.g.
   smurfing requires `amount < smurfing_threshold` and `count >= min_txns`). A
   ₹49,990 transfer against a ₹50,000 limit, or 4 transfers against a 5-transfer
   burst minimum, is dropped entirely. Structurers hug these boundaries on purpose.

Both fixes are additive, CPU-only, and ride the existing alert/tiering spine.

### Goals
- Surface novel anomalies the supervised models miss, as a first-class, demoable lane.
- Stop hard rule boundaries from silently dropping near-miss cases, **without**
  flooding the analyst queue.
- Zero regression by default: every new behaviour is config-gated and defaults to
  "off" / "identical to today".

### Non-goals
- Temporal Graph Networks / continuous-time modelling (separate future track).
- Replacing the supervised models or the F2 operating point.
- Re-architecting the tiering or incident-clustering spine.

---

## 2. Feature 1 — Unsupervised Anomaly Lane

### 2.1 Approach (decided)

- **Algorithm:** `sklearn.ensemble.IsolationForest`. CPU-cheap (seconds on 100k
  edges), no GPU, no torch dependency, well-suited to tabular edge features.
  *Rejected:* autoencoder (heavier, GPU-leaning, no demonstrated win over IF on
  tabular data, conflicts with the documented single-CPU design ethos).

- **Granularity:** **edge-level**, reusing the exact feature matrix the XGB lane
  already builds — `extract_features(graph, transactions)` returns a DataFrame
  indexed by `"u->v"` over the fixed 30-column `FEATURE_COLUMNS`
  ([ml_model.py:157](../../../src/ml_model.py#L157), [ml_model.py:339](../../../src/ml_model.py#L339)).
  No new feature engineering; IF consumes `X.values`.

- **Integration as a lane, not an ensemble base (decided):** the IF anomaly score is
  *not* a fraud probability, so feeding it into the supervised stacked ensemble would
  dilute the meta-learner. Its entire value is the **disagreement region** — edges
  that are anomalous **and** that the supervised ensemble scored *below* its F2
  threshold. Those become a standalone **"Novel Anomaly"** alert lane.

### 2.2 New module: `src/unsupervised_model.py`

```
train_unsupervised(graph, transactions, data_dir, variant="ibm_aml",
                   contamination="auto", random_state=42) -> dict (metrics)
```
- Builds `X = extract_features(graph, transactions)` (reuse — do not duplicate).
- Fits `IsolationForest(n_estimators=200, contamination=..., random_state=42)`.
- Converts `score_samples(X)` (higher = more normal) into a **normalized anomaly
  score in [0,1]** (higher = more anomalous), via min-max over the population.
- Persists:
  - `data/ml/{variant}/unsupervised/edge_scores.json` — `{"u->v": anomaly_score}`
  - `data/ml/{variant}/unsupervised/metrics.json` — n_edges, contamination,
    anomaly-score distribution percentiles, the chosen `anomaly_threshold`
    (default: the score at the 99th percentile, i.e. flag the top ~1% as anomalous;
    config-overridable).
- `load_unsupervised_scores(data_dir, variant)` loader, mirroring
  `load_edge_scores` ([ml_model.py:437](../../../src/ml_model.py#L437)).

**Ordering constraint:** IsolationForest is pure sklearn (no torch), so it trains
**right after XGB and before the GNN import** in the pipeline — respecting the
documented "XGB must be fit before torch loads in a process" rule.

### 2.3 Alert generation: extend `src/ml_alert_generator.py`

New function alongside `generate_ml_alerts`:

```
generate_novel_alerts(graph, unsup_scores, supervised_scores, sup_threshold,
                      anomaly_threshold, max_alerts=None) -> List[dict]
```
- Emits an alert **only** for edges where
  `unsup_scores[e] >= anomaly_threshold` **AND**
  `supervised_scores.get(e, 0) < sup_threshold` (the novel/disagreement set).
- Ranked by anomaly score, capped by `max_alerts` (default cap to bound volume —
  e.g. 500; logged when truncated, no silent cap).
- Canonical alert schema, matching `generate_ml_alerts`, with:
  - `pattern_type: "Novel Anomaly"`
  - `source: "unsupervised"`
  - `algorithm: "isolation_forest"`
  - `tier: 4` (new — see tier semantics below)
  - `anomaly_score` (and `ml_score` set to the same value so existing
    sort/filter code that reads `ml_score` keeps working)
  - `description` explaining "no supervised model flagged this; statistically
    anomalous on <top-contributing features>".

**Reason/explanation:** the lazy SHAP endpoint (`/api/alerts/{id}/explain`) is
`TreeExplainer`-based and XGB-specific. Novel alerts instead carry a **lightweight,
precomputed reason**: the top-3 features by absolute z-score vs the population mean
(computed cheaply from the same `X` at train time, stored per flagged edge). This
avoids wiring SHAP-for-IsolationForest and keeps the explanation honest.

### 2.4 Tiering: extend `apply_tiers` / `fuse_ml_alerts`

- Novel-anomaly alerts are a **new orthogonal tier 4** ("Novel Anomaly —
  unsupervised, unseen typology"). They are appended after the existing T1/T2/T3
  logic in `apply_tiers` ([ml_alert_generator.py:125](../../../src/ml_alert_generator.py#L125)).
- If a novel alert shares an entity with a rule alert, it is **upgraded to T1**
  (cross-lane agreement, same rule as ML alerts) and carries `corroborated_by`.
  Otherwise it stays T4.
- `fuse_ml_alerts` ([ml_alert_generator.py:218](../../../src/ml_alert_generator.py#L218))
  loads the unsupervised scores + the supervised (ensemble/XGB) scores, calls
  `generate_novel_alerts`, and merges. Degrades gracefully (no novel alerts) if the
  unsupervised artefacts are absent — same pattern as the existing
  "no ML scores → rule-only" fallback.
- `tier_summary` gains a `tier4` / `novel` count.

### 2.5 Pipeline wiring: `src/run_pipeline.py`

- After the XGB step ([run_pipeline.py:184](../../../src/run_pipeline.py#L184)), before
  the GNN import, add:
  ```
  from unsupervised_model import train_unsupervised
  train_unsupervised(graph, df, data_dir, variant=dataset)
  ```
  wrapped in try/except like the other trainers (graceful skip on failure).
- No change needed to the fuse call itself — `fuse_ml_alerts` internally picks up
  the new scores.

### 2.6 Backend + frontend

- **Backend:** alerts are served generically from `fraud_alerts.json`, so novel
  alerts flow through with no endpoint change. Verify `_alerts_with_case_status` in
  [backend/main.py](../../../backend/main.py) preserves `anomaly_score`/`ml_score`
  and `tier` (it already must not overwrite `ml_score` per the existing constraint).
- **Frontend:** add a `TIER_META[4]` entry + filter chip in
  [frontend/src/pages/Cases.jsx:56](../../../frontend/src/pages/Cases.jsx#L56) (and the
  Incidents tier render at
  [frontend/src/pages/Incidents.jsx:214](../../../frontend/src/pages/Incidents.jsx#L214)).
  Distinct colour (e.g. amber) and tooltip: "Novel Anomaly — flagged by unsupervised
  model, no supervised/typology match". Sort: T4 sits after T3 by default (it is a
  recall/exploration lane, not a priority queue).

### 2.7 Edge cases / risks
- **Volume:** IF flags ~contamination fraction; the `< sup_threshold` filter and
  `max_alerts` cap keep it bounded. Log truncation.
- **Synthetic variant:** synthetic data is clean by construction (XGB scores ~1.0
  AUPRC); the novel lane will mostly be empty there — acceptable, it is meaningful on
  real/IBM data.
- **Determinism:** `random_state=42` for reproducibility (consistent with the rest
  of the pipeline).

### 2.8 Tests (`tests/`)
- `train_unsupervised` produces a scores file keyed by edge, scores in [0,1].
- `generate_novel_alerts` emits **only** edges that are anomalous AND below the
  supervised threshold (no overlap with the supervised alert set on a fixture).
- `apply_tiers` assigns `tier: 4`, upgrades to `tier: 1` on shared-entity rule
  corroboration.
- Graceful degradation: missing unsupervised artefacts → no novel alerts, pipeline
  unaffected.

---

## 3. Feature 2 — Fuzzy Rule Thresholds

### 3.1 Approach (decided)

A shared membership helper turns the hard fire/no-fire boundary into a **soft ramp**:
values within a configurable margin of the boundary are admitted at **reduced
confidence proportional to their membership degree μ ∈ (0, 1]**. Outside the margin,
behaviour is unchanged.

**Critical safety property:** the margin defaults to **0**, which reproduces today's
exact behaviour — a true zero-regression default. We turn it up only with measured
evidence.

**Why this won't flood the queue:** the ML-led tiering already **suppresses
rule-only alerts unless ML corroborates** (low-precision rule-only is dropped;
[ml_alert_generator.py:164](../../../src/ml_alert_generator.py#L164)). So fuzzy-admitted
near-misses mostly survive only as **Tier 1** (ML agrees) — fuzzy widens
recall/explanation where it is corroborated, rather than adding raw noise.

### 3.2 New module: `src/fuzzy.py`

Two pure, unit-testable membership functions:

```
ramp_upper(value, hard_limit, margin) -> float   # for "must be BELOW a cap"
    # 1.0 if value <= hard_limit
    # linear 1.0 -> 0.0 across (hard_limit, hard_limit*(1+margin)]
    # 0.0 above
    # margin == 0  =>  step function (1.0 at/below limit, 0.0 above) == today

ramp_lower(value, hard_min, margin) -> float     # for "must be AT LEAST N"
    # 1.0 if value >= hard_min
    # linear 0.0 -> 1.0 across [hard_min*(1-margin), hard_min)
    # 0.0 below
    # margin == 0  =>  step function == today
```

A small `combine(*mus)` (product or min — **min** chosen, weakest-link semantics)
folds multiple memberships into one factor.

### 3.3 Detectors to update (first pass)

**Smurfing** ([fraud_detector.py:435](../../../src/fraud_detector.py#L435),
fan-out at [fraud_detector.py:622](../../../src/fraud_detector.py#L622)):
- Amount-near-limit: `ramp_upper(amount, smurfing_threshold, margin)` instead of the
  hard `amount < threshold`.
- Count thresholds: `ramp_lower(count, min_txns, margin)`,
  `ramp_lower(n_receivers, min_receivers, margin)`.
- Final `confidence *= combine(...)`; admitted near-misses tagged `fuzzy: true`.

**Funnel** ([fraud_detector.py:728](../../../src/fraud_detector.py#L728)):
- `ramp_lower(imbalance, funnel_imbalance_threshold, margin)` instead of the hard
  `imbalance >= threshold`; fold into confidence.

*Other detectors (cycle, layering, recruiter, dormant, profile) are out of scope for
this first pass* — they are higher-precision typologies where boundary-hugging is
less of an evasion vector. Revisit only with evidence.

### 3.4 Config

New `ConfigStore` keys, default **0.0** (= off / no behaviour change):
- `smurfing_fuzzy_margin`
- `funnel_fuzzy_margin`

Read via the existing `self._cfg("key", 0.0)` pattern — **never hardcode**
([fraud_detector.py:50](../../../src/fraud_detector.py#L50)). Exposed in the config
admin surface like every other threshold.

### 3.5 Edge cases / risks
- Confidence must stay in [0,1] and respect existing caps (e.g. the
  `min(0.95, ...)` at [fraud_detector.py:574](../../../src/fraud_detector.py#L574)) —
  apply the fuzzy factor *before* the cap.
- A near-miss must never out-rank a clean hit: since μ < 1 strictly inside the
  margin, fuzzy alerts always score below an equivalent exact hit.
- Severity buckets are amount/flow-based and unchanged.

### 3.6 Tests (`tests/`)
- `ramp_upper` / `ramp_lower`: boundary values, margin=0 step behaviour, midpoint
  μ=0.5, clamping outside the margin.
- **Regression guard:** with both margins = 0, smurfing and funnel produce
  byte-identical alert sets to the pre-change detectors on the synthetic fixture.
- With margin > 0, a known near-miss (₹49,990 vs ₹50k; 4 vs 5 burst min) now fires
  at confidence strictly below the equivalent exact hit.

---

## 4. Shared concerns

- **Retrain required:** both features regenerate artefacts via
  `python src/run_pipeline.py` (the one command). Feature 1 adds a scores file;
  Feature 2 changes rule output at fuse time. Document in CLAUDE.md/README after build.
- **Independence:** the two features touch disjoint code except for the shared
  `ml_alert_generator` / pipeline fuse step; they can be built and tested in either
  order. Implementation plan may sequence Feature 2 first (smaller, no new model).
- **Docs:** update CLAUDE.md module table (`unsupervised_model.py`, `fuzzy.py`),
  the data-files table (`data/ml/{variant}/unsupervised/`), and the tier description
  (add Tier 4) once implemented.

## 5. Out of scope (parked)
- Feature 3 — Temporal/TGN graph modelling. Separate spec + plan when revisited.
- SHAP-for-IsolationForest explanations (lightweight z-score reason used instead).
- Extending fuzzy logic beyond smurfing + funnel.

## 5a. Measured outcome + decision (post-implementation, 2026-06-29)

Both features were built with TDD (32 new tests) and evaluated system-level via
`evaluate_detection.evaluate_alert_entities` (entity coverage vs `fraud_count`
labels) on IBM AML, reusing persisted ensemble scores:

| alert set | recall | precision | F1 | F2 |
|---|---|---|---|---|
| before (shipped) | 0.675 | 0.302 | 0.417 | 0.541 |
| + novel lane (F1) | 0.676 | 0.290 | 0.406 | 0.534 |
| + fuzzy (F2) | 0.675 | 0.302 | 0.417 | 0.541 |

**Neither improved the confusion matrix.** Root cause: IsolationForest anomalies
are enriched for fraud (15% vs 5.6% base), but *every* fraud-correlated anomaly is
already caught by the supervised model — so the novel lane's disagreement region
("anomalous AND supervised-below-threshold") is **0% fraud (0/640)**, pure FP
residue. Fuzzy admitted only 7 near-misses (IBM AML has few boundary-huggers). The
~2,066 missed fraud entities are structureless transfers invisible to both
supervised and unsupervised models — the recall ceiling.

**Decision (revised):** the unsupervised novel lane was **removed entirely** — on
this benchmark it only added FP noise (its disagreement region is 0% fraud), so it
earned no place even as an opt-in lane. **Feature 2 (fuzzy) was kept**: off by
default (margins 0), zero-cost, sound evasion-resistance for boundary-hugging data.
The eval harness (`evaluate_detection.py`) was kept. Recorded in CLAUDE.md and
project memory. The structureless-fraud ceiling needs a different attack (see the
papers review) — not bolt-on detectors.

## 6. Testing strategy summary
All new logic is unit-tested in `tests/`; the regression guard (margin=0 identical
output) is the key safety net for Feature 2, and the "novel set excludes the
supervised set" assertion is the key correctness check for Feature 1. Existing
suite (136 tests) must stay green. Final verification: run the app and confirm the
Tier 4 lane renders and is filterable on the Cases page (per the "verify like a
user" standard).
