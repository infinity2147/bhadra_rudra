# Temporal GNN (TGN) Fraud Prediction — Design Spec

**Date:** 2026-07-01
**Status:** Approved (design); pending implementation plan
**Origin:** Phase-2 predictive analytics — answers Evaluator-1 ("use of AI for predictive analysis"). Based on Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs* (2020).

---

## 1. Why

RUDRA's ML today is **static edge classification** (XGBoost + GraphSAGE + ensemble) — it scores an edge from a snapshot of the graph. A **Temporal Graph Network (TGN)** adds an axis the static models cannot see: the **continuous-time evolution** of each account's behaviour (a recurrent per-node memory updated by every transaction). This is the honest form of "predictive analytics" — a forward-looking, time-aware fraud model — and it is directly comparable to the existing models on the same held-out metric.

## 2. Objective (decided) and the reframe it forces

**Objective: temporal FRAUD prediction**, not link-existence prediction.

The original request's *goal* ("predict a **fraudulent** transaction") contradicted its *method* (temporal negative sampling of random fake edges, which trains link **existence** — "will A transact with B" — and never sees the fraud label). Resolving in favour of the goal changes one requirement:

- **Replaced:** random-pair "temporal negative sampling."
- **With:** natural class supervision — every real edge is an example, label = `is_fraud`, class imbalance (5.18% fraud on IBM AML) handled by `BCEWithLogitsLoss(pos_weight=...)`.

This makes the task **temporal edge classification with a TGN backbone** — a direct upgrade of the existing static SAGE classifier, evaluated on the **same AUPRC yardstick**, with no "does the edge exist vs. is it fraud" conflation. Everything else from the TGN recipe stays: node memory (GRU), message function with edge features, temporal-attention embedding, MLP decoder, strict chronological split, `memory.detach()` per batch, and predict-then-update ordering.

## 3. Goals / Non-goals

**Goals**
- A paper-faithful TGN (memory + temporal attention) trained to classify edges as fraud, with a strictly chronological 70/15/15 split (no temporal leakage).
- Selection + reporting on **val/test AUPRC** (threshold-free), with an **F2** operating point via the shared `ml_model.fbeta_optimal_threshold` — same conventions as XGB/SAGE/ensemble, so the numbers are comparable.
- Persisted artifacts + a ModelMetrics comparison and a small "top predicted future-fraud" list.
- Optional dependency: skips cleanly (like SAGE) when torch/PyG absent.

**Non-goals**
- **No live/interactive runtime prediction.** TGN's node-memory state makes A→B-at-request-time a stateful path that violates RUDRA's "serve from persisted JSON, never reload weights at runtime" rule. MVP serves persisted rankings; live scoring is a documented later phase.
- **No promise to beat the static models.** Per RUDRA's recorded null-result history (bolt-on lanes haven't cracked the recall ceiling; feature-injection was a no-op), TGN's metrics are reported transparently — win, tie, or loss.
- **No random-pair link-existence objective** (see §2).
- No change to the existing detectors, alerts, incidents, or the RCA/collusion lanes.

## 4. Architecture

Four small, independently testable units.

### 4.1 `src/temporal_data_loader.py`
- Load `data/{variant}/transactions.csv`; **sort strictly by `timestamp`**.
- Map `sender_id`/`receiver_id` → contiguous integer node ids (persist the mapping for later id↔name display).
- Build a PyG `TemporalData(src, dst, t, msg, y)` where:
  - `t` = integer seconds from the first timestamp.
  - `msg` = edge features: normalized `amount`, `log1p(amount)`, and a compact one-hot of `transaction_type` (small, fixed order).
  - `y` = `is_fraud` (float 0/1).
- Chronological split via `TemporalData.train_val_test_split(val_ratio=0.15, test_ratio=0.15)` (ratio split in time order — no shuffle).
- Public: `load_temporal_data(csv_path) -> {"data": TemporalData, "train": ..., "val": ..., "test": ..., "num_nodes": int, "msg_dim": int, "id_map": {...}}`.

### 4.2 `src/tgn_model.py`
Architecture per Rossi et al., using PyG primitives:
- `TGNMemory(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module=IdentityMessage(...), aggregator_module=LastAggregator())` — GRU-backed per-node memory `s_i(t)`, updated on each event.
- `LastNeighborLoader(num_nodes, size=K)` — maintains each node's K most-recent temporal neighbors.
- `GraphAttentionEmbedding` — a `TransformerConv` over the temporal neighborhood, fed `(last_update, t)` through the memory's `TimeEncoder`, producing `Z_i(t)`.
- `FraudDecoder` — MLP on `concat(Z_a(t), Z_b(t))` → **1 logit** (fraud). (Not a link-existence "two-tower" scorer — a single supervised fraud head.)
- Pure `nn.Module` definitions; no training logic here.

### 4.3 `src/train_tgn.py`
- Chronological `TemporalDataLoader` batches over the train split.
- **Per batch:** (1) look up recent neighbors, (2) compute `Z` from **current** memory + attention, (3) decode fraud logits for the batch's edges, (4) `BCEWithLogitsLoss(pos_weight)` vs `y`, (5) backward/step, (6) **then** `memory.update_state(...)` + `neighbor_loader.insert(...)` with this batch (predict-then-update — the model never sees the memory update caused by the edge it is scoring), (7) **`memory.detach()`** to cut BPTT and prevent VRAM growth.
- Eval: reset memory, replay train→val (or train→test) to warm memory, compute AUPRC on the held-out split. **Early-stop on val AUPRC**; cap epochs (~20–50).
- **F2 threshold** via `ml_model.fbeta_optimal_threshold(y_true, y_prob, beta=2.0)`.
- Persist to `data/ml/{variant}/tgn/`:
  - `metrics.json` — test `auprc, auc, f1, f2, precision, recall, threshold, n_train/val/test, epochs`.
  - `model.pt` — weights (NOT loaded at runtime).
  - `predictions.json` — ranked top-N test-window edges by predicted fraud probability (`{src_id, dst_id, src_name, dst_name, prob, is_fraud}`) for the UI.
- Public: `train_tgn(data_dir, variant, epochs=..., seed=...) -> metrics dict`; `load_tgn_metrics(...)`, `load_tgn_predictions(...)` (JSON readers, mirroring `gnn_model.load_gnn_metrics`).

### 4.4 Integration + serving
- **`src/run_pipeline.py`**: train TGN in the torch section **after** XGBoost (torch loads post-XGB — respects the segfault-ordering rule), alongside/after SAGE, wrapped in `try/except ImportError` (optional). Time-bounded (epoch cap).
- **`backend/main.py`**: read endpoints serving persisted artifacts only (never load `.pt`):
  - extend the ML-metrics response (or add `GET /api/ml/tgn`) with TGN `metrics.json`.
  - `GET /api/tgn/predictions` → `predictions.json` (top predicted future-fraud edges).
- **Frontend**: TGN column on the **ModelMetrics** page (XGB / SAGE / ensemble / **TGN** side by side) + a compact "Predicted future fraud" list from `predictions.json`.

## 5. Data flow
`transactions.csv` → temporal_data_loader (sorted, split) → train_tgn (predict-then-update, detach, AUPRC early-stop, F2) → `data/ml/{variant}/tgn/{metrics.json, model.pt, predictions.json}` → backend serves JSON → ModelMetrics comparison + predicted-fraud list.

## 6. Evaluation
- Strictly chronological 70/15/15; memory warmed by replaying earlier events before scoring a later split (no future leakage).
- Report AUPRC (primary, threshold-free) + F1/F2/precision/recall at the F2 threshold, on the **test** split, next to the other models. If TGN loses, `metrics.json` and the UI show it plainly.

## 7. Integration constraints (designed around)
- **XGB before torch** in a process (segfault otherwise) — TGN trains in the post-XGB torch block; TGN test runs in an isolated subprocess like the SAGE test.
- **Optional dependency** — `ImportError` → skip, pipeline continues.
- **Serve from JSON, never reload `.pt`** at runtime.
- **Slowest trainer** (sequential memory dependency) — epoch-capped + AUPRC early-stop; fully optional.

## 8. Testing
- `tests/test_temporal_data_loader.py`: timestamps strictly sorted; split is chronological (max train t ≤ min val t ≤ min test t); `y`/`msg` shapes; id map round-trips.
- `tests/test_tgn_model.py` (isolated subprocess, per the torch/XGB rule): a tiny synthetic `TemporalData` trains a few steps without error; `memory.detach()` keeps memory grad-free; decoder outputs one logit per edge; a short run persists a well-formed `metrics.json` + `predictions.json`; predictions are sorted by prob desc.
- Endpoint test: `GET /api/tgn/predictions` and the TGN metrics endpoint return the documented shape (or a clean "not trained" payload when artifacts absent, mirroring `/api/ml/tabular`).

## 9. Risks & mitigations
- **Doesn't beat static models** → framed as a new temporal capability, reported honestly (§3, §6).
- **Training time** → epoch cap + early-stop + optional; 100k events is tractable.
- **Temporal leakage** → ratio split in time order + memory-warm-then-score; asserted by a loader test.
- **Torch/XGB segfault** → post-XGB training + isolated-subprocess test.
- **Runtime statefulness** → explicitly out of scope; serve persisted artifacts only.

## 10. Open questions (resolve during planning)
- Exact `msg` feature set (keep lean: amount, log-amount, transaction_type one-hot) — finalize the one-hot column order at plan time.
- Whether TGN metrics extend the existing `/api/ml/metrics` payload or get a dedicated `/api/ml/tgn` (mirror whichever the ModelMetrics page consumes most cleanly).
- `predictions.json` top-N size (default 50).
