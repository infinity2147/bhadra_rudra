# Design: Recurrence-Based Severity Escalation (IRB)

**Date:** 2026-06-29
**Status:** Approved design (pre-implementation)
**Lineage:** Adapted from the IRB ("Intimation Rule Based") alert-escalation idea in
Ahmed et al., *A semantic rule based digital fraud detection* (PeerJ CS 2021).

## 1. Why

A purely point-in-time alert list treats a one-shot flag and a serial re-offender
the same. Real triage cares about **recurrence over time**: an account flagged
across many windows is a stronger lead than one flagged once. This adds a third,
**temporal** triage axis — orthogonal to the existing **tier** (ML+rule confidence)
and **severity** (amount) — so an analyst instantly sees who keeps reoffending.

It is a **legibility / triage upgrade, explicitly not a detection-numbers change**:
- **Recall: unchanged** — nothing is suppressed; one-shot alerts remain, just ranked lower.
- **Precision: unchanged** — no alerts added or removed; this re-ranks and labels.

Validated on IBM AML (24h windows): ~88% of alerted entities are L1 (1 window),
~9% L2 (2 windows), ~3% L3 (≥3 windows), with standout hubs flagged across up to
10 distinct days — a clean triage pyramid, real signal, not a dud.

## 2. Recurrence definition

- **Alert time** = the max `last_seen` over graph edges among the alert's entities
  (9,242/9,307 resolve on IBM AML; alerts with no derivable time default to L1).
- Bucket times into `recurrence_window_hours` (default **24h**), origin = the
  earliest alert time.
- **Per entity**, `hit_count` = number of *distinct* windows it appears in across
  all alerts. Levels (config-driven):
  - **L1 "Suspected"** — 1 window
  - **L2 "Investigate"** — ≥ `recurrence_l2_windows` (default 2)
  - **L3 "Recurring Pattern"** — ≥ `recurrence_l3_windows` (default 3)
- An **alert's** escalation = the **max level over its entities** (a hub's alerts
  inherit the hub's recurrence; a one-shot leaf stays L1).

## 3. Components (small, pure, testable)

`src/recurrence.py`:
- `derive_alert_times(alerts, graph) -> {alert_id: datetime|None}` — graph-based time
  derivation (max `last_seen` of edges among entities). The only graph-coupled part.
- `compute_recurrence(alerts, alert_times, window_hours, l2_windows, l3_windows)
  -> {entity: {hit_count, windows, level}}` — **pure** (takes times, not a graph),
  so unit tests pass times directly.
- `apply_escalation(alerts, recurrence_map) -> alerts` — adds, in place:
  `escalation: {level, label, hit_count, entity, windows}` where `entity` is the
  driving (max-level) recurring entity and `windows` its window list (the
  legibility trail — "seen in windows [d1, d4, d7]").

Defaults live in `ConfigStore` (never hardcode): `recurrence_window_hours` (24),
`recurrence_l2_windows` (2), `recurrence_l3_windows` (3).

## 4. Integration

- **Pipeline** (`run_pipeline.py`): after `fuse_ml_alerts`, before writing
  `fraud_alerts.json` — `derive_alert_times` → `compute_recurrence` →
  `apply_escalation`. Re-clustering is unaffected (escalation is additive metadata).
- **Backend:** no change — alerts flow through `_alerts_with_case_status` as-is and
  carry `escalation` to the UI.
- **Frontend** (`Cases.jsx`): an escalation badge (L1/L2/L3, amber→red) beside the
  tier badge; a filter chip; default sort becomes `(tier, escalation_level, severity,
  flow)` so recurring high-confidence alerts float up. Alert detail shows the
  recurrence windows. No suppression of L1 (protects recall).

## 5. Orthogonality (decision)

Escalation is its **own field/badge**, not an overwrite of `severity` (amount) or
`tier` (confidence) — three distinct axes. It only influences default sort order.

## 6. Out of scope (deferred)

- A persisted cross-run / live-stream recurrence ledger in SQLite (batch computation
  + UI delivers the demo win; streaming accumulation is a clean follow-up).
- A per-alert `parent_id` pointer chain (the `entity`+`windows` trail already gives
  the recurrence timeline; YAGNI for v1).

## 7. Testing (TDD)

- `compute_recurrence`: distinct-window counting (two alerts same window → 1),
  L1/L2/L3 boundaries, ≥3 → L3.
- `apply_escalation`: alert level = max over entities; escalation block shape; a
  one-shot entity → L1.
- `derive_alert_times`: max `last_seen` over entity edges; no edge → None → L1.
- Regression: alerts unchanged in count (no suppression) — recall guard.
