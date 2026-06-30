# Forensic RCA + Foresight — Design Spec

**Date:** 2026-07-01
**Status:** Approved (strategy); pending implementation plan
**Author:** RUDRA team
**Origin:** PSB Hackathon evaluator feedback (post-presentation round)

---

## 1. Why

Two evaluators converged on the same axis:

- **Evaluator 1:** *"Can consider more features, use of AI for predictive analysis, recommendation to be given to solve for problem from its root."*
- **Evaluator 2:** *"Team can focus on Forensic usages of the solution as a part of RCA, which will widen the scope."*

RUDRA today answers **"is this fraud?"** in sub-second. The feedback asks it to answer the three questions a bank risk committee asks *next*:

1. **How did it get through?** — forensic reconstruction (Eval 2: forensic / RCA)
2. **Who's next?** — explainable forward-looking prediction (Eval 1: predictive analysis)
3. **What do we change so it can't recur?** — prescriptive remediation (Eval 1: solve from the root)

This is a **legibility/assembly** gap, not a capability gap: most building blocks already exist but aren't assembled into one narrative. That makes it the highest-leverage, lowest-risk thing to build.

## 2. The deliverable

A **per-incident RCA dossier** attached to each clustered incident (RUDRA already clusters alerts into incidents via `incident_clustering.cluster_alerts`, served at `GET /api/incidents/{id}`). The dossier has four sections, each mapped to an evaluator ask:

| # | Section | Evaluator ask | Primary data source (existing) |
|---|---------|---------------|--------------------------------|
| 1 | **Forensic reconstruction** — *how it happened* | Eval 2 | `fund_tracer.trace_journey` / `trace_for_alert`, `fatf_typology.TYPOLOGY` |
| 2 | **Root-cause diagnosis** — *why it succeeded* | Eval 1 (root) | NEW control-gap taxonomy (extends `fatf_typology.TYPOLOGY`) |
| 3 | **Foresight** — *who's next* | Eval 1 (predictive) | `taint_store`, `recurrence`, `risk_score_learner` |
| 4 | **Prescriptive recommendations** — *what to fix* | Eval 1 (recommendation) | §2 taxonomy + §3 ranking |

## 3. Goals / Non-goals

**Goals**
- One coherent feature answering all four evaluator points, demoable end-to-end.
- Reuse existing engine modules; minimal new ML.
- Every RCA line bound to **specific evidence** (this txn / this threshold / this account) — never generic boilerplate.
- An AI-narrated summary (via existing `llm_copilot.LLMCopilot`) with a deterministic local fallback when no `ANTHROPIC_API_KEY`.

**Non-goals**
- **No new "fraud forecasting" ML model.** Per `memory/project_ml_gnn_levers`, feature injection is a no-op and the recall ceiling is structural. Foresight is framed as *explainable graph-propagation* (taint × behavioral risk), NOT a black-box predictor. We will not claim a metrics lift we cannot defend.
- No change to detection recall/precision. RCA is a post-detection investigation layer; it adds no alerts and suppresses none.
- No new datasets.

## 4. Architecture

### 4.1 New engine module: `src/rca_engine.py`
Single responsibility: assemble an RCA dossier for an incident. Pure functions; no I/O beyond reading `state`-provided objects (graph, transactions, taint store, config).

```
build_rca(incident, *, graph, transactions, taint, config, risk_map) -> dict
    ├─ reconstruct(incident, graph, transactions)        # §4.2 — wraps fund_tracer
    ├─ diagnose_root_cause(incident)                      # §4.3 — control-gap taxonomy
    ├─ foresight(incident, graph, taint, risk_map)        # §4.4 — next targets + exposure
    └─ recommend(diagnosis, foresight)                    # §4.5 — actions tied to gaps
```

`build_rca` returns a JSON-serializable dict; the LLM narrative is layered on top by the endpoint (lazy, like SHAP reasons) so the core dossier works without an API key.

### 4.2 Forensic reconstruction (§ dossier 1)
Wrap `fund_tracer.trace_for_alert(incident["primary_alert_id"], ...)` to get the Sankey node/link list + chronological timeline + terminal classification already produced for the Journey page. Add:
- **Origin / cash-out attribution**: terminal classification (`_classify_terminal`) labels source vs integration nodes.
- **Method classification**: `incident["primary_pattern"]` → `fatf_typology.TYPOLOGY[pattern]` gives the FATF code + typology label already used in SAR generation.

This section is ~80% assembly of existing tracer output.

### 4.3 Root-cause diagnosis (§ dossier 2) — the new substance
Extend `fatf_typology.TYPOLOGY` (or a parallel `CONTROL_GAP` map keyed by the same pattern names) with two fields per pattern: `control_gap` and `remediation`. Grounded in RUDRA's actual detectors:

| Pattern (TYPOLOGY key) | `control_gap` | `remediation` |
|---|---|---|
| Smurfing / Structuring | CTR threshold evaluated per-txn, not aggregated across window + branches | Rolling per-beneficiary aggregation; lower effective threshold |
| Dormant Activation | No re-KYC / step-up trigger on dormancy reactivation | Auto-EDD when dormant > N days reactivates with > X volume |
| Rapid Layering / Circular Transaction | Cross-branch / cross-bank velocity not correlated in real time | Real-time velocity rollup (RUDRA does this — recommend rollout) |
| Shell Company Funnel | Weak account-opening due diligence; BO unverified | Fan-in flag within N days of opening; BO verification gate |
| Profile Mismatch | Stale behavioral KYC | Periodic behavioral re-score |
| Recruiter / Coordinator | Mule-network linkage not surfaced at onboarding | Graph-proximity check to known mules at onboarding |

Each diagnosis line is **bound to evidence**: e.g. "7 deposits of ₹1.9L each (all under the ₹2L CTR threshold) across 3 branches in 4 days" — pulled from the actual transactions in the incident, not templated.

### 4.4 Foresight (§ dossier 3) — explainable, not black-box
- **Next-target ranking**: `taint_store.get_all()` already propagates decaying suspicion over the graph from confirmed-bad seeds. Surface the top-K warm-but-unflagged neighbors of the incident's entities, each with `taint × risk_score_learner` score and the hop-distance explanation ("2 hops from confirmed mule, taint 0.7").
- **Ring-growth trajectory**: `recurrence.compute_recurrence` → is this incident's entity set expanding across time windows? Flag "growing" vs "contained".
- **Exposure projection**: sum of outbound flow capacity through predicted-next entities → "₹X at risk if uncontained over next N days." Deterministic, explainable.

Framing in the UI and narrative: *explainable graph-propagation prediction* — defensible to bank risk officers, and already real in the codebase.

### 4.5 Prescriptive recommendations (§ dossier 4)
Two tiers, each tied to the gap it closes:
- **Account-level (immediate)**: freeze / EDD on the §4.4 predicted-next entities.
- **Policy-level (root)**: the `remediation` from §4.3, expressed as a concrete config change (threshold/window/trigger).

### 4.6 The demo loop — recommendation → proof (headline)
RUDRA already has `POST /api/simulate/scenario/{name}` and a SimulationStudio page. Close the loop:

> A policy-level recommendation (e.g. "lower CTR aggregation window") is applied to a `ConfigStore` override → the **same incident's transactions are replayed** through detection under the new config → the dossier shows "this incident would have been flagged 4 hops / 2 days earlier."

This turns "we recommend X" into "here's proof X works" — the most literal answer to *"solve the problem from its root."* Build last, demo first.

### 4.7 New endpoint
`GET /api/incidents/{id}/rca` → returns the dossier dict. RBAC: same read permission as `GET /api/incidents/{id}`. The LLM narrative is generated lazily here (cached in `state` view cache, invalidated on retrain/rerun like other derived views).

### 4.8 Frontend
A new **RCA** view (tab on the existing Incident detail, or a dedicated page reachable from Incidents). Four collapsible sections matching the dossier. The SimulationStudio "prove the fix" loop links out from the recommendation section.

## 5. Phasing (leverage-ordered)

| Phase | Scope | Effort | Rationale |
|---|---|---|---|
| **1** | Forensic reconstruction + AI RCA narrative + `GET /api/incidents/{id}/rca` + RCA view | ~hours | 80% assembly; instant demo value |
| **2** | Control-gap taxonomy → prescriptive recommendations (`rca_engine` §4.3/§4.5) | ~half day | The "root cause" substance |
| **3** | Foresight: next-target ranking + exposure projection (§4.4) | ~half day | Reuses taint/recurrence; the "predictive" box |
| **4** | Recommendation → SimulationStudio "prove the fix" loop (§4.6) | ~day | The wow moment |

**Must-haves:** Phases 1 + 2 (hit 3 of 4 evaluator points, low risk). **Headline:** Phase 4 if a day is available. Phase 3 is real but least visually striking — sequence after 4 if time is tight.

## 6. Testing
- `tests/test_rca_engine.py`: dossier shape; evidence-binding (diagnosis references a real txn/threshold from the incident); control-gap present for every TYPOLOGY pattern; foresight excludes already-flagged entities; exposure is non-negative.
- Endpoint test: `GET /api/incidents/{id}/rca` returns 200 with all four sections; RBAC parity with incident detail; graceful behavior with no `ANTHROPIC_API_KEY` (deterministic fallback narrative).
- Phase 4: replay-under-new-config detects the incident at least as early as baseline.

## 7. Risks & mitigations
- **Predictive theater** → foresight is graph-propagation + taint only; no fabricated ML forecast (see §3 non-goals).
- **Generic-feeling RCA** → hard requirement that every diagnosis line cites specific incident evidence; enforced by a test.
- **Scope creep** → Phases 1–2 are the committed MVP; 3–4 are explicitly optional and independent.

## 8. Open questions (resolve during planning)
- RCA view: tab on Incident detail vs. standalone page?
- Foresight K (number of predicted next-targets surfaced) — fixed or severity-scaled?
- Phase 4 config-override mechanism: transient `ConfigStore` override vs. a dedicated sandbox path.
