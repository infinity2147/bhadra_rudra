# Collusion Rings (Shared-Identifier Detection) — Design Spec

**Date:** 2026-07-01
**Status:** Approved (design); pending implementation plan
**Origin:** Evaluator/teammate suggestion — a graph DB should make accounts sharing a compromised KYC document or device/IP "clump together" visibly.

---

## 1. Why

RUDRA's transaction graph links accounts by **money flow** (A paid B). A disciplined mule farm deliberately avoids money links between its mules — but it cannot avoid sharing the **forged KYC document** or the **device/IP** the accounts were opened from. Linking accounts by **shared identity attributes** catches rings that the money-flow detectors are structurally blind to (accounts that never transact with each other but share an identifier). This is a new, complementary detection dimension and a strong forensic visual.

## 2. Honest data constraint (drives the architecture)

The live stack runs on the **IBM AML** dataset, which has **no** device/IP or KYC-document fields (and the only `branch`-like field is the coarse *bank*, which would over-cluster — thousands of accounts share one bank). It must **stay on IBM AML**.

Therefore this feature is a **standalone, clearly-labeled synthetic *identity* lane**:
- It operates on a small **synthetic identity dataset** we generate (accounts carrying device/IP/KYC + a few injected collusion rings).
- It is **independent of `ACTIVE_VARIANT`** — it serves the same synthetic identity data while the rest of the backend runs IBM AML.
- The **detection logic is real graph analysis**; only the identifiers are synthetic. It is labeled "synthetic identity demo" in the API and UI so it never pretends to mine IBM AML.
- On any account set lacking the identifier columns, the detector returns **zero rings** (graceful no-op) — a tested safety net.

## 3. Goals / Non-goals

**Goals**
- Detect groups of accounts linked by a shared `device_id`, `ip`, or `kyc_doc_hash`, even with no money flow between them.
- A dedicated "Collusion Rings" view showing each clump and the identifier that binds it.
- Pure, testable detector; deterministic seeded synthetic data.
- Zero impact on the IBM AML pipeline (no new alerts, no state mutation, no variant change).

**Non-goals**
- **No integration into IBM AML incidents/RCA** — the two datasets share no accounts, so a cross-link would be meaningless. Deliberately out of scope.
- **No bank-`branch` keying** — too coarse; would over-cluster. (A *granular* synthetic onboarding identifier was considered and deferred; device/IP + KYC are the chosen signals.)
- No fabrication of identity fields onto IBM AML accounts (that would be theater).
- No new ML model.

## 4. Architecture

Four small, independently-testable units.

### 4.1 Synthetic identity dataset — `src/identity_generator.py`
Produces `data/collusion/identity.json` (a **standalone** artifact, not under a variant dir — it is variant-independent). Seeded for reproducibility.

- ~150 accounts, each: `account_id`, `name`, `device_id`, `ip`, `kyc_doc_hash`, `is_mule` (ground-truth label).
- Most accounts get **unique** identifiers.
- **Injected rings** (the ground truth the detector must recover), e.g.:
  - Ring K: 8 accounts share one `kyc_doc_hash` (forged PAN).
  - Ring D: 12 accounts share one `device_id` + `ip` (one-phone mule farm).
  - Ring X: a small ring sharing **both** a device and a KYC doc (tests transitive merge).
- **Benign noise**: one 2-account device share (below `min_ring_size`) so the detector isn't trivially perfect.
- Names follow the existing synthetic style (à la `data_generator.py`).

### 4.2 Detector — `src/collusion_detector.py` (pure)
```
detect_collusion_rings(accounts, *, identifiers=("device_id","ip","kyc_doc_hash"),
                       min_ring_size=3) -> list[dict]
```
- Union-find over accounts: for each identifier column, group accounts by value; for any value shared by ≥2 accounts, `union` them. (Same union-find shape as `incident_clustering.py`.)
- Connected components of size ≥ `min_ring_size` are rings.
- Each ring: `{ring_id, account_ids: [...], size, shared_identifiers: [{type, value, count}, ...]}` — `shared_identifiers` lists every identifier value that binds ≥2 members (the "smoking gun").
- Pure (no I/O); `min_ring_size` supplied by the caller (backend reads it from `ConfigStore`, key `collusion_min_ring_size`, default 3).
- Accounts missing an identifier key contribute no shared value → never unioned on it. An account set with none of the columns → returns `[]`.

### 4.3 Endpoint — `GET /api/collusion/rings`
- Read-open (parity with other read endpoints; no RBAC gate).
- Loads the synthetic identity dataset (pre-warmed into `state["identity_accounts"]` at startup, regenerating via `identity_generator` if `data/collusion/identity.json` is absent — wrapped in try/except like the rest of startup).
- Runs `detect_collusion_rings` with `min_ring_size` from `ConfigStore`.
- Returns `{"dataset": "synthetic_identity", "rings": [...], "total": N, "n_accounts": M}`.
- **Independent of `ACTIVE_VARIANT`** — always serves the synthetic identity lane.

### 4.4 Frontend — `frontend/src/pages/Collusion.jsx` + nav entry
- A prominent **"Synthetic identity demo"** banner (states plainly this lane uses generated identity data; the rest of RUDRA runs on IBM AML).
- A list of rings, each headlined by its binding identifier in plain language, e.g. *"8 accounts share KYC document PAN-7F3A — likely one forged document."*
- Each ring renders a small `react-force-graph-2d` (the library already used in `pages/Graph.jsx`) showing the member accounts clustered around the shared-identifier hub node.
- API helper `getCollusionRings()` added to `src/api.js` in the existing `fetchAPI` style; route + nav entry added wherever the app registers pages.

## 5. Data flow
startup → load/generate `identity.json` into `state` → `GET /api/collusion/rings` runs detector → frontend renders rings + per-ring graph.

## 6. Testing
- `tests/test_collusion_detector.py`:
  - planted KYC ring of 8 → one ring, size 8, `shared_identifiers` names that `kyc_doc_hash`.
  - transitive: A&B share device, B&C share kyc → single ring `{A,B,C}`.
  - a 2-account share with `min_ring_size=3` → not flagged.
  - account set with none of the identifier columns → `[]` (the real-dataset no-op).
  - `identity_generator` (seeded) plants exactly the designed rings → detector recovers them.
- Endpoint test (`tests/test_api_integration.py`): `GET /api/collusion/rings` → 200, `dataset == "synthetic_identity"`, rings have the documented shape, total ≥ the planted ring count.

## 7. Risks & mitigations
- **Looking like theater** → explicit "synthetic identity" labeling in API + UI; detector logic is real and tested; no identity fields are fabricated onto real data.
- **Disturbing the IBM AML stack** → the lane is variant-independent and additive (new module, new endpoint, new page); no change to detectors, alerts, or state used by the existing pipeline.
- **Over-clustering** → only high-signal identifiers (device/IP, KYC doc); bank-branch explicitly excluded; `min_ring_size` gate (default 3) configurable.

## 8. Open questions (resolve during planning)
- Exact nav placement / route path for the new page.
- Whether `identity.json` is generated at startup-if-missing vs. by an explicit `run_pipeline` step (default: startup-if-missing, committed generator).
