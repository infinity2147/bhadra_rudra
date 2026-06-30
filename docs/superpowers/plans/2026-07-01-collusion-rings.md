# Collusion Rings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect accounts secretly linked by a shared device/IP or KYC document (even with no money flow between them) on a standalone synthetic identity dataset, and show each clump in a dedicated "Collusion Rings" view — leaving the IBM AML stack untouched.

**Architecture:** A seeded synthetic identity dataset (`identity_generator`) → a pure union-find detector (`collusion_detector`) → a variant-independent read endpoint (`GET /api/collusion/rings`) → a dedicated React page with a per-ring force-graph. Real-dataset accounts (no identifier columns) yield zero rings — a tested no-op.

**Tech Stack:** Python 3 / FastAPI / pytest + FastAPI `TestClient` (backend); React 19 + Vite + Tailwind 4 + `react-force-graph-2d` (frontend).

## Global Constraints

- The collusion lane is **standalone + variant-independent**: it never reads or writes the IBM AML pipeline, adds no alerts, and mutates no state the existing pipeline uses.
- Detector is **pure** (no I/O). `min_ring_size` comes from config (`state["config"].get("collusion_min_ring_size", 3)`), default **3**. Identifiers: **`device_id`, `ip`, `kyc_doc_hash`** only — never bank-`branch`.
- Synthetic generation is **seeded (`seed=42`)** and deterministic.
- Endpoint is **read-open** (no `Depends(get_role)`/`require`) and returns `dataset: "synthetic_identity"`.
- Identity data is fabricated **only** for the standalone lane — never written onto IBM AML accounts.
- Commit after each task. Co-author trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Local commits only — do NOT push.

## File Structure

- `src/identity_generator.py` — **create**: seeded synthetic identity accounts + injected rings; writes `data/collusion/identity.json`.
- `src/collusion_detector.py` — **create**: pure `detect_collusion_rings` (union-find).
- `backend/main.py` — **modify**: imports; load identity accounts into `state` at startup; `GET /api/collusion/rings`.
- `frontend/src/api.js` — **modify**: `getCollusionRings()`.
- `frontend/src/pages/Collusion.jsx` — **create**: the view + per-ring force-graph.
- `frontend/src/App.jsx` — **modify**: import, route, nav entry.
- `tests/test_collusion_detector.py` — **create**: generator + detector tests.
- `tests/test_api_integration.py` — **modify**: endpoint test.

---

### Task 1: Synthetic identity generator

**Files:**
- Create: `src/identity_generator.py`
- Test: `tests/test_collusion_detector.py`

**Interfaces:**
- Produces: `generate_identity_accounts(*, n_accounts=150, seed=42) -> list[dict]` (each dict: `account_id, name, device_id, ip, kyc_doc_hash, is_mule`); `write_identity_dataset(path, *, n_accounts=150, seed=42) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collusion_detector.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import identity_generator


def test_generate_plants_expected_rings():
    accts = identity_generator.generate_identity_accounts(seed=42)
    assert len(accts) == 150
    kyc_ring = [a for a in accts if a["kyc_doc_hash"] == "PAN-FORGED-7F3A"]
    dev_ring = [a for a in accts if a["device_id"] == "DEV-FARM-01"]
    assert len(kyc_ring) == 8
    assert len(dev_ring) == 12
    assert all(a["is_mule"] for a in kyc_ring + dev_ring)
    # deterministic: same seed -> identical account_ids in order
    again = identity_generator.generate_identity_accounts(seed=42)
    assert [a["account_id"] for a in accts] == [a["account_id"] for a in again]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_collusion_detector.py::test_generate_plants_expected_rings -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'identity_generator'`.

- [ ] **Step 3: Implement `src/identity_generator.py`**

```python
"""Synthetic IDENTITY dataset for the Collusion Rings demo.

Standalone + variant-independent: a small set of accounts each carrying
device/IP/KYC identifiers, with a few collusion rings injected (accounts that
share an identifier). This is the only data the collusion lane runs on; the
rest of RUDRA runs on the active (IBM AML) variant. Seeded for reproducibility.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

_FIRST = ["Aarav", "Vivaan", "Aditya", "Ananya", "Diya", "Kabir", "Ishaan",
          "Myra", "Sara", "Reyansh", "Anaya", "Vihaan", "Arjun", "Saanvi",
          "Aryan", "Riya", "Dhruv", "Kiara", "Ayaan", "Navya"]
_LAST = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Gupta", "Patel",
         "Khan", "Bose", "Rao"]


def _gen_account(i: int) -> Dict:
    return {
        "account_id": f"ACC{i:05d}",
        "name": f"{_FIRST[i % len(_FIRST)]} {_LAST[(i // len(_FIRST)) % len(_LAST)]}",
        "device_id": f"DEV-{i:05d}",
        "ip": f"10.{i % 256}.{(i // 256) % 256}.{(i * 7) % 256}",
        "kyc_doc_hash": f"PAN-{i:05d}",
        "is_mule": False,
    }


def generate_identity_accounts(*, n_accounts: int = 150, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    accounts = [_gen_account(i) for i in range(n_accounts)]

    # Ring K: 8 accounts share one forged KYC document.
    for a in accounts[10:18]:
        a["kyc_doc_hash"] = "PAN-FORGED-7F3A"
        a["is_mule"] = True
    # Ring D: 12 accounts share one device + IP (one-phone mule farm).
    for a in accounts[40:52]:
        a["device_id"] = "DEV-FARM-01"
        a["ip"] = "10.66.66.66"
        a["is_mule"] = True
    # Ring X: 4 accounts share BOTH a device and a KYC doc (transitive case).
    for a in accounts[80:84]:
        a["device_id"] = "DEV-X-9"
        a["kyc_doc_hash"] = "PAN-X-9"
        a["is_mule"] = True
    # Benign noise: 2 accounts share a device (below default min_ring_size=3).
    for a in accounts[100:102]:
        a["device_id"] = "DEV-FAMILY-2"

    rng.shuffle(accounts)
    return accounts


def write_identity_dataset(path: str, *, n_accounts: int = 150, seed: int = 42) -> List[Dict]:
    accounts = generate_identity_accounts(n_accounts=n_accounts, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(accounts, f, indent=2)
    return accounts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_collusion_detector.py::test_generate_plants_expected_rings -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/identity_generator.py tests/test_collusion_detector.py
git commit -m "feat(collusion): seeded synthetic identity dataset with injected rings"
```

---

### Task 2: Collusion detector

**Files:**
- Create: `src/collusion_detector.py`
- Test: `tests/test_collusion_detector.py`

**Interfaces:**
- Produces: `detect_collusion_rings(accounts, *, identifiers=("device_id","ip","kyc_doc_hash"), min_ring_size=3) -> list[dict]`; each ring `{ring_id, account_ids, size, shared_identifiers:[{type,value,count}]}`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_collusion_detector.py
import collusion_detector as cd


def _acc(aid, **kw):
    base = {"account_id": aid, "device_id": None, "ip": None, "kyc_doc_hash": None}
    base.update(kw)
    return base


def test_planted_kyc_ring_detected():
    accts = [_acc(f"A{i}", kyc_doc_hash="PAN-FORGED") for i in range(5)] + \
            [_acc(f"B{i}", kyc_doc_hash=f"PAN-{i}") for i in range(3)]
    rings = cd.detect_collusion_rings(accts, min_ring_size=3)
    assert len(rings) == 1
    assert rings[0]["size"] == 5
    assert any(s["type"] == "kyc_doc_hash" and s["value"] == "PAN-FORGED"
               for s in rings[0]["shared_identifiers"])


def test_transitive_merge_across_identifiers():
    # A&B share a device; B&C share a kyc doc -> one ring {A,B,C}
    accts = [
        _acc("A", device_id="D1"),
        _acc("B", device_id="D1", kyc_doc_hash="K1"),
        _acc("C", kyc_doc_hash="K1"),
    ]
    rings = cd.detect_collusion_rings(accts, min_ring_size=3)
    assert len(rings) == 1
    assert set(rings[0]["account_ids"]) == {"A", "B", "C"}


def test_below_threshold_not_flagged():
    accts = [_acc("A", device_id="D1"), _acc("B", device_id="D1")]
    assert cd.detect_collusion_rings(accts, min_ring_size=3) == []


def test_no_identifier_columns_is_noop():
    accts = [{"account_id": f"A{i}"} for i in range(10)]
    assert cd.detect_collusion_rings(accts, min_ring_size=3) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_collusion_detector.py -k "kyc_ring or transitive or below_threshold or noop" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'collusion_detector'`.

- [ ] **Step 3: Implement `src/collusion_detector.py`**

```python
"""Detect collusion rings: accounts linked by a shared identifier
(device/IP/KYC document) even with no money flow between them.

Pure function. Same union-find shape as incident_clustering.py, but the
linking edge is a shared *identity attribute*, not a transaction.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

DEFAULT_IDENTIFIERS = ("device_id", "ip", "kyc_doc_hash")


def detect_collusion_rings(accounts: List[Dict], *,
                           identifiers: Sequence[str] = DEFAULT_IDENTIFIERS,
                           min_ring_size: int = 3) -> List[Dict]:
    n = len(accounts)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # (identifier, value) -> indices of accounts carrying it
    value_to_idx: Dict[tuple, List[int]] = defaultdict(list)
    for i, acc in enumerate(accounts):
        for key in identifiers:
            v = acc.get(key)
            if v is None or v == "":
                continue
            value_to_idx[(key, v)].append(i)

    for idxs in value_to_idx.values():
        for j in range(1, len(idxs)):
            union(idxs[0], idxs[j])

    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)

    rings: List[Dict] = []
    ring_no = 0
    for _, members in sorted(comps.items()):
        if len(members) < min_ring_size:
            continue
        member_set = set(members)
        shared = []
        for (key, value), idxs in value_to_idx.items():
            in_comp = sum(1 for i in idxs if i in member_set)
            if in_comp >= 2:
                shared.append({"type": key, "value": value, "count": in_comp})
        ring_no += 1
        rings.append({
            "ring_id": f"RING-{ring_no:03d}",
            "account_ids": [accounts[i]["account_id"] for i in members],
            "size": len(members),
            "shared_identifiers": sorted(shared, key=lambda s: -s["count"]),
        })
    return rings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_collusion_detector.py -v`
Expected: PASS (all tests, including Task 1's generator test).

- [ ] **Step 5: Commit**

```bash
git add src/collusion_detector.py tests/test_collusion_detector.py
git commit -m "feat(collusion): union-find shared-identifier ring detector"
```

---

### Task 3: Backend endpoint + startup load

**Files:**
- Modify: `backend/main.py` (imports near the other `src/` imports; a `state` slot; a load block in `load_or_generate`; the route near other read endpoints)
- Test: `tests/test_api_integration.py`

**Interfaces:**
- Consumes: `identity_generator.write_identity_dataset`, `collusion_detector.detect_collusion_rings` (Tasks 1–2); existing `state`, `DATA_DIR`, `state["config"]` (a `ConfigStore` with `.get(key, default)`).
- Produces: `GET /api/collusion/rings` → `{dataset:"synthetic_identity", rings:[...], total:int, n_accounts:int}`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_api_integration.py
def test_collusion_rings_endpoint(client):
    r = client.get("/api/collusion/rings", headers=INV)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dataset"] == "synthetic_identity"
    assert body["total"] >= 3  # planted: forged-KYC, device-farm, both-share rings
    ring = body["rings"][0]
    assert {"ring_id", "account_ids", "size", "shared_identifiers"} <= set(ring)
    assert ring["size"] >= 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_api_integration.py -k collusion -v`
Expected: FAIL — 404 (route not defined) → assertion error on status.

- [ ] **Step 3: Add imports**

Near `backend/main.py`'s other `src/` imports (e.g. by the `from fraud_detector import ...` group), add:

```python
from collusion_detector import detect_collusion_rings
from identity_generator import write_identity_dataset
```

- [ ] **Step 4: Add a `state` slot + startup load**

Add `"identity_accounts": None,` to the `state = {...}` dict initializer (alongside the other slots).

Inside `load_or_generate()`, after the existing loads, add (the try/except mirrors the project's startup-resilience pattern so a failure here can't break boot):

```python
    # Synthetic identity lane for collusion detection (variant-independent,
    # standalone — does NOT touch the active-variant pipeline). Generate on
    # first boot if absent.
    try:
        identity_path = os.path.join(DATA_DIR, "collusion", "identity.json")
        if os.path.exists(identity_path):
            with open(identity_path) as f:
                state["identity_accounts"] = json.load(f)
        else:
            state["identity_accounts"] = write_identity_dataset(identity_path)
    except Exception as e:
        print(f"[backend] collusion identity lane unavailable: {e}")
        state["identity_accounts"] = []
```

- [ ] **Step 5: Add the route**

Place near the other read endpoints (e.g. after the incidents endpoints):

```python
@app.get("/api/collusion/rings")
def get_collusion_rings():
    accounts = state.get("identity_accounts") or []
    min_size = int(state["config"].get("collusion_min_ring_size", 3))
    rings = detect_collusion_rings(accounts, min_ring_size=min_size)
    return {"dataset": "synthetic_identity", "rings": rings,
            "total": len(rings), "n_accounts": len(accounts)}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/test_api_integration.py -k collusion -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/main.py tests/test_api_integration.py
git commit -m "feat(collusion): GET /api/collusion/rings + startup identity load"
```

---

### Task 4: Frontend Collusion Rings view

**Files:**
- Modify: `frontend/src/api.js` (add helper next to the other `fetchAPI` calls)
- Create: `frontend/src/pages/Collusion.jsx`
- Modify: `frontend/src/App.jsx` (import + `<Route>` + a `NAV_GROUPS` item — read the file first to match the item shape `{to,label,icon}`)
- Verify: browser (no component test harness — manual per project convention)

**Interfaces:**
- Consumes: `GET /api/collusion/rings` (Task 3); `react-force-graph-2d` (already used in `pages/Graph.jsx`).
- Produces: `getCollusionRings()` in `api.js`; `Collusion` page at route `/collusion`.

- [ ] **Step 1: Add the API helper**

In `frontend/src/api.js`, add (matches the existing `fetchAPI` style):

```js
export const getCollusionRings = () => fetchAPI('/api/collusion/rings');
```

- [ ] **Step 2: Create `frontend/src/pages/Collusion.jsx`**

```jsx
import { useEffect, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { getCollusionRings } from '../api';

function RingGraph({ ring }) {
  const nodes = ring.account_ids.map((id) => ({ id, kind: 'account' }));
  const links = [];
  (ring.shared_identifiers || []).forEach((s) => {
    const hubId = `${s.type}:${s.value}`;
    nodes.push({ id: hubId, kind: 'identifier' });
    ring.account_ids.forEach((a) => links.push({ source: a, target: hubId }));
  });
  return (
    <div className="h-56 border border-gray-200 rounded bg-gray-50">
      <ForceGraph2D
        graphData={{ nodes, links }}
        width={420}
        height={220}
        nodeRelSize={5}
        nodeLabel="id"
        nodeColor={(n) => (n.kind === 'identifier' ? '#dc2626' : '#4f46e5')}
        linkColor={() => '#cbd5e1'}
      />
    </div>
  );
}

export default function Collusion() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    getCollusionRings().then(setData).catch(() => setErr('Failed to load collusion rings'));
  }, []);

  if (err) return <div className="p-6 text-red-600">{err}</div>;
  if (!data) return <div className="p-6 text-gray-500">Loading…</div>;

  return (
    <div className="p-6 space-y-4">
      <div>
        <h2 className="text-xl font-bold text-gray-900">Collusion Rings</h2>
        <p className="text-sm text-gray-600">
          Accounts secretly linked by a shared device/IP or KYC document — caught even when
          they never transact with each other.
        </p>
      </div>
      <div className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
        Synthetic identity demo — this lane runs on a generated identity dataset
        ({data.n_accounts} accounts). The rest of RUDRA runs on the IBM AML dataset.
      </div>

      {data.rings.length === 0 && (
        <div className="text-gray-500">No collusion rings detected.</div>
      )}

      <div className="space-y-6">
        {data.rings.map((ring) => (
          <div key={ring.ring_id} className="border border-gray-200 rounded-lg p-4 bg-white">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">{ring.ring_id}</h3>
              <span className="text-xs text-gray-500">{ring.size} accounts</span>
            </div>
            <ul className="mt-1 text-sm text-gray-700 list-disc ml-5">
              {ring.shared_identifiers.map((s, i) => (
                <li key={i}>
                  {s.count} accounts share <span className="font-mono">{s.type}</span> ={' '}
                  <span className="font-mono">{s.value}</span>
                </li>
              ))}
            </ul>
            <div className="mt-3">
              <RingGraph ring={ring} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Wire route + nav into `App.jsx`**

Read `frontend/src/App.jsx`. Then: (a) add `import Collusion from './pages/Collusion';` with the other page imports; (b) add `<Route path="/collusion" element={<Collusion />} />` inside `<Routes>`; (c) add an item to the appropriate `NAV_GROUPS` group, matching the existing item shape — use this entry:

```js
{ to: '/collusion', label: 'Collusion Rings', icon: 'M18 18.72a9.094 9.094 0 003.741-.479 3 3 0 00-4.682-2.72m.94 3.198l.001.031c0 .225-.012.447-.037.666A11.944 11.944 0 0112 21c-2.17 0-4.207-.576-5.963-1.584A6.062 6.062 0 016 18.719m12 0a5.971 5.971 0 00-.941-3.197m0 0A5.995 5.995 0 0012 12.75a5.995 5.995 0 00-5.058 2.772m0 0a3 3 0 00-4.681 2.72 8.986 8.986 0 003.74.477m.94-3.197a5.971 5.971 0 00-.94 3.197M15 6.75a3 3 0 11-6 0 3 3 0 016 0zm6 3a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0zm-13.5 0a2.25 2.25 0 11-4.5 0 2.25 2.25 0 014.5 0z' }
```

- [ ] **Step 4: Lint + build**

Run: `cd frontend && npm run lint && npm run build`
Expected: no errors.

- [ ] **Step 5: Manual verification (project convention: verify in the running app)**

Start the stack, open `/collusion`, and confirm: the amber synthetic-demo banner shows; at least 3 rings render (forged-KYC ring of 8, device-farm ring of 12, the both-share ring); each ring lists its binding identifier(s) and draws a star clump around the red identifier hub(s).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.js frontend/src/pages/Collusion.jsx frontend/src/App.jsx
git commit -m "feat(collusion): Collusion Rings view"
```

---

### Task 5: Full-suite regression

- [ ] **Step 1: Backend suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (existing + new collusion tests).

- [ ] **Step 2: Frontend lint + build (final)**

Run: `cd frontend && npm run lint && npm run build`
Expected: clean.

---

## Self-Review

**Spec coverage:**
- Synthetic identity dataset (§4.1) → Task 1. ✓
- Detector incl. transitive merge + no-op-on-missing-columns (§4.2) → Task 2. ✓
- Variant-independent read endpoint + startup load + config-driven threshold (§4.3) → Task 3. ✓
- Dedicated view + banner + per-ring graph + nav (§4.4) → Task 4. ✓
- Testing (§6) → Tasks 1–3 tests + Task 5 regression. ✓
- No IBM AML integration / no bank-branch / no fabrication onto real data (§3 non-goals) → enforced by Global Constraints; the endpoint reads only `state["identity_accounts"]`.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — every code step is literal. ✓

**Type consistency:** `detect_collusion_rings` signature + ring keys (`ring_id, account_ids, size, shared_identifiers`) produced in Task 2 are consumed verbatim by the endpoint (Task 3), the endpoint test, and the frontend (Task 4). Account dict keys (`account_id, device_id, ip, kyc_doc_hash`) produced in Task 1 match the detector's `identifiers` + output. `state["identity_accounts"]` set in Task 3 startup, read by the Task 3 route. ✓
