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
