"""
Geographic aggregation for the India fund-flow map.

Branch names in the data carry their city ("Mumbai Fort", "Pune FC Road"), so we
extract the city and aggregate the real transaction network into inter-city
flows + per-city fraud hotspots. Unknown branch strings map deterministically to
a city so nothing is dropped from the map.

The amounts and fraud counts are real; only the city *assignment* of an unknown
branch is synthetic — which we surface honestly in the UI.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

# Approximate (lat, lng) for major Indian banking hubs — all within India's
# bounding box (~6-37N, 68-98E).
INDIA_CITIES: Dict[str, tuple] = {
    "Mumbai": (19.07, 72.88),
    "Delhi": (28.61, 77.21),
    "Bangalore": (12.97, 77.59),
    "Chennai": (13.08, 80.27),
    "Kolkata": (22.57, 88.36),
    "Hyderabad": (17.39, 78.49),
    "Pune": (18.52, 73.86),
    "Ahmedabad": (23.03, 72.58),
    "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95),
    "Surat": (21.17, 72.83),
    "Nagpur": (21.15, 79.09),
    "Kanpur": (26.45, 80.33),
    "Chandigarh": (30.73, 76.78),
    "Bhopal": (23.26, 77.41),
    "Patna": (25.59, 85.14),
    "Kochi": (9.93, 76.27),
    "Guwahati": (26.14, 91.74),
    "Indore": (22.72, 75.86),
    "Coimbatore": (11.02, 76.96),
}

_CITY_LIST = list(INDIA_CITIES.keys())


def branch_to_city(branch: str) -> str:
    """Resolve a branch string to a city. Prefers an explicit city name in the
    branch ('Mumbai Fort' -> 'Mumbai'); otherwise hashes deterministically so a
    given branch always lands on the same city."""
    if not branch:
        return _CITY_LIST[0]
    low = str(branch).lower()
    for city in _CITY_LIST:
        if city.lower() in low:
            return city
    # Deterministic, stable fallback (no Math.random-style instability).
    idx = sum(ord(c) for c in str(branch)) % len(_CITY_LIST)
    return _CITY_LIST[idx]


def city_flows(transactions: pd.DataFrame) -> Dict:
    """Aggregate transactions into inter-city flows + per-city hotspots.

    Returns ``{"cities": [...], "flows": [...]}`` where each city has
    inflow/outflow/fraud_volume/txn_count/fraud_rate + coords, and each flow has
    source/target city, total amount, txn_count and fraud_count/fraud_amount.
    """
    needed = {"sender_branch", "receiver_branch", "amount"}
    if transactions is None or len(transactions) == 0 or not needed.issubset(transactions.columns):
        return {"cities": [], "flows": []}

    df = transactions[["sender_branch", "receiver_branch", "amount"]].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["is_fraud"] = (
        pd.to_numeric(transactions["is_fraud"], errors="coerce").fillna(0).astype(int)
        if "is_fraud" in transactions.columns else 0
    )
    df["src_city"] = df["sender_branch"].map(branch_to_city)
    df["dst_city"] = df["receiver_branch"].map(branch_to_city)

    # Inter-city flows
    flows = []
    grouped = df.groupby(["src_city", "dst_city"])
    for (src, dst), g in grouped:
        amt = float(g["amount"].sum())
        fraud_amt = float(g.loc[g["is_fraud"] > 0, "amount"].sum())
        flows.append({
            "source": src,
            "target": dst,
            "amount": round(amt, 2),
            "txn_count": int(len(g)),
            "fraud_count": int((g["is_fraud"] > 0).sum()),
            "fraud_amount": round(fraud_amt, 2),
        })

    # Per-city hotspots
    cities = []
    all_cities = set(df["src_city"]) | set(df["dst_city"])
    for city in all_cities:
        out_g = df[df["src_city"] == city]
        in_g = df[df["dst_city"] == city]
        outflow = float(out_g["amount"].sum())
        inflow = float(in_g["amount"].sum())
        fraud_volume = float(
            out_g.loc[out_g["is_fraud"] > 0, "amount"].sum()
            + in_g.loc[in_g["is_fraud"] > 0, "amount"].sum()
        )
        total = inflow + outflow
        lat, lng = INDIA_CITIES.get(city, (None, None))
        cities.append({
            "city": city,
            "lat": lat,
            "lng": lng,
            "inflow": round(inflow, 2),
            "outflow": round(outflow, 2),
            "txn_count": int(len(out_g) + len(in_g)),
            "fraud_volume": round(fraud_volume, 2),
            "fraud_rate": round(fraud_volume / total, 4) if total > 0 else 0.0,
        })

    cities.sort(key=lambda c: c["fraud_volume"], reverse=True)
    flows.sort(key=lambda f: f["amount"], reverse=True)
    return {"cities": cities, "flows": flows}
