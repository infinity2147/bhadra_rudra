"""Geo aggregation: project the (real) branch network onto Indian cities and
aggregate inter-city fund flows + fraud hotspots for the map view. Branch names
already carry their city (e.g. 'Mumbai Fort'), so the flows are real."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from geo import INDIA_CITIES, branch_to_city, city_flows  # noqa: E402


def test_known_city_extracted_from_branch():
    assert branch_to_city("Mumbai Fort") == "Mumbai"
    assert branch_to_city("Pune FC Road") == "Pune"
    assert branch_to_city("Delhi Connaught Place") == "Delhi"


def test_unknown_branch_maps_deterministically():
    a = branch_to_city("Zeta Sector 9")
    b = branch_to_city("Zeta Sector 9")
    assert a == b and a in INDIA_CITIES        # stable + valid city


def test_every_city_has_valid_coordinates():
    for city, (lat, lng) in INDIA_CITIES.items():
        assert 6.0 <= lat <= 37.5, f"{city} lat out of India range"
        assert 68.0 <= lng <= 98.0, f"{city} lng out of India range"


def test_city_flows_aggregate_amounts_and_fraud():
    df = pd.DataFrame([
        {"sender_branch": "Mumbai Fort", "receiver_branch": "Delhi Connaught Place", "amount": 100.0, "is_fraud": 1},
        {"sender_branch": "Mumbai Fort", "receiver_branch": "Delhi Connaught Place", "amount": 50.0, "is_fraud": 0},
        {"sender_branch": "Pune FC Road", "receiver_branch": "Mumbai Fort", "amount": 30.0, "is_fraud": 0},
    ])
    out = city_flows(df)
    flows = {(f["source"], f["target"]): f for f in out["flows"]}
    assert ("Mumbai", "Delhi") in flows
    assert flows[("Mumbai", "Delhi")]["amount"] == 150.0
    assert flows[("Mumbai", "Delhi")]["fraud_count"] == 1
    cities = {c["city"]: c for c in out["cities"]}
    assert "Mumbai" in cities and "Delhi" in cities
    assert "lat" in cities["Mumbai"] and "lng" in cities["Mumbai"]
    assert cities["Delhi"]["inflow"] == 150.0       # both Mumbai->Delhi txns land here


def test_fraud_volume_is_receiver_attributed_and_sums_to_total():
    """Fraud is attributed to the RECEIVING city only, so Σ(city fraud_volume)
    equals the dataset's total fraud volume — no double-counting across endpoints."""
    df = pd.DataFrame([
        {"sender_branch": "Mumbai Fort", "receiver_branch": "Delhi Connaught Place", "amount": 100.0, "is_fraud": 1},
        {"sender_branch": "Pune FC Road", "receiver_branch": "Mumbai Fort", "amount": 40.0, "is_fraud": 1},
        {"sender_branch": "Mumbai Fort", "receiver_branch": "Pune FC Road", "amount": 25.0, "is_fraud": 0},
    ])
    out = city_flows(df)
    cities = {c["city"]: c for c in out["cities"]}
    total_fraud = float(df.loc[df["is_fraud"] > 0, "amount"].sum())  # 140
    assert round(sum(c["fraud_volume"] for c in out["cities"]), 2) == round(total_fraud, 2)
    assert cities["Delhi"]["fraud_volume"] == 100.0   # fraud RECEIVED by Delhi
    assert cities["Mumbai"]["fraud_volume"] == 40.0   # fraud RECEIVED by Mumbai
    assert cities["Pune"]["fraud_volume"] == 0.0      # Pune only sent fraud, received none
    for c in out["cities"]:                           # rate = fraud/inflow, always in [0,1]
        assert 0.0 <= c["fraud_rate"] <= 1.0
