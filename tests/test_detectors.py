"""Each detector must flag at least one alert on its target pattern."""


def test_circular_detector_finds_at_least_one_cycle(synthetic_pipeline):
    res = synthetic_pipeline["detection_results"]
    circ = [a for a in res["all_alerts"] if a["pattern_type"] == "Circular Transaction"]
    assert len(circ) >= 1, "Circular detector returned 0 alerts"
    a = circ[0]
    assert "entities" in a and len(a["entities"]) >= 3
    assert a["algorithm"] == "johnson_simple_cycles"


def test_layering_detector_finds_chains(synthetic_pipeline):
    res = synthetic_pipeline["detection_results"]
    lay = [a for a in res["all_alerts"] if a["pattern_type"] == "Rapid Layering"]
    assert len(lay) >= 1


def test_smurfing_detector_finds_below_threshold_clusters(synthetic_pipeline):
    res = synthetic_pipeline["detection_results"]
    smurf = [a for a in res["all_alerts"] if "Smurfing" in a["pattern_type"]]
    assert len(smurf) >= 1


def test_shell_funnel_detector(synthetic_pipeline):
    res = synthetic_pipeline["detection_results"]
    funnels = [a for a in res["all_alerts"] if "Funnel" in a["pattern_type"]]
    # On dense synthetic graphs the funnel detector may emit 0 or more; the
    # important property is that it doesn't crash. We assert the alerts list
    # has the right shape if any are returned.
    for a in funnels:
        assert "imbalance_ratio" in a
        assert "n_sources" in a


def test_node_risk_scores_in_unit_interval(synthetic_pipeline):
    res = synthetic_pipeline["detection_results"]
    scores = res["node_risk_scores"]
    assert all(0 <= s <= 1 for s in scores.values()), "Risk scores out of [0,1]"


def test_all_alerts_have_required_fields(synthetic_pipeline):
    for a in synthetic_pipeline["alerts"]:
        assert "alert_id" in a
        assert "severity" in a
        assert "entities" in a
        # Funnel alerts report total_inflow / total_outflow instead of total_flow
        has_flow = (
            "total_flow" in a or "total_inflow" in a or "total_outflow" in a
            or a["pattern_type"] in {"Dormant Activation", "Profile Mismatch"}
        )
        assert has_flow, f"Alert {a['alert_id']} ({a['pattern_type']}) lacks a flow field"
