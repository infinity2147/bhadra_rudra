"""Alert clustering must collapse overlapping alerts into incidents correctly."""


def test_disjoint_alerts_stay_separate():
    from incident_clustering import cluster_alerts
    alerts = [
        {"alert_id": "A1", "pattern_type": "P1", "severity": "HIGH", "entities": ["E1"], "total_flow": 100},
        {"alert_id": "A2", "pattern_type": "P2", "severity": "MEDIUM", "entities": ["E2"], "total_flow": 50},
    ]
    incs = cluster_alerts(alerts)
    assert len(incs) == 2


def test_overlapping_alerts_merge():
    from incident_clustering import cluster_alerts
    alerts = [
        {"alert_id": "A1", "pattern_type": "Circular", "severity": "HIGH", "entities": ["E1", "E2"], "total_flow": 1000},
        {"alert_id": "A2", "pattern_type": "Layering", "severity": "MEDIUM", "entities": ["E2", "E3"], "total_flow": 800},
        {"alert_id": "A3", "pattern_type": "Funnel", "severity": "CRITICAL", "entities": ["E5", "E6"], "total_flow": 500},
    ]
    incs = cluster_alerts(alerts)
    assert len(incs) == 2  # A1+A2 share E2 → merge; A3 separate
    big_inc = next(i for i in incs if i["alert_count"] == 2)
    assert big_inc["severity"] == "HIGH"  # worst severity in the cluster


def test_shared_hub_does_not_merge_independent_rings():
    """Two independent rings that both touch a high-degree utility node (a
    payment gateway) must NOT collapse into one 'super-incident' just because
    they share that hub."""
    import networkx as nx
    from incident_clustering import cluster_alerts

    g = nx.DiGraph()
    g.add_edges_from([
        ("A1", "A2"), ("A2", "A3"), ("A3", "A1"),
        ("B1", "B2"), ("B2", "B3"), ("B3", "B1"),
        ("A1", "G"), ("B1", "G"),
    ])
    for i in range(50):           # G is common infrastructure: very high degree
        g.add_edge("G", f"D{i}")

    alerts = [
        {"alert_id": "A1a", "pattern_type": "Circular", "severity": "HIGH", "entities": ["A1", "A2", "G"], "total_flow": 1000},
        {"alert_id": "A2a", "pattern_type": "Layering", "severity": "HIGH", "entities": ["A2", "A3"], "total_flow": 900},
        {"alert_id": "B1a", "pattern_type": "Circular", "severity": "HIGH", "entities": ["B1", "B2", "G"], "total_flow": 800},
        {"alert_id": "B2a", "pattern_type": "Layering", "severity": "HIGH", "entities": ["B2", "B3"], "total_flow": 700},
    ]
    incs = cluster_alerts(alerts, graph=g)
    assert len(incs) == 2, f"hub gateway merged independent rings into {len(incs)} incident(s)"
    # The gateway is still recorded on the incidents it actually touches.
    assert any("G" in inc["entities"] for inc in incs)


def test_alert_to_incident_map():
    from incident_clustering import cluster_alerts, alert_to_incident_map
    alerts = [
        {"alert_id": "A1", "pattern_type": "P", "severity": "HIGH", "entities": ["E1"], "total_flow": 1},
        {"alert_id": "A2", "pattern_type": "P", "severity": "HIGH", "entities": ["E1"], "total_flow": 1},
    ]
    incs = cluster_alerts(alerts)
    mp = alert_to_incident_map(incs)
    assert mp["A1"] == mp["A2"]
