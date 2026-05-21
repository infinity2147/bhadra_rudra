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


def test_alert_to_incident_map():
    from incident_clustering import cluster_alerts, alert_to_incident_map
    alerts = [
        {"alert_id": "A1", "pattern_type": "P", "severity": "HIGH", "entities": ["E1"], "total_flow": 1},
        {"alert_id": "A2", "pattern_type": "P", "severity": "HIGH", "entities": ["E1"], "total_flow": 1},
    ]
    incs = cluster_alerts(alerts)
    mp = alert_to_incident_map(incs)
    assert mp["A1"] == mp["A2"]
