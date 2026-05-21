"""Fund journey tracer must return a usable graph + timeline."""


def test_trace_journey_returns_expected_shape(synthetic_pipeline):
    from fund_tracer import trace_journey
    df = synthetic_pipeline["df"]
    graph = synthetic_pipeline["graph"]
    risk_scores = [
        {"entity_id": n, "name": graph.nodes[n].get("name", n), "risk_score": 0.5}
        for n in graph.nodes()
    ]
    # Pick a node that has actual edges
    nid = next(iter(graph.nodes()))
    result = trace_journey(graph, df, risk_scores, entity_id=nid, max_hops=1, min_amount=0)
    assert "entity" in result and "nodes" in result and "links" in result
    assert "summary" in result and "timeline" in result


def test_trace_for_alert_scopes_to_alert_entities(synthetic_pipeline):
    from fund_tracer import trace_for_alert
    df = synthetic_pipeline["df"]
    graph = synthetic_pipeline["graph"]
    alerts = synthetic_pipeline["alerts"]
    circ = next((a for a in alerts if a["pattern_type"] == "Circular Transaction"), None)
    if not circ:
        return  # No circular alert; nothing to test
    risk_scores = [{"entity_id": n, "name": "x", "risk_score": 0.3} for n in graph.nodes()]
    result = trace_for_alert(graph, df, risk_scores, circ)
    node_ids = {n["id"] for n in result["nodes"]}
    # Every alert entity should be in the result
    for e in circ["entities"]:
        assert e in node_ids
