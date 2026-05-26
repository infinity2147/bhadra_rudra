"""
Shared benchmark utilities for perf-testing graph engine + fund tracer changes.
Import into individual test files or run directly.
"""
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def timeit(label: str, fn, repeat: int = 5):
    """Run fn() `repeat` times, report mean ± best."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    mean_ms = 1000 * sum(times) / len(times)
    best_ms = 1000 * min(times)
    print(f"  {label:55s}  mean={mean_ms:7.2f}ms  best={best_ms:7.2f}ms")
    return result, mean_ms, best_ms


def make_pipeline():
    from data_generator import TransactionGenerator
    from graph_engine import FundFlowGraph
    from fraud_detector import FraudDetector
    from advanced_detectors import DormantActivationDetector

    gen = TransactionGenerator(seed=42)
    df, fraud_cases = gen.generate_all_data()
    ffg = FundFlowGraph()
    graph = ffg.build_graph(df)
    detector = FraudDetector(graph)
    results = detector.run_all_detections()
    dormant = DormantActivationDetector(graph, df).detect()
    alerts = results["all_alerts"] + dormant
    risk_scores = [
        {"entity_id": n, "name": graph.nodes[n].get("name", n), "risk_score": rs}
        for n, rs in results["node_risk_scores"].items()
    ]
    return df, graph, alerts, risk_scores
