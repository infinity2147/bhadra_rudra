"""Pytest fixtures shared across test modules."""

import os
import sys
import tempfile
import shutil

import pytest

# Make src/ importable for tests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


@pytest.fixture(scope="session")
def synthetic_pipeline():
    """Generate data + graph + alerts once, share across all tests."""
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
    dormant_alerts = DormantActivationDetector(graph, df).detect()
    all_alerts = results["all_alerts"] + dormant_alerts
    return {
        "df": df,
        "graph": graph,
        "fraud_cases": fraud_cases,
        "alerts": all_alerts,
        "detection_results": results,
        "ffg": ffg,
    }


@pytest.fixture
def temp_data_dir():
    """Create a fresh data dir for tests that write to disk."""
    d = tempfile.mkdtemp(prefix="rudra_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)
