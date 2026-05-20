"""
Pipeline runner — generates data, builds graph, runs detection.
Run this before starting the Streamlit app.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data_generator import TransactionGenerator, save_data
from graph_engine import FundFlowGraph
from fraud_detector import FraudDetector
from advanced_detectors import DormantActivationDetector, ProfileMismatchDetector


def main():
    print("=" * 60)
    print("  RUDRA — Fund Flow Tracking Pipeline")
    print("=" * 60)

    # Step 1: Generate data
    print("\n[1/4] Generating synthetic transaction data...")
    generator = TransactionGenerator(seed=42)
    df, fraud_cases = generator.generate_all_data()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    save_data(df, fraud_cases, data_dir)

    # Step 2: Build graph
    print("\n[2/4] Building fund flow graph...")
    ffg = FundFlowGraph()
    graph = ffg.build_graph(df)
    stats = ffg.get_graph_stats()
    print(f"  Nodes: {stats['total_nodes']}")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Density: {stats['density']}")

    # Step 3: Run fraud detection
    print("\n[3/4] Running fraud detection (core + advanced)...")
    detector = FraudDetector(graph)
    results = detector.run_all_detections()

    # Advanced detectors
    print("\n  Running Dormant Activation Detection...")
    dormant_detector = DormantActivationDetector(graph, df)
    dormant_alerts = dormant_detector.detect()
    print(f"    Found {len(dormant_alerts)} dormant activation alerts")

    print("  Running Profile Mismatch Detection...")
    risk_scores_data = []
    for node_id, score in results["node_risk_scores"].items():
        node_data = dict(graph.nodes[node_id])
        risk_scores_data.append({
            "entity_id": node_id,
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
            "risk_score": score,
            "risk_level": "CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.5 else "MEDIUM" if score >= 0.3 else "LOW",
        })

    profile_detector = ProfileMismatchDetector(graph, df, risk_scores_data)
    profile_alerts = profile_detector.detect()
    print(f"    Found {len(profile_alerts)} profile mismatch alerts")

    # Merge advanced alerts
    all_alerts = results["all_alerts"] + dormant_alerts + profile_alerts
    results["all_alerts"] = all_alerts
    results["dormant_activation"] = dormant_alerts
    results["profile_mismatch"] = profile_alerts
    results["summary"]["total_alerts"] = len(all_alerts)
    results["summary"]["dormant_count"] = len(dormant_alerts)
    results["summary"]["profile_count"] = len(profile_alerts)
    results["summary"]["critical_alerts"] = sum(1 for a in all_alerts if a["severity"] == "CRITICAL")
    results["summary"]["high_alerts"] = sum(1 for a in all_alerts if a["severity"] == "HIGH")
    results["summary"]["medium_alerts"] = sum(1 for a in all_alerts if a["severity"] == "MEDIUM")

    detector.save_results(results, data_dir)

    # Step 4: Generate SAR reports
    print("\n[4/4] Generating SAR reports...")
    from sar_generator import SARGenerator
    sar_gen = SARGenerator(graph, df, all_alerts, fraud_cases)
    sar_reports = sar_gen.generate_all_sars(min_severity="HIGH")
    sar_dir = os.path.join(data_dir, "sar_reports")
    for sar in sar_reports:
        path = sar_gen.export_sar_pdf(sar, sar_dir)
        print(f"    Generated: {sar['report_id']} → {path}")

    print("\n" + "=" * 60)
    print("  Pipeline complete! Run the dashboard with:")
    print("  streamlit run src/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
