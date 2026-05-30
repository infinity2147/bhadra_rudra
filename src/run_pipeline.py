"""
Pipeline runner — generates data, builds graph, runs detection, trains models.

Order of operations:
  1. Generate synthetic transactions + entities + fraud cases
  2. Build the fund-flow graph
  3. Run all 6 heuristic detectors → alerts
  4. Compute risk scores and cluster alerts into incidents
  5. Train XGBoost on the synthetic graph (variant="synthetic")
  6. Train GraphSAGE GNN on the same graph (if torch + PyG installed)
  7. If real datasets present in data/real/, train extra model variants
  8. Generate SAR PDFs for HIGH+ alerts
"""

import json
import os
import sys

# Windows terminals default to cp1252; force UTF-8 so ₹, →, etc. don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from data_generator import TransactionGenerator, save_data
from graph_engine import FundFlowGraph
from fraud_detector import FraudDetector
from advanced_detectors import DormantActivationDetector, ProfileMismatchDetector
from incident_clustering import cluster_alerts
from sar_generator import SARGenerator


def run_full_pipeline(data_dir: str, seed: int = 42, verbose: bool = True) -> dict:
    """Run the complete RUDRA pipeline and write artefacts under data_dir."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    os.makedirs(data_dir, exist_ok=True)

    log("\n[1/7] Generating synthetic transaction data...")
    generator = TransactionGenerator(seed=seed)
    df, fraud_cases = generator.generate_all_data()
    save_data(df, fraud_cases, data_dir, entities=generator.entities)

    log("\n[2/7] Building fund flow graph...")
    ffg = FundFlowGraph()
    graph = ffg.build_graph(df)
    if verbose:
        stats = ffg.get_graph_stats()
        log(f"  Nodes: {stats['total_nodes']}")
        log(f"  Edges: {stats['total_edges']}")
        log(f"  Density: {stats['density']}")

    log("\n[3/7] Running fraud detection (core + advanced)...")
    detector = FraudDetector(graph)
    results = detector.run_all_detections()

    log("  Running Dormant Activation Detection...")
    dormant_alerts = DormantActivationDetector(graph, df).detect()
    log(f"    {len(dormant_alerts)} dormant activation alerts")

    log("  Running Profile Mismatch Detection...")
    risk_scores_data = []
    for node_id, score in results["node_risk_scores"].items():
        node_data = dict(graph.nodes[node_id])
        risk_scores_data.append({
            "entity_id": node_id,
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
            "risk_score": score,
            "risk_level": (
                "CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.5
                else "MEDIUM" if score >= 0.3 else "LOW"
            ),
        })
    profile_alerts = ProfileMismatchDetector(graph, df, risk_scores_data).detect()
    log(f"    {len(profile_alerts)} profile mismatch alerts")

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

    log("\n[4/7] Clustering alerts into incidents...")
    incidents = cluster_alerts(all_alerts, graph=graph)
    with open(os.path.join(data_dir, "incidents.json"), "w") as f:
        json.dump(incidents, f, indent=2, default=str)
    log(f"  {len(all_alerts)} alerts → {len(incidents)} incidents")

    log("\n[5/7] Training XGBoost classifier (variant=synthetic)...")
    ml_metrics = None
    try:
        from ml_model import train_and_save
        ml_metrics = train_and_save(
            graph, df, data_dir,
            variant="synthetic",
            dataset_name="RUDRA Synthetic Generator",
        )
        if verbose and ml_metrics:
            log(
                f"  F1: {ml_metrics['f1']:.3f} | AUC: {ml_metrics['auc']:.3f} | "
                f"Precision: {ml_metrics['precision']:.3f} | Recall: {ml_metrics['recall']:.3f}"
            )
    except Exception as e:
        log(f"  XGBoost training failed: {e}")

    log("\n[6/7] Training GraphSAGE GNN (variant=synthetic)...")
    try:
        from gnn_model import train_gnn
        gnn_metrics = train_gnn(graph, data_dir, variant="synthetic", epochs=60)
        if verbose:
            log(
                f"  GNN F1: {gnn_metrics['f1']:.3f} | AUC: {gnn_metrics['auc']:.3f} | "
                f"Final Loss: {gnn_metrics['final_loss']:.4f}"
            )
    except ImportError as e:
        log(f"  GNN skipped (PyTorch Geometric not installed): {str(e).splitlines()[0]}")
    except Exception as e:
        log(f"  GNN training failed: {e}")

    from real_data_loader import available_datasets, load_ibm_aml, load_paysim
    avail = available_datasets()
    for ds_key, loader, name in [
        ("ibm_aml", load_ibm_aml, "IBM AML"),
        ("paysim", load_paysim, "PaySim"),
    ]:
        if not avail.get(ds_key, {}).get("present"):
            if verbose:
                log(f"  [{name}] not present — skipping. Download to {avail[ds_key]['expected_dir']}")
            continue
        log(f"\n  Training on {name}...")
        try:
            real_df, _real_cases = loader(sample_size=100_000)
            real_graph = FundFlowGraph().build_graph(real_df)
            from ml_model import train_and_save
            m = train_and_save(
                real_graph, real_df, data_dir,
                variant=ds_key, dataset_name=name,
            )
            log(f"    {name}: F1 {m['f1']:.3f} | AUC {m['auc']:.3f} on {m['n_edges']} edges")
        except Exception as e:
            log(f"    {name} training failed: {e}")

    if avail.get("ieee_cis", {}).get("present"):
        log("\n  Training IEEE-CIS tabular baseline...")
        try:
            from tabular_baseline import train_tabular_baseline
            m = train_tabular_baseline(data_dir, sample_size=100_000)
            log(f"    IEEE-CIS: F1 {m['f1']:.3f} | AUC {m['auc']:.3f} on {m['n_features']} features")
        except Exception as e:
            log(f"    IEEE-CIS training failed: {e}")

    log("\n[7/7] Generating SAR reports (HIGH+ severity)...")
    sar_gen = SARGenerator(graph, df, all_alerts, fraud_cases)
    sar_reports = sar_gen.generate_all_sars(min_severity="HIGH")
    sar_dir = os.path.join(data_dir, "sar_reports")
    os.makedirs(sar_dir, exist_ok=True)
    for sar in sar_reports:
        sar_gen.export_sar_pdf(sar, sar_dir)
    log(f"  {len(sar_reports)} SAR PDFs in {sar_dir}/")

    return {
        "alerts": all_alerts,
        "incidents": incidents,
        "risk_scores": risk_scores_data,
        "summary": results["summary"],
        "fraud_cases": fraud_cases,
        "ml_metrics": ml_metrics,
    }


def main():
    print("=" * 70)
    print("  RUDRA — Full Pipeline")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    run_full_pipeline(data_dir, seed=42, verbose=True)

    print("\n" + "=" * 70)
    print("  Pipeline complete!")
    print()
    print("  Start the backend:    cd backend && uvicorn main:app --reload --port 8000")
    print("  Start the frontend:   cd frontend && npm run dev")
    print("=" * 70)


if __name__ == "__main__":
    main()
