"""
Dataset-driven pipeline runner.

Pick the dataset variant with `--dataset` (or RUDRA_DATASET env). For each
variant it loads source transactions from `data/real/<dataset>/`, builds
the fund-flow graph, runs all 6 detectors, clusters incidents, trains
XGBoost (+ GraphSAGE + stacked ensemble when PyTorch Geometric is
available). SAR PDFs are generated on-request by the backend, not here.

Outputs land under `data/<variant>/` (operational artefacts: transactions,
alerts, risk scores, rudra.db) and `data/ml/<variant>/` (model
weights + metrics). No file is ever written to the top-level `data/` dir.

    python src/run_pipeline.py                    # ibm_aml (default)
    python src/run_pipeline.py --dataset paysim
"""

import argparse
import json
import os
import sys

# Windows terminals default to cp1252; force UTF-8 so ₹, →, etc. don't crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

from dataset_config import VALID_VARIANTS, variant_data_dir
from data_generator import save_data
from graph_engine import FundFlowGraph
from fraud_detector import FraudDetector
from advanced_detectors import DormantActivationDetector, ProfileMismatchDetector
from incident_clustering import cluster_alerts


def _load_source(dataset: str, data_dir: str):
    """Return (df, fraud_cases, entities) for the given real dataset.

    The loaders read the raw CSV from `data/real/<dataset>/` and map it
    onto our internal schema. The cached pre-sampled IBM AML CSV under
    `data/real/ibm_aml/HI-Small_Trans_100k_sampled.csv` is preferred when
    present so re-runs don't re-stratify the full 30M-row file.
    """
    if dataset == "ibm_aml":
        from real_data_loader import load_ibm_aml
        cached = os.path.join(data_dir, "real", "ibm_aml", "HI-Small_Trans_100k_sampled.csv")
        if os.path.exists(cached):
            print(f"[{dataset}] loading cached sample: {cached}")
            import pandas as pd
            df = pd.read_csv(cached)
            # The cached CSV is already in the internal schema; fraud cases
            # are derived on the fly because the IBM dataset has a single
            # labelled bucket rather than per-pattern groupings.
            fraud_cases = [{
                "case_id": "IBM_AML_CASE",
                "pattern": "ibm_labeled_aml",
                "entities": [],
                "total_amount": float(df.loc[df["is_fraud"].astype(bool), "amount"].sum()),
                "description": "Transactions labelled as money laundering in IBM AML.",
            }]
        else:
            print(f"[{dataset}] sampling 100k rows from HI-Small_Trans.csv...")
            df, fraud_cases = load_ibm_aml(sample_size=100_000, seed=42)
            os.makedirs(os.path.dirname(cached), exist_ok=True)
            df.to_csv(cached, index=False)
        return df, fraud_cases, []  # IBM AML has no separate entity list; types inferred in loader

    if dataset == "paysim":
        from real_data_loader import load_paysim
        df, fraud_cases = load_paysim(sample_size=100_000, seed=42)
        return df, fraud_cases, []

    raise ValueError(f"Unknown dataset {dataset!r}; expected one of {VALID_VARIANTS}")


def main(dataset: str = None, force_retrain_ml: bool = False):
    dataset = (dataset or os.environ.get("RUDRA_DATASET") or "ibm_aml").lower()
    if dataset not in VALID_VARIANTS:
        raise SystemExit(f"--dataset must be one of {VALID_VARIANTS} (got {dataset!r})")
    force_retrain_ml = (
        force_retrain_ml
        or os.environ.get("RUDRA_FORCE_RETRAIN", "").lower() in ("1", "true", "yes")
    )

    print("=" * 70)
    print(f"  RUDRA Pipeline — variant={dataset}")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    variant_dir = variant_data_dir(data_dir, dataset)
    os.makedirs(variant_dir, exist_ok=True)

    # ── 1. Source data ───────────────────────────────────────────────────────
    print(f"\n[1/5] Loading source data for {dataset}...")
    df, fraud_cases, entities = _load_source(dataset, data_dir)
    save_data(df, fraud_cases, variant_dir, entities=entities)

    # ── 2. Graph ─────────────────────────────────────────────────────────────
    print("\n[2/5] Building fund-flow graph...")
    ffg = FundFlowGraph()
    graph = ffg.build_graph(df)
    print(f"  Nodes: {graph.number_of_nodes()}  Edges: {graph.number_of_edges()}")

    # ── 3. Detectors ─────────────────────────────────────────────────────────
    print("\n[3/5] Running detectors (core + advanced)...")
    detector = FraudDetector(graph, transactions=df)
    results = detector.run_all_detections()

    dormant_alerts = DormantActivationDetector(graph, df).detect()
    print(f"  Dormant: {len(dormant_alerts)}")

    risk_scores_data = []
    for node_id, score in results["node_risk_scores"].items():
        node_data = dict(graph.nodes[node_id])
        risk_scores_data.append({
            "entity_id": node_id,
            "name": node_data.get("name", ""),
            "type": node_data.get("type", ""),
            "risk_score": score,
            "risk_level": ("CRITICAL" if score >= 0.7 else "HIGH" if score >= 0.5
                            else "MEDIUM" if score >= 0.3 else "LOW"),
        })

    # ProfileMismatch wants ML edge scores when they exist (rule + ML compose).
    # First pipeline run has none yet; detector falls back cleanly.
    try:
        from ml_model import load_edge_scores
        edge_scores = load_edge_scores(data_dir, variant=dataset) or {}
    except Exception:
        edge_scores = {}
    profile_alerts = ProfileMismatchDetector(
        graph, df, risk_scores_data, edge_scores=edge_scores,
    ).detect()
    print(f"  Profile mismatch: {len(profile_alerts)}")

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

    # Learned risk-score weights (LR over fraud_count labels)
    try:
        from risk_score_learner import train_risk_weights, load_risk_weights
        rw_metrics = train_risk_weights(graph, data_dir, variant=dataset)
        # Only ADOPT the learned weights if they clear a sane quality floor.
        # A node-risk LR below this is worse than the structural hand-tuned
        # heuristic and, when poorly calibrated, assigns a moderate risk to
        # almost every node (e.g. IBM AML: F1 0.17 -> ~36k nodes over 0.5,
        # which is meaningless). Below the floor we keep the hand-tuned scores.
        MIN_RW_F1 = 0.40
        if rw_metrics.get("trained") and rw_metrics.get("f1", 0.0) >= MIN_RW_F1:
            print(f"  Risk-weight LR adopted: F1 {rw_metrics['f1']:.3f}  AUC {rw_metrics['auc']:.3f}")
            bundle = load_risk_weights(data_dir, variant=dataset)
            new_scores = detector.compute_node_risk_scores(risk_weights_bundle=bundle)
            results["node_risk_scores"] = new_scores
        elif rw_metrics.get("trained"):
            print(f"  Risk-weight LR rejected (F1 {rw_metrics['f1']:.3f} < {MIN_RW_F1}); keeping hand-tuned risk scores.")
        else:
            print(f"  Risk-weight training skipped: {rw_metrics.get('reason')}")
    except Exception as e:
        print(f"  Risk-weight training failed (hand-tuned fallback): {e}")

    detector.save_results(results, variant_dir)

    # ── 4. Incidents ─────────────────────────────────────────────────────────
    print("\n[4/5] Clustering alerts into incidents...")
    incidents = cluster_alerts(all_alerts, graph=graph)
    with open(os.path.join(variant_dir, "incidents.json"), "w") as f:
        json.dump(incidents, f, indent=2, default=str)
    print(f"  {len(all_alerts)} alerts -> {len(incidents)} incidents")

    # ── 5. ML training ───────────────────────────────────────────────────────
    print(f"\n[5/5] Training XGBoost (variant={dataset})...")
    try:
        from ml_model import train_and_save
        ml_metrics = train_and_save(
            graph, df, data_dir, variant=dataset, dataset_name=dataset,
        )
        print(f"  XGB F1={ml_metrics['f1']:.3f}  AUC={ml_metrics['auc']:.3f}  "
              f"P={ml_metrics['precision']:.3f}  R={ml_metrics['recall']:.3f}")
    except Exception as e:
        print(f"  XGBoost training failed: {e}")

    print(f"\nTraining GraphSAGE GNN (variant={dataset})...")
    gnn_metrics_path = os.path.join(data_dir, "ml", dataset, "gnn", "metrics.json")
    if os.path.exists(gnn_metrics_path) and not force_retrain_ml:
        print(f"  SAGE already trained — skipping (delete {gnn_metrics_path} or set RUDRA_FORCE_RETRAIN=1 to redo).")
    else:
        try:
            from gnn_model import train_gnn
            gnn_metrics = train_gnn(graph, data_dir, variant=dataset, epochs=200)
            print(f"  SAGE F1={gnn_metrics['f1']:.3f}  AUC={gnn_metrics['auc']:.3f}")
        except ImportError as e:
            print(f"  SAGE skipped (PyTorch Geometric not installed): {e}")
        except Exception as e:
            print(f"  SAGE failed: {e}")

    # Stacked ensemble — only worth running when both XGB and SAGE trained.
    if dataset in ("ibm_aml", "paysim"):
        ens_metrics_path = os.path.join(data_dir, "ml", dataset, "ensemble", "metrics.json")
        if os.path.exists(ens_metrics_path) and not force_retrain_ml:
            print(f"\nEnsemble already trained — skipping (delete {ens_metrics_path} or set RUDRA_FORCE_RETRAIN=1 to redo).")
        else:
            print(f"\nTraining stacked ensemble (variant={dataset})...")
            try:
                from ensemble_model import train_ensemble
                ens_m = train_ensemble(
                    graph, df, data_dir, variant=dataset, dataset_name=dataset,
                    n_folds=3, gnn_epochs=200,
                )
                ens = ens_m["ensemble"]
                print(f"  ENS F1={ens['f1']:.3f}  AUC={ens['auc']:.3f}  "
                      f"P={ens['precision']:.3f}  R={ens['recall']:.3f}")
            except Exception as e:
                print(f"  Ensemble failed: {e}")

    # SAR PDFs are generated on-request by the backend (GET /api/sar/generate
    # and the FIU package endpoint), so the pipeline no longer pre-bakes them.

    print("\n" + "=" * 70)
    print(f"  Done. Backend will pick this up via RUDRA_DATASET={dataset}.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=VALID_VARIANTS,
        default=os.environ.get("RUDRA_DATASET", "ibm_aml"),
        help="Which dataset to run the pipeline on (default: ibm_aml).",
    )
    parser.add_argument(
        "--force-retrain-ml",
        action="store_true",
        help="Force retrain GNN + ensemble even if metrics file exists (default: skip if present).",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, force_retrain_ml=args.force_retrain_ml)
