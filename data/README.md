# Data directory

This directory holds the generated pipeline outputs. **All files except this README are regenerable.**

To regenerate everything, from the repo root run:

```bash
python src/run_pipeline.py
```

That will produce:

- `transactions.csv` / `transactions.parquet` — synthetic transactions
- `entities.json` — synthetic entities
- `fraud_cases.json` — ground-truth fraud cases
- `fraud_alerts.json` — detector outputs
- `risk_scores.json` — per-entity risk
- `detection_summary.json` — high-level counts
- `ml/model.pkl` + `ml/metrics.json` + `ml/edge_scores.json` — trained XGBoost
- `sar_reports/` — optional on-demand SAR PDFs (not created by the pipeline)
- `cases.json` — case workflow state (auto-created on backend startup)

Anything stored elsewhere in this directory is ignored by git on purpose; we keep the repository light and let the pipeline create the data fresh on every laptop.

The `data/real/` subdirectory is reserved for real public benchmark datasets (IBM AML, PaySim, IEEE-CIS). It is not populated by default — see the project README for download instructions.
