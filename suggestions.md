# RUDRA — Low-Hanging Fruits to Replace Synthetic with Real

Ranked by effort. The goal: flip the demo from "synthetic data + real algorithms" to "real public datasets + real streaming + real LLM + real algorithms" with minimal work.

## Tier 1 — Zero external dependency, immediate swap

### 1. Replace synthetic transactions with IBM AML on the main dashboard ★ biggest impact
- `HI-Small_Trans.csv` (475 MB) and the 100k sample are already on disk under `data/real/ibm_aml/`.
- Models are already trained under `data/ml/ibm_aml/` (XGB + GraphSAGE + GAT + ensemble).
- But the backend `load_or_generate()` in `backend/main.py:104-106, 168-170` still loads `data/transactions.csv` — the synthetic file — at startup.

**Fix:** Make `load_or_generate()` accept a `RUDRA_DATASET=ibm_aml` env var and load `HI-Small_Trans_100k_sampled.csv` instead. Build the graph + run the 6 detectors on IBM AML. ~30–60 lines, mostly wiring.

**Impact:** Every page (Dashboard, Graph, Journey, Alerts, Incidents, Cases) shows real ML4FinCrime data. F1=1.00 vanity number disappears.

### 2. Kill the synthetic metrics from the dashboard
The Dashboard + ModelMetrics page currently fetches `/api/ml/metrics` defaulting to `variant=synthetic` → F1=1.00. Evaluators spot this in 5 seconds.

**Fix:** Default `?variant=ibm_aml` in the frontend API client, or remove the synthetic variant from the dropdown. One file, one line.

### 3. Stream IBM AML over Kafka (broker already in docker-compose)
`docker-compose.yml` already has Bitnami Kafka 3.7. Producer CLI exists in `src/streaming/kafka_producer.py`. Compose file header already documents the command:

```
docker compose exec backend python -m streaming.kafka_producer \
    --source data/real/ibm_aml/HI-Small_Trans_100k.csv --rate 10 --total 500
```

**Fix:** `docker compose up` → run that command. Live page shows real IBM AML txns flowing through real Kafka. **Zero code changes.**

### 4. Wire Gemini key — free tier, 5 minutes
Google Gemini has a free tier. Get a key → `export GEMINI_API_KEY=...` → restart. Copilot stops doing keyword fallback, starts doing real tool-calling. Code already there.

## Tier 2 — Small code change

### 5. Run detectors on IBM AML, not synthetic
6 detectors (`fraud_detector.py`, `advanced_detectors.py`) run on whatever graph you feed them. If you do #1, they run on IBM AML automatically. Thresholds in `ConfigStore` were tuned for synthetic → expect noisier output initially. Re-tune via Settings page or `/api/config/rerun`.

### 6. Remove the synthetic generator from the startup path
`load_or_generate()` regenerates synthetic data if `data/transactions.csv` is missing. Replace with: "if no real dataset is configured, fail loud — don't silently fall back to fake data." Forces honesty.

### 7. Regenerate SAR PDFs against IBM AML alerts
5 PDFs in `data/sar_reports/` are templated against synthetic alerts. After #1, regenerate from IBM AML alerts. `sar_generator.py` doesn't care about data source.

### 8. Use the IBM AML risk-score learner
Backend currently loads `data/ml/synthetic/risk_weights.pkl`. There's a learner that produces `risk_weights.pkl` for `ibm_aml` too — point backend at that one.

## Tier 3 — External signup, but free/cheap

### 9. DiliSense free tier
Freemium tier for sanctions/PEP checks. Sign up → get API key → `export DILISENSE_API_KEY=...`. Adapter already coded. Real sanctions screening against OFAC/UN/EU lists, on real names from IBM AML.

### 10. Sahamati AA sandbox
**Harder than the others.** Code currently lacks JWS signing, which the real sandbox enforces. Env-var path is there; actual JWS implementation is ~50 lines + ECC-256 keys you'd have to register with Sahamati.

## Tier 4 — Other public datasets

`data/real/paysim/` and `data/real/ieee_cis/` directories exist but are **empty**. Loaders exist in `real_data_loader.py`. Download Kaggle CSVs → drop in → `python src/run_pipeline.py` picks them up. Extra ML variants for comparison.

---

## Recommended order (≈1 hour total)

1. **Backend dataset switch** (Tier 1, #1) — 30 min, kills the biggest credibility issue
2. **Dashboard default to ibm_aml** (Tier 1, #2) — 5 min, honest numbers everywhere
3. **Kafka stream IBM AML** (Tier 1, #3) — 10 min, Live page becomes real
4. **Gemini key** (Tier 1, #4) — 5 min, copilot stops faking

After this, only the AA + DiliSense + FIU submission paths remain mocked (and those are always labeled as such with `_real: false` / `_mock_disclaimer`).
