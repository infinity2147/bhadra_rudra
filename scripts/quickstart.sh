#!/usr/bin/env bash
#
# RUDRA — one-command quickstart for evaluators.
#
#   ./scripts/quickstart.sh
#
# Generates the dataset if it isn't present yet, then launches the full stack
# (backend :8000 + frontend :5173 + Kafka) via docker compose with healthcheck-
# gated startup. Falls back to printing local-dev commands if Docker is absent.
#
# Override the dataset with:  RUDRA_DATASET=paysim ./scripts/quickstart.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${RUDRA_DATASET:-ibm_aml}"

echo "=================================================================="
echo "  RUDRA quickstart — dataset=${DATASET}"
echo "=================================================================="

if [ ! -f "data/${DATASET}/transactions.csv" ]; then
  echo "[quickstart] No artefacts for '${DATASET}' yet — running the pipeline"
  echo "             (one-time; trains XGBoost + GNN + ensemble, a few minutes)..."
  python src/run_pipeline.py --dataset "${DATASET}"
else
  echo "[quickstart] Artefacts for '${DATASET}' already present — skipping generation."
fi

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  echo "[quickstart] Launching the stack (backend :8000, frontend :5173)."
  echo "             Compose waits on /api/health before marking the backend ready."
  exec docker compose up --build
else
  echo "[quickstart] Docker not found. Start the two services locally instead:"
  echo
  echo "    Terminal 1:  cd backend && uvicorn main:app --reload --port 8000"
  echo "    Terminal 2:  cd frontend && npm install && npm run dev"
  echo
  echo "Then open http://localhost:5173"
fi
