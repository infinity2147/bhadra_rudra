"""
Active-dataset configuration.

Single source of truth for which dataset the running RUDRA stack is bound
to and where its generated artefacts live on disk.

Layout under `data/`:

    data/
      ml/                       # per-variant ML artefacts
        ibm_aml/ ...
        paysim/ ...
      real/                     # raw downloaded datasets
        ibm_aml/HI-Small_Trans.csv ...
        paysim/ ...
        ieee_cis/ ...
      ibm_aml/                  # per-variant operational artefacts
        transactions.csv
        fraud_alerts.json
        risk_scores.json
        detection_summary.json
        fraud_cases.json
        incidents.json
        entities.json
        rudra.db                # cases + audit log + config thresholds
        sar_reports/*.pdf
      paysim/                   # same shape, different dataset

Pick the active variant via the RUDRA_DATASET env var. Defaults to
`ibm_aml` — the IBM AML 100k public benchmark is the production dataset
RUDRA is built to detect against. PaySim is supported as a secondary
benchmark when its CSV is dropped under `data/real/paysim/`.
"""

from __future__ import annotations

import os
from typing import Iterable


VALID_VARIANTS = ("ibm_aml", "paysim")
DEFAULT_VARIANT = "ibm_aml"


def get_active_variant() -> str:
    """Return the variant the backend should bind to at startup."""
    v = (os.environ.get("RUDRA_DATASET") or DEFAULT_VARIANT).strip().lower()
    if v not in VALID_VARIANTS:
        raise ValueError(
            f"RUDRA_DATASET={v!r} is not one of {VALID_VARIANTS}. "
            f"Set it to one of those values or unset it (defaults to {DEFAULT_VARIANT})."
        )
    return v


def variant_data_dir(data_dir: str, variant: str | None = None) -> str:
    """Return the per-variant subdirectory under `data_dir`.

    Caller is responsible for `os.makedirs(..., exist_ok=True)` before
    writing — `load_or_generate()` in the backend does this once at startup.
    """
    variant = variant or get_active_variant()
    return os.path.join(data_dir, variant)


def required_artefacts(variant_dir: str) -> Iterable[str]:
    """Files the backend needs to be present before it can serve a variant.

    Used by the startup check to fail loud (with the exact missing path +
    the command to regenerate) instead of silently regenerating synthetic
    data.
    """
    return (
        os.path.join(variant_dir, "transactions.csv"),
        os.path.join(variant_dir, "fraud_alerts.json"),
        os.path.join(variant_dir, "risk_scores.json"),
        os.path.join(variant_dir, "detection_summary.json"),
        os.path.join(variant_dir, "fraud_cases.json"),
    )


def regenerate_command(variant: str) -> str:
    """Shell command the user should run to (re)generate a variant's data."""
    return f"python src/run_pipeline.py --dataset {variant}"
