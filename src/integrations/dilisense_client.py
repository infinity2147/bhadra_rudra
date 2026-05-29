"""
DiliSense client — real KYC/sanctions screening with mock fallback.

DiliSense exposes (https://api.dilisense.com/v1/):
    GET  /checkIndividual?names=<name>&dob=<yyyy-mm-dd>
    GET  /checkEntity?names=<name>
    GET  /sources                              — list of watchlists screened

Real responses come back with hit arrays scored by source confidence;
we re-shape them to the same envelope our mock returns so the UI is
indifferent to the source.

Set DILISENSE_API_KEY in the environment to switch to real mode.
Set USE_REAL_DILISENSE=false to force mock mode (e.g., for offline demo).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, Optional

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

from aa_kyc_mock import dilisense_screen as _mock_screen


logger = logging.getLogger(__name__)


DILISENSE_BASE = "https://api.dilisense.com/v1"
DEFAULT_TIMEOUT = 6.0


class DilisenseClient:
    """KYC/sanctions screening client.

    Env vars:
        DILISENSE_API_KEY    — provided by DiliSense after contract sign-off
        DILISENSE_BASE_URL   — optional override, defaults to prod API
        USE_REAL_DILISENSE   — set to "false" to force mock even when key present
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("DILISENSE_API_KEY")
        self.base_url = (base_url or os.getenv("DILISENSE_BASE_URL") or DILISENSE_BASE).rstrip("/")
        self._force_mock = os.getenv("USE_REAL_DILISENSE", "true").lower() in ("false", "0", "no")

    @property
    def is_real(self) -> bool:
        return bool(self.api_key) and _HTTPX and not self._force_mock

    def mode(self) -> Dict:
        return {
            "provider": "DiliSense",
            "real": self.is_real,
            "base_url": self.base_url,
            "missing_creds": [] if self.api_key else ["DILISENSE_API_KEY"],
            "httpx_installed": _HTTPX,
        }

    def _normalize_real_response(self, raw: Dict, queried_name: str, entity_type: str) -> Dict:
        """Map DiliSense's real response shape onto our mock envelope.

        Real DiliSense returns:
            {
              "total_hits": int,
              "found_records": [{"source_type": "...", "source_id": "...", "name": "...", ...}],
              ...
            }
        We pick the highest-severity source_type per hit and squash to our
        mock's {type, source, severity, notes} shape.
        """
        # Severity ranking — DiliSense's source_type values
        severity_for = {
            "SANCTION": "CRITICAL",
            "TERRORISM": "CRITICAL",
            "PEP": "HIGH",
            "CRIMINAL": "HIGH",
            "MOST_WANTED": "HIGH",
            "OTHER_OFFICIAL_LISTS": "MEDIUM",
            "OTHER_EXEMPTIONS": "LOW",
            "ADVERSE_MEDIA": "MEDIUM",
        }
        hits = []
        worst_severity_rank = 0
        rank_for = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        for rec in raw.get("found_records", []):
            src_type = (rec.get("source_type") or "OTHER_OFFICIAL_LISTS").upper()
            sev = severity_for.get(src_type, "MEDIUM")
            hits.append({
                "type": "SANCTIONS" if src_type in ("SANCTION", "TERRORISM") else src_type,
                "source": rec.get("source_id") or rec.get("name") or src_type,
                "severity": sev,
                "notes": rec.get("name", queried_name),
            })
            worst_severity_rank = max(worst_severity_rank, rank_for.get(sev, 0))

        risk = ["LOW", "LOW", "MEDIUM", "HIGH", "CRITICAL"][worst_severity_rank]
        # Crude 0–100 score from hit count and worst severity
        risk_score = min(100, raw.get("total_hits", 0) * 10 + worst_severity_rank * 20)

        return {
            "queried_name": queried_name,
            "entity_type": entity_type,
            "risk": risk,
            "risk_score_0_100": int(risk_score),
            "hits": hits,
            "checked_lists": raw.get("sources_checked", ["OFAC_SDN", "UN_1267", "EU_Sanctions", "PEP", "AdverseMedia"]),
            "queried_at": datetime.utcnow().isoformat(),
            "_real": True,
            "_raw_total_hits": raw.get("total_hits", 0),
        }

    def screen(self, name: str, entity_type: str = "individual") -> Dict:
        if not self.is_real:
            out = _mock_screen(name, entity_type)
            out["_real"] = False
            return out

        endpoint = "/checkIndividual" if entity_type == "individual" else "/checkEntity"
        url = f"{self.base_url}{endpoint}"
        try:
            with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
                r = c.get(url, params={"names": name}, headers={"x-api-key": self.api_key})
            r.raise_for_status()
            raw = r.json()
            return self._normalize_real_response(raw, name, entity_type)
        except Exception as e:
            logger.warning("DiliSense real call failed (%s); falling back to mock.", e)
            out = _mock_screen(name, entity_type)
            out["_real"] = False
            out["_fallback_reason"] = str(e)
            return out
