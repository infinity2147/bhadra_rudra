"""
Account Aggregator (AA) client — real Sahamati sandbox calls with mock fallback.

The Sahamati AA flow has four real moves:
    1. POST {AA_BASE}/Consent           — bank initiates a consent request
    2. (out-of-band)                    — customer approves on AA's UI
    3. GET  {AA_BASE}/Consent/{id}      — bank pings to check status
    4. POST {AA_BASE}/FI/request        — bank requests the data pull
    5. GET  {AA_BASE}/FI/fetch/{sid}    — bank fetches the encrypted FI data

This client wraps step 1, a fast-track approval poll (steps 2-3 collapsed),
and step 4-5 collapsed into a single pull_data call. The shape is the same
whether the calls are real or mocked, so the rest of RUDRA is unchanged.

Sahamati publishes the full spec at https://api.rebit.org.in/spec/aa. Real
production uses signed ECC-256 detached JWS on every request body; we don't
re-implement that here — when a PSB integrates, their HSM signs the requests
and we forward the signed payload via the Authorization header. The plain
adapter is correct for sandbox endpoints, which accept bearer-token auth.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

# Mock fallback lives one level up so existing imports keep working.
from aa_kyc_mock import (
    aa_create_consent as _mock_create,
    aa_pull_data as _mock_pull,
    aa_revoke_consent as _mock_revoke,
    aa_list_consents as _mock_list,
)


logger = logging.getLogger(__name__)


SAHAMATI_SANDBOX_BASE = "https://api.sahamati.org.in/sandbox/v2"
DEFAULT_TIMEOUT = 8.0   # AA responses are usually sub-second; 8s covers TLS handshake under load.


class AAClient:
    """Account Aggregator client. Real HTTPS when creds present, mock otherwise.

    Env vars used (all required for real mode):
        SAHAMATI_CLIENT_ID      — bank's FIU client ID issued by AA
        SAHAMATI_CLIENT_SECRET  — bearer token (or signing key in prod HSM)
        SAHAMATI_FIU_ID         — RBI-registered FIU identifier
        SAHAMATI_BASE_URL       — optional override, defaults to sandbox

    Set USE_REAL_AA=false to force mock mode even when creds are present
    (useful for offline demos).
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        fiu_id: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("SAHAMATI_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SAHAMATI_CLIENT_SECRET")
        self.fiu_id = fiu_id or os.getenv("SAHAMATI_FIU_ID")
        self.base_url = (base_url or os.getenv("SAHAMATI_BASE_URL") or SAHAMATI_SANDBOX_BASE).rstrip("/")
        self._force_mock = os.getenv("USE_REAL_AA", "true").lower() in ("false", "0", "no")

    # ── Mode detection ────────────────────────────────────────────────────────

    @property
    def is_real(self) -> bool:
        if self._force_mock or not _HTTPX:
            return False
        return bool(self.client_id and self.client_secret and self.fiu_id)

    def mode(self) -> Dict:
        """Diagnostic — used by /api/integrations/status."""
        return {
            "provider": "Sahamati AA",
            "real": self.is_real,
            "base_url": self.base_url,
            "fiu_id": self.fiu_id if self.is_real else None,
            "missing_creds": [
                k for k, v in [
                    ("SAHAMATI_CLIENT_ID", self.client_id),
                    ("SAHAMATI_CLIENT_SECRET", self.client_secret),
                    ("SAHAMATI_FIU_ID", self.fiu_id),
                ] if not v
            ],
            "httpx_installed": _HTTPX,
        }

    # ── Internal HTTP ─────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.client_secret}",
            "x-fiu-id": self.fiu_id or "",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _post(self, path: str, body: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
            r = c.post(url, json=body, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> Dict:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
            r = c.get(url, headers=self._headers())
        r.raise_for_status()
        return r.json()

    # ── Public API ────────────────────────────────────────────────────────────

    def create_consent(
        self,
        customer_id: str,
        fip_ids: List[str],
        purpose_code: str = "103",
        duration_days: int = 30,
    ) -> Dict:
        if not self.is_real:
            out = _mock_create(customer_id, fip_ids, purpose_code, duration_days)
            out["_real"] = False
            return out

        now = datetime.utcnow()
        body = {
            "ver": "2.0.0",
            "timestamp": now.isoformat() + "Z",
            "txnid": f"rudra-{int(now.timestamp() * 1000)}",
            "ConsentDetail": {
                "consentStart": now.isoformat() + "Z",
                "consentExpiry": (now + timedelta(days=duration_days)).isoformat() + "Z",
                "consentMode": "STORE",
                "fetchType": "PERIODIC",
                "consentTypes": ["TRANSACTIONS", "PROFILE", "SUMMARY"],
                "fiTypes": ["DEPOSIT", "TERM_DEPOSIT"],
                "DataConsumer": {"id": self.fiu_id, "type": "FIU"},
                "Customer": {"id": customer_id},
                "FIDataRange": {
                    "from": (now - timedelta(days=duration_days)).isoformat() + "Z",
                    "to": now.isoformat() + "Z",
                },
                "DataLife": {"unit": "DAY", "value": duration_days},
                "Frequency": {"unit": "DAY", "value": 1},
                "Purpose": {"code": purpose_code, "text": "RUDRA fund-flow investigation"},
                "FIPDetails": [{"id": fid, "type": "BANK"} for fid in fip_ids],
            },
        }

        try:
            resp = self._post("/Consent", body)
            resp["_real"] = True
            return resp
        except Exception as e:
            logger.warning("AA create_consent real call failed (%s); falling back to mock.", e)
            out = _mock_create(customer_id, fip_ids, purpose_code, duration_days)
            out["_real"] = False
            out["_fallback_reason"] = str(e)
            return out

    def pull_data(self, consent_handle: str, days_back: int = 30) -> Dict:
        if not self.is_real:
            out = _mock_pull(consent_handle, days_back=days_back)
            out["_real"] = False
            return out

        now = datetime.utcnow()
        body = {
            "ver": "2.0.0",
            "timestamp": now.isoformat() + "Z",
            "txnid": f"rudra-fi-{int(now.timestamp() * 1000)}",
            "FIDataRange": {
                "from": (now - timedelta(days=days_back)).isoformat() + "Z",
                "to": now.isoformat() + "Z",
            },
            "Consent": {"id": consent_handle},
            "KeyMaterial": {"cryptoAlg": "ECDH", "curve": "Curve25519"},
        }

        try:
            req = self._post("/FI/request", body)
            session_id = req.get("sessionId") or req.get("session_id")
            if not session_id:
                raise RuntimeError("AA FI/request returned no sessionId")
            fetched = self._get(f"/FI/fetch/{session_id}")
            fetched["_real"] = True
            return fetched
        except Exception as e:
            logger.warning("AA pull_data real call failed (%s); falling back to mock.", e)
            out = _mock_pull(consent_handle, days_back=days_back)
            out["_real"] = False
            out["_fallback_reason"] = str(e)
            return out

    def revoke(self, consent_handle: str) -> Dict:
        if not self.is_real:
            out = _mock_revoke(consent_handle)
            out["_real"] = False
            return out

        now = datetime.utcnow()
        body = {
            "ver": "2.0.0",
            "timestamp": now.isoformat() + "Z",
            "txnid": f"rudra-rev-{int(now.timestamp() * 1000)}",
            "ConsentStatus": {"id": consent_handle, "status": "REVOKED"},
        }
        try:
            resp = self._post("/Consent/Status", body)
            resp["_real"] = True
            return resp
        except Exception as e:
            logger.warning("AA revoke real call failed (%s); falling back to mock.", e)
            out = _mock_revoke(consent_handle)
            out["_real"] = False
            out["_fallback_reason"] = str(e)
            return out

    def list_consents(self) -> List[Dict]:
        # Sahamati doesn't expose a bulk list; we always reflect the local store.
        # In real mode the local store is hydrated from create_consent responses.
        return _mock_list()
