"""
Real-API adapter layer for RUDRA.

Two integrations every PSB has to plumb in: the Account Aggregator (AA)
framework for consent-based cross-bank financial-data pull, and a
KYC/sanctions enrichment provider (we use DiliSense — same vendor most
PSBs use; equivalent for Refinitiv World-Check or LexisNexis would
follow the same pattern).

Each client makes real HTTPS calls when its credentials are present in
the environment and falls back to the schema-accurate mock implementation
in src.aa_kyc_mock when they aren't. Every response carries a
`"_real": true|false` flag so the UI can show the user which mode the
system is operating in.

The cred names are:
    SAHAMATI_CLIENT_ID, SAHAMATI_CLIENT_SECRET, SAHAMATI_FIU_ID
    DILISENSE_API_KEY
"""

from .aa_client import AAClient
from .dilisense_client import DilisenseClient

__all__ = ["AAClient", "DilisenseClient"]
