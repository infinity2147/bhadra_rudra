"""Synthetic IDENTITY dataset for the Collusion Rings demo.

Standalone + variant-independent: a small set of accounts each carrying
device/IP/KYC identifiers, with a few collusion rings injected (accounts that
share an identifier). This is the only data the collusion lane runs on; the
rest of RUDRA runs on the active (IBM AML) variant. Seeded for reproducibility.
"""
from __future__ import annotations

import json
import os
import random
from typing import Dict, List

_FIRST = ["Aarav", "Vivaan", "Aditya", "Ananya", "Diya", "Kabir", "Ishaan",
          "Myra", "Sara", "Reyansh", "Anaya", "Vihaan", "Arjun", "Saanvi",
          "Aryan", "Riya", "Dhruv", "Kiara", "Ayaan", "Navya"]
_LAST = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Gupta", "Patel",
         "Khan", "Bose", "Rao"]


def _gen_account(i: int) -> Dict:
    return {
        "account_id": f"ACC{i:05d}",
        "name": f"{_FIRST[i % len(_FIRST)]} {_LAST[(i // len(_FIRST)) % len(_LAST)]}",
        "device_id": f"DEV-{i:05d}",
        "ip": f"10.{i % 256}.{(i // 256) % 256}.{(i * 7) % 256}",
        "kyc_doc_hash": f"PAN-{i:05d}",
        "is_mule": False,
    }


def generate_identity_accounts(*, n_accounts: int = 150, seed: int = 42) -> List[Dict]:
    rng = random.Random(seed)
    accounts = [_gen_account(i) for i in range(n_accounts)]

    # Ring K: 8 accounts share one forged KYC document.
    for a in accounts[10:18]:
        a["kyc_doc_hash"] = "PAN-FORGED-7F3A"
        a["is_mule"] = True
    # Ring D: 12 accounts share one device + IP (one-phone mule farm).
    for a in accounts[40:52]:
        a["device_id"] = "DEV-FARM-01"
        a["ip"] = "10.66.66.66"
        a["is_mule"] = True
    # Ring X: 4 accounts share BOTH a device and a KYC doc (transitive case).
    for a in accounts[80:84]:
        a["device_id"] = "DEV-X-9"
        a["kyc_doc_hash"] = "PAN-X-9"
        a["is_mule"] = True
    # Benign noise: 2 accounts share a device (below default min_ring_size=3).
    for a in accounts[100:102]:
        a["device_id"] = "DEV-FAMILY-2"

    rng.shuffle(accounts)
    return accounts


def write_identity_dataset(path: str, *, n_accounts: int = 150, seed: int = 42) -> List[Dict]:
    accounts = generate_identity_accounts(n_accounts=n_accounts, seed=seed)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(accounts, f, indent=2)
    return accounts
