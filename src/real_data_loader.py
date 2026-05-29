"""
Real-dataset loaders for fund-flow benchmarking.

The synthetic generator is fine for development, but the only credible way to
defend an ML claim to a hackathon judge is to show metrics on a public,
non-synthetic dataset. This module provides loaders that map three real
datasets onto our internal transaction schema:

  1. IBM AML — Anti-Money-Laundering Transaction Data
     The PoA's primary benchmark (HI-Small variant is laptop-runnable).
     Files: HI-Small_Trans.csv (or LI-Small / HI-Medium variants)
     Source: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

  2. PaySim — Mobile money transactions, J.P. Morgan synthetic
     Realistic sender/receiver structure, smaller and easier to handle.
     File: any .csv inside data/real/paysim/
     Source: https://www.kaggle.com/datasets/ealtman2019/paysim1

  3. IEEE-CIS Fraud Detection
     Tabular (no graph) — used separately for the "tabular ML baseline" page.
     Files: train_transaction.csv, train_identity.csv
     Source: https://www.kaggle.com/c/ieee-fraud-detection/data

We deliberately do not bundle the data files (they are 100s of MB to GB). Each
loader fails honestly with a download URL if its file is missing.

Place downloaded files under: data/real/{ibm_aml,paysim,ieee_cis}/
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REAL_DATA_BASE = os.path.join(os.path.dirname(__file__), "..", "data", "real")


# Internal schema every loader must produce. Same columns the synthetic
# generator emits, so the rest of the pipeline (graph builder, detectors,
# ML feature extractor) is unchanged.
INTERNAL_COLUMNS = [
    "transaction_id", "timestamp",
    "sender_id", "sender_name", "sender_type", "sender_branch", "sender_product",
    "receiver_id", "receiver_name", "receiver_type", "receiver_branch", "receiver_product",
    "amount", "currency", "transaction_type", "channel", "purpose_code",
    "is_fraud", "fraud_pattern", "fraud_case_id",
]


class DatasetNotFound(Exception):
    """Raised when expected data files are absent. Includes a clear download URL."""

    def __init__(self, dataset: str, expected_path: str, url: str):
        msg = (
            f"\nDataset '{dataset}' is not present at:\n"
            f"    {expected_path}\n\n"
            f"Download it from:\n"
            f"    {url}\n\n"
            f"Then place the extracted CSV files in that directory."
        )
        super().__init__(msg)
        self.dataset = dataset
        self.expected_path = expected_path
        self.url = url


# ── IBM AML loader ────────────────────────────────────────────────────────────

def load_ibm_aml(
    sample_size: Optional[int] = 200_000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Load IBM AML and map to our internal schema.

    Args:
        sample_size: cap the sample size for tractable training (None = all).
        seed: random seed for sampling.
    """
    ibm_dir = os.path.join(REAL_DATA_BASE, "ibm_aml")
    candidates = [
        os.path.join(ibm_dir, "HI-Small_Trans.csv"),
        os.path.join(ibm_dir, "LI-Small_Trans.csv"),
        os.path.join(ibm_dir, "HI-Medium_Trans.csv"),
        os.path.join(ibm_dir, "LI-Medium_Trans.csv"),
    ]
    if os.path.isdir(ibm_dir):
        for f in os.listdir(ibm_dir):
            if f.lower().endswith(".csv") and f not in [os.path.basename(c) for c in candidates]:
                candidates.append(os.path.join(ibm_dir, f))
    trans_path = next((p for p in candidates if os.path.exists(p)), None)
    if trans_path is None:
        raise DatasetNotFound(
            "IBM AML",
            os.path.join(ibm_dir, "HI-Small_Trans.csv"),
            "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        )

    df = pd.read_csv(trans_path)
    if sample_size and len(df) > sample_size:
        # Stratified-by-label sample to preserve the fraud rate
        if "Is Laundering" in df.columns:
            fraud_df = df[df["Is Laundering"] == 1]
            normal_df = df[df["Is Laundering"] == 0]
            normal_take = max(0, sample_size - len(fraud_df))
            normal_df = normal_df.sample(min(normal_take, len(normal_df)), random_state=seed)
            df = pd.concat([fraud_df, normal_df], ignore_index=True)
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            df = df.sample(sample_size, random_state=seed).reset_index(drop=True)

    df = df.rename(columns={
        "Timestamp": "_ts",
        "From Bank": "_from_bank",
        "Account": "_from_acct",
        "To Bank": "_to_bank",
        "Account.1": "_to_acct",
        "Amount Received": "_amt_recv",
        "Receiving Currency": "_currency",
        "Amount Paid": "_amt_paid",
        "Payment Format": "_payment_format",
        "Is Laundering": "_is_fraud",
    })

    out = pd.DataFrame()
    out["transaction_id"] = [f"IBM{i:08d}" for i in range(len(df))]
    out["timestamp"] = pd.to_datetime(df["_ts"], errors="coerce").fillna(
        pd.Timestamp("2022-01-01")
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    out["sender_id"] = df["_from_bank"].astype(str) + "-" + df["_from_acct"].astype(str)
    out["sender_name"] = out["sender_id"]
    out["sender_type"] = "individual"   # will be overwritten below
    out["sender_branch"] = "Bank-" + df["_from_bank"].astype(str)
    out["sender_product"] = "CurrentAccount"

    out["receiver_id"] = df["_to_bank"].astype(str) + "-" + df["_to_acct"].astype(str)
    out["receiver_name"] = out["receiver_id"]
    out["receiver_type"] = "individual"  # will be overwritten below
    out["receiver_branch"] = "Bank-" + df["_to_bank"].astype(str)
    out["receiver_product"] = "CurrentAccount"

    out["amount"] = df["_amt_paid"].fillna(df.get("_amt_recv", 0)).astype(float)

    # IBM AML uses long currency names ("US Dollar"). Normalise to 3-letter
    # ISO codes so downstream feature extractors that key on currency.upper()
    # == "USD" pick up the right reporting-threshold (US CTR $10k vs RBI ₹2L).
    ibm_to_iso = {
        "US Dollar": "USD", "Euro": "EUR", "Swiss Franc": "CHF", "Yuan": "CNY",
        "Rupee": "INR", "Shekel": "ILS", "UK Pound": "GBP", "Ruble": "RUB",
        "Yen": "JPY", "Australian Dollar": "AUD", "Canadian Dollar": "CAD",
        "Bitcoin": "BTC", "Mexican Peso": "MXN", "Saudi Riyal": "SAR",
        "Brazil Real": "BRL",
    }
    out["currency"] = df["_currency"].fillna("USD").map(lambda c: ibm_to_iso.get(c, "USD"))

    # Map IBM payment formats to our rail / channel taxonomy
    pf = df["_payment_format"].fillna("").str.lower()
    rail_map = {
        "wire": "Wire Transfer", "credit card": "UPI", "ach": "NEFT",
        "cheque": "NEFT", "cash": "Branch", "bitcoin": "Wire Transfer",
        "reinvestment": "RTGS",
    }
    channel_map = {
        "wire": "Branch", "credit card": "MobileApp", "ach": "NetBanking",
        "cheque": "Branch", "cash": "Branch", "bitcoin": "ThirdPartyAPI",
        "reinvestment": "NetBanking",
    }
    out["transaction_type"] = pf.map(rail_map).fillna("NEFT")
    out["channel"] = pf.map(channel_map).fillna("NetBanking")

    out["purpose_code"] = "Business Payment"
    out["is_fraud"] = df["_is_fraud"].fillna(0).astype(int).astype(bool)
    out["fraud_pattern"] = np.where(out["is_fraud"], "ibm_labeled_aml", "none")
    out["fraud_case_id"] = np.where(out["is_fraud"], "IBM_AML_CASE", "")

    # ── Infer entity types from graph behaviour ───────────────────────────────
    # IBM AML has no entity labels — everyone would be "individual" by default,
    # making sender_is_shell / sender_is_business always 0 (dead features).
    # We infer types from transaction patterns instead:
    #   shell_company → fans out to many unique receivers vs. how many send to it
    #   business      → appears in many transactions (top 25% by activity)
    #   individual    → everything else
    sender_unique_recv = out.groupby("sender_id")["receiver_id"].nunique()
    recv_unique_send   = out.groupby("receiver_id")["sender_id"].nunique()
    all_accounts       = pd.concat([out["sender_id"], out["receiver_id"]])
    account_activity   = all_accounts.value_counts()
    activity_75th      = float(np.percentile(account_activity.values, 75))

    def _infer_type(acct_id: str) -> str:
        out_fan  = int(sender_unique_recv.get(acct_id, 0))
        in_fan   = int(recv_unique_send.get(acct_id, 0))
        activity = int(account_activity.get(acct_id, 0))
        # Pass-through / fan-out pattern → shell company
        if out_fan >= 5 and out_fan >= 3 * max(in_fan, 1):
            return "shell_company"
        # High transaction activity → business
        if activity >= activity_75th and activity >= 5:
            return "business"
        return "individual"

    out["sender_type"]   = out["sender_id"].map(_infer_type)
    out["receiver_type"] = out["receiver_id"].map(_infer_type)
    # ─────────────────────────────────────────────────────────────────────────

    out = out[INTERNAL_COLUMNS]

    fraud_cases = [{
        "case_id": "IBM_AML_CASE",
        "pattern": "ibm_labeled_aml",
        "entities": [],
        "total_amount": float(out.loc[out["is_fraud"], "amount"].sum()),
        "description": "Transactions labelled as money laundering in IBM AML.",
    }]
    return out, fraud_cases


# ── PaySim loader ─────────────────────────────────────────────────────────────

def load_paysim(
    sample_size: Optional[int] = 200_000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Load PaySim mobile-money transactions and map to our internal schema."""
    paysim_dir = os.path.join(REAL_DATA_BASE, "paysim")
    candidates = []
    if os.path.isdir(paysim_dir):
        for f in os.listdir(paysim_dir):
            if f.lower().endswith(".csv"):
                candidates.append(os.path.join(paysim_dir, f))
    trans_path = next((p for p in candidates if os.path.exists(p)), None)
    if trans_path is None:
        raise DatasetNotFound(
            "PaySim",
            os.path.join(paysim_dir, "paysim.csv"),
            "https://www.kaggle.com/datasets/ealtman2019/paysim1",
        )

    df = pd.read_csv(trans_path)
    if sample_size and len(df) > sample_size:
        fraud_df = df[df["isFraud"] == 1]
        normal_df = df[df["isFraud"] == 0]
        normal_take = max(0, sample_size - len(fraud_df))
        normal_df = normal_df.sample(min(normal_take, len(normal_df)), random_state=seed)
        df = pd.concat([fraud_df, normal_df], ignore_index=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    base_ts = pd.Timestamp("2024-01-01")
    out = pd.DataFrame()
    out["transaction_id"] = [f"PS{i:08d}" for i in range(len(df))]
    out["timestamp"] = (base_ts + pd.to_timedelta(df["step"].astype(int), unit="h")).dt.strftime("%Y-%m-%d %H:%M:%S")
    out["sender_id"] = df["nameOrig"]
    out["sender_name"] = df["nameOrig"]
    # PaySim convention: nameDest starting with "M" => merchant
    out["sender_type"] = np.where(df["nameOrig"].str.startswith("M"), "business", "individual")
    out["sender_branch"] = "PaySim-Mobile"
    out["sender_product"] = "MobileWallet"
    out["receiver_id"] = df["nameDest"]
    out["receiver_name"] = df["nameDest"]
    out["receiver_type"] = np.where(df["nameDest"].str.startswith("M"), "business", "individual")
    out["receiver_branch"] = "PaySim-Mobile"
    out["receiver_product"] = "MobileWallet"
    out["amount"] = df["amount"].astype(float)
    out["currency"] = "USD"

    rail_map = {
        "CASH-IN": "Branch", "CASH-OUT": "Branch",
        "TRANSFER": "IMPS", "DEBIT": "UPI", "PAYMENT": "UPI",
    }
    channel_map = {
        "CASH-IN": "Branch", "CASH-OUT": "Branch",
        "TRANSFER": "MobileApp", "DEBIT": "MobileApp", "PAYMENT": "MobileApp",
    }
    out["transaction_type"] = df["type"].map(rail_map).fillna("UPI")
    out["channel"] = df["type"].map(channel_map).fillna("MobileApp")
    out["purpose_code"] = "Mobile Money Transfer"
    out["is_fraud"] = df["isFraud"].astype(bool)
    out["fraud_pattern"] = np.where(out["is_fraud"], "paysim_labeled_fraud", "none")
    out["fraud_case_id"] = np.where(out["is_fraud"], "PAYSIM_CASE", "")
    out = out[INTERNAL_COLUMNS]

    fraud_cases = [{
        "case_id": "PAYSIM_CASE",
        "pattern": "paysim_labeled_fraud",
        "entities": [],
        "total_amount": float(out.loc[out["is_fraud"], "amount"].sum()),
        "description": "Transactions labelled as fraud in PaySim mobile-money dataset.",
    }]
    return out, fraud_cases


# ── IEEE-CIS loader (tabular — for the separate baseline page) ───────────────

def load_ieee_cis(
    sample_size: Optional[int] = 100_000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load IEEE-CIS Fraud Detection and return (features_df, labels).

    Unlike the graph datasets, IEEE-CIS is tabular: each row is one transaction
    with ~400 anonymised features. We return the raw feature matrix so the
    tabular-baseline trainer can use it directly — there is no graph mapping.
    """
    ieee_dir = os.path.join(REAL_DATA_BASE, "ieee_cis")
    trans_path = os.path.join(ieee_dir, "train_transaction.csv")
    ident_path = os.path.join(ieee_dir, "train_identity.csv")
    if not os.path.exists(trans_path):
        raise DatasetNotFound(
            "IEEE-CIS",
            trans_path,
            "https://www.kaggle.com/c/ieee-fraud-detection/data",
        )

    df = pd.read_csv(trans_path)
    if os.path.exists(ident_path):
        ident = pd.read_csv(ident_path)
        df = df.merge(ident, on="TransactionID", how="left")
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed).reset_index(drop=True)

    y = df["isFraud"].astype(int)
    drop_cols = ["isFraud", "TransactionID", "TransactionDT"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))
    return X, y


# ── Availability checks (used by the UI / pipeline to decide which to train) ─

def available_datasets() -> Dict[str, Dict]:
    """Report which real datasets are present on disk."""
    out = {}
    for ds, expected, url in [
        ("ibm_aml", "HI-Small_Trans.csv",
         "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"),
        ("paysim", "*.csv",
         "https://www.kaggle.com/datasets/ealtman2019/paysim1"),
        ("ieee_cis", "train_transaction.csv",
         "https://www.kaggle.com/c/ieee-fraud-detection/data"),
    ]:
        ds_dir = os.path.join(REAL_DATA_BASE, ds)
        present = False
        if os.path.isdir(ds_dir):
            files = os.listdir(ds_dir)
            present = any(f.lower().endswith(".csv") for f in files)
        out[ds] = {
            "present": present,
            "expected_dir": ds_dir,
            "download_url": url,
        }
    return out
