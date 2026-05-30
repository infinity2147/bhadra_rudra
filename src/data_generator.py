"""
Synthetic transaction generator — test fixtures only.

The production RUDRA stack runs against real public AML datasets
(IBM AML 100k, optionally PaySim) loaded by `src/real_data_loader.py`.
This generator is kept so unit tests have small, fully-labelled inputs
with known fraud patterns embedded — it is NOT a runtime data source.

Patterns embedded for test coverage:
- Circular transactions (round-tripping)
- Rapid layering (money moving through multiple accounts quickly)
- Smurfing (structuring large amounts into small transactions)
- Shell company funnels
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import json


CUSTOMER_TYPES = ["individual", "business", "shell_company"]
BUSINESS_NAMES = [
    "Apex Trading Co.", "Blue Horizon Exports", "Crystal Clear Logistics",
    "Delta Ventures Pvt Ltd", "Evergreen Suppliers", "Frontline Distributors",
    "Golden Gate Imports", "Highland Merchants", "Iron Bridge Consulting",
    "Jupiter Enterprises", "Keystone Solutions", "Lunar Imports Pvt Ltd",
    "Metro Traders", "Nova Corp", "Omega Industries",
    "Pacific Exports Ltd", "Quantum Trading", "Redwood Suppliers",
    "Silverline Logistics", "Titan Group", "Unity Merchants",
    "Vertex Holdings", "Western Trading Co.", "Zenith Enterprises",
]
INDIVIDUAL_NAMES = [
    "Amit Sharma", "Priya Patel", "Rahul Verma", "Sneha Reddy",
    "Vikram Singh", "Anjali Gupta", "Ravi Kumar", "Meera Joshi",
    "Arjun Nair", "Kavita Desai", "Deepak Rao", "Pooja Iyer",
    "Suresh Menon", "Lata Krishnan", "Rajesh Pillai", "Nita Bhatt",
    "Kiran Naik", "Divya Saxena", "Mohan Chauhan", "Ritu Agarwal",
    "Sanjay Mishra", "Anita Kulkarni", "Pradeep Hegde", "Sunita Rao",
    "Manoj Tiwari", "Swati Mukherjee", "Gaurav Jain", "Neha Kapoor",
    "Ashok Shetty", "Rekha Panda", "Harsh Vardhan", "Pallavi Das",
    "Naveen Reddy", "Shruti Bhat", "Tarun Grover", "Vandana Sethi",
    "Siddharth Das", "Rashmi Kaur", "Ganesh Patil", "Aparna Menon",
    "Bharat Shah", "Monica Dutt", "Chetan Joshi", "Geeta Rangan",
    "Dinesh Panda", "Jaya Nambiar", "Eshwar Rao", "Kalyani Bose",
    "Farhan Sheikh", "Lakshmi Iyengar",
]
BRANCHES = [
    "Mumbai Fort", "Delhi Connaught Place", "Bangalore MG Road",
    "Chennai Anna Nagar", "Hyderabad Banjara Hills", "Pune FC Road",
    "Kolkata Park Street", "Ahmedabad CG Road", "Jaipur MI Road",
    "Lucknow Hazratganj",
]
PAYMENT_RAILS = ["NEFT", "RTGS", "IMPS", "UPI", "Wire Transfer"]
CHANNELS = ["Branch", "NetBanking", "MobileApp", "ATM", "UPI", "ThirdPartyAPI"]
PRODUCTS_INDIVIDUAL = ["SavingsAccount", "CurrentAccount", "CreditCardAccount"]
PRODUCTS_BUSINESS = ["CurrentAccount", "OverdraftAccount", "LoanAccount"]
PRODUCTS_SHELL = ["CurrentAccount", "OverdraftAccount"]
PURPOSE_CODES = [
    "Business Payment", "Salary", "Vendor Payment", "Consulting Fee",
    "Import Payment", "Export Receipt", "Loan Disbursement", "Investment",
    "Insurance Premium", "Rent Payment", "Utility Payment", "Tax Payment",
]


def _pick_product(entity_type: str) -> str:
    if entity_type == "business":
        return random.choice(PRODUCTS_BUSINESS)
    if entity_type == "shell_company":
        return random.choice(PRODUCTS_SHELL)
    return random.choice(PRODUCTS_INDIVIDUAL)


def _pick_channel(rail: str, entity_type: str) -> str:
    """Pick a realistic initiation channel given the payment rail and entity type."""
    if rail == "UPI":
        return "MobileApp" if random.random() < 0.85 else "UPI"
    if rail == "RTGS" or rail == "Wire Transfer":
        return random.choices(["Branch", "NetBanking"], weights=[0.6, 0.4])[0]
    if rail == "NEFT":
        return random.choices(["NetBanking", "Branch", "MobileApp"], weights=[0.5, 0.3, 0.2])[0]
    if rail == "IMPS":
        return random.choices(["MobileApp", "NetBanking"], weights=[0.7, 0.3])[0]
    return random.choice(CHANNELS)


class TransactionGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.entities = self._generate_entities()
        self.fraud_cases: List[Dict] = []
        # Map entity_id → declared (KYC) product for behavior-vs-profile checks
        self._entity_product = {e["entity_id"]: e["primary_product"] for e in self.entities}

    def _generate_entities(self) -> List[Dict]:
        """Generate the entity roster.

        Note: we deliberately do NOT pre-assign a `risk_score` to entities.
        The risk score is computed downstream by
        `FraudDetector.compute_node_risk_scores` based on actual graph
        behaviour (centrality, fraud-edge count, type). Pre-labelling shells
        with `risk_score=0.6-0.95` was dead state — nothing read it — but
        carrying it in entities.json was misleading because it looked like
        the model was being fed labels. Removed in T1.5.
        """
        entities = []
        entity_id = 1000
        # Business entities
        for i, name in enumerate(BUSINESS_NAMES):
            entities.append({
                "entity_id": f"ENT{entity_id + i:06d}",
                "name": name,
                "type": "business",
                "branch": random.choice(BRANCHES),
                "primary_product": _pick_product("business"),
                "kyc_declared_monthly_volume": round(random.uniform(2000000, 20000000), 2),
                "kyc_declared_purpose": random.choice(["Business Payment", "Vendor Payment", "Import Payment", "Export Receipt"]),
            })
        entity_id += len(BUSINESS_NAMES)
        # Individual entities
        for i, name in enumerate(INDIVIDUAL_NAMES):
            entities.append({
                "entity_id": f"ENT{entity_id + i:06d}",
                "name": name,
                "type": "individual",
                "branch": random.choice(BRANCHES),
                "primary_product": _pick_product("individual"),
                "kyc_declared_monthly_volume": round(random.uniform(50000, 500000), 2),
                "kyc_declared_purpose": random.choice(["Salary", "Rent Payment", "Utility Payment", "Investment"]),
            })
        entity_id += len(INDIVIDUAL_NAMES)
        # Shell companies (will be used in fraud patterns)
        shell_names = [
            "Global Synergy Corp", "Prime Vision Holdings", "Star Light Trading",
            "Alpha Wave Enterprises", "Crown Crest Solutions",
            "Ocean Breeze Imports", "Thunder Bolt Exports", "Pearl Harbor Traders",
        ]
        for i, name in enumerate(shell_names):
            entities.append({
                "entity_id": f"ENT{entity_id + i:06d}",
                "name": name,
                "type": "shell_company",
                "branch": random.choice(BRANCHES),
                "primary_product": _pick_product("shell_company"),
                "kyc_declared_monthly_volume": round(random.uniform(100000, 1000000), 2),
                "kyc_declared_purpose": random.choice(["Consulting Fee", "Business Payment"]),
            })
        return entities

    def _get_entities_by_type(self, etype: str) -> List[Dict]:
        return [e for e in self.entities if e["type"] == etype]

    def _random_datetime(self, start: datetime, end: datetime) -> datetime:
        delta = end - start
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)

    def _build_txn(self, sender: Dict, receiver: Dict, amount: float,
                   ts: datetime, rail: str, purpose: str,
                   is_fraud: bool = False, pattern: str = "none",
                   case_id: str = "") -> Dict:
        channel = _pick_channel(rail, sender["type"])
        return {
            "transaction_id": f"TXN{random.randint(100000, 999999)}",
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "sender_id": sender["entity_id"],
            "sender_name": sender["name"],
            "sender_type": sender["type"],
            "sender_branch": sender["branch"],
            "sender_product": sender["primary_product"],
            "receiver_id": receiver["entity_id"],
            "receiver_name": receiver["name"],
            "receiver_type": receiver["type"],
            "receiver_branch": receiver["branch"],
            "receiver_product": receiver["primary_product"],
            "amount": amount,
            "currency": "INR",
            "transaction_type": rail,
            "channel": channel,
            "purpose_code": purpose,
            "is_fraud": is_fraud,
            "fraud_pattern": pattern,
            "fraud_case_id": case_id,
        }

    def generate_normal_transactions(self, n: int = 2000,
                                      start_date: str = "2025-01-01",
                                      end_date: str = "2025-03-31") -> pd.DataFrame:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []

        # Normal transactions tend to flow business → individual (salary), individual → business (payments)
        for _ in range(n):
            r = random.random()
            if r < 0.4:
                # Business pays individual (salary, refund)
                sender = random.choice(self._get_entities_by_type("business"))
                receiver = random.choice(self._get_entities_by_type("individual"))
                purpose = random.choice(["Salary", "Vendor Payment", "Consulting Fee"])
                amount = round(random.lognormvariate(np.log(40000), 1.0), 2)
            elif r < 0.7:
                # Individual pays business (purchase, rent)
                sender = random.choice(self._get_entities_by_type("individual"))
                receiver = random.choice(self._get_entities_by_type("business"))
                purpose = random.choice(["Vendor Payment", "Rent Payment", "Utility Payment", "Tax Payment", "Insurance Premium"])
                amount = round(random.lognormvariate(np.log(25000), 1.2), 2)
            else:
                # Business pays business (vendor)
                sender = random.choice(self._get_entities_by_type("business"))
                receiver = random.choice(self._get_entities_by_type("business"))
                while receiver["entity_id"] == sender["entity_id"]:
                    receiver = random.choice(self._get_entities_by_type("business"))
                purpose = random.choice(["Vendor Payment", "Business Payment", "Import Payment", "Export Receipt"])
                amount = round(random.lognormvariate(np.log(150000), 1.4), 2)

            amount = min(max(amount, 100), 5000000)
            ts = self._random_datetime(start, end)
            rail = random.choices(PAYMENT_RAILS, weights=[0.25, 0.10, 0.20, 0.40, 0.05])[0]
            transactions.append(self._build_txn(sender, receiver, amount, ts, rail, purpose))

        return pd.DataFrame(transactions)

    def generate_circular_transactions(self, n_rings: int = 4,
                                       start_date: str = "2025-01-15",
                                       end_date: str = "2025-03-15") -> pd.DataFrame:
        """Generate circular/round-tripping patterns where funds loop back to origin."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []
        individuals = self._get_entities_by_type("individual")

        for ring_idx in range(n_rings):
            ring_size = random.choice([3, 4, 5, 6])
            participants = random.sample(individuals, ring_size)
            base_amount = round(random.uniform(500000, 5000000), 2)
            ring_start = self._random_datetime(start, end - timedelta(days=7))

            case_id = f"CIRC_{ring_idx + 1:03d}"
            participants_ids = [p["entity_id"] for p in participants]
            self.fraud_cases.append({
                "case_id": case_id,
                "pattern": "circular_transaction",
                "entities": participants_ids,
                "base_amount": base_amount,
                "ring_size": ring_size,
                "description": f"Circular flow of ₹{base_amount:,.0f} through {ring_size} accounts",
            })

            # Two rounds — second one slightly different to look natural
            for round_num in range(2):
                for step in range(ring_size):
                    sender = participants[step]
                    receiver = participants[(step + 1) % ring_size]
                    if round_num == 0:
                        step_time = ring_start + timedelta(hours=random.uniform(1, 24) * (step + 1))
                        variation = random.uniform(0.95, 1.05)
                    else:
                        step_time = ring_start + timedelta(days=random.uniform(2, 5), hours=random.uniform(1, 24))
                        variation = random.uniform(0.90, 1.10)
                    amount = round(base_amount * variation, 2)
                    rail = random.choice(["NEFT", "RTGS", "IMPS"])
                    purpose = random.choice(["Business Payment", "Consulting Fee", "Vendor Payment"])
                    transactions.append(self._build_txn(
                        sender, receiver, amount, step_time, rail, purpose,
                        is_fraud=True, pattern="circular_transaction", case_id=case_id,
                    ))

        return pd.DataFrame(transactions)

    def generate_layering_patterns(self, n_chains: int = 5,
                                    start_date: str = "2025-01-20",
                                    end_date: str = "2025-03-20") -> pd.DataFrame:
        """Generate rapid layering: funds move through multiple accounts quickly to obscure origin."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []
        businesses = self._get_entities_by_type("business")
        shells = self._get_entities_by_type("shell_company")
        individuals = self._get_entities_by_type("individual")

        for chain_idx in range(n_chains):
            chain_length = random.randint(4, 7)
            source = random.choice(individuals + businesses)
            intermediaries = random.sample(businesses + shells, min(chain_length - 1, len(businesses + shells)))
            final_dest = random.choice(individuals)
            chain = [source] + intermediaries + [final_dest]
            base_amount = round(random.uniform(2000000, 10000000), 2)
            chain_start = self._random_datetime(start, end - timedelta(days=3))

            case_id = f"LAYER_{chain_idx + 1:03d}"
            self.fraud_cases.append({
                "case_id": case_id,
                "pattern": "rapid_layering",
                "entities": [e["entity_id"] for e in chain],
                "base_amount": base_amount,
                "chain_length": chain_length,
                "description": f"Layering chain of ₹{base_amount:,.0f} through {chain_length} entities in rapid succession",
            })

            current_amount = base_amount
            current_ts = chain_start
            for step in range(len(chain) - 1):
                sender = chain[step]
                receiver = chain[step + 1]
                # Amount slightly decreases at each step (fees, splitting)
                current_amount = round(current_amount * random.uniform(0.88, 0.98), 2)
                step_time = current_ts + timedelta(minutes=random.uniform(5, 120))
                rail = random.choice(["NEFT", "RTGS", "Wire Transfer"])
                purpose = random.choice(["Consulting Fee", "Business Payment", "Vendor Payment", "Import Payment"])
                transactions.append(self._build_txn(
                    sender, receiver, current_amount, step_time, rail, purpose,
                    is_fraud=True, pattern="rapid_layering", case_id=case_id,
                ))
                current_ts = step_time

        return pd.DataFrame(transactions)

    def generate_smurfing_patterns(self, n_patterns: int = 4,
                                    start_date: str = "2025-02-01",
                                    end_date: str = "2025-03-15") -> pd.DataFrame:
        """Large amounts broken into smaller transactions just below reporting thresholds."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []
        individuals = self._get_entities_by_type("individual")
        businesses = self._get_entities_by_type("business")

        for pattern_idx in range(n_patterns):
            source = random.choice(individuals)
            target = random.choice(businesses)
            total_amount = round(random.uniform(5000000, 20000000), 2)
            n_splits = int(total_amount / random.uniform(180000, 199000)) + 1
            split_amount = round(total_amount / n_splits, 2)

            # Mules must exclude the source and target (no self-loops).
            available_mules = [m for m in individuals
                                 if m["entity_id"] != source["entity_id"]
                                 and m["entity_id"] != target["entity_id"]]
            mules = random.sample(available_mules, min(n_splits, len(available_mules)))
            if len(mules) < n_splits and mules:
                # Repeat the pool, still skipping source/target
                mules = (mules * ((n_splits // len(mules)) + 1))[:n_splits]

            pattern_start = self._random_datetime(start, end - timedelta(days=5))
            case_id = f"SMURF_{pattern_idx + 1:03d}"
            self.fraud_cases.append({
                "case_id": case_id,
                "pattern": "smurfing",
                "entities": [source["entity_id"]] + [m["entity_id"] for m in mules] + [target["entity_id"]],
                "total_amount": total_amount,
                "n_splits": n_splits,
                "description": f"Structured ₹{total_amount:,.0f} into {n_splits} transactions of ~₹{split_amount:,.0f} each",
            })

            for i, mule in enumerate(mules):
                step_time = pattern_start + timedelta(hours=random.uniform(1, 48) * (i + 1))
                variation = random.uniform(0.95, 1.05)
                amount = min(round(split_amount * variation, 2), 199999)
                # Smurfing tends to use UPI/IMPS for speed and to avoid branch scrutiny
                rail = random.choice(["UPI", "IMPS", "NEFT"])
                purpose = random.choice(["Business Payment", "Vendor Payment"])
                # Source → Mule
                transactions.append(self._build_txn(
                    source, mule, amount, step_time, rail, purpose,
                    is_fraud=True, pattern="smurfing", case_id=case_id,
                ))
                # Mule → Target
                step_time2 = step_time + timedelta(minutes=random.uniform(10, 120))
                rail2 = random.choice(["UPI", "IMPS", "NEFT"])
                transactions.append(self._build_txn(
                    mule, target, round(amount * random.uniform(0.90, 0.99), 2),
                    step_time2, rail2,
                    random.choice(["Business Payment", "Consulting Fee"]),
                    is_fraud=True, pattern="smurfing", case_id=case_id,
                ))

        return pd.DataFrame(transactions)

    def generate_shell_funnel_patterns(self, n_patterns: int = 3,
                                        start_date: str = "2025-01-25",
                                        end_date: str = "2025-03-25") -> pd.DataFrame:
        """Multiple sources feed into shell companies."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []
        shells = self._get_entities_by_type("shell_company")
        businesses = self._get_entities_by_type("business")
        individuals = self._get_entities_by_type("individual")

        for pattern_idx in range(n_patterns):
            shell = random.choice(shells)
            n_sources = random.randint(3, 6)
            sources = random.sample(businesses + individuals, n_sources)
            total_funneled = round(random.uniform(10000000, 50000000), 2)
            pattern_start = self._random_datetime(start, end - timedelta(days=10))
            case_id = f"FUNNEL_{pattern_idx + 1:03d}"
            self.fraud_cases.append({
                "case_id": case_id,
                "pattern": "shell_funnel",
                "entities": [s["entity_id"] for s in sources] + [shell["entity_id"]],
                "total_amount": total_funneled,
                "n_sources": n_sources,
                "description": f"₹{total_funneled:,.0f} funneled from {n_sources} entities into shell company '{shell['name']}'",
            })

            for source in sources:
                n_txns = random.randint(2, 5)
                for _ in range(n_txns):
                    step_time = pattern_start + timedelta(
                        days=random.uniform(0, 10),
                        hours=random.uniform(8, 20),
                    )
                    amount = round(total_funneled / (n_sources * n_txns) * random.uniform(0.8, 1.2), 2)
                    rail = random.choice(["NEFT", "RTGS", "Wire Transfer"])
                    purpose = random.choice(["Consulting Fee", "Vendor Payment", "Business Payment"])
                    transactions.append(self._build_txn(
                        source, shell, amount, step_time, rail, purpose,
                        is_fraud=True, pattern="shell_funnel", case_id=case_id,
                    ))

        return pd.DataFrame(transactions)

    def generate_dormant_activation_pattern(self, n_patterns: int = 3,
                                             start_date: str = "2025-01-01",
                                             end_date: str = "2025-03-31") -> pd.DataFrame:
        """Pick a few entities, give them an initial period of low activity then a sudden spike.

        This adds a per-account time signature that the dormant detector can pick up.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []
        individuals = self._get_entities_by_type("individual")
        businesses = self._get_entities_by_type("business")

        for idx in range(n_patterns):
            target = random.choice(individuals)
            counterparty = random.choice(businesses)
            case_id = f"DORM_{idx + 1:03d}"

            # 1–3 small txns at the start of the window
            for _ in range(random.randint(1, 3)):
                ts = start + timedelta(days=random.uniform(0, 7), hours=random.uniform(8, 20))
                small = round(random.uniform(2000, 25000), 2)
                rail = random.choice(["UPI", "IMPS", "NEFT"])
                # Not flagged as fraud — just legitimate prior activity
                transactions.append(self._build_txn(
                    counterparty, target, small, ts, rail, "Salary",
                    is_fraud=False, pattern="none", case_id="",
                ))

            # Big spike near end of window — flagged as fraud
            spike_count = random.randint(3, 5)
            spike_start = end - timedelta(days=random.randint(3, 10))
            for j in range(spike_count):
                ts = spike_start + timedelta(hours=random.uniform(1, 24) * (j + 1))
                big = round(random.uniform(800000, 3000000), 2)
                rail = random.choice(["RTGS", "NEFT"])
                transactions.append(self._build_txn(
                    target, counterparty, big, ts, rail, "Investment",
                    is_fraud=True, pattern="dormant_activation", case_id=case_id,
                ))

            self.fraud_cases.append({
                "case_id": case_id,
                "pattern": "dormant_activation",
                "entities": [target["entity_id"]],
                "total_amount": 0,
                "description": f"Account '{target['name']}' dormant then sudden high-value spike",
            })

        return pd.DataFrame(transactions)

    def generate_all_data(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """Generate complete dataset with both normal and fraudulent transactions."""
        print("Generating normal transactions...")
        normal = self.generate_normal_transactions(n=2000)

        print("Generating circular transaction patterns...")
        circular = self.generate_circular_transactions(n_rings=4)

        print("Generating rapid layering patterns...")
        layering = self.generate_layering_patterns(n_chains=5)

        print("Generating smurfing patterns...")
        smurfing = self.generate_smurfing_patterns(n_patterns=4)

        print("Generating shell company funnel patterns...")
        funnels = self.generate_shell_funnel_patterns(n_patterns=3)

        print("Generating dormant activation patterns...")
        dormant = self.generate_dormant_activation_pattern(n_patterns=3)

        all_data = pd.concat(
            [normal, circular, layering, smurfing, funnels, dormant],
            ignore_index=True,
        )
        all_data = all_data.sort_values("timestamp").reset_index(drop=True)
        all_data["transaction_id"] = [f"TXN{i:07d}" for i in range(len(all_data))]

        print(f"\nGenerated {len(all_data)} total transactions")
        print(f"  Normal: {len(normal)}")
        print(f"  Circular: {len(circular)}")
        print(f"  Layering: {len(layering)}")
        print(f"  Smurfing: {len(smurfing)}")
        print(f"  Shell Funnel: {len(funnels)}")
        print(f"  Dormant Activation: {len(dormant)}")
        print(f"  Fraud cases: {len(self.fraud_cases)}")

        return all_data, self.fraud_cases


def save_data(df: pd.DataFrame, fraud_cases: List[Dict],
              data_dir: str = "data", entities: List[Dict] = None):
    """Save generated data to files."""
    import os
    os.makedirs(data_dir, exist_ok=True)

    df.to_csv(os.path.join(data_dir, "transactions.csv"), index=False)
    df.to_parquet(os.path.join(data_dir, "transactions.parquet"), index=False)

    with open(os.path.join(data_dir, "fraud_cases.json"), "w") as f:
        json.dump(fraud_cases, f, indent=2)

    # Caller may pass entities to avoid re-seeding the generator
    if entities is None:
        gen = TransactionGenerator()
        entities = gen.entities

    with open(os.path.join(data_dir, "entities.json"), "w") as f:
        json.dump(entities, f, indent=2)

    print(f"\nData saved to {data_dir}/")
    print(f"  transactions.csv ({len(df)} rows)")
    print(f"  fraud_cases.json ({len(fraud_cases)} cases)")
    print(f"  entities.json ({len(entities)} entities)")


if __name__ == "__main__":
    generator = TransactionGenerator(seed=42)
    df, fraud_cases = generator.generate_all_data()
    save_data(df, fraud_cases, entities=generator.entities)
    print("\n[OK] Data generation complete!")
