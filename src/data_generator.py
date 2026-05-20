"""
Synthetic Banking Transaction Data Generator
Generates realistic fund flow data with embedded fraud patterns:
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
TRANSACTION_TYPES = ["NEFT", "RTGS", "IMPS", "UPI", "Wire Transfer"]
PURPOSE_CODES = [
    "Business Payment", "Salary", "Vendor Payment", "Consulting Fee",
    "Import Payment", "Export Receipt", "Loan Disbursement", "Investment",
    "Insurance Premium", "Rent Payment", "Utility Payment", "Tax Payment",
]


class TransactionGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.entities = self._generate_entities()
        self.fraud_cases: List[Dict] = []

    def _generate_entities(self) -> List[Dict]:
        entities = []
        entity_id = 1000
        # Business entities
        for i, name in enumerate(BUSINESS_NAMES):
            entities.append({
                "entity_id": f"ENT{entity_id + i:06d}",
                "name": name,
                "type": "business",
                "branch": random.choice(BRANCHES),
                "risk_score": round(random.uniform(0.1, 0.5), 2),
            })
        entity_id += len(BUSINESS_NAMES)
        # Individual entities
        for i, name in enumerate(INDIVIDUAL_NAMES):
            entities.append({
                "entity_id": f"ENT{entity_id + i:06d}",
                "name": name,
                "type": "individual",
                "branch": random.choice(BRANCHES),
                "risk_score": round(random.uniform(0.05, 0.3), 2),
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
                "risk_score": round(random.uniform(0.6, 0.95), 2),
            })
        return entities

    def _get_entities_by_type(self, etype: str) -> List[Dict]:
        return [e for e in self.entities if e["type"] == etype]

    def _random_datetime(self, start: datetime, end: datetime) -> datetime:
        delta = end - start
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)

    def generate_normal_transactions(self, n: int = 2000,
                                      start_date: str = "2025-01-01",
                                      end_date: str = "2025-03-31") -> pd.DataFrame:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []

        for _ in range(n):
            sender = random.choice(self.entities)
            receiver = random.choice(self.entities)
            while receiver["entity_id"] == sender["entity_id"]:
                receiver = random.choice(self.entities)

            amount = round(random.lognormvariate(np.log(50000), 1.5), 2)
            amount = min(amount, 5000000)

            ts = self._random_datetime(start, end)

            transactions.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "sender_id": sender["entity_id"],
                "sender_name": sender["name"],
                "sender_type": sender["type"],
                "sender_branch": sender["branch"],
                "receiver_id": receiver["entity_id"],
                "receiver_name": receiver["name"],
                "receiver_type": receiver["type"],
                "receiver_branch": receiver["branch"],
                "amount": amount,
                "currency": "INR",
                "transaction_type": random.choice(TRANSACTION_TYPES),
                "purpose_code": random.choice(PURPOSE_CODES),
                "is_fraud": False,
                "fraud_pattern": "none",
            })

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

            for step in range(ring_size):
                sender = participants[step]
                receiver = participants[(step + 1) % ring_size]
                step_time = ring_start + timedelta(hours=random.uniform(1, 24))
                variation = random.uniform(0.95, 1.05)
                amount = round(base_amount * variation, 2)

                transactions.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "timestamp": step_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": sender["entity_id"],
                    "sender_name": sender["name"],
                    "sender_type": sender["type"],
                    "sender_branch": sender["branch"],
                    "receiver_id": receiver["entity_id"],
                    "receiver_name": receiver["name"],
                    "receiver_type": receiver["type"],
                    "receiver_branch": receiver["branch"],
                    "amount": amount,
                    "currency": "INR",
                    "transaction_type": random.choice(["NEFT", "RTGS", "IMPS"]),
                    "purpose_code": random.choice(["Business Payment", "Consulting Fee", "Vendor Payment"]),
                    "is_fraud": True,
                    "fraud_pattern": "circular_transaction",
                    "fraud_case_id": case_id,
                })

            # Add a second round with slightly different amounts
            for step in range(ring_size):
                sender = participants[step]
                receiver = participants[(step + 1) % ring_size]
                step_time = ring_start + timedelta(days=random.uniform(2, 5), hours=random.uniform(1, 24))
                variation = random.uniform(0.90, 1.10)
                amount = round(base_amount * variation, 2)

                transactions.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "timestamp": step_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": sender["entity_id"],
                    "sender_name": sender["name"],
                    "sender_type": sender["type"],
                    "sender_branch": sender["branch"],
                    "receiver_id": receiver["entity_id"],
                    "receiver_name": receiver["name"],
                    "receiver_type": receiver["type"],
                    "receiver_branch": receiver["branch"],
                    "amount": amount,
                    "currency": "INR",
                    "transaction_type": random.choice(["NEFT", "RTGS", "IMPS"]),
                    "purpose_code": random.choice(["Business Payment", "Consulting Fee", "Vendor Payment"]),
                    "is_fraud": True,
                    "fraud_pattern": "circular_transaction",
                    "fraud_case_id": case_id,
                })

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
            # Build a chain mixing businesses and shell companies
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
            for step in range(len(chain) - 1):
                sender = chain[step]
                receiver = chain[step + 1]
                # Amount slightly decreases at each step (fees, splitting)
                current_amount = round(current_amount * random.uniform(0.88, 0.98), 2)
                step_time = chain_start + timedelta(minutes=random.uniform(5, 120))

                transactions.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "timestamp": step_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": sender["entity_id"],
                    "sender_name": sender["name"],
                    "sender_type": sender["type"],
                    "sender_branch": sender["branch"],
                    "receiver_id": receiver["entity_id"],
                    "receiver_name": receiver["name"],
                    "receiver_type": receiver["type"],
                    "receiver_branch": receiver["branch"],
                    "amount": current_amount,
                    "currency": "INR",
                    "transaction_type": random.choice(["NEFT", "RTGS", "Wire Transfer"]),
                    "purpose_code": random.choice(["Consulting Fee", "Business Payment", "Vendor Payment", "Import Payment"]),
                    "is_fraud": True,
                    "fraud_pattern": "rapid_layering",
                    "fraud_case_id": case_id,
                })
                chain_start = step_time

        return pd.DataFrame(transactions)

    def generate_smurfing_patterns(self, n_patterns: int = 4,
                                    start_date: str = "2025-02-01",
                                    end_date: str = "2025-03-15") -> pd.DataFrame:
        """Generate smurfing/structuring: large amounts broken into smaller transactions just below reporting thresholds."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        transactions = []
        individuals = self._get_entities_by_type("individual")
        businesses = self._get_entities_by_type("business")

        for pattern_idx in range(n_patterns):
            source = random.choice(individuals)
            target = random.choice(businesses)
            total_amount = round(random.uniform(5000000, 20000000), 2)
            # Split into transactions just below ₹200,000 (common reporting threshold)
            n_splits = int(total_amount / random.uniform(180000, 199000)) + 1
            split_amount = round(total_amount / n_splits, 2)

            # Use multiple mules (smurfs)
            mules = random.sample(individuals, min(n_splits, len(individuals)))
            if len(mules) < n_splits:
                mules = mules * ((n_splits // len(mules)) + 1)
            mules = mules[:n_splits]

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
                step_time = pattern_start + timedelta(
                    hours=random.uniform(1, 48) * (i + 1)
                )
                variation = random.uniform(0.95, 1.05)
                amount = round(split_amount * variation, 2)
                amount = min(amount, 199999)  # Keep below threshold

                # Source -> Mule
                transactions.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "timestamp": step_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": source["entity_id"],
                    "sender_name": source["name"],
                    "sender_type": source["type"],
                    "sender_branch": source["branch"],
                    "receiver_id": mule["entity_id"],
                    "receiver_name": mule["name"],
                    "receiver_type": mule["type"],
                    "receiver_branch": mule["branch"],
                    "amount": amount,
                    "currency": "INR",
                    "transaction_type": random.choice(["UPI", "IMPS", "NEFT"]),
                    "purpose_code": random.choice(["Business Payment", "Vendor Payment"]),
                    "is_fraud": True,
                    "fraud_pattern": "smurfing",
                    "fraud_case_id": case_id,
                })

                # Mule -> Target
                step_time2 = step_time + timedelta(minutes=random.uniform(10, 120))
                transactions.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "timestamp": step_time2.strftime("%Y-%m-%d %H:%M:%S"),
                    "sender_id": mule["entity_id"],
                    "sender_name": mule["name"],
                    "sender_type": mule["type"],
                    "sender_branch": mule["branch"],
                    "receiver_id": target["entity_id"],
                    "receiver_name": target["name"],
                    "receiver_type": target["type"],
                    "receiver_branch": target["branch"],
                    "amount": round(amount * random.uniform(0.90, 0.99), 2),
                    "currency": "INR",
                    "transaction_type": random.choice(["UPI", "IMPS", "NEFT"]),
                    "purpose_code": random.choice(["Business Payment", "Consulting Fee"]),
                    "is_fraud": True,
                    "fraud_pattern": "smurfing",
                    "fraud_case_id": case_id,
                })

        return pd.DataFrame(transactions)

    def generate_shell_funnel_patterns(self, n_patterns: int = 3,
                                        start_date: str = "2025-01-25",
                                        end_date: str = "2025-03-25") -> pd.DataFrame:
        """Generate shell company funnel patterns: multiple sources feed into shell companies."""
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

            for i, source in enumerate(sources):
                n_txns = random.randint(2, 5)
                for j in range(n_txns):
                    step_time = pattern_start + timedelta(
                        days=random.uniform(0, 10),
                        hours=random.uniform(8, 20)
                    )
                    amount = round(total_funneled / (n_sources * n_txns) * random.uniform(0.8, 1.2), 2)

                    transactions.append({
                        "transaction_id": f"TXN{random.randint(100000, 999999)}",
                        "timestamp": step_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "sender_id": source["entity_id"],
                        "sender_name": source["name"],
                        "sender_type": source["type"],
                        "sender_branch": source["branch"],
                        "receiver_id": shell["entity_id"],
                        "receiver_name": shell["name"],
                        "receiver_type": shell["type"],
                        "receiver_branch": shell["branch"],
                        "amount": amount,
                        "currency": "INR",
                        "transaction_type": random.choice(["NEFT", "RTGS", "Wire Transfer"]),
                        "purpose_code": random.choice(["Consulting Fee", "Vendor Payment", "Business Payment"]),
                        "is_fraud": True,
                        "fraud_pattern": "shell_funnel",
                        "fraud_case_id": case_id,
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

        all_data = pd.concat([normal, circular, layering, smurfing, funnels],
                             ignore_index=True)
        all_data = all_data.sort_values("timestamp").reset_index(drop=True)

        # Regenerate unique transaction IDs
        all_data["transaction_id"] = [f"TXN{i:07d}" for i in range(len(all_data))]

        print(f"\nGenerated {len(all_data)} total transactions")
        print(f"  Normal: {len(normal)}")
        print(f"  Circular: {len(circular)}")
        print(f"  Layering: {len(layering)}")
        print(f"  Smurfing: {len(smurfing)}")
        print(f"  Shell Funnel: {len(funnels)}")
        print(f"  Fraud cases: {len(self.fraud_cases)}")

        return all_data, self.fraud_cases


def save_data(df: pd.DataFrame, fraud_cases: List[Dict],
              data_dir: str = "data"):
    """Save generated data to files."""
    import os
    os.makedirs(data_dir, exist_ok=True)

    df.to_csv(os.path.join(data_dir, "transactions.csv"), index=False)
    df.to_parquet(os.path.join(data_dir, "transactions.parquet"), index=False)

    with open(os.path.join(data_dir, "fraud_cases.json"), "w") as f:
        json.dump(fraud_cases, f, indent=2)

    with open(os.path.join(data_dir, "entities.json"), "w") as f:
        gen = TransactionGenerator()
        json.dump(gen.entities, f, indent=2)

    print(f"\nData saved to {data_dir}/")
    print(f"  transactions.csv ({len(df)} rows)")
    print(f"  fraud_cases.json ({len(fraud_cases)} cases)")
    print(f"  entities.json ({len(gen.entities)} entities)")


if __name__ == "__main__":
    generator = TransactionGenerator(seed=42)
    df, fraud_cases = generator.generate_all_data()
    save_data(df, fraud_cases)
    print("\n✓ Data generation complete!")
