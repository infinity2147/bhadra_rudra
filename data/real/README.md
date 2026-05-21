# Real benchmark datasets

This directory is where you place the public datasets that RUDRA trains
its real-data ML models on. The files themselves are not committed — they
are 100s of MB to GB each, and most have license terms that don't allow
redistribution.

To enable a real-data model variant, download the CSVs into the matching
subdirectory below and re-run `python src/run_pipeline.py`. The pipeline
will detect which datasets are present and train a model variant for each.

## 1. IBM AML — Anti-Money-Laundering Transaction Data

The PoA's primary benchmark. The HI-Small variant is laptop-runnable
(~5 M rows ungzipped).

- Download: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
- Place files into: `data/real/ibm_aml/`
- Expected: `HI-Small_Trans.csv` (or `LI-Small_Trans.csv`)

## 2. PaySim — Mobile money fraud

Synthetic-but-realistic mobile-money transactions with sender→receiver
structure. About 6 M rows.

- Download: https://www.kaggle.com/datasets/ealtman2019/paysim1
- Place files into: `data/real/paysim/`
- Expected: any `.csv` file

## 3. IEEE-CIS Fraud Detection

Tabular (no graph) — used by RUDRA's separate "tabular ML baseline" page
so we can claim ML metrics on a public Kaggle dataset suggested by the
evaluators.

- Download: https://www.kaggle.com/c/ieee-fraud-detection/data
- Place files into: `data/real/ieee_cis/`
- Expected: `train_transaction.csv` (and optionally `train_identity.csv`)

## Why these specifically?

Our problem statement is *fund flow tracking* — money moving between
accounts. The credit-card-fraud datasets in the evaluators' resource list
(Kaggle Credit Card Fraud, IEEE-CIS) are tabular with no sender→receiver
structure, so we use them only for the tabular baseline. IBM AML and
PaySim are the public datasets that match our graph-structured problem.
