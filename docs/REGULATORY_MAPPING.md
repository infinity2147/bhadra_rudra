# RUDRA — Regulatory Mapping & Legal Basis

Every detector in RUDRA maps to a recognised **FATF money-laundering typology**
and the specific **Indian-regulatory hooks** an FIU-IND filer cites — and, unlike
a slide deck, **every mapping here is backed by running code** (file references
given). The tagging is applied automatically to each alert by
[`src/fatf_typology.py`](../src/fatf_typology.py) and travels all the way into
the STR XML.

> Design stance: we don't *claim* regulatory alignment, we *substantiate* it.
> Each row below points at the code that implements it and the test that proves it.

---

## The two-stage legal lever (why severity → action matters)

A bank doesn't have to wait for a predicate offence to act. RUDRA grades every
alert into a **legal basis for action** (see `_legal_basis()` in
[`src/fatf_typology.py`](../src/fatf_typology.py)):

| Tier | Trigger | Legal basis | What the bank may do |
|---|---|---|---|
| Pre-emptive | severity = HIGH | **RBI KYC Master Direction ¶38** | Enhanced monitoring / partial operating restriction *before* PMLA applies |
| Mandatory | severity = CRITICAL or confidence ≥ 85 | **PMLA 2002 §12** | File a Suspicious Transaction Report (STR) with FIU-IND |
| Internal | otherwise | Internal EDD | Monitor, re-KYC, re-score |

This is the "restrict early under KYC rules, report once suspicion is firm under
PMLA" framing — surfaced on every alert (`legal_basis` field) and in the UI
badges, and written into `<LegalBasis>` in the STR XML.

---

## Detector → typology → regulation

| Pattern (alert `pattern_type`) | Detector (implementing code) | FATF typology | FIU-IND red flag | PMLA | RBI |
|---|---|---|---|---|---|
| Circular Transaction | `detect_circular_transactions` — Johnson's cycles over SCCs ([fraud_detector.py](../src/fraud_detector.py)) | Round-tripping / circular movement | Funds returning to originator via intermediaries | §3 r/w §12 | KYC MD |
| Rapid Layering | `detect_rapid_layering` — temporal-causal money-following walk | Layering — rapid multi-hop to break the trail | Rapid pass-through, no economic rationale | §3 r/w §12 | KYC MD |
| Smurfing / Structuring | `detect_smurfing` — edge-cluster + temporal burst + fan-out | Structuring (smurfing) | Many sub-threshold transfers to/from many parties | §12 r/w Rule 3 | KYC MD + PMLA Records Rules 2005 |
| Shell Company Funnel | `detect_shell_funnels` — imbalance + pass-through (FIFO holding) | Use of shell / funnel accounts | High-velocity pass-through, no genuine business | §3 r/w §12 | KYC MD (beneficial ownership) |
| Dormant Activation | `DormantActivationDetector` ([advanced_detectors.py](../src/advanced_detectors.py)) | Dormant-account reactivation / takeover | Dormant account, abrupt high-value activity | §12 | KYC MD (ongoing DD) |
| Profile Mismatch | `ProfileMismatchDetector` | Activity inconsistent with customer profile | Activity disproportionate to KYC profile | §12 | KYC MD (risk categorisation) |
| Recruiter / Coordinator | `detect_recruiters` — dominant-funder fan-out | Money-mule network (coordinator/herder) | One funder seeding many forwarding accounts | §3 r/w §12 | KYC MD + RBI mule-account guidance |

The machine-readable source of truth for this table is the `TYPOLOGY` dict in
[`src/fatf_typology.py`](../src/fatf_typology.py); it is unit-tested in
[`tests/test_fatf.py`](../tests/test_fatf.py) (every pattern maps to a complete
typology; legal basis tiers by severity).

---

## The evidence package — what an investigator actually files

`GET /api/fiu/package/{alert_id}` ([backend/main.py](../backend/main.py)) streams a
zip built by [`src/fiu_package.py`](../src/fiu_package.py):

- **`STR.xml`** — FIU-IND-shaped Suspicious Transaction Report. Now carries a
  `<RegulatoryClassification>` block (FATF typology + FIU advisory + PMLA section
  + RBI ref + the graded `<LegalBasis>`) **and** a `<SHAPExplainability>` block
  embedding the model's top signed feature contributions for the flagged edge —
  so the *why* travels with the filing, not just a probability. (Tested in
  [`tests/test_str_xml.py`](../tests/test_str_xml.py).)
- **`SAR_<id>.pdf`** — ReportLab narrative SAR.
- **`subgraph.json`** — the implicated fund-flow subgraph.
- **`transaction_chain.csv`** — the chronological transaction lineage.
- **`pmla_citations.txt`** — pattern-specific PMLA/RBI citations.
- **`case_audit_log.json`** — the full case trail, **SHA-256 hash-chained** and
  tamper-evident (`GET /api/cases/{id}/verify`; tested in
  `tests/test_case_store.py::test_hash_chain_detects_tampering`).

PII fields are emitted as `REDACTED` placeholders by default (DPDP Act data
minimisation); the filing bank fills them in just before submission.

---

## PCI DSS Compliance (Data Security)

While RUDRA operates on transaction metadata, we demonstrate readiness for secure environments (such as processing environments touching credit card data) by fulfilling core Payment Card Industry Data Security Standard (PCI DSS) requirements within the application layer:

| Requirement | Implementation in RUDRA | Verification Code |
|---|---|---|
| **Req 3: Protect Stored Account Data** | Mock card numbers are injected into the API payload but masked (`453271XXXXXX2345`) before leaving the backend, ensuring plaintext data is never exposed by default. | `backend/main.py::_compute_alerts_with_case_status` |
| **Req 7: Restrict Access to Cardholder Data** | The `/api/cases/{alert_id}/reveal-card` endpoint enforces Role-Based Access Control (RBAC). Only `SUPERVISOR` or `ADMIN` roles can unmask the card. `INVESTIGATOR` attempts return HTTP 403 Forbidden. | `backend/main.py::reveal_card`, `tests/test_pci_dss.py` |
| **Req 10: Log and Monitor All Access** | Every successful card unmasking action writes a secure, tamper-evident `REVEAL_CARD` event to the `CaseStore` hash-chain audit log. | `src/case_manager.py::audit_action` |

---

## "Backed by running code" — the contrast that matters

Several systems in this space *describe* these capabilities; RUDRA ships them.
Concretely:

| Capability | Real in RUDRA (file) | Common substitute elsewhere |
|---|---|---|
| Tamper-evident audit | SHA-256 hash chain, `src/case_manager.py` + verify endpoint | DB triggers / nothing |
| ML explainability in the filing | SHAP embedded in STR XML, `src/shap_explainer.py` → `src/fiu_package.py` | a probability number, or none |
| Multi-hop detection | graph detectors + GraphSAGE + stacked ensemble | edge-only / unused ML |
| Sanctions + AA | credential-driven real adapters, `src/integrations/` | `md5()` mocks |
| Persistent cross-run memory | decaying taint store, `src/taint_store.py` | per-run only / stub |
| Coordinator (mule-herder) detection | `detect_recruiters` | mule-only |

---

## Caveats (stated plainly)

- The `fatf_code` values (e.g. `ML-LAYERING`) are RUDRA's **internal handles** for
  the UI; FATF publishes typologies as descriptive families, not numbered codes —
  the human-readable `fatf_typology` carries the actual wording.
- Demonstrations run on synthetic data and the public IBM AML 100k sample; the
  regulatory mapping is dataset-independent.
- Legal references are provided to orient an investigator; they are not legal
  advice and a bank's compliance team owns the final filing decision.
