# Results directory

Files already in this directory predate the canonical evaluation contract and
are retained as provenance.  Do not quote them as current evidence.  New
results should include the contract version, split dates, costs/slippage, and
random seed in their filename or metadata.  `scripts/audit_pipeline.py` writes
the first contract-level artifact as `canonical_data_audit.json`.

The reviewed canonical retraining evidence is documented in
[`canonical/dqn/README.md`](canonical/dqn/README.md). It is marked
`research_only`; the Stage 1 validation gate is held and the portfolio remains
unchanged.
