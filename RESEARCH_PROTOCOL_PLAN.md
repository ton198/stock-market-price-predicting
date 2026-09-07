# Price-Only Research Protocol Plan

## Objective

Replace the current fixed-origin, macro-feature research path with a **price-only,
leakage-aware Stage 1 protocol**. The first deliverable is an auditable
walk-forward evaluation; it is not a claim of trading alpha. DQN remains
`research_only` until Stage 1 clears the protocol and baseline gates.

## Security prerequisite

- The hard-coded FRED credential was removed from
  `source_code/data_processing/data_pipeline.ipynb`; the notebook now requires
  `FRED_API_KEY` from the environment.
- The exposed key must be revoked/rotated in FRED by its owner. Do not add a
  replacement key to source control.

## Proposed protocol (confirm before implementation)

| Area | Proposed rule |
| --- | --- |
| Features | Keep OHLCV and features derived solely from OHLCV. Exclude VIX, TNX, FedRate, CPI_MoM, and all macro-derived features. |
| Information time | A signal at close on day `t` may use only information available by that close. |
| Execution | Execute at the next trading day's open (`t+1`), then measure the holding return from `t+1` to `t+2`. Start in cash and discard incomplete final intervals. |
| Split isolation | At both train/validation and validation/test boundaries use `purge=20` and `embargo=20` trading days. Retain every excluded range in the manifest. |
| Validation | Use expanding-window walk-forward folds. Fit scaler and label threshold independently inside each fold's training partition. |
| Model selection | Select seed/epoch only with validation. Evaluate each fold's test partition once; never aggregate test results to choose parameters. |
| DQN | Do not include DQN in v1 walk-forward claims. Keep its current outputs explicitly `research_only`. |
| Legacy code | Retain old macro and same-day execution paths only when visibly marked `legacy`; canonical commands must reject them. |

## Implementation sequence

### 1. Establish a price-only data contract

Files: `quant_pipeline/data.py`, `quant_pipeline/stage1.py`,
`quant_pipeline/dqn_env.py`, `scripts/train_stage1.py`,
`scripts/train_dqn.py`.

- Define an explicit price-only feature list and protocol name/version.
- Ensure feature engineering does not read macro columns when using the
  canonical protocol; a CSV with only OHLCV columns must work.
- Replace the macro branch of `HybridLSTM` with a single price-only sequence
  model, or isolate the old model as legacy.
- Store `protocol`, complete `feature_names`, and `feature_count` in every
  checkpoint, prediction manifest, and audit record. Reject incompatible
  checkpoints.

### 2. Make execution timing explicit

Files: `quant_pipeline/data.py`, `quant_pipeline/backtest.py`,
`quant_pipeline/evaluator.py`, `quant_pipeline/dqn_env.py`.

- Preserve `Open` in the market frame and separate signal dates, execution
  dates/prices, and realized-return intervals.
- Add a canonical delayed-execution simulator. An action must not earn a
  same-timestamp return; turnover cost is charged at execution.
- Evaluate buy-and-hold over the identical executable window.
- Keep `simulate_positions()` only as a legacy compatibility function, or
  remove it after all callers migrate.

### 3. Add purge and embargo to chronological splits

Files: `quant_pipeline/data.py`, `quant_pipeline/stage1.py`,
`scripts/audit_pipeline.py`.

- Extend split metadata to record train, validation, test, and excluded ranges.
- Implement separate `horizon`, `purge`, and `embargo` parameters; defaults are
  20, 20, and 20 days respectively.
- Reject datasets too small to form the requested partitions.
- Permit prior historical rows as sequence context only; no row after a split
  boundary may enter that split's features or labels.

### 4. Add an expanding walk-forward Stage 1 runner

New file: `quant_pipeline/walk_forward.py`.

Files to update: `scripts/train_stage1.py`,
`scripts/evaluate_predictions.py`, `quant_pipeline/evaluator.py`.

- Define fold objects with train/validation/test slices, dates, purge ranges,
  embargo ranges, and a fold ID.
- For each fold: fit preprocessing on train, select only on validation, predict
  the isolated test slice, and evaluate it with delayed execution.
- Write a stable run layout:

  ```text
  runs/walk_forward/<run_id>/
    run_manifest.json
    dataset_audit.json
    fold-000/{split_manifest.json,stage1/,evaluation.json}
    aggregate_evaluation.json
  ```

- Record Git revision, dataset SHA-256, feature contract, protocol settings,
  dependency versions, fold boundaries, and hashes of generated artifacts.
- Aggregate only fold-level test metrics; include cash, buy-and-hold, and at
  least one simple price-only baseline.

## Required automated evidence

1. **Feature contract:** price-only CSVs work; macro columns are not required
   or read; manifests identify the exact feature set.
2. **Execution:** synthetic price tests prove that actions execute one trading
   day after signals, costs apply then, and incomplete last intervals are
   excluded.
3. **Split isolation:** tests prove no 20-day forward label crosses a split
   boundary and every purge/embargo row is recorded.
4. **Walk-forward:** fold dates are ordered; no test row is used for fitting or
   selection; per-fold preprocessing is independent.
5. **Artifacts:** manifests are JSON-serializable, include protocol and hashes,
   and aggregation rejects missing, duplicate, or mismatched folds.
6. Run the full unit suite and a small deterministic synthetic walk-forward
   smoke run. Do not treat a real-data training run as evidence of correctness.

## Acceptance criteria

- Canonical Stage 1 cannot silently use macro data or same-day execution.
- Every canonical result states its protocol, executable date window, feature
  contract, data fingerprint, and fold boundaries.
- Test results are produced only from isolated fold test periods.
- README and result documents state that DQN is `research_only` and that this
  work does not establish an investable strategy.

## Explicit non-goals for v1

- ALFRED/release-vintage macro data support.
- Per-fold DQN retraining or claims about DQN alpha.
- Statistical confidence intervals, cost sensitivity sweeps, and a full
  baseline matrix (these follow after the v1 protocol is stable).
- Rewriting historical notebooks or presenting their old results as canonical.

## Decisions for the implementation agent

Unless the owner changes them, implement the proposed protocol above. Before
starting, confirm that the dataset's `Open` field is complete enough for
next-open execution. If it is not, stop and request a decision rather than
silently falling back to same-day or next-close execution.
