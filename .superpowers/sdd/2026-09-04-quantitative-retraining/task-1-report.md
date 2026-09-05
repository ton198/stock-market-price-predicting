# Task 1 report: canonical Stage 1 data/model API

## Status

Implemented and committed the canonical Stage 1 Hybrid LSTM data/model API, deterministic per-seed trainer, CLI, and contract tests. The implementation commit is `2b76508` (`feat: extract canonical hybrid LSTM trainer`). No real Stage 1 training run was started.

## Implementation

- Extended `build_sequence_windows` with `stride: int = 1`, positive-stride validation, and anchor subsampling after the canonical anchor range is constructed. The default daily behavior is unchanged.
- Added the seven-name macro feature contract in the required order:
  `FedRate`, `FedRate_chg20`, `FedRate_chg60`, `TNX`, `Yield_slope`, `VIX`, `VIX_percentile`.
- Added `HybridLSTM` with a 30-feature sequence branch, hidden size 32 LSTM, `7 -> 16` ReLU macro branch, dropout 0.3, and one output logit per row.
- Added `build_stage1_windows`, using stride 3 only for train fit windows and stride 1 for validation and test windows.
- Added `fit_stage1_seed`, which:
  - seeds Python, NumPy, and PyTorch;
  - enables deterministic PyTorch behavior and uses a seeded `torch.Generator` for shuffled train batches;
  - trains only on sparse train fit windows with `BCEWithLogitsLoss`, Adam at `1e-3`, gradient clipping at `1.0`, and `ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-5)`;
  - chooses and restores the best checkpoint by validation loss with early stopping;
  - performs separate dense stride-1 inference for every eligible train anchor;
  - emits exactly `checkpoint.pt`, `history.json`, `predictions_train_fit.csv`, `predictions_train_dense.csv`, `predictions_validation.csv`, `predictions_test.csv`, `metrics.json`, and immutable `manifest.json` in the supplied per-seed directory;
  - writes each prediction CSV with exactly `Date,probability`, deriving tuple dates through explicit integer anchor lookup;
  - records dataset fingerprint, revision, split metadata, device, hyperparameters, stream cadence/role, and artifact names in the manifest.
- Added a multi-seed CLI with the planned arguments. It prepares the canonical dataset once and passes each `run-dir/seed-NNN` directory to the per-seed trainer.

## Files

- `quant_pipeline/data.py` — added the stride API and validation.
- `quant_pipeline/stage1.py` — added model, window builder, deterministic trainer, predictions, metrics, checkpoint, history, and manifest generation.
- `scripts/train_stage1.py` — added the canonical multi-seed training CLI.
- `tests/test_stage1.py` — added stride, macro ordering, shape, canonical alignment/count, artifact, probability, reproducibility, and CLI tests.
- `.superpowers/sdd/2026-09-04-quantitative-retraining/task-1-report.md` — this report.

The checkout already had unrelated modified files and untracked canonical pipeline files. Only the four implementation paths above were staged for the implementation commit; legacy notebooks/models and unrelated paths were untouched.

## TDD evidence

### RED cycle 1: stride and macro selection

Command:

```text
.venv-quant/bin/python -m unittest tests.test_stage1 -v
```

Observed result: exit 1. Four tests exercised the initial contract. The existing default-stride test passed; stride-2 and non-positive-stride cases errored because `stride` was not accepted, and macro selection errored because `quant_pipeline.stage1` did not exist. This was the expected missing-feature failure.

### GREEN cycle 1

After adding only stride handling and `macro_feature_indices`, the same command exited 0: 4 tests ran and all passed.

### RED cycle 2: model, windows, trainer, and CLI

Command:

```text
.venv-quant/bin/python -m unittest tests.test_stage1 -v
```

Observed result: exit 1. The four first-cycle tests passed. Three tests errored on missing `HybridLSTM`, `build_stage1_windows`, and `fit_stage1_seed`; the CLI test failed because `scripts/train_stage1.py` did not exist. This was the expected missing-interface failure.

### GREEN cycle 2

After implementing those interfaces, the same command exited 0: 8 tests ran and all passed in 2.929 seconds.

An additional self-review assertion confirmed that `metrics.json` validation loss equals the best validation loss recorded in `history.json`. It passed immediately and is therefore recorded as a characterization check, not misrepresented as a RED/GREEN cycle.

## Test coverage and final evidence

The Stage 1 tests verify:

- stride-2 anchors are exactly `[3, 5, 7, 9, 11]`;
- stride 0 and -1 are rejected;
- omitted/default stride retains all daily anchors;
- macro columns resolve in the exact required order;
- `HybridLSTM.forward` returns shape `(batch_size,)` logits;
- canonical dense train, validation, and test lengths are exactly 6133, 884, and 1771;
- dense train indices have cadence 1, sparse fit indices have cadence 3, and all streams map exactly to canonical tuple dates;
- all eight per-seed artifacts are present;
- prediction probabilities are finite and in `[0, 1]`;
- synthetic train-fit, train-dense, validation, and test CSV dates are exact;
- two CPU runs with the same seed produce identical dense prediction arrays and validation loss;
- the CLI exposes the planned training arguments.

Final commands:

```text
.venv-quant/bin/python -m compileall -q quant_pipeline scripts tests
.venv-quant/bin/python -m unittest tests.test_stage1 -v
.venv-quant/bin/python -m unittest discover -s tests -v
git diff --cached --check
```

Results:

- compilation exited 0;
- Stage 1 suite: 8/8 passed;
- full existing suite: 19/19 passed in 3.377 seconds;
- staged whitespace check exited 0 before commit.

## Self-review

- Checked every brief item against the implementation and tests.
- Confirmed the trainer receives a per-seed output directory; only the CLI creates `seed-NNN` children.
- Confirmed train stride affects optimizer sampling only. Dense train inference is separately generated with stride 1 and explicitly marked as the only DQN-eligible Stage 1 stream.
- Confirmed checkpoint selection and early stopping read validation loss only; test labels are used only for post-selection diagnostic metrics.
- Confirmed predictions are generated after restoring the best validation state and applying `torch.sigmoid` to logits.
- Confirmed the manifest uses exclusive creation and records stream cadence so sparse fit output cannot be mistaken for dense DQN input.
- Confirmed no renaming or duplication of `fit_signal_normalizer` or `normalize_signal` and no edits to legacy notebooks/models.
- Mutation review: removing stride slicing, allowing non-positive stride, changing macro order, returning a scalar/two-dimensional logit, making dense cadence sparse, shifting split dates, omitting artifacts, emitting invalid probabilities, or dropping deterministic seeding is covered by a test.

## Concerns

- The task intentionally stops before Task 2: no pilot, full ten-seed training, summarization, seed selection, or canonical evaluation run was performed.
- CPU deterministic reproducibility is tested. CUDA execution and cross-device bitwise identity were not tested in this CPU-only environment.
- The shared checkout remains deliberately dirty. Several supporting canonical pipeline files were already untracked and were preserved outside this task commit, exactly as requested; the implementation was verified against the complete shared checkout state.

## Review fix report — clean-checkout dependency and protocol manifest

### Review findings verified

The clean-checkout dependency was reproduced from a `git archive HEAD` extraction. Running the Task 1 environment's Python from that extraction with:

```text
/home/doctor235/intern-workspace/stock-market-price-predicting/.venv-quant/bin/python -c "import quant_pipeline.stage1"
```

exited 1 with `ModuleNotFoundError: No module named 'quant_pipeline.metrics'`. `git ls-tree` confirmed that the reviewed head contained only `quant_pipeline/data.py` and `quant_pipeline/stage1.py` under the package, while the dirty shared checkout's `quant_pipeline/metrics.py` was untracked and masked the defect.

Inspection also confirmed that `manifest.json` recorded split boundaries but had no fields qualifying the split protocol as fixed-origin, prediction-anchor based, unpurged at label-overlap boundaries, and not an exact retraining-at-boundary simulation.

### Fix implementation

- Removed the import and use of untracked `quant_pipeline.metrics.classification_metrics`. `metrics.json` retains the Task 1 artifact contract and its train-fit, train-dense, validation, and test loss diagnostics. Canonical classification evaluation remains outside this trainer and is performed by the planned Task 2 canonical evaluator.
- Strengthened the CLI test to run from an isolated temporary package containing only Task 1's committed `data.py`, `stage1.py`, and `train_stage1.py`. This prevents unrelated dirty files from masking future clean-checkout dependencies.
- Added a `protocol` object to every per-seed manifest with these explicit fields:
  - `origin: "fixed"`;
  - `split_basis: "prediction_anchor"`;
  - `label_overlap_purged_at_boundaries: false`;
  - `exact_retraining_at_each_boundary: false`;
  - qualification: `Fixed-origin, anchor-based, unpurged offline research contract; not an exact retraining-at-boundary simulation.`
- Added a focused artifact assertion for the complete literal protocol object.
- Preserved the user's untracked `quant_pipeline/metrics.py` and all other unrelated dirty paths; none were staged.

### Fix TDD evidence

Clean-checkout dependency RED:

```text
.venv-quant/bin/python -m unittest tests.test_stage1.Stage1ModelContractTests.test_stage1_cli_exposes_canonical_arguments_with_only_task1_modules -v
```

Result: exit 1. The isolated CLI failed while importing `quant_pipeline.stage1`, with `ModuleNotFoundError: No module named 'quant_pipeline.metrics'`.

Clean-checkout dependency GREEN after removing the dependency:

```text
.venv-quant/bin/python -m unittest tests.test_stage1.Stage1ModelContractTests.test_stage1_cli_exposes_canonical_arguments_with_only_task1_modules -v
```

Result: exit 0; 1/1 passed in 1.420 seconds.

Protocol metadata RED:

```text
.venv-quant/bin/python -m unittest tests.test_stage1.Stage1ModelContractTests.test_seed_trainer_writes_aligned_finite_reproducible_predictions -v
```

Result: exit 1 with `KeyError: 'protocol'`, proving the generated manifest lacked the required observable contract.

Protocol metadata GREEN after adding the explicit fields:

```text
.venv-quant/bin/python -m unittest tests.test_stage1.Stage1ModelContractTests.test_seed_trainer_writes_aligned_finite_reproducible_predictions -v
```

Result: exit 0; 1/1 passed in 0.815 seconds.

### Fix verification

Focused suite:

```text
.venv-quant/bin/python -m unittest tests.test_stage1 -v
```

Result: exit 0; 8/8 passed in 2.748 seconds on the final pre-commit rerun.

Full suite:

```text
.venv-quant/bin/python -m unittest discover -s tests -v
```

Result: exit 0; 19/19 passed in 3.332 seconds on the final pre-commit rerun.

`.venv-quant/bin/python -m compileall -q quant_pipeline scripts tests` and `git diff --check` also exited 0.

### Fix self-review and concerns

- The isolated CLI test exercises actual imports and argument parsing rather than searching source text or mocking dependencies.
- Removing or reintroducing a non-Task-1 package import now breaks the isolated test.
- Omitting or changing any mandatory protocol field now breaks the manifest artifact test.
- The trainer still does not perform Task 2 canonical classification evaluation or seed aggregation.
- No real 300-epoch or CUDA training run was performed as part of this review fix.
