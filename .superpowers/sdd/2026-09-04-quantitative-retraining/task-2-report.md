# Task 2 Report: Stage 1 Pilot and Full Seed Audit

## Status

`DONE_WITH_CONCERNS`

Task 2 implementation, TDD verification, three-seed pilot, ten-seed full run,
canonical test evaluation, summary generation, and artifact audit completed.
Both pilot and full technical gates passed. The validation-only Stage 2 gate is
`hold` because the aggregate Stage 1 signal did not beat all frozen
training-prior criteria.

Code/test commit: `49ffe7fa96b2f005d38b2be93306f0a2bbcc6fb9`
(`feat: audit stage 1 seed runs`). Generated run artifacts remain ignored under
`runs/canonical_retrain/20260905T053626Z/` and are not committed.

## Requirements and scope observed

- Read and followed `task-2-brief.md` before implementation.
- Used `datasets/nasdaq_multivariate.csv` and frozen run ID
  `20260905T053626Z`.
- Used `.venv-quant/bin/python` for every test, evaluation, summary, and
  training command.
- Used the exact pilot seeds `0 17 34` and full seeds
  `0 17 34 51 68 85 102 119 136 153` with `300` requested epochs,
  `window-size=60`, `train-stride=3`, `batch-size=32`, and `device=auto`.
- Did not dispatch subagents.
- Did not modify `models/`, notebooks, portfolio files, README files,
  `METHODOLOGY.md`, or Task 3+ DQN code.
- Did not create `results/canonical/stage1/` or copy result files because the
  post-run review condition has not been satisfied.
- Preserved and did not stage unrelated pre-existing user changes and
  untracked canonical modules.

## Implementation

### `quant_pipeline/evaluator.py`

- Added `align_split_predictions(...)` for exact canonical train,
  validation, or test anchor alignment.
- Added `evaluate_probability_stream_for_split(...)` for split-specific
  classification and costed strategy diagnostics.
- Preserved `align_test_predictions(...)` and
  `evaluate_probability_stream(...)` as exact-test wrappers with the original
  test report schema (`test_start`, `test_end`, and no `split` field).

### `scripts/summarize_stage1.py`

- Discovers only exact `seed-NNN` child directories and requires the exact
  declared pilot or full seed set.
- Requires exactly these prediction artifacts per seed:
  `predictions_train_fit.csv`, `predictions_train_dense.csv`,
  `predictions_validation.csv`, and `predictions_test.csv`; missing or extra
  `predictions_*.csv` files fail the run.
- Validates all four date streams, probability bounds/finiteness, artifact
  inventory, seed metadata, dense-train role metadata, cadence, row counts,
  and date bounds.
- Runs the shared exact-test evaluator before reading `metrics.json`, then
  atomically writes or verifies immutable `canonical_evaluation.json`.
- Rejects mismatched existing evaluations and non-finite values in manifest,
  history, metrics, predictions, or the generated summary.
- Writes `summary.json` atomically with per-seed rows and population
  mean/std/median metric tables.
- Includes the frozen training-prior, cash, buy-and-hold, and raw
  `probability_positions` validation readouts. The raw policy is recorded
  verbatim as `clip(2 * probability - 1, -1, 1)` with position range
  `[-1.0, 1.0]`, transaction cost `0.001`, and slippage `0.0005`.
- Emits no pilot selection. For exactly ten full seeds, selection is
  `validation_loss ascending, seed ascending tie-break`.
- Uses only validation aggregates for the explicit Stage 2 gate:
  median validation loss must be below training-prior log loss, median Brier
  must be below training-prior Brier, and median ROC-AUC must exceed the
  training-prior ROC-AUC. Test metrics are never referenced by this gate.

### Tests

Added focused coverage in `tests/test_stage1.py` for:

- exact validation split alignment/evaluation;
- unchanged exact-test CLI report behavior;
- pilot discovery and absence of a Stage 1 selection;
- full-run validation-loss selection and seed tie-break;
- exact prediction artifact inventory;
- immutable evaluator recomputation/mismatch rejection;
- atomic summary output cleanup; and
- rejection of non-finite history diagnostics.

## TDD evidence

### Baseline

Command:

```bash
.venv-quant/bin/python -m unittest tests.test_pipeline tests.test_stage1 -v
```

Result: `Ran 18 tests in 2.858s` / `OK`.

### RED

Initial command:

```bash
.venv-quant/bin/python -m unittest tests.test_stage1.Stage1EvaluationTests.test_split_evaluator_uses_exact_validation_anchors tests.test_stage1.Stage1SummaryTests -v
```

The first run correctly failed for the absent split helper but also exposed a
test-fixture slice-precedence error (`TypeError: slice indices must be
integers or None or have an __index__ method`). The fixture was corrected
without production changes. The rerun then produced six assertion failures
solely for missing production behavior:

- `split-parameterized evaluator helper is missing`; and
- `can't open file .../scripts/summarize_stage1.py: [Errno 2] No such file or directory`.

The non-finite-history test was introduced in a separate RED cycle:

```bash
.venv-quant/bin/python -m unittest tests.test_stage1.Stage1SummaryTests.test_summary_rejects_nonfinite_history_diagnostics -v
```

Result: expected failure, `AssertionError: 0 == 0`, proving the initial
summarizer incorrectly accepted `NaN` in `history.json`.

### GREEN

Focused command:

```bash
.venv-quant/bin/python -m unittest tests.test_stage1.Stage1EvaluationTests tests.test_stage1.Stage1SummaryTests -v
```

Result: `Ran 7 tests in 5.285s` / `OK`.

Complete command before the code commit:

```bash
.venv-quant/bin/python -m unittest discover -s tests -v
.venv-quant/bin/python -m py_compile quant_pipeline/evaluator.py scripts/evaluate_predictions.py scripts/summarize_stage1.py
```

Result: `Ran 26 tests in 8.622s` / `OK`; compilation exited `0`.

## Run commands and results

### Pilot training

```bash
.venv-quant/bin/python scripts/train_stage1.py \
  --data datasets/nasdaq_multivariate.csv \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-pilot \
  --seeds 0 17 34 \
  --device auto \
  --epochs 300 \
  --window-size 60 \
  --train-stride 3 \
  --batch-size 32
```

Exit `0`. All seeds resolved to CPU and early-stopped:

| Seed | Best epoch | Epochs completed | Validation loss | Test loss |
|---:|---:|---:|---:|---:|
| 0 | 1 | 21 | 0.6958279014 | 0.6923524141 |
| 17 | 2 | 22 | 0.6963351369 | 0.7029217482 |
| 34 | 1 | 21 | 0.7003245354 | 0.7067430019 |

### Pilot exact-test evaluation

The following command was run separately for seeds `000`, `017`, and `034`:

```bash
.venv-quant/bin/python scripts/evaluate_predictions.py \
  --predictions runs/canonical_retrain/20260905T053626Z/stage1-pilot/seed-NNN/predictions_test.csv \
  --output runs/canonical_retrain/20260905T053626Z/stage1-pilot/seed-NNN/canonical_evaluation.json
```

All three exited `0` and accepted exactly 1,771 dates from `2019-02-01` to
`2026-02-18`.

| Seed | Test accuracy | Test Brier | Test ROC-AUC | Raw-policy test return |
|---:|---:|---:|---:|---:|
| 0 | 0.5296442688 | 0.2496559525 | 0.5346556767 | -0.0133085499 |
| 17 | 0.4686617730 | 0.2548485620 | 0.5124300974 | -0.1279933181 |
| 34 | 0.4601919819 | 0.2567560806 | 0.5128592154 | -0.1150983279 |

These test values were recorded only after evaluator acceptance and did not
affect the pilot technical gate or Stage 2 decision.

### Pilot summary

```bash
.venv-quant/bin/python scripts/summarize_stage1.py \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-pilot \
  --data datasets/nasdaq_multivariate.csv \
  --output runs/canonical_retrain/20260905T053626Z/stage1-pilot/summary.json
```

Exit `0`; technical gate `pass`; no Stage 1 selection emitted. Validation-only
Stage 2 decision: `hold`.

Pilot validation medians:

- loss `0.6963351369` versus training-prior loss `0.6931471806`;
- Brier `0.2515768562` versus training-prior Brier `0.25`;
- ROC-AUC `0.5085900974` versus training-prior ROC-AUC `0.5`;
- raw-policy total return `-0.0436963173` versus cash `0.0` and buy-and-hold
  `0.4223861515`.

Because the pilot technical gate passed, the exact full run proceeded even
though the separate validation-only research gate was weak.

### Full ten-seed training

```bash
.venv-quant/bin/python scripts/train_stage1.py \
  --data datasets/nasdaq_multivariate.csv \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-full \
  --seeds 0 17 34 51 68 85 102 119 136 153 \
  --device auto \
  --epochs 300 \
  --window-size 60 \
  --train-stride 3 \
  --batch-size 32
```

Exit `0`; every seed early-stopped after 21 or 22 epochs.

### Full summary

```bash
.venv-quant/bin/python scripts/summarize_stage1.py \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-full \
  --data datasets/nasdaq_multivariate.csv \
  --output runs/canonical_retrain/20260905T053626Z/stage1-full/summary.json
```

Exit `0`; all ten canonical evaluations were generated and verified. The same
command was rerun after generation and exited `0`, proving that every existing
immutable evaluation matched fresh shared-evaluator output.

Per-seed audited results:

| Seed | Val loss | Val Brier | Val ROC-AUC | Val policy return | Test accuracy | Test ROC-AUC | Test policy return |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.6958279014 | 0.2513216475 | 0.5217658475 | -0.0324286022 | 0.5296442688 | 0.5346556767 | -0.0133085499 |
| 17 | 0.6963351369 | 0.2515768562 | 0.5085900974 | -0.0436963173 | 0.4686617730 | 0.5124300974 | -0.1279933181 |
| 34 | 0.7003245354 | 0.2535685493 | 0.4522667104 | -0.0553554305 | 0.4601919819 | 0.5128592154 | -0.1150983279 |
| 51 | 0.6953690052 | 0.2511057223 | 0.5023888408 | -0.0264313256 | 0.5070581592 | 0.5124971674 | -0.0222086965 |
| 68 | 0.7016122341 | 0.2542125661 | 0.4661286441 | -0.0531051091 | 0.4568040655 | 0.5303827290 | -0.0642758375 |
| 85 | 0.6931626797 | 0.2500084465 | 0.5039045897 | -0.0006109038 | 0.5590062112 | 0.5345788392 | 0.0028657357 |
| 102 | 0.6953774095 | 0.2511100599 | 0.4976418838 | -0.0446363201 | 0.4562394128 | 0.4916715938 | -0.1177886240 |
| 119 | 0.6920855045 | 0.2494707442 | 0.5229487462 | -0.0239523334 | 0.5081874647 | 0.5143041516 | -0.0516099818 |
| 136 | 0.6929973364 | 0.2499257895 | 0.5189391806 | -0.0297016935 | 0.5064935065 | 0.5176583048 | 0.0126982472 |
| 153 | 0.6966410279 | 0.2517420447 | 0.4763343455 | -0.0360826044 | 0.4839073970 | 0.5449128350 | -0.0563272218 |

Selected Stage 1 artifact for any later Stage 2 work:

```json
{
  "criterion": "validation_loss ascending, seed ascending tie-break",
  "seed": 119,
  "seed_directory": "seed-119",
  "validation_loss": 0.6920855045318604
}
```

The selection is entirely validation-based. Test metrics were evaluated and
reported only after each file passed exact canonical alignment.

## Full aggregate result

Key full-run mean/std/median values from `summary.json`:

| Metric | Mean | Std | Median |
|---|---:|---:|---:|
| Validation loss | 0.6959732771 | 0.0028968934 | 0.6956026554 |
| Validation accuracy | 0.5009049774 | 0.0223272335 | 0.5050904977 |
| Validation Brier | 0.2514042426 | 0.0014411723 | 0.2512158537 |
| Validation ROC-AUC | 0.4970908886 | 0.0231528877 | 0.5031467153 |
| Validation raw-policy return | -0.0346000640 | 0.0152412692 | -0.0342556033 |
| Test accuracy | 0.4936194241 | 0.0327399272 | 0.4952004517 |
| Test ROC-AUC | 0.5205950610 | 0.0146875879 | 0.5159812282 |
| Test raw-policy return | -0.0553046574 | 0.0487088844 | -0.0539686018 |
| Test buy-and-hold return | 2.1277939592 | 0.0000000000 | 2.1277939592 |

The full validation-only gate is `hold`:

- median loss `0.6956026554` is not below `0.6931471806`;
- median Brier `0.2512158537` is not below `0.25`;
- median ROC-AUC `0.5031467153` is above `0.5`; and
- median raw-policy validation return `-0.0342556033` trails cash `0.0` and
  buy-and-hold `0.4223861515`.

The gate explicitly records `uses_test_metrics: false`.

## Artifact verification

An independent inventory check found:

- pilot: 3 exact seed directories, exactly 9 files per seed after canonical
  evaluation;
- full: 10 exact seed directories, exactly 9 files per seed after canonical
  evaluation;
- all 13 manifests record code revision
  `49ffe7fa96b2f005d38b2be93306f0a2bbcc6fb9`;
- all 13 canonical evaluations report exactly 1,771 rows over
  `2019-02-01` through `2026-02-18`;
- pilot summary SHA-256:
  `b41b5b7302e4ff615987728749069fdc1b0769dddf0465837d9087468c498f90`;
- full summary SHA-256:
  `0e9fdb9bf761b49b734212f44b70577000239c864885fcd401111d3c1e77359e`;
- the full summary retained the same SHA-256 after the immutable verification
  rerun; and
- `git status --short models` was empty. The run root appeared only as
  ignored (`!! runs/canonical_retrain/20260905T053626Z/`).

## Self-review

- Confirmed the Stage 2 gate references only validation aggregates and frozen
  validation baselines.
- Confirmed full selection uses only `(validation_loss, seed)` and the pilot
  has no selection key.
- Confirmed `evaluate_probability_stream(...)` retains its prior exact-test
  output schema and `scripts/evaluate_predictions.py` was not changed.
- Confirmed the summarizer recomputes shared-evaluator output before opening
  `metrics.json` and reads the written/verified immutable JSON for aggregation.
- Confirmed atomic writes use a same-directory temporary file, `fsync`, and
  `os.replace`, with temporary-file cleanup.
- Confirmed only `quant_pipeline/evaluator.py`,
  `scripts/summarize_stage1.py`, and `tests/test_stage1.py` were staged in the
  code/test commit.
- Confirmed no reviewed results were copied and no forbidden documentation or
  model files were modified.

## Concerns and follow-up gate

1. The full Stage 1 aggregate is weak on validation. It misses the frozen loss
   and Brier baselines, average ROC-AUC is below 0.5, median ROC-AUC is only
   0.5031, and every seed has a negative raw-policy validation return. The
   predeclared decision is therefore `hold`; Stage 2 should not proceed without
   explicit review and a new authorized task.
2. Seed 119 wins the predeclared selection and individually beats the
   training-prior loss/Brier/AUC thresholds, but its validation policy return
   is still `-2.3952%`. This does not override the aggregate hold decision.
3. Test-set results are also weak (median raw-policy return `-5.3969%` versus
   buy-and-hold `+212.7794%`), but they were not used for selection or the
   Stage 2 gate.
4. An optional report-inspection command using `jq` failed with
   `zsh:1: command not found: jq`. It was replaced by a read-only
   `.venv-quant/bin/python -c ...` extraction; no dependency was installed and
   the training/evaluation workflow was unaffected.
5. Publication under `results/canonical/stage1/` remains intentionally pending
   external review. No result copies or results README were created.

## Review fix round: provenance and canonical-contract enforcement

### Fix status

The three blocking review findings were addressed in one code/test round.
Existing pilot and full artifacts pass the strengthened audit, so the change
does not invalidate checkpoints or predictions and the ten-seed training was
not rerun.

### Root-cause verification

The reviewed implementation previously:

- treated manifest `window_size` and `train_stride` as trusted inputs when
  deriving expected dates;
- did not require the canonical protocol, checkpoint-selection criterion,
  raw dataset identity, prepared-dataset fingerprint, feature order, or split
  metadata;
- checked only two manifest hyperparameters; and
- selected directly from `metrics["validation_loss"]` without corroborating
  history, best epoch, checkpoint metadata, or validation prediction BCE.

The frozen artifacts were measured before choosing a tolerance. Across all 13
pilot/full seed artifacts, the maximum absolute metrics-to-history validation
loss delta was `7.01e-8`, the maximum metrics-to-prediction BCE delta was
`6.38e-8`, and every metrics/history best epoch agreed. The fix therefore uses
a documented absolute tolerance of `1e-7` with zero relative tolerance.

### Production changes

`scripts/summarize_stage1.py` now requires:

- canonical raw data SHA-256
  `3f8f3dcd571fceca215778333d6c97d45352bb2544ebc4c2d7467d79cf540ef7`;
- prepared-dataset fingerprint
  `f4e3fe6d621f82525c977c427e4f61deb96b420ab2c3b34361419f1cc63c4f02`;
- horizon `20`, exactly 30 features in `DEFAULT_FEATURE_COLS` order, and exact
  chronological slices train `[0,6192)`, validation `[6192,7076)`, and test
  `[7076,8847)`;
- requested epochs `300`, batch size `32`, window size `60`, train stride `3`,
  learning rate `1e-3`, and early-stopping patience `20`;
- the exact fixed-origin, prediction-anchor, unpurged, not-exact-boundary-
  retraining protocol object;
- manifest checkpoint selection exactly `lowest validation loss`, rejecting
  absent, changed, or test-based criteria; and
- checkpoint seed/feature/model metadata plus internally consistent history
  epoch sequence, best epoch, metrics epoch count, history best loss, metrics
  validation loss, selected history row loss, and independently recomputed
  validation prediction BCE.

Expected prediction anchors now come only from canonical constants, never from
manifest-provided window or stride values. Aggregate validation loss and the
full-run selection key use `corroborated_validation_loss`; ties still resolve
by ascending seed.

The synthetic fixture now contains the complete canonical manifest protocol,
configuration, fingerprint, features, splits, loadable checkpoint metadata,
history epochs, and predictions whose validation BCE equals the declared loss.

### Focused RED evidence

Command:

```bash
.venv-quant/bin/python -m unittest \
  tests.test_stage1.Stage1SummaryTests.test_summary_rejects_missing_or_altered_selection_provenance \
  tests.test_stage1.Stage1SummaryTests.test_summary_rejects_self_consistent_noncanonical_training_configuration \
  tests.test_stage1.Stage1SummaryTests.test_summary_rejects_noncanonical_dataset_sha \
  tests.test_stage1.Stage1SummaryTests.test_summary_rejects_uncorroborated_validation_selection_evidence \
  -v
```

Before production changes, result: `FAILED (failures=14)` in `8.769s`.
Every mutation was incorrectly accepted with return code `0`:

- missing protocol;
- test-based checkpoint selection;
- wrong feature order, split metadata, or prepared-dataset fingerprint;
- self-consistent window size `30` or train stride `2` streams;
- batch size `64` or requested epochs `299`;
- canonical CSV with an appended newline and therefore a different raw SHA;
- metrics/history validation-loss mismatches;
- best-epoch mismatch; and
- checkpoint seed mismatch.

The noncanonical window/stride fixtures deliberately rewrote their prediction
dates and manifest stream metadata consistently, proving the old summarizer
trusted the altered configuration rather than merely catching incidental date
misalignment.

### Focused GREEN evidence

The same four-method command after implementation produced:

```text
Ran 4 tests in 24.483s
OK
```

The complete Task 2 evaluator/summarizer classes were then run:

```bash
.venv-quant/bin/python -m unittest \
  tests.test_stage1.Stage1EvaluationTests \
  tests.test_stage1.Stage1SummaryTests \
  -v
```

Result:

```text
Ran 11 tests in 37.093s
OK
```

This includes the unchanged exact-test evaluator behavior, validation split
helper, evaluator immutability, exact artifact inventory, atomic output,
pilot non-selection, and full validation-loss/seed tie-break tests.

Full-suite command:

```bash
.venv-quant/bin/python -m unittest discover -s tests -v
```

Result:

```text
Ran 30 tests in 41.353s
OK
```

### Existing artifact compatibility

The strengthened summarizer was run against both frozen directories:

```bash
.venv-quant/bin/python scripts/summarize_stage1.py \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-pilot \
  --data datasets/nasdaq_multivariate.csv \
  --output runs/canonical_retrain/20260905T053626Z/stage1-pilot/summary.json

.venv-quant/bin/python scripts/summarize_stage1.py \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-full \
  --data datasets/nasdaq_multivariate.csv \
  --output runs/canonical_retrain/20260905T053626Z/stage1-full/summary.json
```

Both exited `0`. The regenerated summaries record all strengthened technical
checks as passed. The pilot still emits no selection; the full run still
selects seed `119` at corroborated validation loss `0.6920855045318604`; both
validation-only Stage 2 decisions remain `hold`.

Updated ignored summary SHA-256 values:

- pilot: `266f6b847b225d15259c43a13a0e0eb1b435861c1b83a820157f1e0b5b462225`;
- full: `3360c7d8de35336e34ae58fa7f1d04d346ce61d4379c1d14cbbd943f3e1f02ba`.

No training command was rerun because all existing immutable per-seed
checkpoint, history, metric, manifest, and prediction evidence satisfied the
strengthened contract. No result artifact, model, portfolio, README,
methodology file, notebook, or DQN file is included in this fix.

## Scoped re-review fix: complete trainer hyperparameter contract

The scoped re-review found that four trainer-recorded parameters were present
in valid manifests and synthetic fixtures but absent from
`CANONICAL_HYPERPARAMETERS`. The canonical contract now additionally requires:

- `gradient_clip_norm: 1.0`;
- `scheduler_factor: 0.5`;
- `scheduler_patience: 10`; and
- `scheduler_min_lr: 1e-5`.

The existing self-consistent noncanonical-configuration test was extended with
`scheduler_factor: 0.4`. This exercises the summarizer through its CLI and
asserts a nonzero exit, canonical-hyperparameter error, and no summary output.

### RED

Command:

```bash
.venv-quant/bin/python -m unittest \
  tests.test_stage1.Stage1SummaryTests.test_summary_rejects_self_consistent_noncanonical_training_configuration \
  -v
```

Output before the production change:

```text
test_summary_rejects_self_consistent_noncanonical_training_configuration ...
  (configuration={'scheduler_factor': 0.4}) ... FAIL
AssertionError: 0 == 0
Ran 1 test in 9.040s
FAILED (failures=1)
```

The pre-existing window, stride, batch-size, and epoch mutations continued to
pass their rejection assertions; only the newly covered scheduler mutation was
incorrectly accepted.

### GREEN

Focused regression command:

```bash
.venv-quant/bin/python -m unittest \
  tests.test_stage1.Stage1SummaryTests.test_summary_rejects_self_consistent_noncanonical_training_configuration \
  -v
```

Output:

```text
Ran 1 test in 9.187s
OK
```

Complete Task 2 evaluator/summarizer command:

```bash
.venv-quant/bin/python -m unittest \
  tests.test_stage1.Stage1EvaluationTests \
  tests.test_stage1.Stage1SummaryTests \
  -v
```

Output:

```text
Ran 11 tests in 39.619s
OK
```

Full-suite command:

```bash
.venv-quant/bin/python -m unittest discover -s tests -v
```

Output:

```text
Ran 30 tests in 44.617s
OK
```

### Frozen artifact re-verification

Commands:

```bash
.venv-quant/bin/python scripts/summarize_stage1.py \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-pilot \
  --data datasets/nasdaq_multivariate.csv \
  --output runs/canonical_retrain/20260905T053626Z/stage1-pilot/summary.json

.venv-quant/bin/python scripts/summarize_stage1.py \
  --run-dir runs/canonical_retrain/20260905T053626Z/stage1-full \
  --data datasets/nasdaq_multivariate.csv \
  --output runs/canonical_retrain/20260905T053626Z/stage1-full/summary.json
```

Both exited `0`. The ignored summary hashes remain
`266f6b847b225d15259c43a13a0e0eb1b435861c1b83a820157f1e0b5b462225`
for pilot and
`3360c7d8de35336e34ae58fa7f1d04d346ce61d4379c1d14cbbd943f3e1f02ba`
for full. No run artifact values were changed, no training was rerun, and no
run result is staged.
