# NASDAQ forecasting and trading project

This repository contains a two-stage research project: a 20-trading-day
NASDAQ direction model followed by an RL trading policy.  The checked-in
notebooks are useful historical artifacts, but their old result cells were
generated with inconsistent targets and evaluation windows.  The canonical
data contract now lives in `quant_pipeline/` and must be used for any new
number reported in the paper, portfolio, or resume.

## Quick start

The data contract needs only Python 3.10+ and NumPy:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/audit_pipeline.py --output results/canonical_data_audit.json
```

The audit prints and saves the dataset SHA-256, exact date boundaries, training-fitted target
threshold, training-fitted feature statistics, and a training-prior reference
metric. It removes feature warm-up rows and the final 20 rows from the anchor
set because those rows do not have a complete feature/forward-outcome record.

Once a model produces a probability CSV with the canonical test dates, evaluate
it with identical classification and costed trading metrics:

```bash
python scripts/evaluate_predictions.py \
  --predictions results/hybrid_test_predictions.csv \
  --output results/hybrid_canonical_evaluation.json
```

The evaluator rejects missing, reordered, duplicated, or extra dates instead of
silently aligning them.

The optional ML stack is intentionally separate:

```bash
python -m pip install -r requirements-ml.txt
```

This is required only for retraining the PyTorch/ARIMA/DQN experiments; it is
not required to inspect or test the data contract.

## Canonical rules

1. The prediction target is the 20-day close-to-close forward return, with the
   30-feature Hybrid LSTM contract (16 checked-in features plus 14 causal
   rolling features) reproduced in `quant_pipeline.data`.
2. The binary threshold is the median of **training forward returns only**.
3. Train, validation, and test anchors are chronological, non-overlapping, and
   exhaustive (70% / 10% / 20% of labelled anchors by default).
4. Sequence windows may use historical context immediately before a split, but
   never a feature after the prediction anchor.
5. Every feature scaler and RL signal normalizer is fitted on the training
   period and then reused unchanged on validation/test periods.
6. Strategy positions are applied to the next price interval.  Turnover costs
   and slippage are explicit, and comparisons use the same exposure budget.

## Current evidence status

The existing `models/`, notebook output cells, report tables, and
`results/arima_v2_walk_forward_predictions.csv` are legacy artifacts.  They are
kept for provenance but should not be cited as a new out-of-sample result until
all models are retrained under the rules above.  In particular, the previous
hybrid accuracy comparison used different prediction targets, and the previous
DQN evaluation normalized probabilities with statistics from the evaluation
period and omitted trading frictions.

The canonical Stage 1 trainer and DQN evaluator have now been extracted and
audited. The resulting DQN evidence is deliberately marked research-only
because the Stage 1 validation gate is still `hold`; see
[`results/canonical/dqn/README.md`](results/canonical/dqn/README.md) for the
run identifier, seed aggregate, baselines, and protocol limitations.
