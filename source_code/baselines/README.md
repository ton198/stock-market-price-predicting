# Historical baselines

The ARIMA and vanilla-LSTM files in this directory are retained as provenance
for the original project. They use the retired one-day-return contract and
some scripts reference the former `datasets_aligned/NASDAQCOM.csv` layout,
which is not checked in anymore.

Do not use their saved output as a current model comparison. New baselines
must be evaluated on the same labelled anchors and with the same target,
split, and cost-aware backtest rules documented in the repository-level
[`METHODOLOGY.md`](../../METHODOLOGY.md). Use `quant_pipeline` and
[`scripts/evaluate_predictions.py`](../../scripts/evaluate_predictions.py) for
new prediction artifacts.
