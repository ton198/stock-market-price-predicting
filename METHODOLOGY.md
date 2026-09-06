# Reproducible evaluation contract

## Why the old numbers are retired

The historical report compared an ARIMA/vanilla-LSTM one-day return forecast
with a hybrid model trained on a 20-day label.  The samples, target definition,
and test windows were therefore different.  The hybrid label threshold was
also computed from all forward returns before the chronological split.  In the
RL notebook, probability z-scores were recomputed from each input subset,
including the test period; the backtest used unconstrained leverage and no
transaction costs or slippage.  These choices make the old accuracy and Sharpe
figures unsuitable for an apples-to-apples claim.

## Contract used by new experiments

`quant_pipeline.prepare_dataset` is the source of truth:

- It loads `datasets/nasdaq_multivariate.csv` and computes
  `Close[t + 20] / Close[t] - 1`.
- It removes only feature warm-up rows and the unlabelled tail, then splits the
  remaining labelled anchors in time order.
- It fits the median label threshold and standardization statistics on the
  training anchors only.
- It creates sequence windows ending at an anchor, allowing only past/current
  observations.

`quant_pipeline.signals` provides the equivalent train-fitted transform for a
model probability stream feeding an RL policy.  `quant_pipeline.backtest`
applies a position at time `t` over the next price interval and charges turnover
cost plus slippage.  `quant_pipeline.metrics` reports classification metrics
and annualized strategy metrics (total return, Sharpe, Sortino, maximum
drawdown, and win rate) without a hidden dependency on scikit-learn.

Before publishing a model number, record:

1. the commit and dataset hash;
2. target horizon, threshold, split dates, and feature list;
3. model seed(s) and hyperparameters;
4. the same labelled anchors for every model;
5. transaction cost, slippage, exposure limit, and baseline definitions; and
6. mean and dispersion across multiple seeds where a stochastic model is used.

## Reviewed canonical retraining evidence

Run `20260905T053626Z` uses dataset SHA-256
`3f8f3dcd571fceca215778333d6c97d45352bb2544ebc4c2d7467d79cf540ef7`, the
2019-02-01 through 2026-02-18 test anchors, transaction cost `0.001`,
slippage `0.0005`, and five predeclared DQN seeds with 1,000,000 timesteps
each. The selected Stage 1 seed is 119 by validation loss. The full result is
reproducible and reviewed, but remains `research_only`: the Stage 1
validation-only gate is `hold`, so its return is not a portfolio-ready claim.

The detailed aggregate and saved-artifact replay command are in
[`results/canonical/dqn/README.md`](results/canonical/dqn/README.md).
