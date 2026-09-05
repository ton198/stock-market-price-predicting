# Canonical DQN retraining evidence

This run is research evidence only. It is not a portfolio or resume claim.
The Stage 1 validation-only gate is `hold`, and the publication gate remains
closed; no portfolio numbers were changed.

## Run contract

- Run: `20260905T053626Z`
- Dataset SHA-256: `3f8f3dcd571fceca215778333d6c97d45352bb2544ebc4c2d7467d79cf540ef7`
- Selected Stage 1 seed: `119`, chosen by lowest validation loss (`0.6920855045318604`)
- DQN full seeds: `42, 59, 76, 93, 110`; `1,000,000` timesteps each
- DQN pilot seeds `42, 59, 76` used `100,000` timesteps and are excluded from the full aggregate
- Test anchors: `2019-02-01` through `2026-02-18`; 1,771 action anchors and 1,770 realized backtest intervals
- Transaction cost: `0.001`; slippage: `0.0005`; position range: `[-1, 1]`

The protocol is fixed-origin, anchor-based, and unpurged at split boundaries.
It is not an exact walk-forward or retraining-at-each-boundary simulation. The
DQN normalizer uses the selected Stage 1 dense training stream, which is
in-sample Stage 1 inference. The final exported action has no realized interval
or forced liquidation charge, and the environment's shaped reward is a
learning diagnostic rather than investment return.

## Full-seed test readout

Across the five predeclared full seeds, costed DQN total return was:

| Metric | Mean | Median | Population std. |
| --- | ---: | ---: | ---: |
| Total return | 92.20% | 91.90% | 31.28 pp |
| Sharpe | 0.6234 | 0.6488 | 0.1176 |
| Sortino | 0.7580 | 0.7672 | 0.1766 |
| Maximum drawdown | 24.53% | 24.72% | 4.57 pp |
| Win rate | 53.39% | 52.88% | 1.28 pp |

Per-seed DQN total returns were `136.25%` (42), `114.69%` (59), `91.90%`
(76), `47.82%` (93), and `70.36%` (110). The full aggregate includes every
planned seed; no seed was selected by test return.

For the same test interval and cost contract, the raw Stage 1 probability
policy returned `-5.16%` (Sharpe `-0.270`), cash returned `0%`, and buy-and-hold
returned `212.78%` (Sharpe `0.797`). These baselines are shared across all
full-seed reports.

## Reproduction artifacts

The generated evidence is kept outside Git under:

`runs/canonical_retrain/20260905T053626Z/dqn-full/`

The main files are `dqn-summary.json`, one `manifest.json` and
`evaluation.json` per seed, and `source_manifest.json`. The summary is
regenerated with:

```bash
.venv-quant/bin/python scripts/summarize_dqn.py \
  --data datasets/nasdaq_multivariate.csv \
  --stage1-summary runs/canonical_retrain/20260905T053626Z/stage1-full/summary.json \
  --run-dir runs/canonical_retrain/20260905T053626Z/dqn-full \
  --output runs/canonical_retrain/20260905T053626Z/dqn-full/dqn-summary.json
```

The technical gate passed, all saved evaluators were rerun from a fresh shell,
and the regenerated DQN summary matched the saved summary exactly. The final
repository test suite passed (`64` tests), and the canonical data audit matched
the frozen baseline.
