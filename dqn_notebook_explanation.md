# DQN Trading Agent — Notebook Explanation

> Source: `source_code/stage2/dqn_252_colab.ipynb`

---

## Overview

This notebook implements a **Deep Q-Network (DQN) trading agent** for the NASDAQ index. It is the second stage of a two-stage pipeline:

1. **Stage 1 (LSTM)** — A pre-trained `ReturnPredictor` LSTM predicts whether NASDAQ will be higher 20 days from now. Its rolling probability output becomes the primary signal fed into Stage 2.
2. **Stage 2 (DQN)** — A reinforcement learning agent trained inside a custom `gymnasium` trading environment uses that signal (and other features) to decide daily position sizes.

The LSTM signal is computed **rolling**: at each day `t`, the model uses the **last 60 days** of data to predict whether the price will be higher 20 days later (day 0 predicts day 20, day 1 predicts day 21, etc.).

---

## Required Files

| File | Purpose |
|------|---------|
| `datasets/nasdaq_multivariate.csv` | Raw NASDAQ price + macro dataset |
| `models/stage1_model.pth` | Pre-trained LSTM weights |
| `models/stage1_scaler.joblib` | Fitted `StandardScaler` for LSTM features |

---

## Section 1 — Install Dependencies (Cell 2)

```
!pip install stable-baselines3[extra] gymnasium joblib -q
```

Three packages are installed:
- `stable-baselines3[extra]` — provides the DQN implementation
- `gymnasium` — the RL environment interface
- `joblib` — used to load the saved `StandardScaler`

---

## Section 2 — Mount Google Drive (Cells 3–4)

The notebook is designed to run on **Google Colab**. All files live under `BASE_DIR = '/content/drive/MyDrive/'`.

### Path constants

| Variable | Path |
|----------|------|
| `CSV_PATH` | `…/datasets/nasdaq_multivariate.csv` |
| `SCALER_PATH` | `…/models/stage1_scaler.joblib` |
| `WEIGHTS_PATH` | `…/models/stage1_model.pth` |
| `DQN_SAVE` | `…/models/dqn_nasdaq_252.zip` |
| `CURVE_SAVE` | `…/results/dqn_252_reward_curve.png` |
| `EQUITY_SAVE` | `…/results/dqn_252_equity_curves.png` |

After mounting, the cell checks each of the three input paths and prints `[OK]` or `[MISSING]`. The run output shows all three as `[OK]`.

---

## Section 3 — Stage 1: LSTM Model Definition (Cells 5–6)

### Device

```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

The run output shows `Device: cuda`, confirming GPU execution.

### Feature columns

30 feature columns are defined in `FEATURE_COLS`:

```
Return, Return_lag1, Return_lag2, Return_lag5,
Price_SMA20, Price_SMA50,
RSI_14, MACD_Hist, BB_pos, BB_width,
Volume_ratio, Volume_change,
VIX, TNX, FedRate, CPI_MoM,
Regime, VIX_percentile,
Momentum_20d, Momentum_60d, Momentum_252d,
Volatility_20d, ATR_14,
Yield_slope, Dist_52w_high, Dist_52w_low,
VIX_change5, Gap,
FedRate_chg20, FedRate_chg60
```

7 macro-only columns are in `MACRO_COLS`:

```
FedRate, FedRate_chg20, FedRate_chg60, TNX, Yield_slope, VIX, VIX_percentile
```

### Hyperparameters (LSTM)

| Constant | Value |
|----------|-------|
| `WINDOW_SIZE` | 60 days |
| `HIDDEN_SIZE` | 32 |
| `DROPOUT` | 0.3 |

### `ReturnPredictor` architecture

```python
class ReturnPredictor(nn.Module):
    def __init__(self, n_features, n_macro, hidden=32, dropout=0.3):
        self.lstm      = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.macro_mlp = nn.Sequential(nn.Linear(n_macro, 16), nn.ReLU())
        self.head      = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + 16, 1))
```

- The LSTM processes a 60-step sequence of the 30 features and outputs a hidden state of size 32.
- The macro MLP takes the 7 macro values at the current timestep and maps them to a 16-dim vector.
- The head concatenates `[h_last (32) | macro_out (16)]` → dropout → linear → scalar logit.
- A sigmoid is applied at inference time to produce a probability.

### `load_and_engineer_features`

Reads the CSV and engineers additional columns that are not in the raw file:

| Column | Formula |
|--------|---------|
| `Forward_20d` | `Close.shift(-20) / Close - 1` |
| `target` | 1 if `Forward_20d > median(Forward_20d)` else 0 |
| `Regime` | 1 if `Close > Close.rolling(60).mean()` |
| `Momentum_20d` | `Close / Close.shift(20) - 1` |
| `Momentum_60d` | `Close / Close.shift(60) - 1` |
| `Momentum_252d` | `Close / Close.shift(252) - 1` |
| `Dist_52w_high` | `Close / Close.rolling(252).max() - 1` |
| `Dist_52w_low` | `Close / Close.rolling(252).min() - 1` |
| `Volatility_20d` | `Return.rolling(20).std()` |
| `ATR_14` | `(High - Low).rolling(14).mean() / Close` |
| `VIX_percentile` | `VIX.rolling(252).rank(pct=True)` |
| `VIX_change5` | `VIX.diff(5)` |
| `Yield_slope` | `TNX - FedRate` |
| `FedRate_chg20` | `FedRate.diff(20)` |
| `FedRate_chg60` | `FedRate.diff(60)` |
| `Gap` | `Open / Close.shift(1) - 1` |

After dropping NaN rows the dataset has **8,848 rows** with a target positive rate of **0.505** (nearly balanced binary classification).

### `load_lstm_model`

Loads weights from disk with `torch.load(..., map_location='cpu')`, applies `load_state_dict`, and sets the model to `eval()` mode.

---

## Section 4 — LSTM Inference: Rolling Predictions (Cells 7–8)

### `generate_prob_series`

This function runs inference over the entire DataFrame and returns three aligned arrays.

**Step-by-step logic:**

1. Scale all 30 features with the pre-fitted `StandardScaler`.
2. For each index `i` from `window` (60) to `len(df)`, construct a sequence `scaled[i-60 : i]` and feed it to the LSTM along with `macro_raw[i-1]`.
3. Apply `torch.sigmoid` to the logit to get a raw probability.
4. Collect all raw probabilities into `raw_probs`.

**Signal normalization:**

Raw probabilities are concentrated (observed: mean=0.643, std=0.120 for training; mean=0.673, std=0.123 for test). To spread the signal, z-scoring followed by tanh is applied:

```python
z          = (raw_probs - raw_probs.mean()) / (raw_probs.std() + 1e-8)
probs_norm = (0.5 + 0.5 * np.tanh(z)).astype(np.float32)
```

Result: normalized probabilities near mean=0.486, std=0.319 (training) and mean=0.486, std=0.328 (test).

**Extra features computed:**

| Feature | Computation |
|---------|-------------|
| `recent_5d_norm` | 5-day return tanh-normalized: `0.5 + 0.5 * tanh(ret5 / 0.05)` |
| `vol_norm` | `clip(Volatility_20d / 0.03, 0.0, 1.0)` |
| `regime` | `Regime` column cast to float32 |

These three are stacked into `extra_features` of shape `(N, 3)`.

**Returns:**
- `probs_norm`: shape `(N,)` — normalized LSTM probabilities in `[0, 1]`
- `prices`: shape `(N,)` — Close prices aligned to `probs_norm`
- `extra_features`: shape `(N, 3)`

---

## Section 5 — Trading Environment (Cells 9–10)

### Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `EPISODE_LEN` | 252 | Steps per episode (~1 trading year) |
| `SIGNAL_BONUS_W` | 20 | Weight of the signal alignment bonus in reward |
| `VOL_PENALTY_W` | 0 | Weight of the volatility penalty (disabled) |
| `MARGIN_RATE` | `0.02 / 252` | Daily interest rate charged on negative cash |
| `N_ACTIONS` | 31 | Total number of discrete actions |

### Action space

31 discrete actions mapped via `ACTION_MAP = {i: (i - 10) * 0.1 for i in range(31)}`:

- Action 0 → position −1.0 (100% short)
- Action 10 → position 0.0 (all cash)
- Action 20 → position +1.0 (fully invested)
- Action 30 → position +2.0 (2× leveraged long)

### `NASDAQTradingEnv`

Inherits from `gymnasium.Env`.

**Constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prices` | — | Close price array |
| `probs` | — | Normalized LSTM probability array |
| `extra_features` | — | Array of shape `(N, 3)` |
| `initial_cash` | 10,000 | Starting portfolio value in dollars |
| `episode_len` | 252 | Steps per episode |
| `signal_bonus_w` | 20 | Signal bonus weight |
| `vol_penalty_w` | 0 | Volatility penalty weight |

**Observation space:** 6-dimensional `Box`:

| Index | Feature | Range |
|-------|---------|-------|
| 0 | `probs[idx]` (LSTM signal) | [0, 1] |
| 1 | `extra_features[idx, 0]` (5-day return norm) | [0, 1] |
| 2 | `extra_features[idx, 1]` (vol norm) | [0, 1] |
| 3 | `pos_ratio` = shares×price / total | [−1, 2] |
| 4 | `cash_ratio` = cash / total | [−1, 2] |
| 5 | `extra_features[idx, 2]` (regime) | [0, 1] |

**`reset`:** Picks a random start index from `[0, len(prices) - episode_len - 1]`, resets cash to `initial_cash`, and sets shares to 0.

**`step(action)` logic:**

1. Map `action` → `target_pos` via `ACTION_MAP`.
2. Compute `current_pos = shares × price / total`.
3. Compute `delta = target_pos - current_pos`, then `shares_delta = delta × total / price`.
4. Adjust `_shares` and `_cash` accordingly (no transaction costs).
5. Advance `_step_count` by 1 and read the next price.
6. If `_cash < 0`, charge margin: `_cash -= abs(_cash) × MARGIN_RATE`.
7. Compute reward:

```
portfolio_return = daily_return × pos_ratio_new
reward_base      = portfolio_return × 100

desired          = 2.0 × signal_t − 1.0          # maps [0,1] → [−1, +1]
alignment        = (1 + pos_ratio_new × desired) / 2
vol_penalty      = vol_t × |pos_ratio_new| × |daily_return| × vol_penalty_w   # = 0
bonus            = alignment × |daily_return| × signal_bonus_w

reward           = reward_base + bonus − vol_penalty
```

8. Episode terminates when `_step_count >= episode_len` (252 steps).

---

## Section 6 — Load Data & Generate LSTM Signals (Cells 11–12)

### Global seed

```python
set_seeds(42)  # sets random, numpy, and torch seeds
```

### Training data split

```python
mask = (
    (df['Date'] >= '1990-01-01') &
    (df['Date'] <  '2019-01-01') &
    ~((df['Date'] >= '2002-01-01') & (df['Date'] < '2003-01-01')) &
    ~((df['Date'] >= '2008-01-01') & (df['Date'] < '2009-01-01'))
)
```

**Date range:** 1990–2019, with two crisis years excluded:
- 2002 (dot-com crash)
- 2008 (Global Financial Crisis)

**Result:** 6,551 rows for RL training.

### LSTM signal generation (training)

After passing through `generate_prob_series` with window=60:
- Output shape: **6,491 steps** (6,551 − 60)
- Raw prob: mean=0.643, std=0.120
- Normalized prob: mean=0.486, std=0.319
- Valid start positions for 252-day episodes: **6,238**

---

## Section 7 — Train DQN (Cells 13–15)

### DQN Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| `TOTAL_TIMESTEPS` | 1,000,000 |
| `BUFFER_SIZE` | 100,000 |
| `LEARNING_STARTS` | 5,000 |
| `BATCH_SIZE` | 128 |
| `LEARNING_RATE` | 1e-4 |
| `GAMMA` | 0.97 |
| `TARGET_UPDATE_INTERVAL` | 1,000 |
| `EXPLORATION_FRACTION` | 0.3 |
| `EXPLORATION_FINAL_EPS` | 0.05 |
| `TRAIN_FREQ` | 4 |
| `GRADIENT_STEPS` | 1 |
| `NET_ARCH` | [256, 256] |
| `SEED` | 42 |
| `EPISODE_LEN` | 252 |
| `Policy` | MlpPolicy |


The Q-network is a 2-layer MLP with 256 units per hidden layer.

Exploration follows an epsilon-greedy schedule: epsilon starts near 1.0 and decays linearly over the first 30% of timesteps (300,000 steps) to a final value of 0.05, then stays there.

### `RewardCallback`

Accumulates per-step reward during each episode and appends the episode total to `ep_rewards` when `dones[0]` is True. This enables the reward curve plot.

### Environment validation

`check_env(env, warn=True)` from `stable_baselines3.common.env_checker` is called before training and passes without errors.

### Training output (first log entries)

At timestep 1,008 (4 episodes): `ep_rew_mean=42.7`, `exploration_rate=0.997`  
At timestep 5,040 (20 episodes): `ep_rew_mean=28.2`, initial `loss=0.428`  
All episodes have `ep_len_mean=252` (consistent with `EPISODE_LEN`).

### Training reward curve (Cell 15)

A smoothing window of `max(1, len(rewards) // 20)` is applied with `np.convolve`. The plot shows raw episode rewards (semi-transparent) and a smoothed overlay. Saved to `CURVE_SAVE`.

### Model saving

```python
dqn.save(DQN_SAVE)  # saves to …/models/dqn_nasdaq_252.zip
```

---

## Section 8 — Evaluate on 2019–2026 Out-of-Sample (Cells 16–20)

### Test split

The test set is the last **20%** of the full dataset by row index (`t2 = int(n * 0.80)`):
- **1,770 rows**, from **2019-02-04 to 2026-02-18**

After the 60-step LSTM window: **1,710 test steps**.  
Raw prob (test): mean=0.673, std=0.123  
Normalized prob (test): mean=0.486, std=0.328

### Evaluation helper functions

#### `compute_metrics(portfolio_values, label)`

Computes 5 metrics from a portfolio value array:

| Metric | Formula |
|--------|---------|
| Total Return | `(last − first) / first` |
| Sharpe Ratio | `mean(daily_ret) / std(daily_ret) × √252` |
| Sortino Ratio | `mean(daily_ret) / std(negative daily_ret) × √252` |
| Max Drawdown | `max((peak − value) / peak)` over time |
| Win Rate | fraction of days with positive daily return |

#### `run_buy_and_hold(prices, initial_cash)`

Buys at day 0 and holds: `(initial_cash / prices[0]) × prices`.

#### `run_always_cash(prices, initial_cash)`

Returns `np.full(len(prices), initial_cash)` — never invests.

#### `run_fixed_threshold(prices, probs, initial_cash)`

- If `prob > 0.6`: go fully long (pos = 1.0)
- If `prob < 0.4`: go to cash (pos = 0.0)
- Otherwise: hold current position

No transaction costs. Cash is not redeployable past what is available.

#### `run_dqn_eval(prices, probs, extra_features, dqn_model, initial_cash=10_000)`

Runs the DQN as a **single continuous episode** over the full test period (`episode_len = len(prices) - 1`). Uses `deterministic=True` prediction (no exploration). Records portfolio value after every step and the chosen action.

### Results table

```
Strategy                   Total Ret   Sharpe  Sortino    Max DD  Win Rate
--------------------------------------------------------------------------
DQN (ep=252d)               510.90%    0.954    1.087   35.65%   51.14%
Buy & Hold                  182.67%    0.751    0.971   36.40%   55.65%
Always Cash                   0.00%    0.000    0.000    0.00%    0.00%
Fixed Threshold             134.57%    0.699    0.643   34.97%   27.15%
```

The DQN achieves **510.90% total return** (vs. 182.67% for buy-and-hold) with a **Sharpe of 0.954** and **Sortino of 1.087** over the 2019–2026 out-of-sample period. Its max drawdown (35.65%) is slightly better than buy-and-hold (36.40%).

### Action distribution (out-of-sample)

| Action | Position | % of steps |
|--------|----------|-----------|
| 0 | −1.0 (fully short) | 0.1% |
| 1 | −0.9 | 7.8% |
| 2 | −0.8 | 11.8% |
| 3 | −0.7 | 0.1% |
| 5 | −0.5 | 0.9% |
| 6 | −0.4 | 4.8% |
| 7 | −0.3 | 8.5% |
| 9 | −0.1 | 2.8% |
| 10 | 0.0 (flat) | 6.0% |
| 11 | +0.1 | 0.2% |
| 12 | +0.2 | 0.3% |
| 13 | +0.3 | 1.0% |
| 14 | +0.4 | 1.1% |
| 15 | +0.5 | 11.9% |
| 16 | +0.6 | 0.1% |
| 17 | +0.7 | 2.0% |
| 18 | +0.8 | 0.4% |
| 19 | +0.9 | 0.1% |
| 20 | +1.0 (fully long) | 2.1% |
| 22 | +1.2 | 0.3% |
| 24 | +1.4 | 2.2% |
| 25 | +1.5 | 2.6% |
| 26 | +1.6 | 0.2% |
| 27 | +1.7 | 6.1% |
| 28 | +1.8 | 0.2% |
| 29 | +1.9 | 8.0% |
| 30 | +2.0 (2× leveraged) | 18.4% |

Actions 4, 8, 21, and 23 are never chosen. The agent heavily favors either **short positions** (−0.8 at 11.8%, −0.9 at 7.8%, −0.3 at 8.5%) or **high leveraged long positions** (+2.0 at 18.4%, +0.5 at 11.9%, +1.9 at 8.0%). Very few steps are taken at near-zero position, reflecting a bimodal "risk-on / risk-off" strategy.

### Equity curve plot (Cell 20)

Plots all four strategies over 1,710 trading days (x-axis: Trading Day, y-axis: Portfolio Value in $). Saved to `EQUITY_SAVE`. DQN is plotted in **darkorange**, buy-and-hold as dashed, always-cash as dotted, and fixed threshold as dash-dot.

---

## End-to-End Data Flow Summary

```
nasdaq_multivariate.csv
        │
        ▼
load_and_engineer_features()
  → 8,848 rows, 30+ engineered columns
        │
        ├─── Training split (1990–2019, excl. 2002, 2008) → 6,551 rows
        │         │
        │         ▼
        │   generate_prob_series()  [LSTM rolling inference, window=60]
        │     → probs (6491,), prices (6491,), extra (6491, 3)
        │         │
        │         ▼
        │   NASDAQTradingEnv  [gymnasium, 252-step episodes]
        │         │
        │         ▼
        │   DQN.learn(1_000_000 timesteps)
        │     → dqn_nasdaq_252.zip
        │
        └─── Test split (last 20%, 2019–2026) → 1,770 rows
                  │
                  ▼
            generate_prob_series()
              → probs (1710,), prices (1710,), extra (1710, 3)
                  │
                  ▼
            run_dqn_eval()  [deterministic, single 1710-step episode]
              → portfolio_values, actions
                  │
                  ▼
            compute_metrics()  → Total Ret 510.90%, Sharpe 0.954
```
