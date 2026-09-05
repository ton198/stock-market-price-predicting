"""Canonical Gymnasium environment for the research-only DQN stage."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np


OBSERVATION_LOW = np.array([0, 0, 0, -1, -1, 0], dtype=np.float32)
OBSERVATION_HIGH = np.array([1, 1, 1, 1, 2, 1], dtype=np.float32)
RESEARCH_PROTOCOL = {
    "origin": "fixed",
    "split_basis": "prediction_anchor",
    "label_overlap_purged_at_boundaries": False,
    "exact_retraining_at_each_boundary": False,
    "qualification": (
        "Fixed-origin, anchor-based, unpurged offline research protocol; "
        "not an exact retraining-at-boundary simulation."
    ),
}


def _finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not positive and number < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return number


class DQNTradingEnv(gym.Env[np.ndarray, int]):
    """Trade a fixed price stream with the historical shaped learning reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        prices: np.ndarray,
        signal: np.ndarray,
        extra_features: np.ndarray,
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        episode_length: int = 252,
        signal_bonus_weight: float = 20.0,
        vol_penalty_weight: float = 0.0,
        margin_rate: float = 0.02 / 252,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        price_values = np.asarray(prices, dtype=np.float64)
        signal_values = np.asarray(signal, dtype=np.float32)
        extras = np.asarray(extra_features, dtype=np.float32)
        if price_values.ndim != 1 or price_values.size < 2:
            raise ValueError("prices must be a one-dimensional array with at least two rows")
        if not np.isfinite(price_values).all() or (price_values <= 0).any():
            raise ValueError("prices must be finite and positive")
        if signal_values.shape != price_values.shape:
            raise ValueError("signal must have one value per price")
        if extras.shape != (price_values.size, 3):
            raise ValueError("extra_features must have shape (len(prices), 3)")
        if not np.isfinite(signal_values).all() or not np.isfinite(extras).all():
            raise ValueError("signal and extra_features must be finite")
        if ((signal_values < 0.0) | (signal_values > 1.0)).any():
            raise ValueError("signal must lie in [0, 1]")
        initial_cash_value = _finite_scalar("initial_cash", initial_cash, positive=True)
        transaction_cost_value = _finite_scalar("transaction_cost", transaction_cost)
        slippage_value = _finite_scalar("slippage", slippage)
        signal_bonus_value = _finite_scalar(
            "signal_bonus_weight", signal_bonus_weight
        )
        vol_penalty_value = _finite_scalar(
            "vol_penalty_weight", vol_penalty_weight
        )
        margin_rate_value = _finite_scalar("margin_rate", margin_rate)
        if (
            isinstance(episode_length, bool)
            or not np.isscalar(episode_length)
            or not np.isfinite(episode_length)
            or float(episode_length) != int(episode_length)
        ):
            raise ValueError("episode_length must be a finite integer")
        episode_length_value = int(episode_length)
        if episode_length_value < 1 or episode_length_value >= price_values.size:
            raise ValueError("episode_length must be between 1 and len(prices) - 1")

        self.prices = price_values
        self.signal = signal_values
        self.extra_features = extras
        self.initial_cash = initial_cash_value
        self.transaction_cost = transaction_cost_value
        self.slippage = slippage_value
        self.episode_length = episode_length_value
        self.signal_bonus_weight = signal_bonus_value
        self.vol_penalty_weight = vol_penalty_value
        self.margin_rate = margin_rate_value
        self.seed_value = seed
        self.action_space = spaces.Discrete(21)
        self.observation_space = spaces.Box(
            low=OBSERVATION_LOW.copy(),
            high=OBSERVATION_HIGH.copy(),
            dtype=np.float32,
        )
        self._has_reset = False
        self._terminated = False
        self._start = 0
        self._step_count = 0
        self._cash = self.initial_cash
        self._shares = 0.0

    @property
    def max_start(self) -> int:
        """Last start that leaves exactly ``episode_length`` intervals."""

        return self.prices.size - self.episode_length - 1

    def _current_index(self) -> int:
        return self._start + self._step_count

    def _current_price(self) -> float:
        return float(self.prices[self._current_index()])

    def _portfolio_value(self) -> float:
        return self._cash + self._shares * self._current_price()

    def _get_observation(self) -> np.ndarray:
        index = self._current_index()
        price = self._current_price()
        total = self._portfolio_value()
        if total > 0.0:
            position_ratio = self._shares * price / total
            cash_ratio = self._cash / total
        else:
            position_ratio = 0.0
            cash_ratio = 1.0
        observation = np.array(
            [
                self.signal[index],
                self.extra_features[index, 0],
                self.extra_features[index, 1],
                position_ratio,
                cash_ratio,
                self.extra_features[index, 2],
            ],
            dtype=np.float32,
        )
        return np.clip(observation, OBSERVATION_LOW, OBSERVATION_HIGH).astype(
            np.float32, copy=False
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            super().reset(seed=seed)
        elif not self._has_reset:
            super().reset(seed=self.seed_value)
        else:
            super().reset(seed=None)
        fixed_start = None if options is None else options.get("start")
        if fixed_start is None:
            self._start = int(self.np_random.integers(0, self.max_start + 1))
        else:
            if isinstance(fixed_start, bool) or not isinstance(fixed_start, (int, np.integer)):
                raise ValueError("reset start must be an integer")
            if not 0 <= int(fixed_start) <= self.max_start:
                raise ValueError(f"reset start must lie in [0, {self.max_start}]")
            self._start = int(fixed_start)
        self._step_count = 0
        self._cash = self.initial_cash
        self._shares = 0.0
        self._terminated = False
        self._has_reset = True
        return self._get_observation(), {
            "start": self._start,
            "portfolio_value": self.initial_cash,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if self._terminated:
            raise RuntimeError("step cannot be called after the episode terminates")
        if isinstance(action, bool) or not self.action_space.contains(action):
            raise ValueError("action must be an integer in [0, 20]")

        action_index = int(action)
        target_position = action_index / 10.0 - 1.0
        previous_price = self._current_price()
        reward_index = self._current_index()
        total = self._portfolio_value()
        current_position = self._shares * previous_price / total if total > 0.0 else 0.0
        delta = target_position - current_position
        turnover = abs(delta)
        shares_delta = delta * total / previous_price
        self._shares += shares_delta
        self._cash -= shares_delta * previous_price
        trading_cost = total * turnover * (self.transaction_cost + self.slippage)
        self._cash -= trading_cost

        self._step_count += 1
        current_price = self._current_price()
        daily_return = current_price / previous_price - 1.0
        if self._cash < 0.0:
            self._cash -= abs(self._cash) * self.margin_rate

        total_new = self._portfolio_value()
        position_ratio_new = (
            self._shares * current_price / total_new if total_new > 0.0 else 0.0
        )
        reward_base = daily_return * position_ratio_new * 100.0
        desired = 2.0 * float(self.signal[reward_index]) - 1.0
        alignment = (1.0 + position_ratio_new * desired) / 2.0
        volatility_penalty = (
            float(self.extra_features[reward_index, 1])
            * abs(position_ratio_new)
            * abs(daily_return)
            * self.vol_penalty_weight
        )
        signal_bonus = (
            alignment * abs(daily_return) * self.signal_bonus_weight
        )
        normalized_trading_cost = trading_cost / max(total, 1e-8) * 100.0
        reward = (
            reward_base
            + signal_bonus
            - volatility_penalty
            - normalized_trading_cost
        )
        self._terminated = self._step_count >= self.episode_length
        info = {
            "start": self._start,
            "index": self._current_index(),
            "target_position": target_position,
            "turnover": turnover,
            "trading_cost": trading_cost,
            "portfolio_value": total_new,
            "reward_base": reward_base,
            "signal_bonus": signal_bonus,
            "volatility_penalty": volatility_penalty,
            "normalized_trading_cost": normalized_trading_cost,
        }
        return self._get_observation(), float(reward), self._terminated, False, info
