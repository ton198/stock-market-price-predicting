"""Simple, explicit position-based backtesting helpers."""

from __future__ import annotations

import numpy as np


def simulate_positions(
    prices: np.ndarray,
    target_positions: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
    min_position: float = -1.0,
    max_position: float = 1.0,
) -> np.ndarray:
    """Simulate daily target positions with next-interval returns.

    ``target_positions[t]`` is selected using information available at price
    ``t`` and is held over ``prices[t] -> prices[t+1]``.  Turnover costs are
    charged when the target changes, including the initial move from cash.  A
    position is clipped to the declared exposure range so comparisons between
    strategies use the same leverage budget.

    The returned array has one equity value per price, including the initial
    value.  It is a return-level accounting model intended for transparent
    research comparisons, not an execution simulator.
    """

    prices = np.asarray(prices, dtype=np.float64)
    positions = np.asarray(target_positions, dtype=np.float64)
    if prices.ndim != 1 or positions.ndim != 1 or prices.shape != positions.shape:
        raise ValueError("prices and target_positions must be one-dimensional and equal length")
    if prices.size < 2:
        raise ValueError("at least two prices are required")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and positive")
    if not np.isfinite(positions).all():
        raise ValueError("target_positions must be finite")
    if initial_equity <= 0:
        raise ValueError("initial_equity must be positive")
    if transaction_cost < 0 or slippage < 0:
        raise ValueError("transaction_cost and slippage cannot be negative")
    if min_position > max_position:
        raise ValueError("min_position must not exceed max_position")

    positions = np.clip(positions, min_position, max_position)
    equity = np.empty(prices.size, dtype=np.float64)
    equity[0] = float(initial_equity)
    previous_position = 0.0
    turnover_rate = float(transaction_cost + slippage)
    for index in range(prices.size - 1):
        turnover = abs(positions[index] - previous_position)
        asset_return = prices[index + 1] / prices[index] - 1.0
        net_return = positions[index] * asset_return - turnover_rate * turnover
        equity[index + 1] = equity[index] * (1.0 + net_return)
        if equity[index + 1] <= 0:
            # Keep the series defined after a catastrophic leveraged loss.
            equity[index + 1 :] = 0.0
            break
        previous_position = positions[index]
    return equity


def buy_and_hold_positions(size: int) -> np.ndarray:
    """Return a fully invested, unlevered buy-and-hold position series."""

    if size < 2:
        raise ValueError("at least two positions are required")
    return np.ones(size, dtype=np.float64)


def probability_positions(probabilities: np.ndarray) -> np.ndarray:
    """Map a calibrated up-probability to a bounded long/short exposure."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be a finite one-dimensional array")
    return np.clip(2.0 * probabilities - 1.0, -1.0, 1.0)
