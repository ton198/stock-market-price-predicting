#!/usr/bin/env python3
"""Evaluate canonical DQN actions through the shared costed backtest."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.backtest import (
    buy_and_hold_positions,
    probability_positions,
    simulate_positions,
)
from quant_pipeline.data import PreparedDataset, prepare_dataset
from quant_pipeline.dqn_env import RESEARCH_PROTOCOL
from quant_pipeline.evaluator import align_test_predictions, load_prediction_csv
from quant_pipeline.metrics import strategy_metrics


CANONICAL_TRANSACTION_COST = 0.001
CANONICAL_SLIPPAGE = 0.0005


def manifest_diagnostics(manifest: dict[str, Any]) -> dict[str, Any]:
    """Keep learning diagnostics separate from investment-return metrics."""

    exports = manifest.get("exports", {})
    if not isinstance(exports, dict):
        exports = {}
    diagnostics: dict[str, Any] = {
        "environment_shaped_reward": {
            "validation_total": exports.get("validation_shaped_reward"),
            "test_total": exports.get("test_shaped_reward"),
            "usage": "learning diagnostics only; not reported investment return",
        },
        "validation_checkpoint_selection": manifest.get(
            "validation_checkpoint_selection", {}
        ),
    }
    if "reviewed_alternative" in manifest:
        diagnostics["reviewed_alternative"] = manifest["reviewed_alternative"]
    return diagnostics


def _validate_cost_assumptions(
    transaction_cost: float,
    slippage: float,
    diagnostics: dict[str, Any] | None,
) -> tuple[float, float]:
    values = (float(transaction_cost), float(slippage))
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("transaction_cost and slippage must be finite and non-negative")
    if values == (CANONICAL_TRANSACTION_COST, CANONICAL_SLIPPAGE):
        return values

    reviewed = (diagnostics or {}).get("reviewed_alternative")
    if not isinstance(reviewed, dict):
        raise ValueError("noncanonical costs require a reviewed alternative record")
    reference = reviewed.get("review_reference")
    reviewed_values = (
        reviewed.get("transaction_cost"),
        reviewed.get("slippage"),
    )
    if (
        reviewed.get("approved") is not True
        or not isinstance(reference, str)
        or not reference.strip()
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in reviewed_values
        )
        or not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in reviewed_values
        )
        or tuple(float(value) for value in reviewed_values) != values
    ):
        raise ValueError(
            "reviewed alternative must be approved, referenced, finite, and match costs"
        )
    return values


def load_action_csv(path: str | Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Load an exact ``Date,position`` action stream without clipping it."""

    action_path = Path(path)
    with action_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if "Date" not in fields or "position" not in fields:
            raise ValueError(f"{action_path} must contain Date and position columns")
        dates: list[str] = []
        positions: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            dates.append(row["Date"])
            try:
                positions.append(float(row["position"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"row {row_number}: position is not numeric") from exc
    return tuple(dates), np.asarray(positions, dtype=np.float64)


def evaluate_dqn_actions(
    dataset: PreparedDataset,
    action_dates: Sequence[str],
    positions: np.ndarray,
    stage1_dates: Sequence[str],
    stage1_probabilities: np.ndarray,
    *,
    initial_equity: float = 10_000.0,
    transaction_cost: float = CANONICAL_TRANSACTION_COST,
    slippage: float = CANONICAL_SLIPPAGE,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strictly align actions and report DQN plus canonical baselines."""

    transaction_cost, slippage = _validate_cost_assumptions(
        transaction_cost, slippage, diagnostics
    )
    expected_dates = tuple(dataset.dates[dataset.splits.test])
    provided_dates = tuple(str(value) for value in action_dates)
    if provided_dates != expected_dates:
        raise ValueError(
            "action dates must exactly match canonical test anchors "
            f"({len(expected_dates)} expected, {len(provided_dates)} provided)"
        )
    action_values = np.asarray(positions, dtype=np.float64)
    if action_values.ndim != 1 or action_values.size != len(expected_dates):
        raise ValueError("positions must have one value for every canonical test anchor")
    if not np.isfinite(action_values).all():
        raise ValueError("positions must be finite")
    if ((action_values < -1.0) | (action_values > 1.0)).any():
        raise ValueError("positions must lie in [-1, 1]; values are never silently clipped")
    _, prices, probabilities = align_test_predictions(
        dataset, stage1_dates, stage1_probabilities
    )

    strategy_positions = {
        "dqn": action_values,
        "raw_stage1_probability_policy": probability_positions(probabilities),
        "cash": np.zeros(prices.size, dtype=np.float64),
        "buy_and_hold": buy_and_hold_positions(prices.size),
    }
    strategies = {
        name: strategy_metrics(
            simulate_positions(
                prices,
                values,
                initial_equity=initial_equity,
                transaction_cost=transaction_cost,
                slippage=slippage,
                min_position=-1.0,
                max_position=1.0,
            )
        )
        for name, values in strategy_positions.items()
    }
    return {
        "status": "research_only",
        "portfolio_publication_allowed": False,
        "protocol": dict(RESEARCH_PROTOCOL),
        "test_start": expected_dates[0],
        "test_end": expected_dates[-1],
        "n_action_anchors": len(expected_dates),
        "n_backtest_intervals": len(expected_dates) - 1,
        "strategies": strategies,
        "backtest": {
            "initial_equity": float(initial_equity),
            "transaction_cost": float(transaction_cost),
            "slippage": float(slippage),
            "position_range": [-1.0, 1.0],
            "position_timing": "position[t] is held over price[t] to price[t+1]",
        },
        "diagnostics": dict(diagnostics or {}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--stage1-predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument(
        "--transaction-cost", type=float, default=CANONICAL_TRANSACTION_COST
    )
    parser.add_argument("--slippage", type=float, default=CANONICAL_SLIPPAGE)
    args = parser.parse_args(argv)

    diagnostics: dict[str, Any] = {}
    if args.manifest is not None:
        try:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read DQN manifest: {args.manifest}") from exc
        diagnostics = manifest_diagnostics(manifest)
    action_dates, positions = load_action_csv(args.actions)
    stage1_dates, probabilities = load_prediction_csv(args.stage1_predictions)
    report = evaluate_dqn_actions(
        prepare_dataset(args.data),
        action_dates,
        positions,
        stage1_dates,
        probabilities,
        initial_equity=args.initial_equity,
        transaction_cost=args.transaction_cost,
        slippage=args.slippage,
        diagnostics=diagnostics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
