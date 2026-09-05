#!/usr/bin/env python3
"""Audit and deterministically summarize canonical DQN seed runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.backtest import simulate_positions
from quant_pipeline.data import PreparedDataset, prepare_dataset
from quant_pipeline.dqn_env import OBSERVATION_HIGH, OBSERVATION_LOW, RESEARCH_PROTOCOL
from scripts.evaluate_dqn import (
    CANONICAL_SLIPPAGE,
    CANONICAL_TRANSACTION_COST,
    evaluate_dqn_actions,
    load_action_csv,
    manifest_diagnostics,
)
from scripts.train_dqn import (
    DQN_HYPERPARAMETERS,
    ENVIRONMENT_DEFAULTS,
    STAGE1_SELECTION_CRITERION,
    VALIDATION_FREQUENCY,
    resolve_stage1_artifacts,
    validation_window_starts,
)
from quant_pipeline.evaluator import load_prediction_csv
from quant_pipeline.signals import SignalNormalizer, fit_signal_normalizer


PILOT_SEEDS = (42, 59, 76)
FULL_SEEDS = (42, 59, 76, 93, 110)
PILOT_TIMESTEPS = 100_000
FULL_TIMESTEPS = 1_000_000
SEED_DIRECTORY_PATTERN = re.compile(r"seed-(\d{3})\Z")
CHECKPOINT_CRITERION = (
    "highest mean costed total_return; earliest checkpoint on ties"
)
ACTION_FILENAMES = {
    "validation": "actions_validation.csv",
    "test": "actions_test.csv",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    _require_finite(value, str(path))
    return value


def _require_finite(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for name, nested in value.items():
            _require_finite(nested, f"{context}.{name}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite(nested, f"{context}[{index}]")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _discover_seed_directories(
    run_dir: Path,
) -> tuple[bool, tuple[int, ...], list[tuple[int, Path]]]:
    discovered: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        match = SEED_DIRECTORY_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            discovered.append((int(match.group(1)), child))
    discovered.sort(key=lambda row: row[0])
    seeds = tuple(seed for seed, _ in discovered)
    if seeds == PILOT_SEEDS:
        return True, seeds, discovered
    if seeds == FULL_SEEDS:
        return False, seeds, discovered
    raise ValueError(
        "exact seed-NNN directories must contain either pilot seeds "
        f"{list(PILOT_SEEDS)} or full seeds {list(FULL_SEEDS)}; "
        f"discovered {list(seeds)}"
    )


def _same_path(left: object, right: Path) -> bool:
    return isinstance(left, str) and Path(left).resolve() == right.resolve()


def _expected_split_metadata(
    dataset: PreparedDataset, split: str, *, train_start: int = 59
) -> dict[str, Any]:
    row_slice = getattr(dataset.splits, split)
    start = int(row_slice.start or 0)
    stop = int(row_slice.stop or len(dataset.dates))
    if split == "train":
        start = max(start, train_start)
    dates = tuple(dataset.dates[start:stop])
    return {
        "rows": len(dates),
        "start_date": dates[0],
        "end_date": dates[-1],
    }


def _validate_checkpoint_records(
    seed_dir: Path, manifest: Mapping[str, Any], timesteps: int
) -> dict[str, Any]:
    checkpoint = manifest.get("validation_checkpoint_selection")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{seed_dir}: missing validation checkpoint records")
    if checkpoint.get("frequency_environment_steps") != VALIDATION_FREQUENCY:
        raise ValueError(f"{seed_dir}: invalid validation checkpoint frequency")
    if checkpoint.get("criterion") != CHECKPOINT_CRITERION:
        raise ValueError(f"{seed_dir}: invalid validation checkpoint criterion")
    expected_starts = validation_window_starts(884)
    if tuple(checkpoint.get("window_starts", ())) != expected_starts:
        raise ValueError(f"{seed_dir}: invalid validation checkpoint window starts")
    records = checkpoint.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{seed_dir}: validation checkpoint records must be a list")
    expected_steps = list(range(VALIDATION_FREQUENCY, timesteps + 1, VALIDATION_FREQUENCY))
    if [row.get("step") for row in records if isinstance(row, Mapping)] != expected_steps:
        raise ValueError(f"{seed_dir}: validation checkpoint steps are incomplete")
    if len(records) != len(expected_steps):
        raise ValueError(f"{seed_dir}: validation checkpoint record count is invalid")

    for row in records:
        if not isinstance(row, Mapping):
            raise ValueError(f"{seed_dir}: validation checkpoint record is not an object")
        windows = row.get("windows")
        if not isinstance(windows, list) or len(windows) != 3:
            raise ValueError(f"{seed_dir}: checkpoint must contain three windows")
        returns: list[float] = []
        for window, start in zip(windows, expected_starts):
            expected = {
                "start": start,
                "stop": start + 253,
                "n_anchors": 253,
                "n_intervals": 252,
            }
            if not isinstance(window, Mapping) or any(
                window.get(name) != value for name, value in expected.items()
            ):
                raise ValueError(f"{seed_dir}: invalid validation checkpoint window")
            try:
                returns.append(float(window["costed_total_return"]))
                float(window["shaped_reward"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{seed_dir}: malformed validation checkpoint window") from exc
        recorded_mean = float(row.get("mean_costed_total_return", math.nan))
        if not math.isclose(recorded_mean, float(np.mean(returns)), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{seed_dir}: validation checkpoint mean is inconsistent")

    selected = max(
        records,
        key=lambda row: (
            float(row["mean_costed_total_return"]),
            -int(row["step"]),
        ),
    )
    if (
        checkpoint.get("selected_step") != selected["step"]
        or not math.isclose(
            float(checkpoint.get("selected_mean_costed_total_return", math.nan)),
            float(selected["mean_costed_total_return"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{seed_dir}: selected checkpoint does not match records")
    return dict(checkpoint)


def _load_and_validate_actions(
    dataset: PreparedDataset,
    seed_dir: Path,
    split: str,
    *,
    transaction_cost: float,
    slippage: float,
) -> tuple[tuple[str, ...], np.ndarray]:
    path = seed_dir / ACTION_FILENAMES[split]
    dates, positions = load_action_csv(path)
    expected_dates = tuple(dataset.dates[getattr(dataset.splits, split)])
    if dates != expected_dates:
        raise ValueError(f"{path}: action dates do not exactly match {split} anchors")
    if positions.ndim != 1 or positions.size != len(expected_dates):
        raise ValueError(f"{path}: one action is required per {split} anchor")
    if not np.isfinite(positions).all():
        raise ValueError(f"{path}: actions must be finite")
    if ((positions < -1.0) | (positions > 1.0)).any():
        raise ValueError(f"{path}: actions must lie in [-1, 1]")
    prices = np.asarray(dataset.close[getattr(dataset.splits, split)], dtype=np.float64)
    equity = simulate_positions(
        prices,
        positions,
        initial_equity=10_000.0,
        transaction_cost=transaction_cost,
        slippage=slippage,
        min_position=-1.0,
        max_position=1.0,
    )
    if not np.isfinite(equity).all() or (equity <= 0.0).any():
        raise ValueError(f"{path}: shared-backtest equity must remain finite and positive")
    return dates, positions


def _validate_seed_manifest(
    dataset: PreparedDataset,
    seed: int,
    seed_dir: Path,
    stage1_summary_path: Path,
    selected_stage1: Path,
    stage1_selection: Mapping[str, Any],
    expected_signal_normalizer: SignalNormalizer,
    *,
    pilot: bool,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    required = {
        "manifest.json",
        "model.zip",
        "actions_validation.csv",
        "actions_test.csv",
    }
    missing = sorted(name for name in required if not (seed_dir / name).is_file())
    if missing or not (seed_dir / "checkpoints").is_dir():
        raise ValueError(f"{seed_dir}: missing required DQN artifacts: {missing}")
    manifest = _read_json(seed_dir / "manifest.json")
    expected_timesteps = PILOT_TIMESTEPS if pilot else FULL_TIMESTEPS
    if manifest.get("seed") != seed:
        raise ValueError(f"{seed_dir}: manifest seed does not match directory")
    if manifest.get("pilot") is not pilot:
        raise ValueError(f"{seed_dir}: pilot marker does not match run kind")
    if manifest.get("timesteps") != expected_timesteps:
        raise ValueError(f"{seed_dir}: timesteps must be exactly {expected_timesteps}")
    if (
        manifest.get("status") != "research_only"
        or manifest.get("portfolio_publication_allowed") is not False
        or manifest.get("protocol") != RESEARCH_PROTOCOL
    ):
        raise ValueError(f"{seed_dir}: research protocol metadata mismatch")
    stage1 = manifest.get("stage1")
    if (
        not isinstance(stage1, Mapping)
        or not _same_path(stage1.get("requested_path"), stage1_summary_path)
        or not _same_path(stage1.get("selected_artifacts"), selected_stage1)
        or stage1.get("selection") != stage1_selection
    ):
        raise ValueError(f"{seed_dir}: Stage 1 summary selection provenance mismatch")
    normalizer = manifest.get("signal_normalizer")
    if (
        not isinstance(normalizer, Mapping)
        or normalizer.get("fit_split") != "train_dense"
        or normalizer.get("fit_filename") != "predictions_train_dense.csv"
        or normalizer.get("rows") != 6133
        or not math.isfinite(float(normalizer.get("mean", math.nan)))
        or not math.isfinite(float(normalizer.get("scale", math.nan)))
        or float(normalizer.get("scale", 0.0)) <= 0.0
    ):
        raise ValueError(f"{seed_dir}: signal normalizer is not train-only metadata")
    try:
        actual_mean = float(normalizer["mean"])
        actual_scale = float(normalizer["scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{seed_dir}: signal normalizer statistics are malformed") from exc
    if not (
        math.isclose(
            actual_mean,
            float(expected_signal_normalizer.mean_),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            actual_scale,
            float(expected_signal_normalizer.scale_),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            f"{seed_dir}: signal normalizer statistics do not match the dense train stream"
        )
    if manifest.get("input_streams") != {
        "train": "predictions_train_dense.csv",
        "validation": "predictions_validation.csv",
        "test": "predictions_test.csv",
    }:
        raise ValueError(f"{seed_dir}: input stream metadata mismatch")
    if manifest.get("observation_bounds") != {
        "low": OBSERVATION_LOW.tolist(),
        "high": OBSERVATION_HIGH.tolist(),
    }:
        raise ValueError(f"{seed_dir}: observation bounds mismatch")
    expected_environment = {
        **ENVIRONMENT_DEFAULTS,
        "signal_bonus_weight": 20.0,
        "action_count": 21,
        "action_mapping": "action_index / 10 - 1",
        "reward_usage": "learning_only",
    }
    if manifest.get("environment") != expected_environment:
        raise ValueError(f"{seed_dir}: exact cost/slippage environment mismatch")
    if manifest.get("dqn_hyperparameters") != DQN_HYPERPARAMETERS:
        raise ValueError(f"{seed_dir}: canonical DQN hyperparameters mismatch")
    expected_alignment = {
        split: _expected_split_metadata(dataset, split)
        for split in ("train", "validation", "test")
    }
    if manifest.get("split_alignment") != expected_alignment:
        raise ValueError(f"{seed_dir}: split alignment metadata mismatch")
    if manifest.get("test_contract") != {
        "action_anchors": 1771,
        "realized_intervals": 1770,
    }:
        raise ValueError(f"{seed_dir}: test 1771-anchor/1770-interval contract mismatch")
    exports = manifest.get("exports")
    if not isinstance(exports, Mapping) or any(
        exports.get(name) != value
        for name, value in {
            "model": "model.zip",
            "validation_actions": "actions_validation.csv",
            "test_actions": "actions_test.csv",
        }.items()
    ):
        raise ValueError(f"{seed_dir}: export inventory mismatch")
    _require_finite(exports, f"{seed_dir}.exports")
    checkpoint = _validate_checkpoint_records(seed_dir, manifest, expected_timesteps)
    wall_clock = manifest.get("wall_clock_seconds")
    device = manifest.get("device")
    if (
        isinstance(wall_clock, bool)
        or not isinstance(wall_clock, (int, float))
        or not math.isfinite(float(wall_clock))
        or float(wall_clock) <= 0.0
        or not isinstance(device, Mapping)
        or not isinstance(device.get("requested"), str)
        or not isinstance(device.get("resolved"), str)
    ):
        raise ValueError(f"{seed_dir}: device/wall-clock metadata is invalid")
    _load_and_validate_actions(
        dataset,
        seed_dir,
        "validation",
        transaction_cost=CANONICAL_TRANSACTION_COST,
        slippage=CANONICAL_SLIPPAGE,
    )
    _, test_positions = _load_and_validate_actions(
        dataset,
        seed_dir,
        "test",
        transaction_cost=CANONICAL_TRANSACTION_COST,
        slippage=CANONICAL_SLIPPAGE,
    )
    return manifest, checkpoint, test_positions


def _statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate metrics must be a finite non-empty sequence")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
    }


def _aggregate_strategy_metrics(
    per_seed: Sequence[Mapping[str, Any]], strategy: str
) -> dict[str, dict[str, float]]:
    metrics = tuple(per_seed[0]["strategies"][strategy])
    return {
        metric: _statistics(
            [float(row["strategies"][strategy][metric]) for row in per_seed]
        )
        for metric in metrics
    }


def _instability_note(total_returns: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(total_returns, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    sign_disagreement = bool((values < 0.0).any() and (values > 0.0).any())
    detected = bool(sign_disagreement or std > abs(mean))
    note = (
        "Across five predeclared seeds, DQN total returns range from "
        f"{float(values.min()):.6g} to {float(values.max()):.6g} "
        f"(mean {mean:.6g}, population std {std:.6g}); return signs "
        f"{'disagree' if sign_disagreement else 'do not disagree'}. "
        "Instability is flagged when signs disagree or population std exceeds "
        f"the absolute mean: {'detected' if detected else 'not detected'}."
    )
    return {
        "detected": detected,
        "rule": "sign disagreement or population std(total_return) > abs(mean(total_return))",
        "sign_disagreement": sign_disagreement,
        "total_return_min": float(values.min()),
        "total_return_max": float(values.max()),
        "note": note,
    }


def summarize_dqn_run(
    data_path: str | Path,
    stage1_summary_path: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Audit one exact pilot/full run and return its deterministic summary."""

    data = Path(data_path)
    stage1_summary_file = Path(stage1_summary_path)
    directory = Path(run_dir)
    if stage1_summary_file.name != "summary.json" or not stage1_summary_file.is_file():
        raise ValueError("--stage1-summary must be the full Stage 1 summary.json path")
    stage1_summary = _read_json(stage1_summary_file)
    gate = stage1_summary.get("stage2_gate")
    if not isinstance(gate, Mapping) or gate.get("decision") != "hold":
        raise ValueError("Task 4 requires the recorded Stage 1 hold gate")
    selected_stage1, selection = resolve_stage1_artifacts(stage1_summary_file)
    if selection.get("criterion") != STAGE1_SELECTION_CRITERION:
        raise ValueError("Stage 1 selection criterion is not canonical")
    dataset = prepare_dataset(data)
    expected_test_dates = tuple(dataset.dates[dataset.splits.test])
    if len(expected_test_dates) != 1771:
        raise ValueError("canonical test split must contain 1771 anchors")
    stage1_dates, stage1_probabilities = load_prediction_csv(
        selected_stage1 / "predictions_test.csv"
    )
    if stage1_dates != expected_test_dates:
        raise ValueError("selected Stage 1 test probabilities are misaligned")
    dense_train_dates, dense_train_probabilities = load_prediction_csv(
        selected_stage1 / "predictions_train_dense.csv"
    )
    expected_train_dates = tuple(dataset.dates[59 : dataset.splits.train.stop])
    if dense_train_dates != expected_train_dates:
        raise ValueError("selected Stage 1 dense train probabilities are misaligned")
    expected_signal_normalizer = fit_signal_normalizer(dense_train_probabilities)

    pilot, seeds, seed_directories = _discover_seed_directories(directory)
    per_seed: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for seed, seed_dir in seed_directories:
        manifest, checkpoint, positions = _validate_seed_manifest(
            dataset,
            seed,
            seed_dir,
            stage1_summary_file,
            selected_stage1,
            selection,
            expected_signal_normalizer,
            pilot=pilot,
        )
        relative_manifest = f"seed-{seed:03d}/manifest.json"
        seed_row: dict[str, Any] = {
            "seed": seed,
            "timesteps": manifest["timesteps"],
            "manifest": relative_manifest,
            "device": dict(manifest["device"]),
            "wall_clock_seconds": float(manifest["wall_clock_seconds"]),
            "selected_checkpoint_step": checkpoint["selected_step"],
            "selected_validation_mean_costed_total_return": checkpoint[
                "selected_mean_costed_total_return"
            ],
        }
        checkpoint_rows.append(
            {
                "seed": seed,
                "manifest": relative_manifest,
                "frequency_environment_steps": checkpoint[
                    "frequency_environment_steps"
                ],
                "criterion": checkpoint["criterion"],
                "window_starts": checkpoint["window_starts"],
                "selected_step": checkpoint["selected_step"],
                "selected_mean_costed_total_return": checkpoint[
                    "selected_mean_costed_total_return"
                ],
                "records": checkpoint["records"],
            }
        )
        if not pilot:
            evaluation_path = seed_dir / "evaluation.json"
            if not evaluation_path.is_file():
                raise ValueError(
                    f"{evaluation_path}: run evaluate_dqn.py for every full seed before aggregation"
                )
            saved = _read_json(evaluation_path)
            reproduced = evaluate_dqn_actions(
                dataset,
                stage1_dates,
                positions,
                stage1_dates,
                stage1_probabilities,
                initial_equity=10_000.0,
                transaction_cost=CANONICAL_TRANSACTION_COST,
                slippage=CANONICAL_SLIPPAGE,
                diagnostics=manifest_diagnostics(manifest),
            )
            if saved != reproduced:
                raise ValueError(f"{evaluation_path} does not match shared evaluator output")
            seed_row["evaluation"] = f"seed-{seed:03d}/evaluation.json"
            seed_row["strategies"] = saved["strategies"]
            evaluations.append(saved)
        per_seed.append(seed_row)

    summary: dict[str, Any] = {
        "status": "research_only",
        "portfolio_publication_allowed": False,
        "pilot": pilot,
        "run_kind": "pilot" if pilot else "full",
        "excluded_from_final_aggregate": pilot,
        "review": {
            "status": "pending",
            "readme_created": False,
            "readme_policy": "results/canonical/dqn/README.md is deferred until code/run review",
        },
        "aggregation_convention": {
            "std_ddof": 0,
            "weighting": "equal weight per predeclared seed; pilots excluded from full aggregate",
        },
        "seeds": list(seeds),
        "stage1": {
            "summary": str(stage1_summary_file),
            "selected_artifacts": str(selected_stage1),
            "selection": selection,
            "gate": dict(gate),
        },
        "test_interval": {
            "start": expected_test_dates[0],
            "end": expected_test_dates[-1],
            "action_anchors": 1771,
            "backtest_intervals": 1770,
            "position_timing": "position[t] is held over price[t] to price[t+1]",
        },
        "backtest": {
            "initial_equity": 10_000.0,
            "transaction_cost": CANONICAL_TRANSACTION_COST,
            "slippage": CANONICAL_SLIPPAGE,
            "position_range": [-1.0, 1.0],
            "turnover": "abs(position[t] - position[t-1]); initial previous position is cash (0)",
            "cost_application": "equity[t] * turnover[t] * (transaction_cost + slippage)",
            "final_anchor": "exported only; no realized interval, turnover charge, or forced liquidation",
        },
        "per_seed": per_seed,
        "validation_checkpoint_records": checkpoint_rows,
        "device_and_timing": {
            "resolved_devices": sorted(
                {str(row["device"]["resolved"]) for row in per_seed}
            ),
            "wall_clock_seconds": _statistics(
                [float(row["wall_clock_seconds"]) for row in per_seed]
            ),
            "total_wall_clock_seconds": float(
                sum(float(row["wall_clock_seconds"]) for row in per_seed)
            ),
        },
        "technical_gate": {
            "status": "pass",
            "checks": [
                "exact pilot/full seed set and timestep budget",
                "fresh output directories and explicit pilot markers",
                "full Stage 1 summary path and selected seed provenance",
                "train-only dense-stream signal normalizer metadata",
                "exact action/date alignment, finite bounds, and finite positive equity",
                "exact costs, slippage, protocol, and observation metadata",
                "complete deterministic validation checkpoint records",
            ],
        },
        "publication_gate": {
            "stage1_decision": "hold",
            "decision": "disabled",
            "reason": gate.get("reason"),
        },
    }
    if pilot:
        return summary

    baseline_names = (
        "raw_stage1_probability_policy",
        "cash",
        "buy_and_hold",
    )
    baselines = {
        name: evaluations[0]["strategies"][name] for name in baseline_names
    }
    for evaluation in evaluations[1:]:
        if any(evaluation["strategies"][name] != baselines[name] for name in baseline_names):
            raise ValueError("shared baseline metrics differ across full seeds")
    summary["aggregate"] = {
        "dqn": _aggregate_strategy_metrics(per_seed, "dqn")
    }
    summary["baselines"] = {
        "raw_stage1_probability_policy": {
            "definition": "position = clip(2 * raw Stage 1 probability - 1, -1, 1)",
            "shared_across_seeds": True,
            "metrics": baselines["raw_stage1_probability_policy"],
        },
        "cash": {
            "definition": "position = 0.0",
            "shared_across_seeds": True,
            "metrics": baselines["cash"],
        },
        "buy_and_hold": {
            "definition": "position = 1.0",
            "shared_across_seeds": True,
            "metrics": baselines["buy_and_hold"],
        },
    }
    summary["instability"] = _instability_note(
        [float(row["strategies"]["dqn"]["total_return"]) for row in per_seed]
    )
    summary["technical_gate"]["checks"].extend(
        [
            "every full seed evaluation reproduced through the shared evaluator",
            "raw Stage 1 probability, cash, and buy-and-hold baselines shared exactly",
            "pilot metrics excluded from all full aggregates",
        ]
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage1-summary", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize_dqn_run(args.data, args.stage1_summary, args.run_dir)
    _write_json_atomic(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
