#!/usr/bin/env python3
"""Validate and summarize canonical Stage 1 seed runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.backtest import simulate_positions
from quant_pipeline.data import DEFAULT_FEATURE_COLS, PreparedDataset, prepare_dataset
from quant_pipeline.evaluator import (
    evaluate_probability_stream,
    evaluate_probability_stream_for_split,
    load_prediction_csv,
)
from quant_pipeline.metrics import classification_metrics, strategy_metrics


PILOT_SEEDS = (0, 17, 34)
FULL_SEEDS = (0, 17, 34, 51, 68, 85, 102, 119, 136, 153)
SEED_DIRECTORY_PATTERN = re.compile(r"seed-(\d{3})\Z")
PREDICTION_FILENAMES = {
    "train_fit": "predictions_train_fit.csv",
    "train_dense": "predictions_train_dense.csv",
    "validation": "predictions_validation.csv",
    "test": "predictions_test.csv",
}
TRAINING_ARTIFACTS = {
    "checkpoint.pt",
    "history.json",
    "metrics.json",
    "manifest.json",
    *PREDICTION_FILENAMES.values(),
}
TRAIN_DENSE_ROLE = "in-sample Stage 1 stream eligible for DQN"
POSITION_RULE = "clip(2 * probability - 1, -1, 1)"
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
CANONICAL_DATA_SHA256 = "3f8f3dcd571fceca215778333d6c97d45352bb2544ebc4c2d7467d79cf540ef7"
CANONICAL_DATASET_FINGERPRINT = (
    "f4e3fe6d621f82525c977c427e4f61deb96b420ab2c3b34361419f1cc63c4f02"
)
CANONICAL_HORIZON = 20
CANONICAL_FEATURE_COUNT = 30
CANONICAL_SPLITS = {
    "train": {"start": 0, "stop": 6192, "step": None},
    "validation": {"start": 6192, "stop": 7076, "step": None},
    "test": {"start": 7076, "stop": 8847, "step": None},
}
CANONICAL_PROTOCOL = {
    "origin": "fixed",
    "split_basis": "prediction_anchor",
    "label_overlap_purged_at_boundaries": False,
    "exact_retraining_at_each_boundary": False,
    "qualification": (
        "Fixed-origin, anchor-based, unpurged offline research contract; "
        "not an exact retraining-at-boundary simulation."
    ),
}
CANONICAL_HYPERPARAMETERS = {
    "epochs": 300,
    "batch_size": 32,
    "window_size": 60,
    "train_stride": 3,
    "learning_rate": 1e-3,
    "early_stopping_patience": 20,
    "gradient_clip_norm": 1.0,
    "scheduler_factor": 0.5,
    "scheduler_patience": 10,
    "scheduler_min_lr": 1e-5,
}
CHECKPOINT_SELECTION = "lowest validation loss"
# Saved prediction losses are float32 while history accumulation is float64.
# The observed maximum delta across the frozen pilot/full artifacts is 7.02e-8.
VALIDATION_LOSS_ABS_TOLERANCE = 1e-7


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def _write_json_atomic(path: Path, payload: object) -> None:
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


def _require_finite_json(value: Any, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context} contains NaN or infinity")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _require_finite_json(nested, f"{context}.{key}")
        return
    if isinstance(value, list):
        for index, nested in enumerate(value):
            _require_finite_json(nested, f"{context}[{index}]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice_metadata(row_slice: slice) -> dict[str, int | None]:
    return {"start": row_slice.start, "stop": row_slice.stop, "step": row_slice.step}


def _validate_canonical_dataset(dataset: PreparedDataset, data_path: Path) -> None:
    if _sha256(data_path) != CANONICAL_DATA_SHA256:
        raise ValueError(
            f"{data_path} does not match canonical dataset SHA-256 {CANONICAL_DATA_SHA256}"
        )
    if dataset.horizon != CANONICAL_HORIZON:
        raise ValueError(f"canonical horizon must be {CANONICAL_HORIZON}")
    if (
        dataset.features.ndim != 2
        or dataset.features.shape[1] != CANONICAL_FEATURE_COUNT
        or tuple(dataset.feature_names) != tuple(DEFAULT_FEATURE_COLS)
    ):
        raise ValueError("canonical feature count/order mismatch")
    split_metadata = {
        name: _slice_metadata(row_slice)
        for name, row_slice in dataset.splits.as_dict().items()
    }
    if split_metadata != CANONICAL_SPLITS:
        raise ValueError("canonical chronological split metadata mismatch")


def _discover_seed_directories(run_dir: Path) -> tuple[str, list[tuple[int, Path]]]:
    discovered: list[tuple[int, Path]] = []
    for child in run_dir.iterdir():
        match = SEED_DIRECTORY_PATTERN.fullmatch(child.name)
        if child.is_dir() and match:
            discovered.append((int(match.group(1)), child))
    discovered.sort(key=lambda item: item[0])
    seeds = tuple(seed for seed, _ in discovered)
    if seeds == PILOT_SEEDS:
        return "pilot", discovered
    if seeds == FULL_SEEDS:
        return "full", discovered
    raise ValueError(
        "exact seed-NNN directories must contain either the pilot seeds "
        f"{list(PILOT_SEEDS)} or full seeds {list(FULL_SEEDS)}; discovered {list(seeds)}"
    )


def _validate_prediction_artifacts(seed_dir: Path) -> None:
    actual = {path.name for path in seed_dir.glob("predictions_*.csv") if path.is_file()}
    expected = set(PREDICTION_FILENAMES.values())
    if actual != expected:
        raise ValueError(
            f"{seed_dir}: prediction artifacts must be exactly {sorted(expected)}; "
            f"found {sorted(actual)}"
        )
    missing = sorted(name for name in TRAINING_ARTIFACTS if not (seed_dir / name).is_file())
    if missing:
        raise ValueError(f"{seed_dir}: missing required Stage 1 artifacts: {missing}")


def _expected_stream_dates(
    dataset: PreparedDataset,
    stream: str,
) -> tuple[str, ...]:
    train_start = CANONICAL_HYPERPARAMETERS["window_size"] - 1
    if stream == "train_fit":
        indices = slice(
            train_start,
            CANONICAL_SPLITS["train"]["stop"],
            CANONICAL_HYPERPARAMETERS["train_stride"],
        )
    elif stream == "train_dense":
        indices = slice(train_start, CANONICAL_SPLITS["train"]["stop"])
    elif stream == "validation":
        indices = slice(
            CANONICAL_SPLITS["validation"]["start"],
            CANONICAL_SPLITS["validation"]["stop"],
        )
    elif stream == "test":
        indices = slice(
            CANONICAL_SPLITS["test"]["start"],
            CANONICAL_SPLITS["test"]["stop"],
        )
    else:
        raise ValueError(f"unknown prediction stream: {stream}")
    return tuple(dataset.dates[indices])


def _validate_manifest_and_streams(
    seed: int,
    seed_dir: Path,
    dataset: PreparedDataset,
    manifest: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    if manifest.get("seed") != seed:
        raise ValueError(f"{seed_dir}/manifest.json seed does not match directory")
    if set(manifest.get("artifacts", ())) != TRAINING_ARTIFACTS:
        raise ValueError(f"{seed_dir}/manifest.json artifact inventory is incomplete or extra")
    if manifest.get("checkpoint_selection") != CHECKPOINT_SELECTION:
        raise ValueError(
            f"{seed_dir}/manifest.json canonical selection evidence requires "
            f"checkpoint_selection={CHECKPOINT_SELECTION!r}"
        )
    if manifest.get("protocol") != CANONICAL_PROTOCOL:
        raise ValueError(f"{seed_dir}/manifest.json canonical protocol mismatch")
    if manifest.get("dataset_fingerprint_sha256") != CANONICAL_DATASET_FINGERPRINT:
        raise ValueError(f"{seed_dir}/manifest.json canonical dataset fingerprint mismatch")
    if tuple(manifest.get("feature_names", ())) != tuple(DEFAULT_FEATURE_COLS):
        raise ValueError(f"{seed_dir}/manifest.json canonical feature order mismatch")
    if manifest.get("splits") != CANONICAL_SPLITS:
        raise ValueError(f"{seed_dir}/manifest.json canonical split metadata mismatch")
    hyperparameters = manifest.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError(f"{seed_dir}/manifest.json lacks hyperparameters")
    invalid_hyperparameters = {
        name: {"expected": expected, "actual": hyperparameters.get(name)}
        for name, expected in CANONICAL_HYPERPARAMETERS.items()
        if hyperparameters.get(name) != expected
    }
    if invalid_hyperparameters:
        raise ValueError(
            f"{seed_dir}/manifest.json canonical hyperparameters mismatch: "
            f"{invalid_hyperparameters}"
        )
    streams = manifest.get("streams")
    if not isinstance(streams, Mapping) or set(streams) != set(PREDICTION_FILENAMES):
        raise ValueError(f"{seed_dir}/manifest.json lacks complete train-only stream metadata")
    dense_metadata = streams["train_dense"]
    if not isinstance(dense_metadata, Mapping) or dense_metadata.get("role") != TRAIN_DENSE_ROLE:
        raise ValueError(f"{seed_dir}/manifest.json lacks train-only dense-stream metadata")

    loaded: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for stream, filename in PREDICTION_FILENAMES.items():
        metadata = streams[stream]
        if not isinstance(metadata, Mapping):
            raise ValueError(f"{seed_dir}/manifest.json stream {stream} is not an object")
        dates, probabilities = load_prediction_csv(seed_dir / filename)
        expected_dates = _expected_stream_dates(dataset, stream)
        if dates != expected_dates:
            raise ValueError(f"{seed_dir}/{filename} does not align to expected {stream} anchors")
        if (
            probabilities.ndim != 1
            or probabilities.size != len(expected_dates)
            or not np.isfinite(probabilities).all()
            or ((probabilities < 0.0) | (probabilities > 1.0)).any()
        ):
            raise ValueError(f"{seed_dir}/{filename} probabilities must be finite and in [0, 1]")
        expected_cadence = (
            CANONICAL_HYPERPARAMETERS["train_stride"] if stream == "train_fit" else 1
        )
        expected_metadata = {
            "filename": filename,
            "rows": len(dates),
            "cadence": expected_cadence,
            "start_date": dates[0],
            "end_date": dates[-1],
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"{seed_dir}/manifest.json stream {stream} has invalid {key}"
                )
        loaded[stream] = (dates, probabilities)
    return loaded


def _close_validation_loss(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=VALIDATION_LOSS_ABS_TOLERANCE,
    )


def _validation_prediction_loss(
    dataset: PreparedDataset, probabilities: np.ndarray
) -> float:
    targets = np.asarray(dataset.targets[dataset.splits.validation], dtype=np.float64)
    values = np.asarray(probabilities, dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(targets * np.log(clipped) + (1 - targets) * np.log1p(-clipped))
    )


def _corroborate_selection_evidence(
    seed: int,
    seed_dir: Path,
    dataset: PreparedDataset,
    history: Mapping[str, Any],
    metrics: Mapping[str, Any],
    validation_probabilities: np.ndarray,
) -> float:
    try:
        checkpoint = torch.load(
            seed_dir / "checkpoint.pt", map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError(f"{seed_dir}: selection evidence checkpoint is unreadable") from exc
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{seed_dir}: selection evidence checkpoint must be a mapping")
    checkpoint_model = checkpoint.get("model")
    if (
        checkpoint.get("seed") != seed
        or tuple(checkpoint.get("feature_names", ())) != tuple(DEFAULT_FEATURE_COLS)
        or not isinstance(checkpoint_model, Mapping)
        or checkpoint_model.get("feature_count") != CANONICAL_FEATURE_COUNT
    ):
        raise ValueError(f"{seed_dir}: selection evidence checkpoint metadata mismatch")

    history_best_epoch = history.get("best_epoch")
    metrics_best_epoch = metrics.get("best_epoch")
    epochs = history.get("epochs")
    if (
        not isinstance(history_best_epoch, int)
        or isinstance(history_best_epoch, bool)
        or metrics_best_epoch != history_best_epoch
        or not isinstance(epochs, list)
        or not epochs
        or metrics.get("epochs_completed") != len(epochs)
    ):
        raise ValueError(f"{seed_dir}: selection evidence best epoch mismatch")
    try:
        epoch_rows = [
            (int(row["epoch"]), float(row["validation_loss"]))
            for row in epochs
            if isinstance(row, Mapping)
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{seed_dir}: selection evidence history is malformed") from exc
    if len(epoch_rows) != len(epochs) or [epoch for epoch, _ in epoch_rows] != list(
        range(1, len(epochs) + 1)
    ):
        raise ValueError(f"{seed_dir}: selection evidence history epoch sequence mismatch")
    selected_epoch, selected_epoch_loss = min(epoch_rows, key=lambda row: (row[1], row[0]))
    try:
        history_loss = float(history["best_validation_loss"])
        metrics_loss = float(metrics["validation_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{seed_dir}: selection evidence validation loss is missing") from exc
    prediction_loss = _validation_prediction_loss(dataset, validation_probabilities)
    if (
        selected_epoch != history_best_epoch
        or not _close_validation_loss(selected_epoch_loss, history_loss)
        or not _close_validation_loss(metrics_loss, history_loss)
        or not _close_validation_loss(metrics_loss, prediction_loss)
    ):
        raise ValueError(
            f"{seed_dir}: selection evidence validation loss mismatch "
            f"(metrics={metrics_loss}, history={history_loss}, "
            f"epoch={selected_epoch_loss}, predictions={prediction_loss}, "
            f"tolerance={VALIDATION_LOSS_ABS_TOLERANCE})"
        )
    return metrics_loss


def _write_or_verify_evaluation(path: Path, evaluation: Mapping[str, Any]) -> Mapping[str, Any]:
    if path.exists():
        saved = _read_json(path)
        if saved != evaluation:
            raise ValueError(f"{path} does not match shared evaluator output")
    else:
        _write_json_atomic(path, evaluation)
        saved = _read_json(path)
    if not isinstance(saved, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return saved


def _statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("aggregate inputs must be non-empty and finite")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
    }


def _metric_aggregate(
    per_seed: Sequence[Mapping[str, Any]],
    path: Sequence[str],
) -> dict[str, dict[str, float]]:
    first: Any = per_seed[0]
    for component in path:
        first = first[component]
    return {
        metric: _statistics(
            _nested_value(seed_row, (*path, metric)) for seed_row in per_seed
        )
        for metric in first
    }


def _nested_value(mapping: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = mapping
    for component in path:
        value = value[component]
    return float(value)


def _build_aggregate(per_seed: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for loss_name in (
        "train_fit_loss",
        "train_dense_loss",
        "validation_loss",
        "test_loss",
    ):
        if loss_name == "validation_loss":
            aggregate[loss_name] = _statistics(
                float(row["corroborated_validation_loss"]) for row in per_seed
            )
        else:
            aggregate[loss_name] = _statistics(
                float(row["metrics"][loss_name]) for row in per_seed
            )
    aggregate["validation_classification"] = _metric_aggregate(
        per_seed, ("validation_evaluation", "classification")
    )
    aggregate["validation_raw_probability_strategy"] = _metric_aggregate(
        per_seed, ("validation_evaluation", "strategy")
    )
    aggregate["validation_buy_and_hold"] = _metric_aggregate(
        per_seed, ("validation_evaluation", "buy_and_hold")
    )
    aggregate["test_classification"] = _metric_aggregate(
        per_seed, ("test_evaluation", "classification")
    )
    aggregate["test_raw_probability_strategy"] = _metric_aggregate(
        per_seed, ("test_evaluation", "strategy")
    )
    aggregate["test_buy_and_hold"] = _metric_aggregate(
        per_seed, ("test_evaluation", "buy_and_hold")
    )
    return aggregate


def _metric_table(aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, values in aggregate.items():
        if set(values) == {"mean", "std", "median"}:
            rows.append({"metric": group, **values})
            continue
        for metric in sorted(values):
            statistics = values[metric]
            rows.append({"metric": f"{group}.{metric}", **statistics})
    return rows


def _load_training_prior() -> dict[str, Any]:
    audit_path = PROJECT_ROOT / "results" / "canonical_data_audit.json"
    audit = _read_json(audit_path)
    return {
        "source": str(audit_path.relative_to(PROJECT_ROOT)),
        "probability": audit["splits"]["train"]["positive_rate"],
        "validation": audit["splits"]["validation"]["train_prior_baseline"],
    }


def _validation_baselines(
    dataset: PreparedDataset, training_prior: Mapping[str, Any]
) -> dict[str, Any]:
    validation_slice = dataset.splits.validation
    prices = np.asarray(dataset.close[validation_slice], dtype=np.float64)
    targets = np.asarray(dataset.targets[validation_slice], dtype=np.int64)
    prior_probability = float(training_prior["probability"])
    prior_probabilities = np.full(targets.size, prior_probability, dtype=np.float64)
    prior_classification = classification_metrics(targets, prior_probabilities)
    declared_prior = training_prior["validation"]
    if prior_classification != declared_prior:
        raise ValueError("canonical audit training-prior validation baseline mismatch")
    prior_loss = float(
        -np.mean(
            targets * np.log(prior_probabilities)
            + (1 - targets) * np.log(1.0 - prior_probabilities)
        )
    )
    cash_equity = simulate_positions(
        prices,
        np.zeros(prices.size),
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
    )
    buy_hold_equity = simulate_positions(
        prices,
        np.ones(prices.size),
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
    )
    return {
        "training_prior": {**training_prior, "validation_loss": prior_loss},
        "cash": {
            "definition": "position = 0.0",
            "validation_strategy": strategy_metrics(cash_equity),
        },
        "buy_and_hold": {
            "definition": "position = 1.0",
            "validation_strategy": strategy_metrics(buy_hold_equity),
        },
    }


def _stage2_gate(aggregate: Mapping[str, Any], baselines: Mapping[str, Any]) -> dict[str, Any]:
    prior = baselines["training_prior"]
    validation_loss = aggregate["validation_loss"]["median"]
    validation_brier = aggregate["validation_classification"]["brier"]["median"]
    validation_auc = aggregate["validation_classification"]["auc_roc"]["median"]
    criteria = {
        "median_validation_loss_below_training_prior": validation_loss
        < prior["validation_loss"],
        "median_validation_brier_below_training_prior": validation_brier
        < prior["validation"]["brier"],
        "median_validation_auc_above_training_prior": validation_auc
        > prior["validation"]["auc_roc"],
    }
    passed = all(criteria.values())
    decision = "proceed_research_only" if passed else "hold"
    reason = (
        "Validation-only gate passed: median loss and Brier beat the frozen training-prior "
        "baselines and median ROC-AUC exceeds the training-prior baseline; Stage 2 remains "
        "research-only."
        if passed
        else "Validation-only gate held: median loss, Brier, and ROC-AUC did not all beat "
        "their frozen training-prior baselines."
    )
    return {
        "decision": decision,
        "uses_test_metrics": False,
        "criteria": criteria,
        "reason": reason,
        "validation_readout": {
            "median_raw_probability_policy": {
                metric: values["median"]
                for metric, values in aggregate[
                    "validation_raw_probability_strategy"
                ].items()
            },
            "cash": baselines["cash"]["validation_strategy"],
            "buy_and_hold": baselines["buy_and_hold"]["validation_strategy"],
            "training_prior": prior,
        },
    }


def summarize_stage1(run_dir: Path, data_path: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise ValueError(f"Stage 1 run directory does not exist: {run_dir}")
    run_kind, seed_directories = _discover_seed_directories(run_dir)
    dataset = prepare_dataset(data_path)
    _validate_canonical_dataset(dataset, data_path)
    per_seed: list[dict[str, Any]] = []
    for seed, seed_dir in seed_directories:
        _validate_prediction_artifacts(seed_dir)
        manifest_path = seed_dir / "manifest.json"
        manifest = _read_json(manifest_path)
        if not isinstance(manifest, Mapping):
            raise ValueError(f"{manifest_path} must contain a JSON object")
        _require_finite_json(manifest, str(manifest_path))
        streams = _validate_manifest_and_streams(seed, seed_dir, dataset, manifest)

        validation_dates, validation_probabilities = streams["validation"]
        validation_evaluation = evaluate_probability_stream_for_split(
            dataset,
            "validation",
            validation_dates,
            validation_probabilities,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
        )
        test_dates, test_probabilities = streams["test"]
        test_evaluation = evaluate_probability_stream(
            dataset,
            test_dates,
            test_probabilities,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
        )
        evaluation_path = seed_dir / "canonical_evaluation.json"
        verified_test_evaluation = _write_or_verify_evaluation(
            evaluation_path, test_evaluation
        )

        history_path = seed_dir / "history.json"
        history = _read_json(history_path)
        _require_finite_json(history, str(history_path))
        metrics_path = seed_dir / "metrics.json"
        metrics = _read_json(metrics_path)
        _require_finite_json(metrics, str(metrics_path))
        if not isinstance(metrics, Mapping) or metrics.get("seed") != seed:
            raise ValueError(f"{metrics_path} seed does not match directory")
        if not isinstance(history, Mapping):
            raise ValueError(f"{history_path} must contain a JSON object")
        corroborated_validation_loss = _corroborate_selection_evidence(
            seed,
            seed_dir,
            dataset,
            history,
            metrics,
            validation_probabilities,
        )
        per_seed.append(
            {
                "seed": seed,
                "seed_directory": seed_dir.name,
                "manifest": str(manifest_path.relative_to(run_dir)),
                "canonical_evaluation": str(evaluation_path.relative_to(run_dir)),
                "corroborated_validation_loss": corroborated_validation_loss,
                "metrics": metrics,
                "validation_evaluation": validation_evaluation,
                "test_evaluation": verified_test_evaluation,
            }
        )

    aggregate = _build_aggregate(per_seed)
    baselines = _validation_baselines(dataset, _load_training_prior())
    summary: dict[str, Any] = {
        "run_kind": run_kind,
        "seeds": [seed for seed, _ in seed_directories],
        "data": str(data_path),
        "test_interval": {
            "start": dataset.dates[dataset.splits.test.start],
            "end": dataset.dates[dataset.splits.test.stop - 1],
            "n_samples": dataset.splits.test.stop - dataset.splits.test.start,
        },
        "policy": {
            "name": "probability_positions",
            "input": "raw probability",
            "position_rule": POSITION_RULE,
            "position_range": [-1.0, 1.0],
            "transaction_cost": TRANSACTION_COST,
            "slippage": SLIPPAGE,
        },
        "technical_gate": {
            "status": "pass",
            "checks": [
                "exact seed-NNN set discovered",
                "required prediction artifact inventory exact",
                "all four prediction streams exactly aligned and finite",
                "dense train stream and train-only metadata present",
                "canonical dataset, feature order, splits, and hyperparameters verified",
                "fixed-origin unpurged protocol and validation-only checkpoint criterion verified",
                "checkpoint, history, metrics, epoch, and prediction loss provenance corroborated",
                "shared evaluator accepted every exact test stream",
                "saved canonical evaluations match shared evaluator",
            ],
        },
        "baselines": baselines,
        "aggregate": aggregate,
        "metric_table": _metric_table(aggregate),
        "per_seed": per_seed,
    }
    summary["stage2_gate"] = _stage2_gate(aggregate, baselines)
    if run_kind == "full":
        selected = min(
            per_seed,
            key=lambda row: (float(row["corroborated_validation_loss"]), row["seed"]),
        )
        summary["stage1_selection"] = {
            "criterion": "validation_loss ascending, seed ascending tie-break",
            "seed": selected["seed"],
            "seed_directory": selected["seed_directory"],
            "validation_loss": selected["corroborated_validation_loss"],
        }
    _require_finite_json(summary, "summary")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summary = summarize_stage1(args.run_dir, args.data)
    _write_json_atomic(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    print(f"\nWrote Stage 1 summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
