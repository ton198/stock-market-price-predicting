#!/usr/bin/env python3
"""Train one canonical research-only DQN seed."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_pipeline.backtest import simulate_positions
from quant_pipeline.data import PreparedDataset, prepare_dataset
from quant_pipeline.dqn_env import DQNTradingEnv, OBSERVATION_HIGH, OBSERVATION_LOW
from quant_pipeline.evaluator import load_prediction_csv
from quant_pipeline.metrics import strategy_metrics
from quant_pipeline.signals import (
    SignalNormalizer,
    fit_signal_normalizer,
    normalize_signal,
)


STREAM_FILENAMES = {
    "train": "predictions_train_dense.csv",
    "validation": "predictions_validation.csv",
    "test": "predictions_test.csv",
}
DENSE_TRAIN_ROLE = "in-sample Stage 1 stream eligible for DQN"
DQN_HYPERPARAMETERS = {
    "buffer_size": 100_000,
    "learning_starts": 5_000,
    "batch_size": 128,
    "learning_rate": 1e-4,
    "gamma": 0.97,
    "target_update_interval": 1_000,
    "exploration_fraction": 0.3,
    "exploration_final_eps": 0.05,
    "train_freq": 4,
    "policy_layers": [256, 256],
}
ENVIRONMENT_DEFAULTS = {
    "initial_cash": 10_000.0,
    "transaction_cost": 0.001,
    "slippage": 0.0005,
    "episode_length": 252,
    "vol_penalty_weight": 0.0,
    "margin_rate": 0.02 / 252,
}
VALIDATION_FREQUENCY = 10_000


def resolve_stage1_artifacts(
    stage1_artifacts: str | Path,
) -> tuple[Path, dict[str, Any]]:
    """Resolve either a selected seed directory or a full Stage 1 summary."""

    path = Path(stage1_artifacts)
    summary_path = path / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read Stage 1 summary: {summary_path}") from exc
        selection = summary.get("stage1_selection")
        if not isinstance(selection, dict):
            raise ValueError("Stage 1 summary lacks stage1_selection")
        seed_directory = selection.get("seed_directory")
        seed = selection.get("seed")
        if not isinstance(seed_directory, str) or not isinstance(seed, int):
            raise ValueError("Stage 1 selection must identify a seed directory and seed")
        selected = path / seed_directory
        if not (selected / "manifest.json").is_file():
            raise ValueError(f"selected Stage 1 artifacts do not exist: {selected}")
        selected_manifest = _load_stage1_manifest(selected)
        if selected_manifest.get("seed") != seed:
            raise ValueError("Stage 1 summary selection does not match the selected manifest")
        return selected, dict(selection)
    if (path / "manifest.json").is_file():
        manifest = _load_stage1_manifest(path)
        return path, {
            "seed": manifest.get("seed"),
            "seed_directory": path.name,
            "criterion": "explicit preselected Stage 1 artifact directory",
        }
    raise ValueError(
        "stage1_artifacts must contain summary.json or a selected seed manifest.json"
    )


def _load_stage1_manifest(stage1_artifacts: Path) -> dict[str, Any]:
    manifest_path = stage1_artifacts / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Stage 1 manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Stage 1 manifest must be a JSON object")
    return manifest


def _expected_dates(
    dataset: PreparedDataset,
    split: str,
    manifest: dict[str, Any],
) -> tuple[str, ...]:
    if split not in STREAM_FILENAMES:
        raise ValueError("split must be one of: train, validation, test")
    split_slice = getattr(dataset.splits, split)
    start = 0 if split_slice.start is None else int(split_slice.start)
    stop = len(dataset.dates) if split_slice.stop is None else int(split_slice.stop)
    if split == "train":
        hyperparameters = manifest.get("hyperparameters")
        if not isinstance(hyperparameters, dict):
            raise ValueError("Stage 1 manifest lacks hyperparameters")
        window_size = hyperparameters.get("window_size")
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 1:
            raise ValueError("Stage 1 manifest has invalid window_size")
        start = max(start, window_size - 1)
    return tuple(dataset.dates[start:stop])


def _validate_stream_metadata(
    manifest: dict[str, Any],
    split: str,
    dates: tuple[str, ...],
) -> None:
    stream_name = "train_dense" if split == "train" else split
    streams = manifest.get("streams")
    if not isinstance(streams, dict) or not isinstance(streams.get(stream_name), dict):
        raise ValueError(f"Stage 1 manifest lacks {stream_name} stream metadata")
    metadata = streams[stream_name]
    if split == "train" and metadata.get("role") != DENSE_TRAIN_ROLE:
        raise ValueError("Stage 1 dense train stream is not eligible for DQN")
    expected = {
        "filename": STREAM_FILENAMES[split],
        "rows": len(dates),
        "cadence": 1,
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
    }
    for name, value in expected.items():
        if metadata.get(name) != value:
            label = "dense train" if split == "train" else split
            raise ValueError(f"Stage 1 {label} stream has invalid {name}")


def _raw_feature(dataset: PreparedDataset, name: str, indices: np.ndarray) -> np.ndarray:
    try:
        column = tuple(dataset.feature_names).index(name)
    except ValueError as exc:
        raise ValueError(f"canonical dataset lacks required feature {name!r}") from exc
    scaled = np.asarray(dataset.features[indices, column], dtype=np.float64)
    mean = float(dataset.scaler.mean_[column])
    scale = float(dataset.scaler.scale_[column])
    return scaled * scale + mean


def build_dqn_observations(
    dataset: PreparedDataset,
    stage1_artifacts: Path,
    split: str,
    normalizer: SignalNormalizer,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Align one dense Stage 1 stream and build the three static DQN features."""

    artifact_directory = Path(stage1_artifacts)
    manifest = _load_stage1_manifest(artifact_directory)
    expected_dates = _expected_dates(dataset, split, manifest)
    prediction_dates, raw_probabilities = load_prediction_csv(
        artifact_directory / STREAM_FILENAMES.get(split, "")
    )
    if prediction_dates != expected_dates:
        label = "dense train" if split == "train" else split
        raise ValueError(
            f"Stage 1 {label} dates must exactly match canonical anchors "
            f"({len(expected_dates)} expected, {len(prediction_dates)} provided)"
        )
    _validate_stream_metadata(manifest, split, prediction_dates)

    split_slice = getattr(dataset.splits, split)
    start = 0 if split_slice.start is None else int(split_slice.start)
    if split == "train":
        start = max(start, int(manifest["hyperparameters"]["window_size"]) - 1)
    stop = len(dataset.dates) if split_slice.stop is None else int(split_slice.stop)
    indices = np.arange(start, stop, dtype=np.int64)
    prices = np.asarray(dataset.close[indices], dtype=np.float64)
    if indices.size != raw_probabilities.size:
        raise ValueError("Stage 1 stream does not have one probability per aligned anchor")

    prior_indices = indices - 5
    if (prior_indices < 0).any():
        raise ValueError("DQN anchors require at least five prior close observations")
    close = np.asarray(dataset.close, dtype=np.float64)
    returns_5d = close[indices] / close[prior_indices] - 1.0
    recent_5d_norm = 0.5 + 0.5 * np.tanh(returns_5d / 0.05)
    volatility = _raw_feature(dataset, "Volatility_20d", indices)
    vol_norm = np.clip(volatility / 0.03, 0.0, 1.0)
    regime = _raw_feature(dataset, "Regime", indices)
    extras = np.column_stack((recent_5d_norm, vol_norm, regime)).astype(np.float32)
    return (
        prediction_dates,
        prices,
        normalize_signal(raw_probabilities, normalizer),
        extras,
    )


def _make_dqn_model(environment: DQNTradingEnv, *, seed: int, device: str):
    """Construct Stable-Baselines3 DQN with the frozen research settings."""

    from stable_baselines3 import DQN

    return DQN(
        "MlpPolicy",
        environment,
        buffer_size=DQN_HYPERPARAMETERS["buffer_size"],
        learning_starts=DQN_HYPERPARAMETERS["learning_starts"],
        batch_size=DQN_HYPERPARAMETERS["batch_size"],
        learning_rate=DQN_HYPERPARAMETERS["learning_rate"],
        gamma=DQN_HYPERPARAMETERS["gamma"],
        target_update_interval=DQN_HYPERPARAMETERS["target_update_interval"],
        exploration_fraction=DQN_HYPERPARAMETERS["exploration_fraction"],
        exploration_final_eps=DQN_HYPERPARAMETERS["exploration_final_eps"],
        train_freq=DQN_HYPERPARAMETERS["train_freq"],
        policy_kwargs={"net_arch": DQN_HYPERPARAMETERS["policy_layers"]},
        seed=seed,
        device=device,
        verbose=0,
    )


def validation_window_starts(
    n_validation: int, episode_length: int = 252
) -> tuple[int, int, int]:
    """Return the fixed first, middle, and last validation window starts."""

    required = episode_length + 1
    if n_validation < required:
        raise ValueError(f"validation requires at least {required} anchors")
    last = n_validation - required
    return (0, last // 2, last)


def checkpoint_is_better(
    candidate_return: float,
    candidate_step: int,
    best_return: float | None,
    best_step: int | None,
) -> bool:
    """Rank checkpoints by return descending and step ascending."""

    if best_return is None or best_step is None:
        return True
    return (float(candidate_return), -int(candidate_step)) > (
        float(best_return),
        -int(best_step),
    )


def rollout_policy_actions(
    policy: Any,
    environment: DQNTradingEnv,
    *,
    start: int,
) -> tuple[np.ndarray, float]:
    """Roll a deterministic policy through portfolio state and include the final anchor."""

    observation, _ = environment.reset(options={"start": start})
    positions: list[float] = []
    shaped_reward = 0.0
    while True:
        action, _ = policy.predict(observation, deterministic=True)
        action_index = int(np.asarray(action).item())
        if not 0 <= action_index <= 20:
            raise ValueError("policy produced an action outside [0, 20]")
        positions.append(action_index / 10.0 - 1.0)
        observation, reward, terminated, truncated, _ = environment.step(action_index)
        shaped_reward += float(reward)
        if truncated:
            raise RuntimeError("canonical DQN rollouts must not truncate")
        if terminated:
            final_action, _ = policy.predict(observation, deterministic=True)
            final_index = int(np.asarray(final_action).item())
            if not 0 <= final_index <= 20:
                raise ValueError("policy produced an action outside [0, 20]")
            positions.append(final_index / 10.0 - 1.0)
            break
    return np.asarray(positions, dtype=np.float64), shaped_reward


def _evaluate_validation_policy(
    policy: Any,
    prices: np.ndarray,
    signal: np.ndarray,
    extras: np.ndarray,
    *,
    seed: int,
    signal_bonus_weight: float,
) -> tuple[float, list[dict[str, Any]]]:
    starts = validation_window_starts(prices.size)
    records: list[dict[str, Any]] = []
    for start in starts:
        environment = DQNTradingEnv(
            prices=prices,
            signal=signal,
            extra_features=extras,
            seed=seed,
            signal_bonus_weight=signal_bonus_weight,
            **ENVIRONMENT_DEFAULTS,
        )
        positions, shaped_reward = rollout_policy_actions(
            policy, environment, start=start
        )
        stop = start + ENVIRONMENT_DEFAULTS["episode_length"] + 1
        metrics = strategy_metrics(
            simulate_positions(
                prices[start:stop],
                positions,
                initial_equity=ENVIRONMENT_DEFAULTS["initial_cash"],
                transaction_cost=ENVIRONMENT_DEFAULTS["transaction_cost"],
                slippage=ENVIRONMENT_DEFAULTS["slippage"],
                min_position=-1.0,
                max_position=1.0,
            )
        )
        records.append(
            {
                "start": start,
                "stop": stop,
                "n_anchors": int(positions.size),
                "n_intervals": int(positions.size - 1),
                "costed_total_return": metrics["total_return"],
                "shaped_reward": shaped_reward,
            }
        )
    return float(np.mean([row["costed_total_return"] for row in records])), records


def _write_action_csv(
    path: Path, dates: Sequence[str], positions: np.ndarray
) -> None:
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 1 or values.size != len(dates):
        raise ValueError("action export requires one position per date")
    if not np.isfinite(values).all() or ((values < -1.0) | (values > 1.0)).any():
        raise ValueError("action positions must be finite and lie in [-1, 1]")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Date", "position"))
        writer.writerows(zip(dates, values))


class _ValidationCheckpointCallback:
    """Factory wrapper that keeps Stable-Baselines3 optional at module import."""

    @staticmethod
    def create(
        output_dir: Path,
        validation_data: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        seed: int,
        signal_bonus_weight: float,
    ):
        from stable_baselines3.common.callbacks import BaseCallback

        prices, signal, extras = validation_data

        class Callback(BaseCallback):
            def __init__(self) -> None:
                super().__init__(verbose=0)
                self.records: list[dict[str, Any]] = []
                self.best_return: float | None = None
                self.best_step: int | None = None
                self.best_path: Path | None = None
                self.next_evaluation = VALIDATION_FREQUENCY

            def _on_step(self) -> bool:
                if self.num_timesteps < self.next_evaluation:
                    return True
                mean_return, windows = _evaluate_validation_policy(
                    self.model,
                    prices,
                    signal,
                    extras,
                    seed=seed,
                    signal_bonus_weight=signal_bonus_weight,
                )
                record = {
                    "step": int(self.num_timesteps),
                    "mean_costed_total_return": mean_return,
                    "windows": windows,
                }
                self.records.append(record)
                if checkpoint_is_better(
                    mean_return,
                    self.num_timesteps,
                    self.best_return,
                    self.best_step,
                ):
                    checkpoint = output_dir / "checkpoints" / f"step-{self.num_timesteps}"
                    self.model.save(checkpoint)
                    self.best_return = mean_return
                    self.best_step = int(self.num_timesteps)
                    self.best_path = checkpoint.with_suffix(".zip")
                self.next_evaluation += VALIDATION_FREQUENCY
                return True

        return Callback()


def _full_split_rollout(
    policy: Any,
    prices: np.ndarray,
    signal: np.ndarray,
    extras: np.ndarray,
    *,
    seed: int,
    signal_bonus_weight: float,
) -> tuple[np.ndarray, float]:
    environment = DQNTradingEnv(
        prices=prices,
        signal=signal,
        extra_features=extras,
        episode_length=prices.size - 1,
        initial_cash=ENVIRONMENT_DEFAULTS["initial_cash"],
        transaction_cost=ENVIRONMENT_DEFAULTS["transaction_cost"],
        slippage=ENVIRONMENT_DEFAULTS["slippage"],
        vol_penalty_weight=ENVIRONMENT_DEFAULTS["vol_penalty_weight"],
        margin_rate=ENVIRONMENT_DEFAULTS["margin_rate"],
        signal_bonus_weight=signal_bonus_weight,
        seed=seed,
    )
    return rollout_policy_actions(policy, environment, start=0)


def train_dqn_seed(
    dataset: PreparedDataset,
    stage1_artifacts: Path,
    output_dir: str | Path,
    *,
    seed: int,
    timesteps: int = 1_000_000,
    device: str = "auto",
    signal_bonus_weight: float = 20.0,
) -> dict[str, object]:
    """Train one DQN seed and write its model, actions, and audit manifest."""

    if timesteps < VALIDATION_FREQUENCY:
        raise ValueError(f"timesteps must be at least {VALIDATION_FREQUENCY}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "checkpoints").mkdir()
    selected_artifacts, stage1_selection = resolve_stage1_artifacts(stage1_artifacts)
    dense_dates, dense_probabilities = load_prediction_csv(
        selected_artifacts / STREAM_FILENAMES["train"]
    )
    normalizer = fit_signal_normalizer(dense_probabilities)
    train = build_dqn_observations(
        dataset, selected_artifacts, "train", normalizer
    )
    validation = build_dqn_observations(
        dataset, selected_artifacts, "validation", normalizer
    )
    test = build_dqn_observations(dataset, selected_artifacts, "test", normalizer)
    if train[0] != dense_dates:
        raise ValueError("normalizer fit stream is not the aligned dense train stream")

    train_environment = DQNTradingEnv(
        prices=train[1],
        signal=train[2],
        extra_features=train[3],
        seed=seed,
        signal_bonus_weight=signal_bonus_weight,
        **ENVIRONMENT_DEFAULTS,
    )
    model = _make_dqn_model(train_environment, seed=seed, device=device)
    callback = _ValidationCheckpointCallback.create(
        destination,
        (validation[1], validation[2], validation[3]),
        seed=seed,
        signal_bonus_weight=signal_bonus_weight,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callback)
    wall_clock_seconds = time.perf_counter() - started
    if callback.best_path is None or callback.best_step is None:
        raise RuntimeError("training produced no validation checkpoint")

    from stable_baselines3 import DQN

    best_model = DQN.load(callback.best_path, device=device)
    best_model.save(destination / "model")
    validation_positions, validation_shaped_reward = _full_split_rollout(
        best_model,
        validation[1],
        validation[2],
        validation[3],
        seed=seed,
        signal_bonus_weight=signal_bonus_weight,
    )
    test_positions, test_shaped_reward = _full_split_rollout(
        best_model,
        test[1],
        test[2],
        test[3],
        seed=seed,
        signal_bonus_weight=signal_bonus_weight,
    )
    _write_action_csv(destination / "actions_validation.csv", validation[0], validation_positions)
    _write_action_csv(destination / "actions_test.csv", test[0], test_positions)

    resolved_device = str(model.device)
    manifest: dict[str, object] = {
        "status": "research_only",
        "portfolio_publication_allowed": False,
        "seed": int(seed),
        "timesteps": int(timesteps),
        "wall_clock_seconds": float(wall_clock_seconds),
        "device": {"requested": device, "resolved": resolved_device},
        "stage1": {
            "requested_path": str(Path(stage1_artifacts)),
            "selected_artifacts": str(selected_artifacts),
            "selection": stage1_selection,
        },
        "signal_normalizer": {
            "fit_split": "train_dense",
            "fit_filename": STREAM_FILENAMES["train"],
            "mean": normalizer.mean_,
            "scale": normalizer.scale_,
            "rows": len(dense_dates),
        },
        "input_streams": {
            "train": STREAM_FILENAMES["train"],
            "validation": STREAM_FILENAMES["validation"],
            "test": STREAM_FILENAMES["test"],
        },
        "source_feature_names": list(dataset.feature_names),
        "static_feature_names": ["recent_5d_norm", "vol_norm", "Regime"],
        "observation_names": [
            "normalized_signal",
            "recent_5d_norm",
            "vol_norm",
            "Regime",
            "position_ratio",
            "cash_ratio",
        ],
        "observation_bounds": {
            "low": OBSERVATION_LOW.tolist(),
            "high": OBSERVATION_HIGH.tolist(),
        },
        "environment": {
            **ENVIRONMENT_DEFAULTS,
            "signal_bonus_weight": float(signal_bonus_weight),
            "action_count": 21,
            "action_mapping": "action_index / 10 - 1",
            "reward_usage": "learning_only",
        },
        "dqn_hyperparameters": DQN_HYPERPARAMETERS,
        "validation_checkpoint_selection": {
            "frequency_environment_steps": VALIDATION_FREQUENCY,
            "criterion": "highest mean costed total_return; earliest checkpoint on ties",
            "window_starts": list(validation_window_starts(len(validation[0]))),
            "records": callback.records,
            "selected_step": callback.best_step,
            "selected_mean_costed_total_return": callback.best_return,
        },
        "split_alignment": {
            name: {
                "rows": len(values[0]),
                "start_date": values[0][0],
                "end_date": values[0][-1],
            }
            for name, values in (("train", train), ("validation", validation), ("test", test))
        },
        "exports": {
            "model": "model.zip",
            "validation_actions": "actions_validation.csv",
            "test_actions": "actions_test.csv",
            "validation_shaped_reward": validation_shaped_reward,
            "test_shaped_reward": test_shaped_reward,
        },
        "test_contract": {
            "action_anchors": len(test[0]),
            "realized_intervals": len(test[0]) - 1,
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage1-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--signal-bonus-weight", type=float, default=20.0)
    args = parser.parse_args(argv)
    manifest = train_dqn_seed(
        prepare_dataset(args.data),
        args.stage1_artifacts,
        args.output_dir,
        seed=args.seed,
        timesteps=args.timesteps,
        device=args.device,
        signal_bonus_weight=args.signal_bonus_weight,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
