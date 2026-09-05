"""Canonical Stage 1 Hybrid LSTM model and training entry points."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    DEFAULT_FEATURE_COLS,
    PreparedDataset,
    SequenceWindows,
    build_sequence_windows,
)
from .metrics import classification_metrics


MACRO_FEATURE_NAMES: tuple[str, ...] = (
    "FedRate",
    "FedRate_chg20",
    "FedRate_chg60",
    "TNX",
    "Yield_slope",
    "VIX",
    "VIX_percentile",
)


def macro_feature_indices(
    feature_names: Sequence[str],
) -> tuple[int, int, int, int, int, int, int]:
    """Return Stage 1 macro-feature positions in the canonical branch order."""

    positions = tuple(feature_names.index(name) for name in MACRO_FEATURE_NAMES)
    return positions  # type: ignore[return-value]


class HybridLSTM(nn.Module):
    """Two-branch classifier combining sequence and current macro features."""

    def __init__(
        self,
        feature_count: int = 30,
        macro_count: int = 7,
        hidden_size: int = 32,
        macro_hidden: int = 16,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.macro_branch = nn.Sequential(
            nn.Linear(macro_count, macro_hidden),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size + macro_hidden, 1)

    def forward(self, sequence: torch.Tensor, macro: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized classification logit per input row."""

        sequence_output, _ = self.lstm(sequence)
        sequence_embedding = sequence_output[:, -1, :]
        macro_embedding = self.macro_branch(macro)
        combined = torch.cat((sequence_embedding, macro_embedding), dim=1)
        return self.output(self.dropout(combined)).squeeze(1)


def build_stage1_windows(
    dataset: PreparedDataset,
    *,
    window_size: int = 60,
    train_stride: int = 3,
) -> dict[str, SequenceWindows]:
    """Build sparse fit windows and dense validation/test inference windows."""

    return {
        "train": build_sequence_windows(
            dataset.features,
            dataset.targets,
            anchor_slice=dataset.splits.train,
            window_size=window_size,
            stride=train_stride,
        ),
        "validation": build_sequence_windows(
            dataset.features,
            dataset.targets,
            anchor_slice=dataset.splits.validation,
            window_size=window_size,
            stride=1,
        ),
        "test": build_sequence_windows(
            dataset.features,
            dataset.targets,
            anchor_slice=dataset.splits.test,
            window_size=window_size,
            stride=1,
        ),
    }


def _set_deterministic_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def _tensor_dataset(
    windows: SequenceWindows,
    macro_indices: tuple[int, ...],
) -> TensorDataset:
    sequence = torch.from_numpy(np.asarray(windows.features, dtype=np.float32))
    macro = sequence[:, -1, list(macro_indices)]
    targets = torch.from_numpy(np.asarray(windows.targets, dtype=np.float32))
    return TensorDataset(sequence, macro, targets)


def _mean_loss(
    model: HybridLSTM,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    row_count = 0
    with torch.no_grad():
        for sequence, macro, targets in loader:
            sequence = sequence.to(device)
            macro = macro.to(device)
            targets = targets.to(device)
            loss = criterion(model(sequence, macro), targets)
            loss_sum += float(loss.item()) * targets.shape[0]
            row_count += targets.shape[0]
    if row_count == 0:
        raise ValueError("Stage 1 windows must contain at least one row")
    return loss_sum / row_count


def _predict(
    model: HybridLSTM,
    windows: SequenceWindows,
    macro_indices: tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        _tensor_dataset(windows, macro_indices),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, macro, _ in loader:
            logits = model(sequence.to(device), macro.to(device))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    if not predictions:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(predictions).astype(np.float32, copy=False)


def _write_predictions(
    path: Path,
    dataset: PreparedDataset,
    windows: SequenceWindows,
    probabilities: np.ndarray,
) -> None:
    dates = tuple(dataset.dates[int(index)] for index in windows.anchor_indices)
    if len(dates) != probabilities.size:
        raise ValueError("prediction count does not match anchor dates")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Date", "probability"))
        for date_value, probability in zip(dates, probabilities):
            writer.writerow((date_value, format(float(probability), ".17g")))


def _write_json(path: Path, payload: object, *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _dataset_fingerprint(dataset: PreparedDataset) -> str:
    digest = hashlib.sha256()
    for date_value in dataset.dates:
        digest.update(date_value.encode("utf-8"))
        digest.update(b"\0")
    for values in (dataset.features, dataset.targets, dataset.close):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _git_revision() -> str | None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _slice_metadata(row_slice: slice) -> dict[str, int | None]:
    return {"start": row_slice.start, "stop": row_slice.stop, "step": row_slice.step}


def _stream_metadata(
    filename: str,
    windows: SequenceWindows,
    cadence: int,
    dataset: PreparedDataset,
    role: str,
) -> dict[str, object]:
    dates = tuple(dataset.dates[int(index)] for index in windows.anchor_indices)
    return {
        "filename": filename,
        "rows": len(dates),
        "cadence": cadence,
        "role": role,
        "start_date": dates[0] if dates else None,
        "end_date": dates[-1] if dates else None,
    }


def fit_stage1_seed(
    dataset: PreparedDataset,
    output_dir: str | Path,
    *,
    seed: int,
    device: str = "auto",
    epochs: int = 300,
    batch_size: int = 32,
    window_size: int = 60,
    train_stride: int = 3,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 20,
) -> dict[str, object]:
    """Train one deterministic seed and write its immutable artifact set."""

    if epochs < 1 or batch_size < 1 or early_stopping_patience < 1:
        raise ValueError("epochs, batch_size, and early_stopping_patience must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable manifest already exists: {manifest_path}")

    _set_deterministic_seed(seed)
    resolved_device = _resolve_device(device)
    macro_indices = macro_feature_indices(dataset.feature_names)
    windows = build_stage1_windows(
        dataset,
        window_size=window_size,
        train_stride=train_stride,
    )
    dense_train = build_sequence_windows(
        dataset.features,
        dataset.targets,
        anchor_slice=dataset.splits.train,
        window_size=window_size,
        stride=1,
    )
    if any(len(value.anchor_indices) == 0 for value in windows.values()):
        raise ValueError("every Stage 1 split must contain at least one eligible window")

    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        _tensor_dataset(windows["train"], macro_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    validation_loader = DataLoader(
        _tensor_dataset(windows["validation"], macro_indices),
        batch_size=batch_size,
        shuffle=False,
    )

    model = HybridLSTM(feature_count=dataset.features.shape[1]).to(resolved_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epoch_history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for sequence, macro, targets in train_loader:
            sequence = sequence.to(resolved_device)
            macro = macro.to(resolved_device)
            targets = targets.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(sequence, macro), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * targets.shape[0]
            train_rows += targets.shape[0]

        validation_loss = _mean_loss(
            model,
            validation_loader,
            criterion,
            resolved_device,
        )
        scheduler.step(validation_loss)
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_rows,
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state)

    prediction_windows = {
        "train_fit": windows["train"],
        "train_dense": dense_train,
        "validation": windows["validation"],
        "test": windows["test"],
    }
    predictions = {
        name: _predict(model, split_windows, macro_indices, batch_size, resolved_device)
        for name, split_windows in prediction_windows.items()
    }
    prediction_filenames = {
        "train_fit": "predictions_train_fit.csv",
        "train_dense": "predictions_train_dense.csv",
        "validation": "predictions_validation.csv",
        "test": "predictions_test.csv",
    }
    for name, filename in prediction_filenames.items():
        _write_predictions(
            destination / filename,
            dataset,
            prediction_windows[name],
            predictions[name],
        )

    losses = {
        name: float(
            nn.functional.binary_cross_entropy(
                torch.from_numpy(predictions[name]),
                torch.from_numpy(np.asarray(prediction_windows[name].targets, dtype=np.float32)),
            ).item()
        )
        for name in prediction_windows
    }
    metrics: dict[str, object] = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(epoch_history),
        "train_fit_loss": losses["train_fit"],
        "train_dense_loss": losses["train_dense"],
        "validation_loss": losses["validation"],
        "test_loss": losses["test"],
        "classification": {
            name: classification_metrics(prediction_windows[name].targets, probabilities)
            for name, probabilities in predictions.items()
        },
    }
    history = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs": epoch_history,
    }
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "feature_names": tuple(dataset.feature_names),
        "macro_feature_names": MACRO_FEATURE_NAMES,
        "macro_feature_indices": macro_indices,
        "model": {
            "feature_count": dataset.features.shape[1],
            "macro_count": len(MACRO_FEATURE_NAMES),
            "hidden_size": 32,
            "macro_hidden": 16,
            "dropout": 0.3,
        },
    }
    torch.save(checkpoint, destination / "checkpoint.pt")
    _write_json(destination / "history.json", history)
    _write_json(destination / "metrics.json", metrics)

    manifest = {
        "seed": seed,
        "code_revision": _git_revision(),
        "dataset_fingerprint_sha256": _dataset_fingerprint(dataset),
        "feature_names": list(dataset.feature_names),
        "macro_feature_names": list(MACRO_FEATURE_NAMES),
        "splits": {
            name: _slice_metadata(row_slice)
            for name, row_slice in dataset.splits.as_dict().items()
        },
        "device": {"requested": device, "resolved": str(resolved_device)},
        "torch_version": torch.__version__,
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "window_size": window_size,
            "train_stride": train_stride,
            "learning_rate": learning_rate,
            "early_stopping_patience": early_stopping_patience,
            "gradient_clip_norm": 1.0,
            "scheduler_factor": 0.5,
            "scheduler_patience": 10,
            "scheduler_min_lr": 1e-5,
        },
        "checkpoint_selection": "lowest validation loss",
        "streams": {
            "train_fit": _stream_metadata(
                prediction_filenames["train_fit"],
                windows["train"],
                train_stride,
                dataset,
                "optimizer-sampling diagnostic only",
            ),
            "train_dense": _stream_metadata(
                prediction_filenames["train_dense"],
                dense_train,
                1,
                dataset,
                "in-sample Stage 1 stream eligible for DQN",
            ),
            "validation": _stream_metadata(
                prediction_filenames["validation"],
                windows["validation"],
                1,
                dataset,
                "validation inference",
            ),
            "test": _stream_metadata(
                prediction_filenames["test"],
                windows["test"],
                1,
                dataset,
                "test inference",
            ),
        },
        "artifacts": [
            "checkpoint.pt",
            "history.json",
            "predictions_train_fit.csv",
            "predictions_train_dense.csv",
            "predictions_validation.csv",
            "predictions_test.csv",
            "metrics.json",
            "manifest.json",
        ],
    }
    _write_json(manifest_path, manifest, exclusive=True)
    return metrics
