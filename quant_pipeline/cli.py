"""Command-line audit for the canonical data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data import PreparedDataset, prepare_dataset
from .metrics import classification_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice_size(row_slice: slice) -> int:
    if row_slice.start is None or row_slice.stop is None:
        raise ValueError("audit slices must have explicit boundaries")
    return row_slice.stop - row_slice.start


def _split_summary(dataset: PreparedDataset, row_slice: slice) -> dict[str, Any]:
    labels = dataset.targets[row_slice]
    # A training-prior probability is a valid, leakage-free reference point.
    train_labels = dataset.targets[dataset.splits.train]
    majority_probability = float(train_labels.mean())
    baseline = classification_metrics(
        labels,
        np.full(labels.shape, majority_probability, dtype=np.float64),
    )
    return {
        "rows": _slice_size(row_slice),
        "start_date": dataset.dates[row_slice.start],
        "end_date": dataset.dates[row_slice.stop - 1],
        "positive_rate": float(labels.mean()),
        "train_prior_baseline": baseline,
    }


def build_summary(dataset: PreparedDataset, data_path: Path) -> dict[str, Any]:
    """Build a JSON-serializable audit summary."""

    return {
        "dataset": data_path.name,
        "dataset_sha256": _sha256(data_path),
        "raw_rows": int(dataset.raw_rows),
        "usable_labelled_rows": int(dataset.close.size),
        "dropped_warmup_or_unlabelled_rows": int(dataset.dropped_rows),
        "date_range": {"start": dataset.dates[0], "end": dataset.dates[-1]},
        "horizon_days": dataset.horizon,
        "feature_count": len(dataset.feature_names),
        "feature_columns": list(dataset.feature_names),
        "target": {
            "definition": "1 if the close-to-close forward return exceeds the training-set median",
            "threshold_fit_on_training_only": dataset.threshold,
        },
        "splits": {
            "train": _split_summary(dataset, dataset.splits.train),
            "validation": _split_summary(dataset, dataset.splits.validation),
            "test": _split_summary(dataset, dataset.splits.test),
        },
        "scaler": {
            "fit_split": "train",
            "means": [float(value) for value in dataset.scaler.mean_],
            "scales": [float(value) for value in dataset.scaler.scale_],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "datasets" / "nasdaq_multivariate.csv",
        help="path to the multivariate market CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional path for a JSON audit artifact",
    )
    args = parser.parse_args(argv)

    dataset = prepare_dataset(args.data)
    summary = build_summary(dataset, args.data)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote audit summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
