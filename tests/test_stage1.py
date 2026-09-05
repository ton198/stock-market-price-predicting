import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from quant_pipeline.data import (
    DEFAULT_FEATURE_COLS,
    PreparedDataset,
    SplitSlices,
    Standardizer,
    build_sequence_windows,
    prepare_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_tiny_dataset() -> PreparedDataset:
    rng = np.random.default_rng(20260904)
    row_count = 18
    feature_count = len(DEFAULT_FEATURE_COLS)
    dates = tuple(
        (date(2024, 1, 1) + timedelta(days=index)).isoformat()
        for index in range(row_count)
    )
    features = rng.normal(size=(row_count, feature_count)).astype(np.float32)
    targets = (np.arange(row_count) % 2).astype(np.int64)
    return PreparedDataset(
        dates=dates,
        close=np.linspace(100.0, 117.0, row_count),
        features=features,
        feature_names=DEFAULT_FEATURE_COLS,
        forward_returns=np.linspace(-0.05, 0.05, row_count),
        targets=targets,
        splits=SplitSlices(slice(0, 10), slice(10, 14), slice(14, 18)),
        threshold=0.0,
        scaler=Standardizer(
            mean_=np.zeros(feature_count, dtype=np.float32),
            scale_=np.ones(feature_count, dtype=np.float32),
        ),
        horizon=2,
        raw_rows=row_count,
        dropped_rows=0,
    )


def read_predictions(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        [row["Date"] for row in rows],
        np.asarray([float(row["probability"]) for row in rows]),
    )


class Stage1DataContractTests(unittest.TestCase):
    def setUp(self):
        self.features = np.arange(24, dtype=np.float32).reshape(12, 2)
        self.targets = np.arange(12, dtype=np.int64)
        self.anchor_slice = slice(0, 12)
        self.window_size = 4

    def test_sequence_window_stride_selects_every_second_anchor(self):
        windows = build_sequence_windows(
            self.features,
            self.targets,
            anchor_slice=self.anchor_slice,
            window_size=self.window_size,
            stride=2,
        )

        np.testing.assert_array_equal(windows.anchor_indices, [3, 5, 7, 9, 11])

    def test_sequence_window_stride_must_be_positive(self):
        for stride in (0, -1):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                build_sequence_windows(
                    self.features,
                    self.targets,
                    anchor_slice=self.anchor_slice,
                    window_size=self.window_size,
                    stride=stride,
                )

    def test_sequence_window_default_stride_preserves_daily_anchors(self):
        windows = build_sequence_windows(
            self.features,
            self.targets,
            anchor_slice=self.anchor_slice,
            window_size=self.window_size,
        )

        np.testing.assert_array_equal(
            windows.anchor_indices,
            [3, 4, 5, 6, 7, 8, 9, 10, 11],
        )

    def test_macro_feature_indices_follow_the_declared_column_order(self):
        from quant_pipeline.stage1 import macro_feature_indices

        indices = macro_feature_indices(DEFAULT_FEATURE_COLS)

        self.assertEqual(
            tuple(DEFAULT_FEATURE_COLS[index] for index in indices),
            (
                "FedRate",
                "FedRate_chg20",
                "FedRate_chg60",
                "TNX",
                "Yield_slope",
                "VIX",
                "VIX_percentile",
            ),
        )


class Stage1ModelContractTests(unittest.TestCase):
    def test_hybrid_lstm_returns_one_logit_per_input_row(self):
        from quant_pipeline.stage1 import HybridLSTM

        model = HybridLSTM()
        logits = model(
            torch.zeros((5, 4, len(DEFAULT_FEATURE_COLS))),
            torch.zeros((5, 7)),
        )

        self.assertEqual(tuple(logits.shape), (5,))

    def test_canonical_windows_have_exact_split_counts_and_date_alignment(self):
        from quant_pipeline.stage1 import build_stage1_windows

        dataset = prepare_dataset(PROJECT_ROOT / "datasets" / "nasdaq_multivariate.csv")
        windows = build_stage1_windows(dataset)
        dense_train = build_sequence_windows(
            dataset.features,
            dataset.targets,
            anchor_slice=dataset.splits.train,
            window_size=60,
            stride=1,
        )

        self.assertEqual(len(dense_train.anchor_indices), 6133)
        self.assertEqual(len(windows["validation"].anchor_indices), 884)
        self.assertEqual(len(windows["test"].anchor_indices), 1771)
        np.testing.assert_array_equal(np.diff(dense_train.anchor_indices), 1)
        np.testing.assert_array_equal(np.diff(windows["train"].anchor_indices), 3)
        self.assertEqual(
            tuple(dataset.dates[int(index)] for index in dense_train.anchor_indices),
            dataset.dates[59:6192],
        )
        self.assertEqual(
            tuple(dataset.dates[int(index)] for index in windows["validation"].anchor_indices),
            dataset.dates[6192:7076],
        )
        self.assertEqual(
            tuple(dataset.dates[int(index)] for index in windows["test"].anchor_indices),
            dataset.dates[7076:8847],
        )

    def test_seed_trainer_writes_aligned_finite_reproducible_predictions(self):
        from quant_pipeline.stage1 import fit_stage1_seed

        dataset = make_tiny_dataset()
        expected_artifacts = {
            "checkpoint.pt",
            "history.json",
            "predictions_train_fit.csv",
            "predictions_train_dense.csv",
            "predictions_validation.csv",
            "predictions_test.csv",
            "metrics.json",
            "manifest.json",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_dir = root / "first" / "seed-007"
            second_dir = root / "second" / "seed-007"
            kwargs = {
                "seed": 7,
                "device": "cpu",
                "epochs": 2,
                "batch_size": 4,
                "window_size": 4,
                "train_stride": 3,
                "early_stopping_patience": 2,
            }

            first_result = fit_stage1_seed(dataset, first_dir, **kwargs)
            second_result = fit_stage1_seed(dataset, second_dir, **kwargs)

            self.assertEqual({path.name for path in first_dir.iterdir()}, expected_artifacts)
            dense_dates, dense_probabilities = read_predictions(
                first_dir / "predictions_train_dense.csv"
            )
            fit_dates, fit_probabilities = read_predictions(
                first_dir / "predictions_train_fit.csv"
            )
            validation_dates, validation_probabilities = read_predictions(
                first_dir / "predictions_validation.csv"
            )
            test_dates, test_probabilities = read_predictions(
                first_dir / "predictions_test.csv"
            )

            self.assertEqual(dense_dates, list(dataset.dates[3:10]))
            self.assertEqual(fit_dates, list(dataset.dates[3:10:3]))
            self.assertEqual(validation_dates, list(dataset.dates[10:14]))
            self.assertEqual(test_dates, list(dataset.dates[14:18]))
            for probabilities in (
                dense_probabilities,
                fit_probabilities,
                validation_probabilities,
                test_probabilities,
            ):
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((0.0 <= probabilities) & (probabilities <= 1.0)).all())

            second_dense_dates, second_dense_probabilities = read_predictions(
                second_dir / "predictions_train_dense.csv"
            )
            self.assertEqual(second_dense_dates, dense_dates)
            np.testing.assert_array_equal(second_dense_probabilities, dense_probabilities)
            self.assertEqual(first_result["validation_loss"], second_result["validation_loss"])
            manifest = json.loads((first_dir / "manifest.json").read_text(encoding="utf-8"))
            history = json.loads((first_dir / "history.json").read_text(encoding="utf-8"))
            metrics = json.loads((first_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed"], 7)
            self.assertEqual(manifest["streams"]["train_dense"]["cadence"], 1)
            self.assertEqual(manifest["streams"]["train_fit"]["cadence"], 3)
            self.assertEqual(
                manifest["protocol"],
                {
                    "origin": "fixed",
                    "split_basis": "prediction_anchor",
                    "label_overlap_purged_at_boundaries": False,
                    "exact_retraining_at_each_boundary": False,
                    "qualification": (
                        "Fixed-origin, anchor-based, unpurged offline research contract; "
                        "not an exact retraining-at-boundary simulation."
                    ),
                },
            )
            self.assertEqual(metrics["validation_loss"], history["best_validation_loss"])

    def test_stage1_cli_exposes_canonical_arguments_with_only_task1_modules(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            isolated_root = Path(temporary_directory)
            isolated_package = isolated_root / "quant_pipeline"
            isolated_scripts = isolated_root / "scripts"
            isolated_package.mkdir()
            isolated_scripts.mkdir()
            (isolated_package / "__init__.py").write_text("", encoding="utf-8")
            for filename in ("data.py", "stage1.py"):
                shutil.copy2(PROJECT_ROOT / "quant_pipeline" / filename, isolated_package)
            shutil.copy2(PROJECT_ROOT / "scripts" / "train_stage1.py", isolated_scripts)

            completed = subprocess.run(
                [sys.executable, "scripts/train_stage1.py", "--help"],
                cwd=isolated_root,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--data",
            "--run-dir",
            "--seeds",
            "--device",
            "--epochs",
            "--batch-size",
            "--window-size",
            "--train-stride",
        ):
            self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
