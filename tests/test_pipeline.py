import unittest
from pathlib import Path

import numpy as np

from quant_pipeline.backtest import simulate_positions
from quant_pipeline.data import (
    chronological_slices,
    fit_standardizer,
    fit_target_threshold,
    make_forward_target,
    build_sequence_windows,
)
from quant_pipeline.signals import fit_signal_normalizer, normalize_signal
from quant_pipeline.evaluator import align_test_predictions, evaluate_probability_stream
from quant_pipeline.data import prepare_dataset
from quant_pipeline.metrics import strategy_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineDataTests(unittest.TestCase):
    def test_checked_in_dataset_uses_the_canonical_label_contract(self):
        dataset = prepare_dataset(PROJECT_ROOT / "datasets" / "nasdaq_multivariate.csv")

        self.assertEqual(dataset.horizon, 20)
        self.assertEqual(dataset.close.size, 8847)
        self.assertEqual(dataset.features.shape, (8847, 30))
        self.assertIn("Momentum_252d", dataset.feature_names)
        self.assertIn("VIX_percentile", dataset.feature_names)
        self.assertLessEqual(dataset.splits.train.stop, dataset.splits.validation.start)
        self.assertLessEqual(dataset.splits.validation.stop, dataset.splits.test.start)
        self.assertTrue(np.isfinite(dataset.threshold))

    def test_chronological_slices_cover_rows_without_overlap(self):
        slices = chronological_slices(10, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        self.assertEqual(slices.train, slice(0, 6))
        self.assertEqual(slices.validation, slice(6, 8))
        self.assertEqual(slices.test, slice(8, 10))

    def test_target_threshold_is_fit_on_training_rows_only(self):
        forward_returns = np.array([0.01, 0.02, 0.03, 0.04, 1.0, 1.1])
        train_slice = slice(0, 4)

        threshold = fit_target_threshold(forward_returns, train_slice)
        targets = make_forward_target(forward_returns, threshold)

        self.assertAlmostEqual(threshold, 0.025)
        np.testing.assert_array_equal(targets, np.array([0, 0, 1, 1, 1, 1]))

    def test_standardizer_uses_training_statistics_only(self):
        values = np.array([[1.0], [3.0], [100.0]])

        scaler = fit_standardizer(values, slice(0, 2))
        transformed = scaler.transform(values)

        self.assertAlmostEqual(float(scaler.mean_[0]), 2.0)
        self.assertAlmostEqual(float(scaler.scale_[0]), 1.0)
        self.assertAlmostEqual(float(transformed[2, 0]), 98.0)

    def test_sequence_windows_use_past_context_for_each_anchor(self):
        features = np.arange(6, dtype=np.float32).reshape(-1, 1)
        targets = np.arange(10, 16, dtype=np.int64)

        windows = build_sequence_windows(
            features,
            targets,
            anchor_slice=slice(4, 6),
            window_size=3,
        )

        np.testing.assert_array_equal(windows.features[0, :, 0], np.array([2, 3, 4]))
        np.testing.assert_array_equal(windows.features[1, :, 0], np.array([3, 4, 5]))
        np.testing.assert_array_equal(windows.targets, np.array([14, 15]))

    def test_signal_normalizer_reuses_training_statistics_on_test_values(self):
        normalizer = fit_signal_normalizer(np.array([0.2, 0.4, 0.6, 0.8]))
        test_values = np.array([0.9, 0.95])

        normalized = normalize_signal(test_values, normalizer)

        self.assertAlmostEqual(normalizer.mean_, 0.5)
        self.assertAlmostEqual(normalizer.scale_, np.std([0.2, 0.4, 0.6, 0.8]))
        self.assertGreater(float(normalized[1]), float(normalized[0]))
        self.assertNotAlmostEqual(float(normalized.mean()), 0.5)


class BacktestTests(unittest.TestCase):
    def test_strategy_metrics_include_downside_risk(self):
        metrics = strategy_metrics(np.array([100.0, 110.0, 100.0, 120.0]))

        self.assertIn("sortino", metrics)
        self.assertTrue(np.isfinite(metrics["sortino"]))

    def test_positions_are_applied_to_next_price_interval_and_costs_are_charged(self):
        prices = np.array([100.0, 110.0, 100.0])
        positions = np.array([1.0, 0.0, 0.0])

        equity = simulate_positions(
            prices,
            positions,
            initial_equity=10_000.0,
            transaction_cost=0.01,
            slippage=0.0,
        )

        # Long exposure earns the first interval's return.  Flattening at t=1
        # incurs one more turnover charge before the second interval.
        self.assertAlmostEqual(float(equity[0]), 10_000.0)
        self.assertAlmostEqual(float(equity[1]), 10_000.0 * (1.0 + 0.10 - 0.01))
        self.assertAlmostEqual(float(equity[2]), float(equity[1]) * (1.0 - 0.01))


class EvaluationTests(unittest.TestCase):
    def test_prediction_dates_must_match_test_anchors_exactly(self):
        dataset = type(
            "Dataset",
            (),
            {
                "dates": ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"),
                "targets": np.array([0, 1, 0, 1]),
                "close": np.array([100.0, 101.0, 100.0, 102.0]),
                "splits": type("Splits", (), {"test": slice(2, 4)})(),
            },
        )()

        with self.assertRaises(ValueError):
            align_test_predictions(dataset, ["2024-01-03"], np.array([0.6]))

    def test_probability_stream_reports_classification_and_costed_strategy_metrics(self):
        dataset = type(
            "Dataset",
            (),
            {
                "dates": ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"),
                "targets": np.array([0, 1, 1, 0]),
                "close": np.array([100.0, 101.0, 102.0, 101.0]),
                "splits": type("Splits", (), {"test": slice(2, 4)})(),
            },
        )()

        report = evaluate_probability_stream(
            dataset,
            ["2024-01-03", "2024-01-04"],
            np.array([0.8, 0.2]),
            transaction_cost=0.0,
            slippage=0.0,
        )

        self.assertAlmostEqual(report["classification"]["accuracy"], 1.0)
        self.assertIn("strategy", report)
        self.assertIn("buy_and_hold", report)


if __name__ == "__main__":
    unittest.main()
