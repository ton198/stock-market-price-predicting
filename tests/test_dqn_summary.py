"""Focused aggregation checks independent of DQN training implementation."""
import unittest

import numpy as np

from scripts.summarize_dqn import _aggregate_strategy_metrics, _instability_note


class SummaryAggregationTests(unittest.TestCase):
    def test_all_five_seeds_receive_equal_weight_and_population_dispersion(self):
        returns = [-0.5, -0.1, 0.0, 0.2, 0.9]
        rows = [{"strategies": {"dqn": {"total_return": value}}} for value in returns]
        metrics = _aggregate_strategy_metrics(rows, "dqn")["total_return"]
        self.assertAlmostEqual(metrics["mean"], 0.1)
        self.assertAlmostEqual(metrics["std"], np.sqrt(0.212))
        self.assertEqual(metrics["median"], 0.0)
        self.assertTrue(_instability_note(returns)["detected"])

    def test_nonfinite_metrics_cannot_enter_aggregate(self):
        rows = [{"strategies": {"dqn": {"total_return": float("nan")}}}]
        with self.assertRaisesRegex(ValueError, "finite"):
            _aggregate_strategy_metrics(rows, "dqn")


if __name__ == "__main__":
    unittest.main()
