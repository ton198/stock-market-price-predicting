import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quant_pipeline.backtest import probability_positions
from quant_pipeline.data import Standardizer, prepare_dataset
from quant_pipeline.evaluator import load_prediction_csv
from quant_pipeline.signals import fit_signal_normalizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELECTED_STAGE1 = (
    PROJECT_ROOT
    / "runs"
    / "canonical_retrain"
    / "20260905T053626Z"
    / "stage1-full"
    / "seed-119"
)


def synthetic_environment(**overrides):
    from quant_pipeline.dqn_env import DQNTradingEnv

    arguments = {
        "prices": np.linspace(100.0, 110.0, 40, dtype=np.float64),
        "extra_features": np.zeros((40, 3), dtype=np.float32),
        "signal": np.zeros(40, dtype=np.float32),
        "transaction_cost": 0.001,
        "slippage": 0.0005,
        "episode_length": 20,
        "seed": 7,
    }
    arguments.update(overrides)
    return DQNTradingEnv(**arguments)


def write_prediction(path, dates, probabilities):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Date", "probability"))
        writer.writerows(zip(dates, probabilities))


def write_synthetic_stage1(directory):
    directory.mkdir()
    streams = {
        "train_dense": (("d1", "d2", "d3"), (0.2, 0.4, 0.6)),
        "train_fit": (("d1", "d3"), (0.2, 0.6)),
        "validation": (("d4", "d5"), (0.3, 0.7)),
        "test": (("d6", "d7"), (0.4, 0.6)),
    }
    filenames = {
        name: f"predictions_{name}.csv"
        for name in streams
    }
    for name, (dates, probabilities) in streams.items():
        write_prediction(directory / filenames[name], dates, probabilities)
    manifest = {
        "hyperparameters": {"window_size": 2, "train_stride": 2},
        "streams": {
            name: {
                "filename": filenames[name],
                "rows": len(streams[name][0]),
                "cadence": 2 if name == "train_fit" else 1,
                "start_date": streams[name][0][0],
                "end_date": streams[name][0][-1],
                "role": (
                    "in-sample Stage 1 stream eligible for DQN"
                    if name == "train_dense"
                    else "optimizer-sampling diagnostic only"
                    if name == "train_fit"
                    else f"{name} inference"
                ),
            }
            for name in streams
        },
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return streams


def synthetic_dataset():
    raw_features = np.array(
        [
            [-0.04, 0.006, 0.0],
            [-0.03, 0.009, 1.0],
            [-0.02, 0.012, 0.0],
            [-0.01, 0.015, 1.0],
            [0.00, 0.018, 0.0],
            [0.01, 0.021, 1.0],
            [0.02, 0.024, 0.0],
            [0.03, 0.027, 1.0],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        dates=tuple(f"d{index}" for index in range(8)),
        close=np.linspace(100.0, 107.0, 8),
        targets=np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        features=raw_features.copy(),
        feature_names=("Return_lag5", "Volatility_20d", "Regime"),
        scaler=Standardizer(
            mean_=np.zeros(3, dtype=np.float32),
            scale_=np.ones(3, dtype=np.float32),
        ),
        splits=SimpleNamespace(
            train=slice(0, 4), validation=slice(4, 6), test=slice(6, 8)
        ),
    )


class DQNTradingEnvironmentTests(unittest.TestCase):
    def test_action_indices_map_to_the_exact_position_grid(self):
        for action, expected_position in ((0, -1.0), (7, -0.3), (10, 0.0), (20, 1.0)):
            with self.subTest(action=action):
                environment = synthetic_environment()
                environment.reset(options={"start": 0})
                _, _, _, _, info = environment.step(action)
                self.assertAlmostEqual(info["target_position"], expected_position)

    def test_actions_outside_the_discrete_grid_are_rejected(self):
        environment = synthetic_environment()
        environment.reset(options={"start": 0})

        for action in (-1, 21):
            with self.subTest(action=action), self.assertRaises(ValueError):
                environment.step(action)

    def test_position_change_charges_transaction_cost_and_slippage(self):
        environment = synthetic_environment(
            prices=np.full(40, 100.0),
            transaction_cost=0.01,
            slippage=0.005,
        )
        environment.reset(options={"start": 0})

        _, reward, _, _, info = environment.step(20)

        self.assertAlmostEqual(info["turnover"], 1.0)
        self.assertAlmostEqual(info["trading_cost"], 150.0)
        self.assertAlmostEqual(reward, -1.5)
        self.assertLess(info["portfolio_value"], 9_850.0)

    def test_reward_matches_the_historical_shaping_equation(self):
        environment = synthetic_environment(
            prices=np.array([100.0, 110.0], dtype=np.float64),
            signal=np.array([1.0, 0.0], dtype=np.float32),
            extra_features=np.array([[0.5, 0.4, 1.0], [0.5, 0.4, 1.0]]),
            transaction_cost=0.0,
            slippage=0.0,
            episode_length=1,
            signal_bonus_weight=20.0,
            vol_penalty_weight=0.0,
        )
        environment.reset(options={"start": 0})

        _, reward, terminated, _, _ = environment.step(20)

        self.assertAlmostEqual(reward, 12.0)
        self.assertTrue(terminated)

    def test_fixed_start_supports_both_endpoints_and_episode_cannot_overrun(self):
        environment = synthetic_environment()
        max_start = 40 - 20 - 1

        environment.reset(options={"start": 0})
        environment.reset(options={"start": max_start})
        for step in range(20):
            _, _, terminated, truncated, _ = environment.step(10)
            self.assertEqual(terminated, step == 19)
            self.assertFalse(truncated)
        with self.assertRaises(RuntimeError):
            environment.step(10)
        with self.assertRaises(ValueError):
            environment.reset(options={"start": max_start + 1})

    def test_observation_order_and_bounds_are_exact(self):
        extras = np.zeros((40, 3), dtype=np.float32)
        extras[0] = (0.25, 0.75, 1.0)
        signal = np.zeros(40, dtype=np.float32)
        signal[0] = 0.6
        environment = synthetic_environment(extra_features=extras, signal=signal)

        observation, _ = environment.reset(options={"start": 0})

        np.testing.assert_array_equal(
            environment.observation_space.low,
            np.array([0, 0, 0, -1, -1, 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            environment.observation_space.high,
            np.array([1, 1, 1, 1, 2, 1], dtype=np.float32),
        )
        np.testing.assert_allclose(observation, [0.6, 0.25, 0.75, 1.0, 0.0, 1.0])
        self.assertTrue(environment.observation_space.contains(observation))


class DQNObservationAssemblyTests(unittest.TestCase):
    def test_test_signal_uses_train_dense_normalizer_without_refitting(self):
        from scripts.train_dqn import build_dqn_observations

        dataset = synthetic_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "seed-119"
            streams = write_synthetic_stage1(artifacts)
            normalizer = fit_signal_normalizer(np.asarray(streams["train_dense"][1]))
            first = build_dqn_observations(dataset, artifacts, "test", normalizer)
            write_prediction(
                artifacts / "predictions_test.csv",
                streams["test"][0],
                (0.4, 0.99),
            )
            perturbed = build_dqn_observations(dataset, artifacts, "test", normalizer)

        self.assertAlmostEqual(float(first[2][0]), float(perturbed[2][0]))
        self.assertNotAlmostEqual(float(first[2][1]), float(perturbed[2][1]))
        self.assertAlmostEqual(normalizer.mean_, 0.4)

    def test_static_features_are_inverted_then_transformed_in_exact_order(self):
        from scripts.train_dqn import build_dqn_observations

        dataset = synthetic_dataset()
        dataset.features = (dataset.features - np.array([0.01, 0.003, 0.5])) / np.array(
            [0.02, 0.006, 0.5]
        )
        dataset.scaler = Standardizer(
            mean_=np.array([0.01, 0.003, 0.5], dtype=np.float32),
            scale_=np.array([0.02, 0.006, 0.5], dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "seed-119"
            streams = write_synthetic_stage1(artifacts)
            normalizer = fit_signal_normalizer(np.asarray(streams["train_dense"][1]))

            dates, prices, _, extras = build_dqn_observations(
                dataset, artifacts, "test", normalizer
            )

        self.assertEqual(dates, ("d6", "d7"))
        np.testing.assert_allclose(prices, [106.0, 107.0])
        expected_recent = 0.5 + 0.5 * np.tanh(
            np.array([106.0 / 101.0 - 1.0, 107.0 / 102.0 - 1.0]) / 0.05
        )
        np.testing.assert_allclose(extras[:, 0], expected_recent, rtol=1e-6)
        np.testing.assert_allclose(extras[:, 1], [0.8, 0.9], rtol=1e-6)
        np.testing.assert_allclose(extras[:, 2], [0.0, 1.0], rtol=1e-6)

    def test_selected_dense_train_stream_has_every_eligible_daily_anchor(self):
        from scripts.train_dqn import build_dqn_observations

        dataset = prepare_dataset(PROJECT_ROOT / "datasets" / "nasdaq_multivariate.csv")
        dense_dates, dense_probabilities = load_prediction_csv(
            SELECTED_STAGE1 / "predictions_train_dense.csv"
        )
        normalizer = fit_signal_normalizer(dense_probabilities)

        dates, prices, signal, extras = build_dqn_observations(
            dataset, SELECTED_STAGE1, "train", normalizer
        )

        self.assertEqual(len(dates), dataset.splits.train.stop - 59)
        self.assertEqual(dates, dense_dates)
        self.assertEqual(prices.shape, signal.shape)
        self.assertEqual(extras.shape, (len(dates), 3))

    def test_sparse_fit_stream_is_rejected_as_train_input(self):
        from scripts.train_dqn import build_dqn_observations

        dataset = synthetic_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "seed-119"
            streams = write_synthetic_stage1(artifacts)
            sparse_dates, sparse_probabilities = streams["train_fit"]
            write_prediction(
                artifacts / "predictions_train_dense.csv",
                sparse_dates,
                sparse_probabilities,
            )
            normalizer = fit_signal_normalizer(np.asarray(streams["train_dense"][1]))

            with self.assertRaisesRegex(ValueError, "dense train"):
                build_dqn_observations(dataset, artifacts, "train", normalizer)


class DQNTrainerTests(unittest.TestCase):
    def test_stable_baselines_model_uses_the_fixed_dqn_hyperparameters(self):
        from scripts.train_dqn import _make_dqn_model

        model = _make_dqn_model(synthetic_environment(), seed=42, device="cpu")

        self.assertEqual(model.buffer_size, 100_000)
        self.assertEqual(model.learning_starts, 5_000)
        self.assertEqual(model.batch_size, 128)
        self.assertEqual(model.learning_rate, 1e-4)
        self.assertEqual(model.gamma, 0.97)
        self.assertEqual(model.target_update_interval, 1_000)
        self.assertEqual(model.exploration_fraction, 0.3)
        self.assertEqual(model.exploration_final_eps, 0.05)
        self.assertEqual(model.train_freq.frequency, 4)
        self.assertEqual(model.policy.q_net.net_arch, [256, 256])

    def test_validation_windows_use_the_three_predeclared_starts(self):
        from scripts.train_dqn import validation_window_starts

        self.assertEqual(validation_window_starts(884), (0, 315, 631))
        with self.assertRaises(ValueError):
            validation_window_starts(252)

    def test_checkpoint_ties_keep_the_earliest_step(self):
        from scripts.train_dqn import checkpoint_is_better

        self.assertTrue(checkpoint_is_better(0.1, 10_000, None, None))
        self.assertFalse(checkpoint_is_better(0.1, 20_000, 0.1, 10_000))
        self.assertTrue(checkpoint_is_better(0.2, 20_000, 0.1, 10_000))

    def test_sequential_rollout_exports_every_anchor_position(self):
        from scripts.train_dqn import rollout_policy_actions

        class AlternatingPolicy:
            def __init__(self):
                self.calls = 0

            def predict(self, observation, deterministic=True):
                self.calls += 1
                return np.array(0 if self.calls % 2 else 20), None

        environment = synthetic_environment(
            prices=np.linspace(100.0, 102.0, 4),
            signal=np.full(4, 0.5, dtype=np.float32),
            extra_features=np.zeros((4, 3), dtype=np.float32),
            episode_length=3,
        )

        positions, shaped_reward = rollout_policy_actions(
            AlternatingPolicy(), environment, start=0
        )

        np.testing.assert_allclose(positions, [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(positions.size, 4)
        self.assertTrue(np.isfinite(shaped_reward))

    def test_stage1_summary_resolves_only_its_preselected_seed(self):
        from scripts.train_dqn import resolve_stage1_artifacts

        selected, selection = resolve_stage1_artifacts(SELECTED_STAGE1.parent)

        self.assertEqual(selected, SELECTED_STAGE1)
        self.assertEqual(selection["seed"], 119)
        self.assertEqual(
            selection["criterion"],
            "validation_loss ascending, seed ascending tie-break",
        )


class DQNEvaluatorTests(unittest.TestCase):
    def test_action_dates_and_position_bounds_are_rejected_before_backtest(self):
        from scripts.evaluate_dqn import evaluate_dqn_actions

        dataset = synthetic_dataset()
        stage1_dates = ("d6", "d7")
        probabilities = np.array([0.4, 0.6])
        with self.assertRaisesRegex(ValueError, "exactly match"):
            evaluate_dqn_actions(
                dataset,
                ("d7", "d6"),
                np.array([0.0, 0.0]),
                stage1_dates,
                probabilities,
            )
        with self.assertRaisesRegex(ValueError, r"\[-1, 1\]"):
            evaluate_dqn_actions(
                dataset,
                stage1_dates,
                np.array([0.0, 1.01]),
                stage1_dates,
                probabilities,
            )

    def test_shared_backtest_reports_dqn_and_all_three_baselines(self):
        from scripts.evaluate_dqn import evaluate_dqn_actions

        dataset = synthetic_dataset()
        dates = ("d6", "d7")
        probabilities = np.array([0.4, 0.6])
        positions = probability_positions(probabilities)

        report = evaluate_dqn_actions(
            dataset,
            dates,
            positions,
            dates,
            probabilities,
            diagnostics={"shaped_reward": 123.0, "validation_checkpoints": []},
        )

        self.assertEqual(report["status"], "research_only")
        self.assertEqual(report["n_action_anchors"], 2)
        self.assertEqual(report["n_backtest_intervals"], 1)
        self.assertEqual(
            set(report["strategies"]),
            {"dqn", "raw_stage1_probability_policy", "cash", "buy_and_hold"},
        )
        self.assertEqual(
            report["strategies"]["dqn"],
            report["strategies"]["raw_stage1_probability_policy"],
        )
        self.assertEqual(report["diagnostics"]["shaped_reward"], 123.0)
        self.assertEqual(report["backtest"]["position_range"], [-1.0, 1.0])

    def test_canonical_test_contract_has_1771_anchors_and_1770_intervals(self):
        from scripts.evaluate_dqn import evaluate_dqn_actions

        dataset = prepare_dataset(PROJECT_ROOT / "datasets" / "nasdaq_multivariate.csv")
        dates, probabilities = load_prediction_csv(
            SELECTED_STAGE1 / "predictions_test.csv"
        )

        report = evaluate_dqn_actions(
            dataset,
            dates,
            probability_positions(probabilities),
            dates,
            probabilities,
        )

        self.assertEqual(report["n_action_anchors"], 1771)
        self.assertEqual(report["n_backtest_intervals"], 1770)


if __name__ == "__main__":
    unittest.main()
