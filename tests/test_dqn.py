import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from quant_pipeline.backtest import probability_positions
from quant_pipeline.data import Standardizer
from quant_pipeline.signals import fit_signal_normalizer


PROTOCOL_QUALIFICATION = (
    "Fixed-origin, anchor-based, unpurged offline research protocol; "
    "not an exact retraining-at-boundary simulation."
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


def write_synthetic_stage1(directory, *, seed=119, write_summary=True):
    directory.mkdir()
    streams = {
        "train_dense": (("d5", "d6", "d7"), (0.2, 0.4, 0.6)),
        "train_fit": (("d5", "d7"), (0.2, 0.6)),
        "validation": (("d8", "d9"), (0.3, 0.7)),
        "test": (("d10", "d11"), (0.4, 0.6)),
    }
    filenames = {
        name: f"predictions_{name}.csv"
        for name in streams
    }
    for name, (dates, probabilities) in streams.items():
        write_prediction(directory / filenames[name], dates, probabilities)
    manifest = {
        "seed": seed,
        "hyperparameters": {"window_size": 6, "train_stride": 2},
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
    if write_summary:
        summary = {
            "stage1_selection": {
                "criterion": "validation_loss ascending, seed ascending tie-break",
                "seed": seed,
                "seed_directory": directory.name,
                "validation_loss": 0.5,
            }
        }
        (directory.parent / "summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
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
            [0.04, 0.030, 0.0],
            [0.05, 0.033, 1.0],
            [0.06, 0.024, 0.0],
            [0.07, 0.027, 1.0],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        dates=tuple(f"d{index}" for index in range(12)),
        close=np.linspace(100.0, 111.0, 12),
        targets=np.arange(12, dtype=np.int64) % 2,
        features=raw_features.copy(),
        feature_names=("Return_lag5", "Volatility_20d", "Regime"),
        scaler=Standardizer(
            mean_=np.zeros(3, dtype=np.float32),
            scale_=np.ones(3, dtype=np.float32),
        ),
        splits=SimpleNamespace(
            train=slice(0, 8), validation=slice(8, 10), test=slice(10, 12)
        ),
    )


def synthetic_evaluation_dataset(test_rows):
    rows = test_rows + 2
    return SimpleNamespace(
        dates=tuple(f"d{index}" for index in range(rows)),
        close=np.linspace(100.0, 120.0, rows),
        targets=np.arange(rows, dtype=np.int64) % 2,
        splits=SimpleNamespace(
            train=slice(0, 1), validation=slice(1, 2), test=slice(2, rows)
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
        extras[1] = (0.2, 0.3, 0.5)
        signal = np.zeros(40, dtype=np.float32)
        signal[0] = 0.6
        environment = synthetic_environment(
            prices=np.full(40, 100.0), extra_features=extras, signal=signal
        )

        observation, _ = environment.reset(options={"start": 0})

        np.testing.assert_array_equal(
            environment.observation_space.low,
            np.array([0, 0, 0, -1, -1, 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            environment.observation_space.high,
            np.array([1, 1, 1, 1, 2, 1], dtype=np.float32),
        )
        np.testing.assert_allclose(observation, [0.6, 0.25, 0.75, 0.0, 1.0, 1.0])
        self.assertTrue(environment.observation_space.contains(observation))
        short_observation, _, _, _, _ = environment.step(0)
        self.assertLess(float(short_observation[3]), 0.0)
        self.assertGreater(float(short_observation[4]), 1.0)
        self.assertAlmostEqual(float(short_observation[5]), 0.5)

    def test_environment_rejects_nan_numeric_configuration(self):
        for name in (
            "initial_cash",
            "transaction_cost",
            "slippage",
            "episode_length",
            "signal_bonus_weight",
            "vol_penalty_weight",
            "margin_rate",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                synthetic_environment(**{name: float("nan")})


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

        self.assertEqual(dates, ("d10", "d11"))
        np.testing.assert_allclose(prices, [110.0, 111.0])
        expected_recent = 0.5 + 0.5 * np.tanh(
            np.array([110.0 / 105.0 - 1.0, 111.0 / 106.0 - 1.0]) / 0.05
        )
        np.testing.assert_allclose(extras[:, 0], expected_recent, rtol=1e-6)
        np.testing.assert_allclose(extras[:, 1], [0.8, 0.9], rtol=1e-6)
        np.testing.assert_allclose(extras[:, 2], [0.0, 1.0], rtol=1e-6)

    def test_selected_dense_train_stream_has_every_eligible_daily_anchor(self):
        from scripts.train_dqn import build_dqn_observations

        dataset = synthetic_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "seed-119"
            streams = write_synthetic_stage1(artifacts)
            normalizer = fit_signal_normalizer(
                np.asarray(streams["train_dense"][1])
            )

            dates, prices, signal, extras = build_dqn_observations(
                dataset, artifacts, "train", normalizer
            )

        self.assertEqual(len(dates), 3)
        self.assertEqual(dates, streams["train_dense"][0])
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

        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            selected_artifacts = run_directory / "seed-119"
            write_synthetic_stage1(selected_artifacts)

            selected, selection = resolve_stage1_artifacts(run_directory)

        self.assertEqual(selected, selected_artifacts)
        self.assertEqual(selection["seed"], 119)
        self.assertEqual(
            selection["criterion"],
            "validation_loss ascending, seed ascending tie-break",
        )

    def test_arbitrary_seed_directory_without_selection_summary_is_rejected(self):
        from scripts.train_dqn import resolve_stage1_artifacts

        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary) / "seed-051"
            write_synthetic_stage1(artifacts, seed=51, write_summary=False)

            with self.assertRaisesRegex(ValueError, "summary"):
                resolve_stage1_artifacts(artifacts)

    def test_stage1_summary_rejects_a_noncanonical_selection_criterion(self):
        from scripts.train_dqn import resolve_stage1_artifacts

        with tempfile.TemporaryDirectory() as temporary:
            run_directory = Path(temporary)
            artifacts = run_directory / "seed-119"
            write_synthetic_stage1(artifacts)
            summary = json.loads((run_directory / "summary.json").read_text())
            summary["stage1_selection"]["criterion"] = "test return descending"
            (run_directory / "summary.json").write_text(json.dumps(summary))

            with self.assertRaisesRegex(ValueError, "criterion"):
                resolve_stage1_artifacts(run_directory)

    def test_manifest_observation_names_follow_environment_slot_order(self):
        from scripts.train_dqn import OBSERVATION_NAMES

        self.assertEqual(
            OBSERVATION_NAMES,
            (
                "normalized_signal",
                "recent_5d_norm",
                "vol_norm",
                "position_ratio",
                "cash_ratio",
                "Regime",
            ),
        )

    def test_training_manifest_metadata_qualifies_the_research_protocol(self):
        from scripts.train_dqn import canonical_manifest_metadata

        metadata = canonical_manifest_metadata()

        self.assertEqual(
            metadata["protocol"]["qualification"], PROTOCOL_QUALIFICATION
        )
        self.assertEqual(metadata["status"], "research_only")
        self.assertFalse(metadata["portfolio_publication_allowed"])


class DQNEvaluatorTests(unittest.TestCase):
    def test_action_dates_and_position_bounds_are_rejected_before_backtest(self):
        from scripts.evaluate_dqn import evaluate_dqn_actions

        dataset = synthetic_dataset()
        stage1_dates = ("d10", "d11")
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
        dates = ("d10", "d11")
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
        self.assertEqual(report["protocol"]["qualification"], PROTOCOL_QUALIFICATION)

    def test_noncanonical_costs_require_matching_reviewed_alternative(self):
        from scripts.evaluate_dqn import evaluate_dqn_actions

        dataset = synthetic_dataset()
        dates = ("d10", "d11")
        probabilities = np.array([0.4, 0.6])
        positions = probability_positions(probabilities)
        for transaction_cost, slippage in ((0.0, 0.0005), (0.001, 0.0)):
            with self.subTest(
                transaction_cost=transaction_cost, slippage=slippage
            ), self.assertRaisesRegex(ValueError, "reviewed alternative"):
                evaluate_dqn_actions(
                    dataset,
                    dates,
                    positions,
                    dates,
                    probabilities,
                    transaction_cost=transaction_cost,
                    slippage=slippage,
                )

        report = evaluate_dqn_actions(
            dataset,
            dates,
            positions,
            dates,
            probabilities,
            transaction_cost=0.0,
            slippage=0.0,
            diagnostics={
                "reviewed_alternative": {
                    "approved": True,
                    "review_reference": "cost-model-review-7",
                    "transaction_cost": 0.0,
                    "slippage": 0.0,
                }
            },
        )
        self.assertEqual(report["backtest"]["transaction_cost"], 0.0)
        self.assertEqual(report["backtest"]["slippage"], 0.0)

    def test_canonical_test_contract_has_1771_anchors_and_1770_intervals(self):
        from scripts.evaluate_dqn import evaluate_dqn_actions

        dataset = synthetic_evaluation_dataset(1771)
        dates = tuple(dataset.dates[dataset.splits.test])
        probabilities = np.linspace(0.25, 0.75, 1771)

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
