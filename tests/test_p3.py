import itertools
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tiki_data.artifacts import write_batch
from tiki_data.contracts import CLASS_ORDER, FeatureBatch, MatchSpec, ProbabilityBatch
from tiki_data.p3.decoder import decode
from tiki_data.p3.rules import predict_features


NAMES = ("home_nearest_ball_m", "away_nearest_ball_m", "ball_speed_mps",
         "home_mean_speed_mps", "away_mean_speed_mps")


def features(spec):
    values = np.tile([1., 5., 4., 2., 2.], (spec.n_frames, 1))
    return FeatureBatch(spec, np.arange(spec.n_frames), values, np.ones(values.shape, bool),
                        NAMES, spec.period_ids())


class RulesTests(unittest.TestCase):
    def test_control_requires_persistence_and_loose_ball_retains_owner(self):
        batch = features(MatchSpec("control", 30, 25))
        batch.values[7:10, :2] = [5, 1]  # Short opposing touch must not change possession.
        batch.values[10:15, :2] = [5, 5]  # Contested ball retains previous team.
        batch.observed[15:20] = False
        batch.values[15:20] = 0  # Missing observations do not imply dead ball.
        batch.values[20:, :2] = [5, 1]
        before = batch.values.copy()
        result = predict_features(batch, {}).labels
        self.assertTrue(np.all(result[4:24] == "H"))
        self.assertTrue(np.all(result[24:] == "A"))
        np.testing.assert_array_equal(before, batch.values)

    def test_stationary_proxy_needs_observations_and_resets_at_halftime(self):
        spec = MatchSpec("stoppage", 120, 25, ((0, 60), (65, 120)))
        batch = features(spec)
        batch.values[:55, 2:] = 0
        batch.values[65:, :2] = [5, 1]
        batch.observed[80:] = False
        batch.values[80:] = 0
        result = predict_features(batch, {}).labels
        self.assertTrue(np.all(result[:55] == "D"))
        self.assertTrue(np.all(result[60:69] == "X"))
        self.assertTrue(np.all(result[69:] == "A"))

    def test_missing_required_channels_and_invalid_settings_fail(self):
        batch = features(MatchSpec("invalid", 10, 25))
        with self.assertRaises(ValueError):
            predict_features(batch, {"control_seconds": float("nan")})
        with self.assertRaises(ValueError):
            predict_features(batch, {"control_seconds": True})
        batch = replace(batch, feature_names=tuple("other_" + str(i) for i in range(5)))
        with self.assertRaises(ValueError):
            predict_features(batch, {})


class DecoderTests(unittest.TestCase):
    def test_viterbi_matches_exhaustive_global_optimum(self):
        p = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7], [.7, .2, .1]])
        spec = MatchSpec("exact", len(p), 25)
        batch = ProbabilityBatch(spec, np.arange(len(p)), p, spec.period_ids())
        settings = {"switch_cost": 1.1, "dead_cost": .4}
        result = decode(batch, {}, settings).labels
        costs = np.array([[0, 1.1, .4], [1.1, 0, .4], [.4, .4, 0]])

        def score(path):
            return sum(np.log(p[i, state]) for i, state in enumerate(path)) - sum(
                costs[a, b] for a, b in zip(path, path[1:]))

        actual = tuple(CLASS_ORDER.index(label) for label in result)
        self.assertAlmostEqual(score(actual), max(score(path) for path in itertools.product(range(3), repeat=4)))

    def test_period_reset_outside_x_and_readonly_inputs(self):
        spec = MatchSpec("halves", 7, 25, ((0, 3), (4, 7)))
        p = np.tile([.99, .005, .005], (7, 1))
        p[4:] = [.005, .99, .005]
        p.setflags(write=False)
        batch = ProbabilityBatch(spec, np.arange(7), p, spec.period_ids())
        result = decode(batch, {}, {"switch_cost": 100}).labels
        np.testing.assert_array_equal(result, list("HHHXAAA"))

    def test_flicker_suppression_without_losing_sustained_change(self):
        spec = MatchSpec("flicker", 60, 25)
        p = np.tile([.9, .05, .05], (60, 1))
        p[10] = [.05, .9, .05]
        p[30:] = [.05, .9, .05]
        result = decode(ProbabilityBatch(spec, np.arange(60), p, spec.period_ids()), {}, {}).labels
        self.assertTrue(np.all(result[:30] == "H"))
        self.assertTrue(np.all(result[30:] == "A"))

    def test_rule_fusion_and_rejecting_hidden_or_invalid_inputs(self):
        spec = MatchSpec("fusion", 12, 25)
        batch = ProbabilityBatch(spec, np.arange(12), np.full((12, 3), 1/3), spec.period_ids())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            write_batch(path, features(spec))
            result = decode(batch, {"features": path}, {"rule_weight": 1}).labels
            self.assertTrue(np.all(result == "H"))
        for settings in ({"switch_cost": -1}, {"dead_cost": float("inf")}, {"rule_weight": 2}, {"typo": 1}):
            with self.assertRaises(ValueError):
                decode(batch, {}, settings)
        with self.assertRaises(ValueError):
            decode(batch, {}, {"rule_weight": .5})


if __name__ == "__main__":
    unittest.main()
