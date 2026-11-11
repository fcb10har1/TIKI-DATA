"""Actual motion/missingness invariants for C and D."""
import unittest

import numpy as np

from tiki_data.contracts import FeatureBatch, MatchSpec
from tiki_data.p1.features import FEATURE_NAMES
from tiki_data.p1.kalman import DEFAULT_SETTINGS, augment, estimate_motion, motion_schema
from tiki_data.p2.data import encode, fit_normalization, hide_observations, validate_schema
from tiki_data.p2.motion_experiment import gap_mask, hide_positions


class MotionTests(unittest.TestCase):
    def fixture(self):
        spec = MatchSpec("motion_fixture", 500, 25, ((0, 250), (250, 500)))
        values = np.random.default_rng(42).uniform(1, 5, (500, 19)).astype(np.float32)
        base = FeatureBatch(spec, np.arange(500), values, np.ones_like(values, dtype=bool),
                            FEATURE_NAMES, spec.period_ids())
        positions = np.column_stack((np.arange(500)*.2, np.arange(500)*.12))
        return base, positions

    def test_c_only_drops_scalar_speed_and_flag_for_entire_window(self):
        base, _ = self.fixture()
        values, observed = hide_observations(base.values, base.observed, np.random.default_rng(3),
                                             {"speed_window_probability":1.}, [], 0)
        self.assertFalse(observed[:, 0].any())
        self.assertTrue((values[:, 0] == 0).all())
        np.testing.assert_array_equal(values[:, 1:], base.values[:, 1:])
        np.testing.assert_array_equal(observed[:, 1:], base.observed[:, 1:])
        self.assertTrue(base.observed.all())

    def test_c_mixes_available_and_missing_windows_reproducibly(self):
        base, _ = self.fixture()
        def sequence():
            rng = np.random.default_rng(42)
            return [hide_observations(base.values[:10], base.observed[:10], rng,
                                     {"speed_window_probability":.5}, [], 0)[1][:, 0].all() for _ in range(100)]
        self.assertEqual(sequence(), sequence())
        self.assertTrue(30 < sum(sequence()) < 70)
        v, o = hide_observations(base.values, base.observed, np.random.default_rng(2), {}, [], 0)
        np.testing.assert_array_equal(v, base.values)
        np.testing.assert_array_equal(o, base.observed)

    def test_kalman_tracks_known_constant_velocity_without_scalar_speed(self):
        base, positions = self.fixture()
        derived = augment(base, positions)
        np.testing.assert_allclose(derived.values[80:240, 19], np.sqrt(5**2+3**2), atol=.01)
        self.assertTrue(derived.observed[80:240, 19].all())
        self.assertFalse(derived.observed[:, 0].any())
        np.testing.assert_array_equal(derived.values[:, 1:19], base.values[:, 1:])
        np.testing.assert_array_equal(derived.observed[:, 1:19], base.observed[:, 1:])
        changed = FeatureBatch(base.match, base.frames, base.values*100, base.observed, base.feature_names, base.periods)
        np.testing.assert_array_equal(augment(changed, positions).values[:, 19:], derived.values[:, 19:])
        validate_schema(motion_schema(DEFAULT_SETTINGS), derived.feature_names)

    def test_filter_is_causal_and_resets_at_halves(self):
        base, positions = self.fixture()
        original = estimate_motion(positions, base.match)
        changed = positions.copy()
        changed[130:250] += 30
        future = estimate_motion(changed, base.match)
        for a, b in zip(original, future):
            np.testing.assert_array_equal(a[:130], b[:130])
            np.testing.assert_array_equal(a[250:], b[250:])
        self.assertFalse(original[1][250:252, 0].any())

    def test_short_gaps_long_gaps_and_large_jump_are_explicitly_invalid(self):
        base, positions = self.fixture()
        positions[100:140] = np.nan
        positions[200] += 1000
        values, valid = estimate_motion(positions, base.match)
        self.assertTrue(valid[100:105, 0].all())
        self.assertFalse(valid[113:142, 0].any())
        self.assertFalse(valid[200:203, 0].any())
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values[~valid] == 0).all())
        self.assertTrue((values[:, 2] <= DEFAULT_SETTINGS["max_prediction_s"]).all())

    def test_d_normalization_finite_when_all_recorded_speed_and_inference_positions_missing(self):
        base, positions = self.fixture()
        train = augment(base, positions)
        norm = fit_normalization([train], allow_unobserved=("ball_speed_mps",))
        self.assertEqual((norm["mean"][0], norm["scale"][0], norm["observed_count"][0]), (0, 1, 0))
        hidden, raw = hide_positions(base, positions, np.ones(base.match.n_frames, dtype=bool))
        inference = augment(hidden, raw)
        tensor = encode(inference.values, inference.observed, norm)
        self.assertTrue(np.isfinite(tensor).all())
        self.assertTrue((tensor[[0, 22]] == 0).all())
        self.assertFalse(inference.observed[:, 19:].any())
        np.testing.assert_array_equal(inference.frames, base.frames)
        np.testing.assert_array_equal(inference.values[:, 11:19], base.values[:, 11:])

    def test_external_gap_masks_and_recomputed_motion_do_not_use_hidden_positions(self):
        base, positions = self.fixture()
        mask = gap_mask(base.match, 2)
        np.testing.assert_array_equal(mask, gap_mask(base.match, 2))
        changed = positions.copy()
        changed[mask] += 1000
        hidden, raw = hide_positions(base, positions, mask)
        _, raw_changed = hide_positions(base, changed, mask)
        np.testing.assert_array_equal(augment(hidden, raw).values, augment(hidden, raw_changed).values)
        self.assertFalse(hidden.observed[mask, :11].any())
        np.testing.assert_array_equal(hidden.values[:, 11:], base.values[:, 11:])


if __name__ == "__main__":
    unittest.main()
