"""P2 regression checks: leakage, alignment, boundaries, actual training/reload."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tiki_data.artifacts import read_batch, read_json, write_json
from tiki_data.contracts import FeatureBatch, MatchSpec
from tiki_data.p1.features import apply_feature_policy
from tiki_data.p2.data import (Windows, encode, fit_normalization, hide_observations,
                               idsse_targets, validate_schema)
from tiki_data.p2.inference import predict, predict_batch
from tiki_data.p2.model import TCN, seed_all
from tiki_data.p2.smoke import prepare
from tiki_data.p2.train import load_data, load_training_config, train
from tiki_data.pipeline import reproduce, run


class P2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        seed_all(42)

    def batch(self):
        match = MatchSpec("alignment_fixture", 31, 25, ((2, 13), (13, 28)))
        values = np.arange(62, dtype=np.float32).reshape(31, 2)
        observed = np.ones_like(values, dtype=bool)
        observed[5, 0] = False
        values[5, 0] = 0
        return FeatureBatch(match, np.arange(31), values, observed,
                            ("ball_speed_mps", "home_width_m"), match.period_ids())

    def test_normalization_observed_train_periods_only(self):
        batch = self.batch()
        norm = fit_normalization([batch])
        for j in range(2):
            x = batch.values[(batch.periods > 0) & batch.observed[:, j], j]
            self.assertAlmostEqual(norm["mean"][j], float(x.mean()), places=5)
            self.assertEqual(norm["observed_count"][j], len(x))
        x = encode(batch.values, batch.observed, norm)
        self.assertEqual(x[0, 5], 0)
        self.assertEqual(x[2, 5], 0)
        self.assertEqual(x[2, 4], 1)
        with self.assertRaisesRegex(ValueError, "observed training"):
            empty = FeatureBatch(batch.match, batch.frames, np.zeros_like(batch.values),
                                 np.zeros_like(batch.observed), batch.feature_names, batch.periods)
            fit_normalization([empty])

    def test_windows_cover_each_label_once_and_never_cross_halves(self):
        batch = self.batch()
        norm = fit_normalization([batch])
        y = np.arange(31) % 3
        y[batch.periods == 0] = -100
        windows = Windows([(batch, y)], norm, 8, 3)
        seen = []
        for i, (_, left, right, start, stop) in enumerate(windows.index):
            x, target = windows[i]
            self.assertEqual(x.shape, (4, 14))
            np.testing.assert_array_equal(target[:stop-start], y[start:stop])
            self.assertTrue(np.all(target[stop-start:].numpy() == -100))
            if start == left:
                self.assertTrue(torch.all(x[:, :3] == 0))
            seen.extend(range(start, stop))
        self.assertEqual(seen, np.flatnonzero(batch.periods > 0).tolist())

    def test_entire_speed_channel_missing_has_neutral_finite_normalization(self):
        batch = self.batch()
        masked = apply_feature_policy(batch, True)
        norm = fit_normalization([masked], allow_unobserved=("ball_speed_mps",))
        self.assertEqual((norm["mean"][0], norm["scale"][0], norm["observed_count"][0]), (0., 1., 0))
        original = fit_normalization([batch])
        for field in ("mean", "scale", "observed_count"):
            self.assertEqual(norm[field][1:], original[field][1:])
        for fitted in (norm, original):
            encoded = encode(masked.values, masked.observed, fitted)
            self.assertTrue(np.isfinite(encoded).all())
            self.assertTrue((encoded[[0, 2]] == 0).all())
        empty = FeatureBatch(batch.match, batch.frames, np.zeros_like(batch.values),
                             np.zeros_like(batch.observed), batch.feature_names, batch.periods)
        with self.assertRaisesRegex(ValueError, "observed training"):
            fit_normalization([empty], allow_unobserved=("ball_speed_mps",))

    def test_masked_training_embeds_inference_policy_and_never_opens_test(self):
        prepare(self.root / "inputs")
        path = self.root / "inputs/train.json"
        config = read_json(path)
        config["p1"] = {"mask_ball_speed": True}
        config["features"]["p2_synthetic_test"] = "unavailable_test.npz"
        write_json(path, config)
        report = train(path, self.root / "masked_training")
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["test_accessed"])
        norm = read_json(self.root / "masked_training/normalization.json")
        self.assertTrue(norm["mask_ball_speed"])
        j = norm["feature_names"].index("ball_speed_mps")
        self.assertEqual(norm["observed_count"][j], 0)
        self.assertEqual(norm["train_match_ids"], ["p2_synthetic_train"])
        manifest = read_json(self.root / "inputs/manifest.json")
        spec = MatchSpec.from_dict(manifest["matches"][1]["match"])
        batch = read_batch(self.root / "inputs/p2_synthetic_validation_features.npz", spec, "features")
        inputs = {"checkpoint": self.root / "masked_training/best.pt"}
        settings = {"allow_synthetic": True}
        original = predict(batch, inputs, settings)
        hidden = predict(apply_feature_policy(batch, True), inputs, settings)
        changed = batch.values.copy()
        changed[batch.observed[:, j], j] = 9999
        altered = FeatureBatch(spec, batch.frames, changed, batch.observed, batch.feature_names, batch.periods)
        explicit_false = predict(altered, inputs, dict(settings, mask_ball_speed=False))
        np.testing.assert_array_equal(original.probabilities, hidden.probabilities)
        np.testing.assert_array_equal(original.probabilities, explicit_false.probabilities)
        np.testing.assert_array_equal(original.frames, batch.frames)

    def test_chunk_reconstruction_and_boundary_independence(self):
        batch = self.batch()
        norm = fit_normalization([batch])
        model = TCN(2, width=4, dilations=[1, 2], dropout=0)
        p = predict_batch(batch, model, norm, core=7)
        other = predict_batch(batch, model, norm, core=29)
        np.testing.assert_allclose(p.probabilities, other.probabilities, atol=2e-7, rtol=0)
        changed = batch.values.copy()
        changed[13:] += 1000
        altered = FeatureBatch(batch.match, batch.frames, changed, batch.observed,
                               batch.feature_names, batch.periods)
        q = predict_batch(altered, model, norm, core=7)
        np.testing.assert_array_equal(p.probabilities[:13], q.probabilities[:13])
        self.assertEqual(p.probabilities.shape, (31, 3))
        p.validate(batch.match)

    def test_hidden_ball_masks_all_dependent_channels_and_preserves_other_inputs(self):
        values = np.ones((100, 4), dtype=np.float32)
        observed = np.ones_like(values, dtype=bool)
        aug = {"point_probability": 0, "ball_stretch_probability": 1, "ball_stretch_frames": [25, 25]}
        v, o = hide_observations(values, observed, np.random.default_rng(42), aug, [0, 1, 2])
        self.assertEqual(int((~o[:, 0]).sum()), 25)
        np.testing.assert_array_equal(o[:, 0], o[:, 1])
        self.assertTrue(o[:, 3].all())
        self.assertTrue((v[~o] == 0).all())
        self.assertTrue(observed.all())
        self.assertTrue((values == 1).all())

    def test_idsse_target_mapping_does_not_turn_unknown_live_into_dead(self):
        np.testing.assert_array_equal(idsse_targets([1, 2, 0, 2, np.nan, 1], [1, 1, 1, 0, 0, np.nan]),
                                      ["H", "A", "X", "D", "D", "X"])

    def test_schema_rejects_targets_and_track_identities(self):
        prepare(self.root / "inputs")
        schema = read_json(self.root / "inputs/feature_schema.json")
        names = [item["name"] for item in schema["features"]]
        validate_schema(schema, names)
        for forbidden in ("BallPossession", "BallStatus", "possession_label", "slot_12_x", "smart_tagging"):
            bad = copy.deepcopy(schema)
            bad["features"][0]["sources"] = [forbidden]
            with self.assertRaises(ValueError):
                validate_schema(bad, names)

    def test_manifest_group_leakage_rejected_before_windowing(self):
        prepare(self.root / "inputs")
        config = load_training_config(self.root / "inputs/train.json")
        manifest = read_json(self.root / "inputs/manifest.json")
        manifest["matches"][1]["group_id"] = manifest["matches"][0]["group_id"]
        write_json(self.root / "inputs/manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "underlying match/group"):
            load_data(config)

    def test_train_reload_and_p4_inference_without_test_access(self):
        prepare(self.root / "inputs")
        path = self.root / "inputs/train.json"
        config = read_json(path)
        # Deliberately invalid test paths prove training does not open the final holdout.
        config["features"]["p2_synthetic_test"] = "does_not_exist.npz"
        write_json(path, config)
        manifest = read_json(self.root / "inputs/manifest.json")
        manifest["matches"][2]["ground_truth"] = "does_not_exist.csv"
        write_json(self.root / "inputs/manifest.json", manifest)
        report = train(path, self.root / "training")
        self.assertEqual(report["status"], "complete")
        self.assertTrue(report["checkpoint_reload_verified"])
        self.assertFalse(report["accuracy_evidence"])
        self.assertFalse(report["test_accessed"])
        norm = read_json(self.root / "training/normalization.json")
        self.assertEqual(norm["train_match_ids"], ["p2_synthetic_train"])
        match = MatchSpec.from_dict(manifest["matches"][1]["match"])
        feature_path = self.root / "inputs/p2_synthetic_validation_features.npz"
        features = read_batch(feature_path, match, "features")
        with self.assertRaisesRegex(ValueError, "synthetic weights"):
            predict(features, {"checkpoint": self.root / "training/best.pt"}, {})
        settings = {"seed": 42, "allow_synthetic": True}
        p = predict(features, {"checkpoint": self.root / "training/best.pt"}, settings)
        wrong = FeatureBatch(match, features.frames, features.values, features.observed,
                             features.feature_names[::-1], features.periods)
        with self.assertRaisesRegex(ValueError, "feature order"):
            predict(wrong, {"checkpoint": self.root / "training/best.pt"}, settings)
        pipeline_config = {"schema_version": 1, "match": match.as_dict(), "method": "tcn",
                           "data_kind": "synthetic", "seed": 42,
                           "predictor": {"kind": "model", "features": {"file": str(feature_path)},
                           "model": {"callable": "tiki_data.p2.inference:predict",
                                     "expected_features": list(features.feature_names),
                                     "inputs": {"checkpoint": str(self.root / "training/best.pt")},
                                     "settings": settings}}}
        write_json(self.root / "p4.json", pipeline_config)
        run(self.root / "p4.json", self.root / "inference")
        exported = read_batch(self.root / "inference/probabilities.npz", match, "probabilities")
        np.testing.assert_array_equal(p.probabilities, exported.probabilities)
        result = reproduce(self.root / "inference", self.root / "repeat")
        self.assertTrue(result["identical_csv"])


if __name__ == "__main__":
    unittest.main()
