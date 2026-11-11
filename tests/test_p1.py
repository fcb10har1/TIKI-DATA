"""P1 regression checks: geometry, missingness, transport and P2 compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tiki_data.artifacts import read_batch, read_json, write_batch
from tiki_data.contracts import FeatureBatch, MatchSpec
from tiki_data.p1.features import (BALL_DEPENDENT, FEATURE_NAMES, apply_feature_policy, compute_frame,
                                   compute_frames, feature_schema)
from tiki_data.p1.idsse import convert_match
from tiki_data.p1.metrica import build_metrica_features
from tiki_data.p2.data import fit_normalization, validate_schema
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.model import TCN, seed_all


class P1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_geometry_and_missingness(self):
        home = np.array([[0.0, 0.0], [4.0, 0.0], [12.0, 5.0]])
        away = np.array([[10.0, 0.0], [20.0, 0.0]])
        values, observed = compute_frame(home, away, [1.0, 2.0, 3.0], [4.0, 6.0],
                                         ball_position=(1.0, 0.0), ball_speed=7.0)
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self.assertAlmostEqual(values[index["home_nearest_ball_m"]], 1.0)
        self.assertAlmostEqual(values[index["home_second_nearest_ball_m"]], 3.0)
        self.assertAlmostEqual(values[index["away_nearest_ball_m"]], 9.0)
        self.assertEqual(values[index["home_players_within_3m"]], 2)
        self.assertEqual(values[index["away_players_within_5m"]], 0)
        self.assertAlmostEqual(values[index["home_width_m"]], 5.0)
        self.assertAlmostEqual(values[index["home_length_m"]], 12.0)
        self.assertAlmostEqual(values[index["away_mean_speed_mps"]], 5.0)
        self.assertTrue(observed.all())

        missing_values, missing_observed = compute_frame(home, away, [1, 2, 3], [4, 6])
        for name in BALL_DEPENDENT:
            self.assertFalse(missing_observed[index[name]])
            self.assertEqual(missing_values[index[name]], 0)
        self.assertTrue(missing_observed[index["home_width_m"]])
        self.assertTrue(missing_observed[index["home_players_observed"]])

    def test_metrica_stream_and_npz_round_trip(self):
        metadata = self.root / "meta.xml"
        metadata.write_text(
            "<Root><Team id='home'><Name>Home</Name></Team>"
            "<Team id='away'><Name>Away</Name></Team>"
            "<Player teamId='home'/><Player teamId='home'/>"
            "<Player teamId='away'/><Player teamId='away'/></Root>", encoding="utf-8")
        raw = self.root / "raw.txt"
        raw.write_text(
            "1:0.50,0.50,1;0.60,0.50,2;0.80,0.50,3;0.90,0.50,4:0.51,0.50,5\n"
            "2:0.51,0.50,NaN;0.61,0.50,NaN;0.79,0.50,3;0.89,0.50,4:NaN,NaN,NaN\n"
            "3:0.52,0.50,1;NaN,NaN,NaN;0.78,0.50,3;0.88,0.50,4:0.77,0.50,6\n"
            "4:0.53,0.50,1;0.63,0.50,2;0.77,0.50,3;0.87,0.50,4:0.76,0.50,6\n",
            encoding="utf-8")
        match = MatchSpec("p1_fixture", 4, 25, ((0, 2), (2, 4)))
        batch = build_metrica_features(match, metadata, raw, (100.0, 50.0))
        self.assertEqual(batch.values.shape, (4, len(FEATURE_NAMES)))
        self.assertEqual(batch.feature_names, FEATURE_NAMES)
        ball_speed = FEATURE_NAMES.index("ball_speed_mps")
        self.assertFalse(batch.observed[1, ball_speed])
        self.assertEqual(batch.values[1, ball_speed], 0)
        self.assertTrue(batch.observed[0, ball_speed])
        home_nearest = FEATURE_NAMES.index("home_nearest_ball_m")
        home_mean_speed = FEATURE_NAMES.index("home_mean_speed_mps")
        self.assertFalse(batch.observed[1, home_nearest])
        self.assertFalse(batch.observed[1, home_mean_speed])
        path = self.root / "features.npz"
        write_batch(path, batch)
        loaded = read_batch(path, match, "features")
        np.testing.assert_array_equal(batch.values, loaded.values)
        np.testing.assert_array_equal(batch.observed, loaded.observed)
        masked = build_metrica_features(match, metadata, raw, (100.0, 50.0), mask_ball_speed=True)
        self.assert_speed_only_change(batch, masked)
        self.assertTrue(masked.observed[0, home_nearest])
        self.assertAlmostEqual(masked.values[0, home_nearest], 1.0)

    def assert_speed_only_change(self, original, masked):
        column = original.feature_names.index("ball_speed_mps")
        other = [i for i in range(len(original.feature_names)) if i != column]
        np.testing.assert_array_equal(original.frames, masked.frames)
        np.testing.assert_array_equal(original.periods, masked.periods)
        np.testing.assert_array_equal(original.values[:, other], masked.values[:, other])
        np.testing.assert_array_equal(original.observed[:, other], masked.observed[:, other])
        self.assertTrue((masked.values[:, column] == 0).all())
        self.assertFalse(masked.observed[:, column].any())
        self.assertEqual(masked.feature_names, original.feature_names)
        self.assertEqual(masked.match, original.match)

    def test_scalar_speed_policy_is_strict_nonmutating_and_idempotent(self):
        match = MatchSpec("policy_fixture", 4, 25)
        batch = FeatureBatch(match, np.arange(4), np.ones((4, 19), dtype=np.float32),
                             np.ones((4, 19), dtype=bool), FEATURE_NAMES, match.period_ids())
        self.assertIs(apply_feature_policy(batch), batch)
        masked = apply_feature_policy(batch, True)
        self.assert_speed_only_change(batch, masked)
        self.assertTrue(batch.observed.all())
        self.assertTrue((batch.values == 1).all())
        np.testing.assert_array_equal(apply_feature_policy(masked, True).values, masked.values)
        for invalid in ("false", 1, None):
            with self.assertRaisesRegex(ValueError, "boolean"):
                apply_feature_policy(batch, invalid)

    def test_missing_scalar_speed_preserves_metrica_ball_position(self):
        from tiki_data.p1.metrica import parse_observation
        x, y, speed = parse_observation("0.51,0.50,NaN")
        self.assertIsNone(speed)
        values, observed = compute_frame(np.array([[50., 25.], [60., 25.]]),
                                         np.array([[80., 25.], [90., 25.]]), [1, 2], [3, 4],
                                         (x * 100, y * 50), speed)
        self.assertFalse(observed[FEATURE_NAMES.index("ball_speed_mps")])
        self.assertTrue(observed[FEATURE_NAMES.index("home_nearest_ball_m")])
        self.assertAlmostEqual(values[FEATURE_NAMES.index("home_nearest_ball_m")], 1.)

    def test_schema_and_p2_tensor_contract(self):
        schema = feature_schema()
        validate_schema(schema, FEATURE_NAMES)
        committed = read_json(Path(__file__).parents[1] / "configs" / "p1_v1_feature_schema.json")
        self.assertEqual(schema, committed)

        seed_all(42)
        match = MatchSpec("p1_p2_fixture", 40, 25, ((0, 20), (20, 40)))
        rng = np.random.default_rng(42)
        values = rng.normal(size=(40, len(FEATURE_NAMES))).astype(np.float32)
        observed = np.ones_like(values, dtype=np.bool_)
        batch = FeatureBatch(match, np.arange(40), values, observed,
                             FEATURE_NAMES, match.period_ids()).validate(match)
        normalization = fit_normalization([batch])
        model = TCN(len(FEATURE_NAMES), width=4, dilations=[1, 2], dropout=0)
        probabilities = predict_batch(batch, model, normalization, core=16)
        self.assertEqual(probabilities.probabilities.shape, (40, 3))
        probabilities.validate(match)

    def test_vectorized_features_match_scalar_definitions(self):
        home = np.array([[[0, 0], [4, 0]], [[1, 0], [np.nan, np.nan]]], dtype=np.float32)
        away = np.array([[[10, 0], [20, 0]], [[9, 0], [19, 0]]], dtype=np.float32)
        home_speed = np.array([[1, 2], [3, np.nan]], dtype=np.float32)
        away_speed = np.array([[4, 6], [5, 7]], dtype=np.float32)
        ball = np.array([[1, 0], [np.nan, np.nan]], dtype=np.float32)
        ball_speed = np.array([7, np.nan], dtype=np.float32)
        values, observed = compute_frames(home, away, home_speed, away_speed, ball, ball_speed,
                                           chunk_size=1)
        for frame in range(2):
            home_valid = np.isfinite(home[frame]).all(axis=1)
            away_valid = np.isfinite(away[frame]).all(axis=1)
            ball_value = ball[frame] if np.isfinite(ball[frame]).all() else None
            expected_values, expected_observed = compute_frame(
                home[frame, home_valid], away[frame, away_valid],
                home_speed[frame, home_valid & np.isfinite(home_speed[frame])],
                away_speed[frame, away_valid & np.isfinite(away_speed[frame])],
                ball_value, ball_speed[frame])
            np.testing.assert_allclose(values[frame], expected_values, rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(observed[frame], expected_observed)

    def test_idsse_xml_conversion_and_target_separation(self):
        info = self.root / "DFL_02_01_matchinformation_DFL-COM-X_DFL-MAT-TEST1.xml"
        info.write_text(
            "<Root><MatchInformation>"
            "<General MatchId='DFL-MAT-TEST1' HomeTeamId='HOME' GuestTeamId='AWAY' "
            "HomeTeamName='Home' GuestTeamName='Away'/>"
            "<Environment PitchX='105' PitchY='68'/>"
            "<Teams><Team TeamId='HOME' TeamName='Home'><Players>"
            "<Player PersonId='h1'/><Player PersonId='h2'/></Players></Team>"
            "<Team TeamId='AWAY' TeamName='Away'><Players>"
            "<Player PersonId='a1'/><Player PersonId='a2'/></Players></Team></Teams>"
            "</MatchInformation></Root>", encoding="utf-8")
        position = self.root / "DFL_04_03_positions_raw_observed_DFL-COM-X_DFL-MAT-TEST1.xml"
        frame_sets = []
        for period, frames in (("firstHalf", (10, 11)), ("secondHalf", (100, 101))):
            for team, person, x in (("HOME", "h1", 0), ("HOME", "h2", 4),
                                    ("AWAY", "a1", 10), ("AWAY", "a2", 20)):
                body = "".join("<Frame N='{}' X='{}' Y='0' S='3.6'/>".format(n, x) for n in frames)
                frame_sets.append("<FrameSet GameSection='{}' TeamId='{}' PersonId='{}'>{}</FrameSet>".format(
                    period, team, person, body))
            labels = ((1, 1), (2, 0)) if period == "firstHalf" else ((2, 1), (-1, 1))
            body = "".join(
                "<Frame N='{}' X='1' Y='0' S='7.2' BallPossession='{}' BallStatus='{}'/>".format(
                    n, possession, status) for n, (possession, status) in zip(frames, labels))
            frame_sets.append("<FrameSet GameSection='{}' TeamId='BALL' PersonId='ball'>{}</FrameSet>".format(
                period, body))
        position.write_text("<Root><Positions>{}</Positions></Root>".format("".join(frame_sets)),
                            encoding="utf-8")
        events = self.root / "DFL_03_02_events_raw_DFL-COM-X_DFL-MAT-TEST1.xml"
        events.write_text("<Root/>", encoding="utf-8")
        out = self.root / "converted"
        match = convert_match("TEST1", {"match_information": info, "events": events,
                                         "positions": position}, out)
        self.assertEqual(match.n_frames, 4)
        self.assertEqual(match.periods, ((0, 2), (2, 4)))
        batch = read_batch(out / "features.npz", match, "features")
        labels = __import__("evaluation.possession_metrics", fromlist=["read_label_csv"]).read_label_csv(
            out / "labels.csv")
        self.assertEqual(labels["label"].tolist(), ["H", "D", "A", "X"])
        self.assertEqual(batch.feature_names, FEATURE_NAMES)
        self.assertNotIn("BallPossession", batch.feature_names)
        self.assertNotIn("BallStatus", batch.feature_names)
        masked_out = self.root / "masked"
        masked_match = convert_match("TEST1", {"match_information": info, "events": events,
                                              "positions": position}, masked_out, mask_ball_speed=True)
        self.assertEqual(match, masked_match)
        masked = read_batch(masked_out / "features.npz", match, "features")
        self.assert_speed_only_change(batch, masked)
        self.assertEqual((out / "labels.csv").read_bytes(), (masked_out / "labels.csv").read_bytes())
        self.assertEqual(read_json(masked_out / "conversion.json")["feature_observed_fraction"]
                         ["ball_speed_mps"], 0.)


if __name__ == "__main__":
    unittest.main()
