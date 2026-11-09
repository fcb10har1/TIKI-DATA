"""P1 regression checks: geometry, missingness, transport and P2 compatibility."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tiki_data.artifacts import read_batch, read_json, write_batch
from tiki_data.contracts import FeatureBatch, MatchSpec
from tiki_data.p1.features import (BALL_DEPENDENT, FEATURE_NAMES, compute_frame,
                                   feature_schema)
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


if __name__ == "__main__":
    unittest.main()
