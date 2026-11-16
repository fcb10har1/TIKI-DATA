"""Input/state checks for the portable selected model release."""
import json

import numpy as np
import pytest

from tiki_data.contracts import FeatureBatch, MatchSpec
from tiki_data.p1.features import FEATURE_NAMES
from tiki_data.p2.selected import DEFAULT_CONFIG, load_ensemble, predict
from tiki_data.p2.selected_features import build_features


def inputs():
    spec = MatchSpec('synthetic-selected', 360, 25, ((0, 180), (180, 360)))
    values = np.zeros((360, 19), np.float32)
    observed = np.zeros_like(values, dtype=bool)
    observed[:, 17:19] = True
    base = FeatureBatch(spec, np.arange(360), values, observed, FEATURE_NAMES, spec.period_ids())
    positions = np.column_stack((np.arange(360) * .1, np.zeros(360)))
    positions[80:100] = np.nan
    return base, positions


def test_kalman_missing_observations_and_half_reset():
    base, positions = inputs()
    batch = build_features(base, positions)
    assert not batch.observed[80:100, 0].any()
    assert not batch.observed[80:100, 20].any()
    assert batch.values[99, 19] == pytest.approx(.8)
    assert batch.values[100, 19] == 0
    changed = positions.copy()
    changed[:180] = np.nan
    other = build_features(base, changed)
    np.testing.assert_array_equal(batch.values[180:], other.values[180:])
    np.testing.assert_array_equal(batch.observed[180:], other.observed[180:])


def test_selected_checkpoints_and_half_safe_inference():
    config, members = load_ensemble()
    assert [saved['seed'] for _, saved, _ in members] == [42, 43]
    assert config['evaluation']['metrics']['spellF1'] == pytest.approx(.5929733850268655)
    base, positions = inputs()
    probabilities, predictions = predict(base, positions)
    changed = positions.copy()
    changed[180:] = np.nan
    other, _ = predict(base, changed)
    np.testing.assert_array_equal(probabilities.probabilities[:180], other.probabilities[:180])
    np.testing.assert_allclose(probabilities.probabilities.sum(axis=1), 1)
    assert set(predictions.labels) <= {'H', 'A', 'D'}


def test_unsupported_ensemble_is_rejected(tmp_path):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config['checkpoints'][0]['weight'] = .75
    path = tmp_path / 'configuration.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='50:50'):
        load_ensemble(path)
