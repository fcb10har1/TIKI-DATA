"""Checks for the single-candidate release; no supplied data or labels."""
from pathlib import Path

import numpy as np
import pytest

from tiki_data.contracts import FeatureBatch, MatchSpec
from tiki_data.p1.features import FEATURE_NAMES
from tiki_data.p2.active import resolve
from tiki_data.p2.verifier import load_bundle, predict
from tiki_data.p2.verifier_edits import Edit, apply_edits
from tiki_data.p2.verifier_decoder import decode


def test_bundle_and_portable_selection():
    selected, path = resolve()
    config, _, members, forest = load_bundle(path)
    assert selected['candidate'] == 'continued_verifier_control_seed42'
    assert config['candidate'] == selected['candidate']
    assert len(members) == 2 and forest.n_features_in_ == 132
    assert config['acceptance_threshold'] == .3
    assert not Path(selected['configuration']).is_absolute()


def test_overlapping_guards_and_strict_threshold():
    base = np.full(100, 'H')
    edits = [Edit(10, 20, 1, 0, 40), Edit(30, 35, 2, 25, 50), Edit(70, 80, 1, 60, 90)]
    result, accepted = apply_edits(base, edits, np.array([.6, .7, .3]), .3)
    assert accepted == [1]
    assert np.all(result[30:35] == 'D')
    assert np.all(result[70:80] == 'H')


def test_decode_keeps_periods_separate_and_outside_unknown():
    spec = MatchSpec('synthetic-release', 160, 25, ((10, 70), (90, 150)))
    probabilities = np.tile([.98, .01, .01], (160, 1))
    probabilities[90:] = [.01, .98, .01]
    labels = decode(probabilities, np.ones(160, bool), spec,
                    dict(duration='hard', minimum=1., transition_penalty=0., missing_weight=0.))
    assert np.all(labels[10:70] == 'H') and np.all(labels[90:150] == 'A')
    assert np.all(labels[spec.period_ids() == 0] == 'X')


def test_real_bundle_synthetic_inference_is_deterministic():
    spec = MatchSpec('synthetic-release', 120, 25, ((0, 50), (70, 120)))
    values = np.zeros((120, len(FEATURE_NAMES)), np.float32)
    observed = np.zeros_like(values, bool)
    features = FeatureBatch(spec, np.arange(120), values, observed, FEATURE_NAMES, spec.period_ids())
    positions = np.full((120, 2), np.nan)
    _, config = resolve()
    _, first = predict(features, positions, config)
    _, second = predict(features, positions, config)
    assert np.array_equal(first.labels, second.labels)
    assert np.all(first.labels[50:70] == 'X')


def test_modified_model_asset_is_rejected(tmp_path):
    import json
    import shutil
    _, path = resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    copy = tmp_path/'configuration.json'
    copy.write_text(json.dumps(config), encoding='utf-8')
    asset = tmp_path/config['historical']['file']
    shutil.copyfile(path.parent/config['historical']['file'], asset)
    with asset.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='asset hash mismatch'):
        load_bundle(copy)
