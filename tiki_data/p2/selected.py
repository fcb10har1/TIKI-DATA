"""Portable inference for the saved highest-external-spellF1 ensemble."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tiki_data.artifacts import read_batch, write_predictions
from tiki_data.contracts import CLASS_ORDER, MatchSpec, PredictionBatch, ProbabilityBatch, require
from .inference import load_checkpoint, predict_batch
from .model import seed_all
from .selected_decoder import decode
from .selected_features import build_features

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'models/possession-kalman-ensemble-20261116/configuration.json'


def load_ensemble(configuration=DEFAULT_CONFIG):
    path = Path(configuration).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    require(config['schema_version'] == 1 and tuple(config['class_order']) == CLASS_ORDER,
            'unsupported selected ensemble configuration')
    require(len(config['checkpoints']) == 2 and
            all(x['weight'] == .5 for x in config['checkpoints']), 'expected frozen 50:50 ensemble')
    members = []
    for entry in config['checkpoints']:
        checkpoint = (path.parent / entry['file']).resolve()
        require(checkpoint.parent == path.parent, 'checkpoint must be in the model directory')
        require(hashlib.sha256(checkpoint.read_bytes()).hexdigest() == entry['sha256'],
                'checkpoint hash mismatch')
        model, saved = load_checkpoint(checkpoint)
        require(saved['seed'] == entry['seed'] and saved['fps'] == config['fps'],
                'checkpoint seed/FPS mismatch')
        members.append((model, saved, entry['weight']))
    return config, members


def predict(base, ball_positions, configuration=DEFAULT_CONFIG):
    """Return ensemble probabilities and labels; accepts no labels or game tags."""
    seed_all(42, 1)
    config, members = load_ensemble(configuration)
    features = build_features(base, ball_positions)
    probability = np.zeros((base.match.n_frames, 3), dtype=np.float64)
    for model, saved, weight in members:
        batch = predict_batch(features, model, saved['normalization'], saved['core_frames'])
        probability += weight * batch.probabilities
    probabilities = ProbabilityBatch(base.match, base.frames.copy(), probability,
                                     base.periods.copy()).validate(base.match)
    visible = np.isfinite(ball_positions).all(axis=1)
    labels = decode(probability, visible, base.match, config['decoder'])
    return probabilities, PredictionBatch(base.match, base.frames.copy(), labels).validate(base.match)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--match', type=Path, required=True, help='MatchSpec JSON with half boundaries')
    parser.add_argument('--features', type=Path, required=True, help='shared 19-feature P1 NPZ')
    parser.add_argument('--ball-positions', type=Path, required=True, help='aligned metric x/y NPY; NaN = missing')
    parser.add_argument('--out', type=Path, required=True, help='new local prediction CSV')
    args = parser.parse_args()
    require(not args.out.exists(), 'refusing to overwrite an existing output')
    spec = MatchSpec.from_dict(json.loads(args.match.read_text(encoding='utf-8')))
    base = read_batch(args.features, spec, 'features')
    positions = np.load(args.ball_positions, allow_pickle=False)
    _, predictions = predict(base, positions, args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out, predictions)
    print('Saved local predictions; no input rows or per-frame predictions printed.')


if __name__ == '__main__':
    main()
