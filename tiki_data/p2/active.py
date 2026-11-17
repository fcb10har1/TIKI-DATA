"""Load the portable, explicitly selected continued verifier."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from tiki_data.artifacts import read_batch, write_predictions
from tiki_data.contracts import MatchSpec, require

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = ROOT / 'models/continued-verifier/selection.json'
MODULES = ('tiki_data.p2.verifier',)


def resolve(pointer=DEFAULT_POINTER):
    value = json.loads(Path(pointer).read_text(encoding='utf-8'))
    path = Path(value['configuration'])
    if not path.is_absolute():
        path = ROOT / path
    require(value['inference_module'] in MODULES, 'unsupported active inference module')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == value['configuration_sha256'],
            'active configuration hash mismatch')
    return value, path.resolve()


def predict(base, ball_positions, pointer=DEFAULT_POINTER):
    value, configuration = resolve(pointer)
    return importlib.import_module(value['inference_module']).predict(base, ball_positions, configuration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, default=DEFAULT_POINTER)
    parser.add_argument('--describe', action='store_true', help='print selected candidate metadata only')
    parser.add_argument('--match', type=Path, help='MatchSpec JSON with half boundaries')
    parser.add_argument('--features', type=Path, help='shared 19-feature P1 NPZ')
    parser.add_argument('--ball-positions', type=Path, help='aligned observed x/y NPY; NaN = missing')
    parser.add_argument('--out', type=Path, help='new local prediction CSV')
    args = parser.parse_args()
    value, configuration = resolve(args.pointer)
    if args.describe:
        print(json.dumps(value, indent=2))
        return
    if not all((args.match, args.features, args.ball_positions, args.out)):
        parser.error('inference requires --match, --features, --ball-positions and --out')
    require(not args.out.exists(), 'refusing to overwrite an existing output')
    spec = MatchSpec.from_dict(json.loads(args.match.read_text(encoding='utf-8')))
    base = read_batch(args.features, spec, 'features')
    positions = np.load(args.ball_positions, allow_pickle=False)
    _, predictions = predict(base, positions, args.pointer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(args.out, predictions)
    print('Saved local predictions with the active model; no per-frame data printed.')


if __name__ == '__main__':
    main()
