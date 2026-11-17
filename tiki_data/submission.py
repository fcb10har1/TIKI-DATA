"""Regenerate the frozen TIKIDATA submission from licensed tactical ATD files."""
import argparse
import hashlib
import json
from pathlib import Path

from evaluation import possession_metrics as pm
from games import GAME, GAMES, asset_paths
from tiki_data.artifacts import write_predictions
from tiki_data.contracts import MatchSpec, require
from tiki_data.p1.ball_observations import metrica_positions
from tiki_data.p1.metrica import build_metrica_features
from tiki_data.p2.verifier import predict

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'models/continued-verifier/configuration.json'


def generate(atd_meta, atd_raw, out, configuration=DEFAULT_CONFIG):
    """Read tracking only, infer all frames, and validate the exact saved CSV."""
    out = Path(out)
    require(not out.exists(), 'refusing to overwrite an existing output')
    require(Path(atd_meta).is_file() and Path(atd_raw).is_file(),
            'both licensed tactical ATD files are required')
    config = json.loads(Path(configuration).read_text(encoding='utf-8'))
    spec = MatchSpec(GAME, GAMES[GAME]['n_video_frames'], GAMES[GAME]['fps'])
    print('Building tracking features for the complete timeline.', flush=True)
    features = build_metrica_features(spec, atd_meta, atd_raw, GAMES[GAME]['pitch'],
                                     mask_ball_speed=True)
    positions = metrica_positions(spec, atd_raw, GAMES[GAME]['pitch'])
    print('Running the frozen continued verifier on CPU.', flush=True)
    _, predictions = predict(features, positions, configuration)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(out, predictions)
    reason = pm.validate_submission(pm.read_label_csv(out), spec.n_frames)
    require(reason is None, 'official submission format validation failed')
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    require(digest == config['reference_submission_sha256'],
            'CSV differs from the frozen reference; check the pinned environment and input bundle')
    result = dict(candidate=config['candidate'], rows=spec.n_frames,
                  official_format_valid=True, matches_frozen_reference=True, sha256=digest)
    print(json.dumps(result), flush=True)
    return result


def main():
    inputs = asset_paths(GAME)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atd-meta', type=Path, default=Path(inputs['atd_meta']))
    parser.add_argument('--atd-raw', type=Path, default=Path(inputs['atd_raw']))
    parser.add_argument('--out', type=Path, default=Path('runs/submission/TIKIDATA.csv'))
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    generate(args.atd_meta, args.atd_raw, args.out, args.config)


if __name__ == '__main__':
    main()
