"""Verify the frozen release and optionally its CSV; print metadata only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'models/continued-verifier'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_materialized(path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Missing model asset: ' + str(path.relative_to(ROOT)))
    prefix = path.read_bytes()[:64]
    if prefix.startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise ValueError('Model asset is an unresolved Git LFS pointer: ' +
                         str(path.relative_to(ROOT)))


def verify(prediction=None):
    selection = json.loads((BUNDLE / 'selection.json').read_text(encoding='utf-8'))
    path = ROOT / selection['configuration']
    if digest(path) != selection['configuration_sha256']:
        raise ValueError('Selected configuration checksum mismatch')
    config = json.loads(path.read_text(encoding='utf-8'))
    assets = [config['historical'], *config['proposers'], config['verifier']]
    for asset in assets:
        model_path = BUNDLE / asset['file']
        require_materialized(model_path)
        if digest(model_path) != asset['sha256']:
            raise ValueError('Model asset checksum mismatch: ' + asset['file'])
    for source in config['source_files']:
        if digest(ROOT / source['file']) != source['sha256']:
            raise ValueError('Source checksum mismatch: ' + source['file'])
    result = dict(candidate=config['candidate'], model_assets_verified=len(assets),
                  frozen_sources_verified=len(config['source_files']))
    if prediction is not None:
        prediction = Path(prediction)
        count = 0
        with prediction.open(encoding='utf-8', newline='') as handle:
            rows = csv.reader(handle)
            if next(rows, None) != ['frame', 'label']:
                raise ValueError('Submission header must be exactly frame,label')
            for count, row in enumerate(rows, 1):
                if len(row) != 2 or row[0] != str(count - 1) or row[1] not in ('H', 'A', 'D', 'X'):
                    raise ValueError('Invalid submission structure or label; no row contents printed')
        if count != 199319:
            raise ValueError('Expected exactly 199319 data rows')
        actual = digest(prediction)
        if actual != config['reference_submission_sha256']:
            raise ValueError('CSV does not match the frozen selected output')
        result.update(rows=count, exact_frozen_csv_match=True, sha256=actual)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, help='optional local generated submission')
    args = parser.parse_args()
    print(json.dumps(verify(args.csv)))


if __name__ == '__main__':
    main()
