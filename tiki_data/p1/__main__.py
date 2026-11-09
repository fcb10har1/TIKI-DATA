"""P1 commands. Run with `python -m tiki_data.p1`."""

import argparse
import json
from pathlib import Path
import sys

from games import GAME, GAMES
from tiki_data.artifacts import read_json, write_batch, write_json
from tiki_data.contracts import MatchSpec

from .features import FEATURE_NAMES, feature_schema
from .metrica import build_metrica_features


def main(argv=None):
    parser = argparse.ArgumentParser(description="P1 tracking-feature production")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-metrica", help="export one Metrica ATD FeatureBatch")
    build.add_argument("--atd-meta", required=True)
    build.add_argument("--atd-raw", required=True)
    build.add_argument("--out", required=True, help="new output directory")
    build.add_argument("--match", help="optional MatchSpec JSON; defaults to BAR-ATH")
    build.add_argument("--pitch-length", type=float, default=GAMES[GAME]["pitch"][0])
    build.add_argument("--pitch-width", type=float, default=GAMES[GAME]["pitch"][1])
    args = parser.parse_args(argv)
    try:
        out = Path(args.out).resolve()
        if out.exists():
            raise FileExistsError("output directory already exists: " + str(out))
        match = (MatchSpec.from_dict(read_json(args.match)) if args.match else
                 MatchSpec(GAME, GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]))
        batch = build_metrica_features(
            match, Path(args.atd_meta).resolve(), Path(args.atd_raw).resolve(),
            (args.pitch_length, args.pitch_width),
        )
        out.mkdir(parents=True, exist_ok=False)
        write_batch(out / "features.npz", batch)
        write_json(out / "match.json", match.as_dict(), exclusive=True)
        write_json(out / "feature_schema.json", feature_schema(), exclusive=True)
        result = {"status": "complete", "match_id": match.match_id,
                  "frames": match.n_frames, "feature_count": len(FEATURE_NAMES),
                  "features": str(out / "features.npz"),
                  "match": str(out / "match.json"),
                  "feature_schema": str(out / "feature_schema.json")}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        print("ERROR {}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
