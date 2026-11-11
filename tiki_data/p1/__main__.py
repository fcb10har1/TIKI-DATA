"""P1 commands. Run with `python -m tiki_data.p1`."""

import argparse
import json
from pathlib import Path
import sys

from games import GAME, GAMES
from tiki_data.artifacts import read_json, write_batch, write_json
from tiki_data.contracts import MatchSpec

from .features import FEATURE_NAMES, feature_schema
from .idsse import build_dataset
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
    build.add_argument("--mask-ball-speed", action="store_true")
    idsse = commands.add_parser("build-idsse", help="convert all seven IDSSE matches for P2")
    idsse.add_argument("--raw-dir", required=True, help="directory containing the 21 IDSSE XML files")
    idsse.add_argument("--out", required=True, help="new processed-data directory")
    idsse.add_argument("--p2-template", default="configs/p2_v1.json")
    idsse.add_argument("--p2-config", default="configs/p2_idsse_v1.json")
    idsse.add_argument("--resume", action="store_true", help="reuse already validated match outputs")
    idsse.add_argument("--mask-ball-speed", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "build-idsse":
            result = build_dataset(Path(args.raw_dir).resolve(), Path(args.out).resolve(),
                                   Path(args.p2_template).resolve(), Path(args.p2_config).resolve(),
                                   resume=args.resume, mask_ball_speed=args.mask_ball_speed)
            print(json.dumps(dict(result, status="complete"), indent=2, sort_keys=True))
            return 0

        out = Path(args.out).resolve()
        if out.exists():
            raise FileExistsError("output directory already exists: " + str(out))
        match = (MatchSpec.from_dict(read_json(args.match)) if args.match else
                 MatchSpec(GAME, GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]))
        batch = build_metrica_features(
            match, Path(args.atd_meta).resolve(), Path(args.atd_raw).resolve(),
            (args.pitch_length, args.pitch_width),
            mask_ball_speed=args.mask_ball_speed,
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
