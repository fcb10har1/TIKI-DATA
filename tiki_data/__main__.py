"""P4 commands. Run with python -m tiki_data from the repository root."""

import argparse
import json
from pathlib import Path
import sys

from games import GAME, GAMES
from .artifacts import read_predictions
from .benchmark import benchmark, load_manifest, smoke
from .contracts import MatchSpec
from .holdout import freeze
from .pipeline import reproduce, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="P4 possession integration and evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    inference = commands.add_parser("run", help="run inference and validate the full CSV")
    inference.add_argument("--config", required=True)
    inference.add_argument("--out", required=True, help="new experiment directory")
    check = commands.add_parser("validate", help="use the official CSV validation contract")
    check.add_argument("csv")
    check.add_argument("--rows", type=int, default=GAMES[GAME]["n_video_frames"])
    compare = commands.add_parser("benchmark", help="compare external matches; never tune on public GT")
    compare.add_argument("--manifest", required=True)
    compare.add_argument("--out", required=True)
    compare.add_argument("--split", choices=("validation", "test"), default="validation")
    compare.add_argument("--bootstrap", type=int, default=0)
    compare.add_argument("--freeze", help="required for test evaluation")
    public = commands.add_parser("smoke", help="public five-minute pipeline check only")
    public.add_argument("--csv", required=True)
    public.add_argument("--out", required=True)
    public.add_argument("--bootstrap", type=int, default=0)
    selection = commands.add_parser("freeze", help="freeze a method from validation before the test split")
    selection.add_argument("--manifest", required=True)
    selection.add_argument("--validation", required=True, help="completed validation experiment directory")
    selection.add_argument("--method", required=True)
    selection.add_argument("--out", required=True, help="new freeze JSON file")
    repeat = commands.add_parser("reproduce", help="rerun inference and compare exact CSV bytes")
    repeat.add_argument("--run", required=True, help="completed inference directory")
    repeat.add_argument("--out", required=True)
    repeat.add_argument("--config", help="equivalent configuration with relocated input paths on another setup")
    manifest = commands.add_parser("check-manifest", help="check split declarations without opening label files")
    manifest.add_argument("manifest")
    demo = commands.add_parser("demo", help="exercise all stages with explicitly synthetic inputs")
    demo.add_argument("--out", required=True)
    index = commands.add_parser("experiments", help="export the shared local experiment index")
    index.add_argument("--root", default="runs")
    index.add_argument("--out", default="runs/experiments.csv")
    full = commands.add_parser("full", help="train, compare P1-P4 methods, freeze, test once and reproduce")
    full.add_argument("--config", required=True, help="P2 training config with the fixed external split")
    full.add_argument("--out", required=True, help="new workflow directory")
    full.add_argument("--checkpoint", help="reuse trained weights instead of training again")
    full.add_argument("--bootstrap", type=int, default=1000)
    full.add_argument("--skip-test", action="store_true", help="freeze but preserve the uninspected holdout")
    full.add_argument("--switch-cost", type=float, default=6.0)
    full.add_argument("--dead-cost", type=float, default=3.0)
    full.add_argument("--supplied-features", help="existing BAR-ATH P1 NPZ; otherwise use installed ATD assets")
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            record = run(args.config, args.out)
            result = {"status": record["status"], "csv": record["output"]["path"],
                      "sha256": record["output"]["sha256"], "seconds": record["elapsed_seconds"]}
        elif args.command == "validate":
            batch = read_predictions(args.csv, MatchSpec("csv_validation", args.rows, 25))
            result = {"status": "valid", "rows": len(batch.labels)}
        elif args.command == "benchmark":
            report = benchmark(args.manifest, args.out, args.split, args.bootstrap, args.freeze)
            result = {"status": "complete", "evidence_kind": report["evidence_kind"],
                      "accuracy_evidence": report["accuracy_evidence"], "split": args.split,
                      "comparison": str(Path(args.out) / "report.md"), "summary": report["summary"]}
        elif args.command == "smoke":
            report = smoke(args.csv, args.out, args.bootstrap)
            result = {"status": report["status"], "purpose": report["purpose"],
                      "graded_frames": report["metrics"]["n_graded"], "accuracy_evidence": False}
        elif args.command == "freeze":
            result = freeze(args.manifest, args.validation, args.method, args.out)
        elif args.command == "reproduce":
            result = reproduce(args.run, args.out, args.config)
        elif args.command == "check-manifest":
            value = load_manifest(args.manifest)
            result = {"status": "valid", "dataset_id": value["dataset_id"], "kind": value["kind"],
                      "splits": {split: sum(entry["split"] == split for entry in value["matches"])
                                 for split in ("train", "validation", "test")}}
        elif args.command == "demo":
            from .demo import demonstrate
            result = demonstrate(args.out)
        elif args.command == "full":
            from .workflow import execute
            result = execute(args.config, args.out, args.checkpoint, args.bootstrap, args.skip_test,
                             args.switch_cost, args.dead_cost, args.supplied_features)
        else:
            from .experiments import export
            result = {"status": "complete", "records": export(args.root, args.out), "index": args.out}
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print("ERROR {}: {}".format(type(error).__name__, error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
