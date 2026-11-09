"""Run from the repository root: python -m tiki_data.p2 --help."""

import argparse
import json
from pathlib import Path

from tiki_data.artifacts import read_batch, read_json, write_batch, write_json
from tiki_data.contracts import MatchSpec
from tiki_data.provenance import file_record
from .inference import load_checkpoint, predict, write_probability_csv
from .smoke import prepare
from .train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-smoke", help="Generate synthetic execution fixtures only")
    p.add_argument("--out", required=True)
    p = sub.add_parser("train", help="Train on external train/validation matches; never read final test")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("infer", help="Export H,A,D probabilities and a runnable P4 model configuration")
    p.add_argument("--features", required=True)
    p.add_argument("--match", required=True, help="MatchSpec JSON")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare-smoke":
        result = prepare(args.out)
    elif args.command == "train":
        report = train(args.config, args.out)
        result = {k: report[k] for k in ("status", "data_kind", "accuracy_evidence", "checkpoint_reload_verified", "best_epoch")}
    else:
        match = MatchSpec.from_dict(read_json(args.match))
        features = read_batch(args.features, match, "features")
        settings = {"seed": args.seed, "device": args.device, "allow_synthetic": args.allow_synthetic}
        probabilities = predict(features, {"checkpoint": Path(args.checkpoint).resolve()}, settings)
        _, checkpoint = load_checkpoint(args.checkpoint)
        out = Path(args.out).resolve()
        out.mkdir(parents=True, exist_ok=False)
        write_batch(out / "probabilities.npz", probabilities)
        write_probability_csv(out / "probabilities.csv", probabilities)
        kind = "supplied" if match.match_id == "BAR-ATH" else checkpoint["data_kind"]
        config = {"schema_version": 1, "match": match.as_dict(), "method": "tcn", "data_kind": kind,
                  "seed": args.seed, "predictor": {"kind": "model", "features": {"file": str(Path(args.features).resolve())},
                      "model": {"callable": "tiki_data.p2.inference:predict",
                                "expected_features": list(features.feature_names),
                                "inputs": {"checkpoint": str(Path(args.checkpoint).resolve())}, "settings": settings}}}
        write_json(out / "p4_model.json", config)
        result = {"status": "complete", "frames": match.n_frames, "class_order": ["H", "A", "D"],
                  "data_kind": checkpoint["data_kind"], "accuracy_evidence": False,
                  "checkpoint": file_record(args.checkpoint),
                  "outputs": [file_record(out / name) for name in ("probabilities.npz", "probabilities.csv")]}
        write_json(out / "inference.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
