"""Train (or reuse a checkpoint), compare, freeze, test once, and reproduce."""

from copy import deepcopy
from pathlib import Path
import math

from games import GAME, GAMES, asset_paths
from .artifacts import write_batch, write_json
from .benchmark import benchmark, load_manifest
from .contracts import MatchSpec, require
from .holdout import freeze
from .pipeline import reproduce, run
from .provenance import file_record, now, source_snapshot


def execute(config_path, out, checkpoint=None, bootstrap=1000, skip_test=False,
            switch_cost=6.0, dead_cost=3.0, supplied_features=None):
    # Import torch only when the complete model workflow is requested.
    from .p2.inference import load_checkpoint
    from .p2.train import load_training_config, train

    require(isinstance(bootstrap, int) and not isinstance(bootstrap, bool) and bootstrap >= 0,
            "bootstrap must be a nonnegative integer")
    for value in (switch_cost, dead_cost):
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value >= 0, "transition costs must be finite and nonnegative")
    config = load_training_config(config_path)
    manifest = load_manifest(config["evaluation_manifest"])
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {"kind": "full_pipeline", "status": "running", "started_at": now(),
              "source": source_snapshot(), "training_config": file_record(config_path),
              "test_attempted": False}
    write_json(out / "run.json", record)
    try:
        if checkpoint is None:
            train(config_path, out / "training")
            checkpoint = out / "training/best.pt"
        checkpoint = Path(checkpoint).resolve()
        _, saved = load_checkpoint(checkpoint)
        require(saved["data_kind"] == manifest["kind"], "checkpoint and dataset kinds differ")
        record["checkpoint"] = file_record(checkpoint)
        names = saved["normalization"]["feature_names"]

        def configuration(spec, features, method, kind):
            if method == "rules":
                predictor = {"kind": "adapter", "stage": {"callable": "tiki_data.p3.rules:predict",
                             "inputs": {"features": str(features)}}}
            else:
                predictor = {"kind": "model", "features": {"file": str(features)},
                             "model": {"callable": "tiki_data.p2.inference:predict",
                                       "inputs": {"checkpoint": str(checkpoint)},
                                       "expected_features": names,
                                       "settings": {"allow_synthetic": kind == "synthetic"}}}
            result = {"schema_version": 1, "match": spec, "method": method,
                      "data_kind": kind, "seed": config["seed"], "predictor": predictor}
            if method == "tcn_viterbi":
                result["decoder"] = {"callable": "tiki_data.p3.decoder:decode",
                                     "settings": {"switch_cost": switch_cost, "dead_cost": dead_cost}}
            return result

        # All paths are absolute in the derived manifest. No test labels are opened here.
        manifest = deepcopy(manifest)
        for entry in manifest["matches"]:
            entry["predictions"] = {}
            if entry["split"] == "train":
                continue
            match_id = entry["match"]["match_id"]
            for method in ("rules", "tcn", "tcn_viterbi"):
                path = out / "inputs" / (match_id + "_" + method + ".json")
                write_json(path, configuration(entry["match"], config["features"][match_id], method,
                                               manifest["kind"]))
                entry["predictions"][method] = {"config": str(path)}
        manifest_path = out / "inputs/manifest.json"
        write_json(manifest_path, manifest)
        print("Comparing rules, TCN, and TCN + Viterbi on validation matches", flush=True)
        validation = benchmark(manifest_path, out / "validation", bootstrap=bootstrap)
        winner = validation["summary"][0]["method"]
        record.update(selected_method=winner, validation=validation["summary"],
                      priorities=validation["priorities"])
        freeze_path = out / "freeze.json"
        freeze(manifest_path, out / "validation", winner, freeze_path)
        if not skip_test:
            print("Frozen method: " + winner + "; evaluating the held-out match once", flush=True)
            record["test_attempted"] = True
            write_json(out / "run.json", record)
            test = benchmark(manifest_path, out / "test", split="test", bootstrap=bootstrap,
                             freeze_path=freeze_path)
            record["test"] = test["summary"]
        # Re-run actual selected-method inference on every validation match.
        for entry in manifest["matches"]:
            if entry["split"] == "validation":
                match_id = entry["match"]["match_id"]
                reproduce(out / "validation/predictions" / match_id / winner,
                          out / "reproduction" / match_id)

        spec = MatchSpec(GAME, GAMES[GAME]["n_video_frames"], 25)
        if supplied_features is None:
            assets = asset_paths()
            missing = [str(assets[key]) for key in ("atd_meta", "atd_raw") if not Path(assets[key]).is_file()]
            if not missing:
                from .p1.adapter import build
                batch = build(spec, {key: Path(assets[key]) for key in ("atd_meta", "atd_raw")}, {})
                supplied_features = out / "supplied/features.npz"
                supplied_features.parent.mkdir(parents=True)
                write_batch(supplied_features, batch)
        else:
            supplied_features = Path(supplied_features).resolve()
            require(supplied_features.is_file(), "supplied feature artifact is missing")
            missing = []
        if supplied_features is not None:
            path = out / "inputs/supplied.json"
            write_json(path, configuration(spec.as_dict(), supplied_features, winner, "supplied"))
            supplied = run(path, out / "supplied/prediction")
            reproduce(out / "supplied/prediction", out / "supplied/reproduction")
            record["supplied"] = {"status": "complete", "output": supplied["output"]}
        else:
            record["supplied"] = {"status": "blocked", "missing_assets": missing}
        record["status"] = "complete" if record["supplied"]["status"] == "complete" else "external_complete"
        record["holdout_skipped"] = bool(skip_test)
    except Exception as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        record["finished_at"] = now()
        write_json(out / "run.json", record)
    return record
