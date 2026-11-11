"""Reproduce the scalar-speed experiment; never open the final test partition.

python -m tiki_data.p2.speed_ablation --help
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from games import GAME, GAMES
from p3.stage import decode_stage
from tiki_data.artifacts import read_json, read_predictions, write_batch, write_json, write_predictions
from tiki_data.benchmark import metric_report, score_match
from tiki_data.contracts import MatchSpec, require
from tiki_data.p1.features import apply_feature_policy
from tiki_data.p1.metrica import build_metrica_features, parse_observation
from tiki_data.provenance import environment, file_record, source_snapshot
from .data import encode, fit_normalization
from .inference import load_checkpoint, predict_batch
from .model import seed_all
from .train import load_data, load_training_config, validate


def audit_mask(original, masked):
    """Check all frames and all non-speed channels, including ball geometry."""
    j = original.feature_names.index("ball_speed_mps")
    other = [k for k in range(len(original.feature_names)) if k != j]
    checks = {
        "frames_equal": np.array_equal(original.frames, masked.frames),
        "periods_equal": np.array_equal(original.periods, masked.periods),
        "feature_names_equal": original.feature_names == masked.feature_names,
        "non_speed_values_equal": np.array_equal(original.values[:, other], masked.values[:, other]),
        "non_speed_observed_equal": np.array_equal(original.observed[:, other], masked.observed[:, other]),
        "speed_zero_placeholder": bool((masked.values[:, j] == 0).all()),
        "speed_observed_false": not bool(masked.observed[:, j].any()),
    }
    require(all(checks.values()), "scalar-speed masking audit failed")
    return dict(checks, match_id=original.match.match_id, frames=len(original.frames),
                original_speed_observed=int(original.observed[:, j].sum()),
                masked_speed_observed=int(masked.observed[:, j].sum()))


def audit_checkpoint(path, config, data, schema):
    """Require matching settings, unchanged inputs and reproduced selection scores."""
    path = Path(path).resolve()
    run = read_json(path.parent / "run.json")
    require(run["status"] == "complete" and run["checkpoint_reload_verified"], "incomplete training run")
    require(run["config"] == config, "checkpoint training settings differ from experiment config")
    require(run["selection"] == "validation mean spellF1, then macroF1", "different checkpoint selection")
    require(file_record(path)["sha256"] == run["checkpoint"]["sha256"], "checkpoint hash changed")
    for record in run["inputs"]:
        current = file_record(record["path"])
        require(current["sha256"] == record["sha256"], "training input changed: " + record["path"])
    current_environment = environment()
    for package in ("torch", "numpy", "pandas"):
        require(run["environment"]["packages"][package] == current_environment["packages"][package],
                "checkpoint environment differs: " + package)
    t = config["training"]
    seed_all(config["seed"], t["threads"])
    model, checkpoint = load_checkpoint(path, t["device"])
    mask = config.get("p1", {}).get("mask_ball_speed", False)
    require(checkpoint["normalization"].get("mask_ball_speed", False) == mask,
            "checkpoint scalar-speed policy differs")
    require(checkpoint["model_config"] == config["model"] and checkpoint["seed"] == config["seed"]
            and checkpoint["fps"] == config["fps"] and checkpoint["core_frames"] == t["core_frames"]
            and checkpoint["feature_schema"] == schema and checkpoint["data_kind"] == "external",
            "checkpoint architecture/schema/settings differ")
    fitted = fit_normalization([batch for batch, _ in data["train"]],
                               allow_unobserved=("ball_speed_mps",) if mask else ())
    for key, value in fitted.items():
        require(value == checkpoint["normalization"][key], "train-only normalization differs: " + key)
    require(checkpoint["normalization"]["train_match_ids"] ==
            [batch.match.match_id for batch, _ in data["train"]], "normalization train split differs")
    repeated = validate(model, data["validation"], checkpoint["normalization"], t["core_frames"], t["device"])
    require(repeated == checkpoint["validation"] == run["best_validation"], "saved validation does not reproduce")
    require(checkpoint["epoch"] == run["best_epoch"], "selected epoch differs")
    current_source = source_snapshot()
    evidence = {
        "checkpoint": file_record(path), "training_run": file_record(path.parent / "run.json"),
        "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
        "settings_inputs_environment_verified": True, "train_only_normalization_verified": True,
        "saved_validation_exactly_reproduced": True,
        "selection": run["selection"], "selection_speed_hidden": mask,
        "changed_source_files_since_training": [p for p, h in run["source"]["files"].items()
                                                if current_source["files"].get(p) != h],
    }
    return model, checkpoint, evidence


def compare(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    configs = {arm: load_training_config(getattr(args, "config_" + arm.lower())) for arm in ("A", "B")}
    require(not configs["A"].get("p1", {}).get("mask_ball_speed", False), "A must train with recorded speed")
    require(configs["B"].get("p1", {}).get("mask_ball_speed") is True, "B must train without scalar speed")
    require({k: v for k, v in configs["A"].items() if k != "p1"} ==
            {k: v for k, v in configs["B"].items() if k != "p1"}, "A/B differ beyond the scalar-speed policy")
    manifest, schema, original, _ = load_data(configs["A"])
    require(manifest["kind"] == "external", "comparison needs real external matches")
    _, _, hidden, _ = load_data(configs["B"])
    checks = []
    for split in ("train", "validation"):
        require(len(original[split]) == len(hidden[split]), "split sizes differ")
        for (batch, labels), (masked, other_labels) in zip(original[split], hidden[split]):
            require(np.array_equal(labels, other_labels) and batch.match == masked.match, "A/B label alignment differs")
            checks.append(dict(audit_mask(batch, masked), split=split))
    settings = read_json(args.decoder)
    settings["seed"] = configs["A"]["seed"]
    rows, checkpoints, metrics = [], {}, {}
    for arm in ("A", "B"):
        print("Auditing checkpoint " + arm, flush=True)
        config = configs[arm]
        t = config["training"]
        model, checkpoint, checkpoints[arm] = audit_checkpoint(
            getattr(args, "checkpoint_" + arm.lower()), config, original if arm == "A" else hidden, schema)
        metrics[arm] = {}
        for batch, _ in hidden["validation"]:
            print("Scoring {} {} with speed hidden".format(arm, batch.match.match_id), flush=True)
            normalized = encode(batch.values, batch.observed, checkpoint["normalization"])
            j = batch.feature_names.index("ball_speed_mps")
            require(np.isfinite(normalized).all() and (normalized[[j, j + len(batch.feature_names)]] == 0).all(),
                    "masked inference tensor is invalid")
            probabilities = predict_batch(batch, model, checkpoint["normalization"], t["core_frames"], t["device"])
            folder = out / arm / batch.match.match_id
            folder.mkdir(parents=True)
            write_batch(folder / "probabilities.npz", probabilities)
            # Position availability is independent of scalar speed. Both arms use this same mask.
            position_observed = (batch.observed[:, batch.feature_names.index("home_nearest_ball_m")] |
                                 batch.observed[:, batch.feature_names.index("away_nearest_ball_m")])
            np.savez_compressed(folder / "ball_observed.npz", ball_observed=position_observed)
            prediction = decode_stage(probabilities, {"ball_observed": folder / "ball_observed.npz"}, settings)
            write_predictions(folder / "submission.csv", prediction)
            entry = next(e for e in manifest["matches"] if e["match"]["match_id"] == batch.match.match_id)
            truth = read_predictions(entry["ground_truth"], batch.match)
            result = score_match(truth, prediction, batch.match)
            metrics[arm][batch.match.match_id] = metric_report(result)
            cm = np.asarray(result["confusion"])
            rows.append({"arm": arm, "match_id": batch.match.match_id,
                         "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
                         "dead_ball_recall": float(cm[2, 2] / cm[2].sum()),
                         "finite_inputs": True, "speed_input_and_flag_zero": True})
    means = [{"arm": arm, "match_id": "mean", **{
        key: float(np.mean([r[key] for r in rows if r["arm"] == arm]))
        for key in ("spellF1", "macroF1", "dead_ball_recall")}} for arm in ("A", "B")]
    report = {"status": "complete", "kind": "external_validation_speed_ablation", "test_accessed": False,
              "public_labels_used_for_fitting_or_scoring": False,
              "public_label_file_hashed_for_leakage_guard": True,
              "evaluation_speed_hidden_for_both": True,
              "configs": configs, "decoder": {"callable": "p3.stage:decode_stage", "settings": settings},
              "decoder_file": file_record(args.decoder), "checkpoints": checkpoints, "mask_checks": checks,
              "matches": rows, "means": means, "metrics": metrics,
              "source": source_snapshot(), "environment": environment(),
              "metric_definition": "spellF1 = mean F1 over IoUs 0.3,0.4,0.5,0.6,0.7; means weight matches equally"}
    write_json(out / "results.json", report)
    with (out / "results.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "match_id", "spellF1", "macroF1", "dead_ball_recall"],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + means)
    print(json.dumps({"matches": rows, "means": means}, indent=2))


def public_check(args):
    """Read only the public tracking window; never read or score its target CSV."""
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = read_json(args.public_meta)
    window = meta["public_window"]
    start, stop = window["first_video_frame"], window["last_video_frame"] + 1
    count, positions = 0, 0
    # The ATD parser requires local contiguous 1-based IDs. Preserve the explicit
    # mapping to original 0-based video frames alongside the window artifacts.
    with Path(args.atd_raw).open(encoding="utf-8") as source, (out / "public_atd.txt").open("x", encoding="utf-8") as target:
        for i, line in enumerate(source):
            if i >= stop:
                break
            raw_id, observations = line.split(":", 1)
            require(int(raw_id) == i + 1, "source ATD frame alignment changed")
            if i >= start:
                target.write(str(count + 1) + ":" + observations)
                positions += parse_observation(observations.split(":", 1)[1]) is not None
                count += 1
    require(count == stop - start == window["n_frames"], "public window is incomplete")
    spec = MatchSpec("BAR-ATH-public-pipeline", count, meta["fps"])
    batch = build_metrica_features(spec, args.atd_meta, out / "public_atd.txt", GAMES[GAME]["pitch"])
    masked = apply_feature_policy(batch, True)
    check = audit_mask(batch, masked)
    require(check["original_speed_observed"] == 0, "public sample unexpectedly contains scalar speed")
    geometry = (masked.observed[:, masked.feature_names.index("home_nearest_ball_m")] |
                masked.observed[:, masked.feature_names.index("away_nearest_ball_m")])
    require(int(geometry.sum()) == positions, "available public ball positions were lost")
    np.save(out / "original_video_frames.npy", batch.frames + start)
    write_batch(out / "features.npz", masked)
    models = {}
    for arm in ("A", "B"):
        seed_all(42, 1)
        path = getattr(args, "checkpoint_" + arm.lower())
        model, checkpoint = load_checkpoint(path)
        encoded = encode(masked.values, masked.observed, checkpoint["normalization"])
        require(np.isfinite(encoded).all(), "nonfinite public inputs")
        probabilities = predict_batch(masked, model, checkpoint["normalization"], checkpoint["core_frames"])
        require(np.array_equal(probabilities.frames + start, np.arange(start, stop)), "public frame IDs changed")
        write_batch(out / (arm + "_probabilities.npz"), probabilities)
        models[arm] = {"finite_inputs_and_probabilities": True, "frame_alignment_verified": True,
                       "checkpoint": file_record(path)}
    result = {"status": "complete", "purpose": "pipeline_check_only", "public_labels_accessed": False,
              "scores_computed": False, "original_video_frame_range": [start, stop],
              "local_to_original_frame_offset": start, "ball_positions_available_and_preserved": positions,
              "mask_check": check, "models": models,
              "inputs": [file_record(args.atd_meta), file_record(args.atd_raw), file_record(args.public_meta)]}
    write_json(out / "checks.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    comparison = sub.add_parser("compare", help="Audit and score A/B on the same speed-hidden validation matches")
    comparison.add_argument("--config-a", default="configs/p2_idsse_v1.json")
    comparison.add_argument("--config-b", default="configs/p2_idsse_no_ball_speed.json")
    comparison.add_argument("--decoder", default="configs/p2_speed_comparison_decoder.json")
    public = sub.add_parser("public-check", help="Check the public five-minute tracking window without scoring")
    public.add_argument("--atd-meta", required=True)
    public.add_argument("--atd-raw", required=True)
    public.add_argument("--public-meta", default="evaluation/gt/BAR-ATH_possession_gt_public_meta.json")
    for command in (comparison, public):
        command.add_argument("--checkpoint-a", required=True)
        command.add_argument("--checkpoint-b", required=True)
        command.add_argument("--out", required=True)
    args = parser.parse_args()
    (compare if args.command == "compare" else public_check)(args)


if __name__ == "__main__":
    main()
