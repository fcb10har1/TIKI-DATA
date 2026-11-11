"""C/D preparation and a frozen, external-validation decoder comparison.

Run from repository root: python -m tiki_data.p2.motion_experiment --help
Public labels are scored only after the external decoder selection is saved.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from p3.stage import decode_stage
from tiki_data.artifacts import (read_batch, read_json, read_predictions, write_batch,
                                 write_json, write_predictions)
from tiki_data.benchmark import load_manifest, score_match
from tiki_data.contracts import CLASS_ORDER, FeatureBatch, MatchSpec, PredictionBatch, require
from tiki_data.p1.features import BALL_DEPENDENT, apply_feature_policy
from tiki_data.p1.kalman import DEFAULT_SETTINGS, augment, metrica_positions, motion_schema
from tiki_data.provenance import environment, file_record, source_snapshot
from .data import encode
from .inference import predict_batch
from .speed_ablation import audit_checkpoint
from .train import load_data, load_training_config

CONFIGS = dict(A="configs/p2_idsse_v1.json", B="configs/p2_idsse_no_ball_speed.json",
               C="configs/p2_idsse_c_optional_speed.json", D="configs/p2_idsse_d_kalman.json")
CHECKPOINTS = dict(A="artifacts/p2-idsse-cd715a3/best.pt",
                   B="artifacts/p2-idsse-no-ball-speed-20261111/best.pt",
                   C="artifacts/p2-idsse-c-optional-speed-20261111/best.pt",
                   D="artifacts/p2-idsse-d-kalman-20261111/best.pt")
WEIGHTS = (0., .25, .5, .75, 1.)
GAP_SECONDS = (2, 10)
GAP_SEED = 20261111


def ball_positions(root, spec):
    with np.load(Path(root) / (spec.match_id + ".npz"), allow_pickle=False) as data:
        require(MatchSpec.from_dict(json.loads(data["match"].item())) == spec, "position match mismatch")
        require(np.array_equal(data["frames"], np.arange(spec.n_frames)), "position frame IDs differ")
        require(np.array_equal(data["periods"], spec.periods), "position periods differ")
        return data["positions"].copy()


def geometry_audit(base, derived):
    require(np.array_equal(base.frames, derived.frames) and np.array_equal(base.periods, derived.periods),
            "D frame/period alignment differs")
    require(np.array_equal(base.values[:, 1:], derived.values[:, 1:19])
            and np.array_equal(base.observed[:, 1:], derived.observed[:, 1:19]), "D changed original geometry")
    require(not derived.observed[:, 0].any() and (derived.values[:, 0] == 0).all(), "D recorded speed is not hidden")
    return {"frame_and_period_ids_equal": True, "original_non_speed_features_preserved": True,
            "recorded_speed_zero_and_unobserved": True,
            "kalman_speed_valid_frames": int(derived.observed[:, 19].sum()),
            "finite_values": bool(np.isfinite(derived.values).all())}


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = load_training_config(CONFIGS["A"])
    manifest = load_manifest(config["evaluation_manifest"])
    write_json(out / "feature_schema.json", motion_schema(DEFAULT_SETTINGS))
    checks = {}
    for entry in manifest["matches"]:
        if entry["split"] == "test":
            continue
        spec = MatchSpec.from_dict(entry["match"])
        base = read_batch(config["features"][spec.match_id], spec, "features")
        positions = ball_positions(args.positions, spec)
        derived = augment(base, positions, DEFAULT_SETTINGS)
        checks[spec.match_id] = dict(geometry_audit(base, derived), split=entry["split"],
                                    source=file_record(Path(args.positions) / (spec.match_id + ".npz")))
        write_batch(out / (spec.match_id + ".npz"), derived)
        print("Prepared D " + spec.match_id, flush=True)
    write_json(out / "checks.json", {"matches": checks, "test_accessed": False,
                                     "kalman": DEFAULT_SETTINGS, "source": source_snapshot()})


def gap_mask(spec, seconds):
    """Label-independent 40% block occlusion, deterministic per match/profile."""
    rng = np.random.default_rng(GAP_SEED + int(seconds) * 1000 + sum(spec.match_id.encode("utf-8")))
    missing = np.zeros(spec.n_frames, dtype=bool)
    width = int(round(seconds * spec.fps))
    for start, stop in spec.periods:
        for left in range(start, stop, width):
            if rng.random() < .4:
                missing[left:min(left+width, stop)] = True
    return missing


def hide_positions(base, positions, missing):
    """Recompute D motion after this operation; never leave estimates of hidden data."""
    values, observed = base.values.copy(), base.observed.copy()
    indices = [base.feature_names.index(name) for name in BALL_DEPENDENT]
    observed[np.ix_(missing, indices)] = False
    values[~observed] = 0
    raw = positions.copy()
    raw[missing] = np.nan
    return FeatureBatch(base.match, base.frames.copy(), values, observed, base.feature_names,
                        base.periods.copy()).validate(base.match), raw


def summarize(truth, prediction):
    result = score_match(truth, prediction, truth.match)
    cm = np.asarray(result["confusion"])
    n = int(cm[:3].sum())
    correct = int(np.trace(cm[:3, :3]))
    return dict(correct_frames=correct, scored_frames=n, accuracy_pct=100*correct/n,
                spellF1=result["spell_f1_mean"], macroF1=result["macro_f1"],
                dead_ball_recall=float(cm[2, 2]/cm[2].sum()), confusion=cm.tolist())


def means(rows):
    result = []
    for scenario, arm in sorted({(r["scenario"], r["arm"]) for r in rows}):
        chosen = [r for r in rows if (r["scenario"], r["arm"]) == (scenario, arm)]
        count = sum(r["scored_frames"] for r in chosen)
        correct = sum(r["correct_frames"] for r in chosen)
        result.append(dict(scenario=scenario, arm=arm, correct_frames=correct, scored_frames=count,
                           pooled_accuracy_pct=100*correct/count, **{key:float(np.mean([r[key] for r in chosen]))
                             for key in ("accuracy_pct", "spellF1", "macroF1", "dead_ball_recall")}))
    return result


def evaluate(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    # Declare selection before new predictions or public scoring. The existing
    # training leakage guard hashes the public CSV but does not use its labels.
    protocol = dict(weights=list(WEIGHTS), fixed_weight=.5, temperature=4., seed=GAP_SEED,
                    gap_block_seconds=list(GAP_SECONDS), missing_block_probability=.4,
                    objective="mean frame accuracy across B,C,D, both external validation matches and both gap profiles",
                    ties="closest to 0.5, then smaller weight", public_used_for_selection=False,
                    test_accessed=False, recorded_speed_hidden_for_primary_comparison=True)
    write_json(out / "protocol.json", protocol)
    configs = {arm:load_training_config(getattr(args, "config_" + arm.lower())) for arm in CONFIGS}
    for arm in ("B", "C", "D"):
        for key in ("evaluation_manifest", "seed", "fps", "model", "training"):
            require(configs[arm][key] == configs["A"][key], "training settings differ: " + key)
    models, checkpoints, audits = {}, {}, {}
    for arm in CONFIGS:
        print("Audit " + arm, flush=True)
        manifest, schema, data, _ = load_data(configs[arm])
        models[arm], checkpoints[arm], audits[arm] = audit_checkpoint(
            getattr(args, "checkpoint_" + arm.lower()), configs[arm], data, schema)
        audits[arm]["selection_speed_hidden"] = bool(configs[arm].get("p1", {}).get("mask_ball_speed", False)
              or configs[arm].get("validation", {}).get("mask_ball_speed", False))
        audits[arm]["parameters"] = sum(p.numel() for p in models[arm].parameters())
        if arm == "A":
            original_validation = [batch for batch, _ in data["validation"]]
    settings = dict(read_json(args.decoder), seed=42, missing_weight=.5)
    require(settings["temperature"] == 4., "comparison temperature must remain 4")
    rows, grid, raw_rows, feature_checks = [], [], [], []
    for base in original_validation:
        spec = base.match
        entry = next(e for e in manifest["matches"] if e["match"]["match_id"] == spec.match_id)
        truth = read_predictions(entry["ground_truth"], spec)
        positions = ball_positions(args.positions, spec)
        for scenario in ("speed_available", "speed_missing", *("gap_%ds" % s for s in GAP_SECONDS)):
            print("External {} {}".format(spec.match_id, scenario), flush=True)
            scenario_base, raw_positions = base, positions
            if scenario != "speed_available":
                scenario_base = apply_feature_policy(base, True)
            if scenario.startswith("gap_"):
                seconds = int(scenario.split("_")[1][:-1])
                scenario_base, raw_positions = hide_positions(scenario_base, positions, gap_mask(spec, seconds))
            derived = augment(scenario_base, raw_positions, DEFAULT_SETTINGS)
            geometry_audit(scenario_base, derived)
            folder = out / "external" / spec.match_id / scenario
            folder.mkdir(parents=True)
            observed = np.isfinite(raw_positions).all(axis=1)
            require(np.array_equal(observed, scenario_base.observed[:, 1] | scenario_base.observed[:, 2]),
                    "raw ball availability differs from base geometry")
            mask_path = folder / "ball_observed.npz"
            np.savez_compressed(mask_path, ball_observed=observed)
            for arm in CONFIGS:
                features = derived if arm == "D" else scenario_base
                features = apply_feature_policy(features, checkpoints[arm]["normalization"].get("mask_ball_speed", False))
                encoded = encode(features.values, features.observed, checkpoints[arm]["normalization"])
                require(np.isfinite(encoded).all(), "nonfinite inference input")
                speed_hidden = not features.observed[:, 0].any()
                if scenario != "speed_available" or arm in ("B", "D"):
                    require(speed_hidden and (encoded[[0, len(features.feature_names)]] == 0).all(), "speed leaked")
                feature_checks.append(dict(match=spec.match_id, scenario=scenario, arm=arm,
                                           finite_inputs=True, scalar_speed_hidden=speed_hidden,
                                           observed_ball_positions=int(observed.sum()),
                                           original_frame_ids_preserved=np.array_equal(features.frames, base.frames)))
                probabilities = predict_batch(features, models[arm], checkpoints[arm]["normalization"])
                write_batch(folder / (arm + "_probabilities.npz"), probabilities)
                raw = PredictionBatch(spec, base.frames.copy(), np.asarray(CLASS_ORDER)[probabilities.probabilities.argmax(axis=1)])
                raw_rows.append(dict(arm=arm, match_id=spec.match_id, scenario=scenario, **summarize(truth, raw)))
                for weight in (WEIGHTS if scenario.startswith("gap_") and arm in ("B", "C", "D") else (.5,)):
                    decoded = decode_stage(probabilities, {"ball_observed": mask_path}, dict(settings, missing_weight=weight))
                    row = dict(arm=arm, match_id=spec.match_id, scenario=scenario, weight=weight, **summarize(truth, decoded))
                    if scenario.startswith("gap_"):
                        grid.append(row)
                    if weight == .5:
                        rows.append(row)
                        write_predictions(folder / (arm + "_decoded.csv"), decoded)
    objectives = {str(w):float(np.mean([r["accuracy_pct"] for r in grid if r["weight"] == w and r["arm"] in ("B", "C", "D")]))
                  for w in WEIGHTS}
    selected = max(WEIGHTS, key=lambda w:(objectives[str(w)], -abs(w-.5), -w))
    selection = dict(protocol, selected_weight=selected, objective_by_weight=objectives,
                     external_fixed_weight_means=means(rows), selected_before_public_scoring=True,
                     public_label_file_hashed_for_training_leakage_guard=True)
    write_json(out / "decoder_selection.json", selection)
    write_json(out / "selected_decoder.json", dict(settings, missing_weight=selected,
               fitted=sorted(set(settings.get("fitted", [])) | {"missing_weight"})))
    write_json(out / "external_results.json", dict(matches=rows, means=means(rows), raw=raw_rows,
                                                   grid=grid, feature_checks=feature_checks, audits=audits))
    print("External decoder selection: " + json.dumps(objectives) + "; selected=" + str(selected), flush=True)
    public = score_public(args.public_inputs, out / "public", models, checkpoints, settings, selected)
    report = dict(status="complete", external_matches=rows, external_means=means(rows), external_raw=raw_rows,
                  decoder_selection=selection, public=public, checkpoint_audits=audits,
                  feature_checks=feature_checks, source=source_snapshot(), environment=environment(),
                  limitations=["single seed; two external validation matches, not an independent final test",
                    "public five minutes is repeatedly inspected development data, not a full-match accuracy estimate",
                    "synthetic position gaps may differ from Barca tracking failures",
                    "D adds 3 features and 6 input channels; hidden TCN architecture and training budget unchanged",
                    "A selected with observed speed; B,C,D selected with recorded speed hidden"])
    write_json(out / "results.json", report)
    print(json.dumps({"public": public["rows"], "decoder_selection": selection}, indent=2), flush=True)


def score_public(source, out, models, checkpoints, settings, selected):
    source, out = Path(source), Path(out)
    out.mkdir(parents=True)
    mapping = np.load(source / "original_video_frames.npy", allow_pickle=False)
    spec = MatchSpec("BAR-ATH-public-pipeline", len(mapping), 25)
    base = read_batch(source / "features.npz", spec, "features")
    positions = metrica_positions(spec, source / "public_atd.txt", (105.3, 68))
    derived = augment(base, positions, DEFAULT_SETTINGS)
    checks = geometry_audit(base, derived)
    require(not base.observed[:, 0].any(), "public speed unexpectedly available")
    observed = np.isfinite(positions).all(axis=1)
    require(np.array_equal(observed, base.observed[:, 1] | base.observed[:, 2]), "public position flags differ")
    np.savez_compressed(out / "ball_observed.npz", ball_observed=observed)
    write_batch(out / "D_features.npz", derived)
    full_spec = MatchSpec("BAR-ATH", 199319, 25)
    truth_path = "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    truth = read_predictions(truth_path, full_spec)
    require(np.array_equal(np.flatnonzero(truth.labels != "X"), mapping), "public frames differ from scored labels")
    require(np.array_equal(mapping, np.arange(mapping[0], mapping[-1]+1)), "public mapping not contiguous")
    rows = []
    for arm in CONFIGS:
        features = derived if arm == "D" else base
        normalized = encode(features.values, features.observed, checkpoints[arm]["normalization"])
        require(np.isfinite(normalized).all() and (normalized[[0, len(features.feature_names)]] == 0).all(),
                "public input nonfinite or speed unmasked")
        probabilities = predict_batch(features, models[arm], checkpoints[arm]["normalization"])
        if arm in ("A", "B"):
            previous = read_batch(source / (arm + "_probabilities.npz"), spec, "probabilities")
            require(np.array_equal(previous.probabilities, probabilities.probabilities), "A/B public baseline changed")
        write_batch(out / (arm + "_probabilities.npz"), probabilities)
        raw_labels = np.asarray(CLASS_ORDER)[probabilities.probabilities.argmax(axis=1)]
        for weight in sorted({.5, selected}):
            decoded = decode_stage(probabilities, {"ball_observed":out / "ball_observed.npz"}, dict(settings, missing_weight=weight))
            full_labels = np.full(full_spec.n_frames, "X", dtype="<U1")
            full_labels[mapping] = decoded.labels
            full = PredictionBatch(full_spec, np.arange(full_spec.n_frames), full_labels)
            write_predictions(out / ("%s_weight_%s.csv" % (arm, weight)), full)
            rows.append(dict(arm=arm, weight=weight, **summarize(truth, full),
                             raw_accuracy_pct=float(100*np.mean(raw_labels == truth.labels[mapping]))))
    return dict(rows=rows, checks=checks, scored_frames=len(mapping), original_video_frame_range_inclusive=[int(mapping[0]), int(mapping[-1])],
                ball_position_missing_frames=int((~observed).sum()), fitted_on_public=False,
                input_files=[file_record(source / "features.npz"), file_record(source / "public_atd.txt"), file_record(truth_path)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--positions", default="runs/cd-ball-inputs")
    p.add_argument("--out", default="runs/cd-features")
    p.set_defaults(run=prepare)
    e = sub.add_parser("evaluate")
    e.add_argument("--positions", default="runs/cd-ball-inputs")
    e.add_argument("--public-inputs", default="runs/speed-ablation-public-20261111")
    e.add_argument("--decoder", default="configs/p2_speed_comparison_decoder.json")
    e.add_argument("--out", default="runs/abcd-comparison-20261111")
    for arm in CONFIGS:
        e.add_argument("--config-" + arm.lower(), default=CONFIGS[arm])
        e.add_argument("--checkpoint-" + arm.lower(), default=CHECKPOINTS[arm])
    e.set_defaults(run=evaluate)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
