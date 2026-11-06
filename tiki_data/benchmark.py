"""Match-level external comparisons and a separate public pipeline smoke check."""

import csv
from pathlib import Path
import time

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, GAMES
from .artifacts import read_json, read_predictions, resolve_path, write_json
from .contracts import MatchSpec, SCHEMA_VERSION, identifier, integer, require
from .pipeline import config_inputs, load_config, name, object_fields, run
from .provenance import ROOT, environment, file_record, now, sha256, source_snapshot

REQUIRED_METHODS = {"rules", "tcn", "tcn_viterbi"}
PUBLIC_GT = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"


def load_manifest(path):
    path = Path(path).resolve()
    value = read_json(path)
    object_fields(value, {"schema_version", "dataset_id", "kind", "matches"},
                  {"schema_version", "dataset_id", "kind", "matches"})
    require(integer(value["schema_version"]) and value["schema_version"] == SCHEMA_VERSION,
            "unsupported manifest schema_version")
    name(value["dataset_id"])
    require(value["kind"] in ("external", "synthetic"), "benchmarks must be external or synthetic")
    require(isinstance(value["matches"], list) and bool(value["matches"]), "manifest needs matches")
    ids, groups, label_paths = set(), set(), set()
    matches = []
    for entry in value["matches"]:
        object_fields(entry, {"match", "group_id", "split", "provenance", "ground_truth", "predictions"},
                      {"match", "group_id", "split", "provenance", "ground_truth"})
        spec = MatchSpec.from_dict(entry["match"])
        name(spec.match_id)
        match_id, group_id = identifier(spec.match_id), identifier(entry["group_id"])
        require(match_id != identifier(GAME) and group_id != identifier(GAME),
                "the supplied BAR-ATH match is not external validation/test data")
        require(match_id not in ids and group_id not in groups,
                "each underlying match/group may appear only once in the split manifest")
        require(entry["split"] in ("train", "validation", "test"), "split must be train, validation or test")
        provenance = entry["provenance"]
        require(isinstance(provenance, dict) and {"dataset", "source", "label_provenance"} <= set(provenance)
                and all(isinstance(provenance[key], str) and provenance[key].strip()
                        for key in ("dataset", "source", "label_provenance")),
                "dataset, source and label_provenance must be documented")
        gt = resolve_path(entry["ground_truth"], path.parent)
        require(gt != PUBLIC_GT.resolve() and gt not in label_paths, "ground-truth file is public or reused")
        predictions = entry.get("predictions", {})
        require(isinstance(predictions, dict), "predictions must map method names to sources")
        resolved = {}
        for method, source in predictions.items():
            name(method)
            require(isinstance(source, dict) and len(source) == 1 and set(source) <= {"csv", "config"},
                    "each prediction source must contain exactly one csv or config path")
            key = next(iter(source))
            resolved[method] = {key: str(resolve_path(source[key], path.parent))}
        matches.append(dict(entry, match=spec.as_dict(), ground_truth=str(gt), predictions=resolved))
        ids.add(match_id)
        groups.add(group_id)
        label_paths.add(gt)
    require({entry["split"] for entry in matches} == {"train", "validation", "test"},
            "declare nonempty train, validation and test partitions before benchmarking")
    return dict(value, matches=matches)


def input_fingerprints(manifest, splits, methods=None, labels=True):
    """Fingerprint only permitted partitions; validation never opens test files."""
    records = {}
    label_hashes = {}
    public_hash = sha256(PUBLIC_GT)
    ground_truth_paths = {entry["ground_truth"] for entry in manifest["matches"]}
    for entry in manifest["matches"]:
        if entry["split"] not in splits:
            continue
        if labels:
            record = file_record(entry["ground_truth"])
            require(record["sha256"] != public_hash, "public BAR-ATH labels cannot be used in an external benchmark")
            require(record["sha256"] not in label_hashes, "duplicate ground-truth contents across match groups")
            label_hashes[record["sha256"]] = entry["group_id"]
            records[record["path"]] = record
        if entry["split"] == "train":
            continue
        for method, source in entry["predictions"].items():
            if methods is not None and method not in methods:
                continue
            key, path = next(iter(source.items()))
            require(path not in ground_truth_paths, "prediction source must not be a ground-truth file")
            record = file_record(path)
            records[record["path"]] = record
            if key == "config":
                config = load_config(path)
                require(config["method"] == method, "prediction method and configuration method disagree")
                require(config["match"] == entry["match"], "prediction configuration belongs to a different match")
                require(config["data_kind"] == manifest["kind"], "configuration and manifest data kinds disagree")
                for record in config_inputs(config):
                    require(record["path"] not in ground_truth_paths, "inference inputs must not include ground truth")
                    records[record["path"]] = record
    return sorted(records.values(), key=lambda record: record["path"])


def score_match(ground_truth, prediction, spec, bootstrap=0):
    require(integer(bootstrap) and bootstrap >= 0, "bootstrap must be a nonnegative integer")
    ground_truth.validate(spec)
    prediction.validate(spec)
    gt = pm.label_codes(ground_truth.labels)
    pred = pm.label_codes(prediction.labels)
    require(np.all(gt[spec.period_ids() == 0] == pm.X), "ground truth outside declared periods must be X")
    taus = tuple(sorted({max(1, int(tau * spec.fps / pm.FPS + 0.5)) for tau in pm.TAUS}))
    # The official transition matcher assumes long gaps between runs. Make that invariant true
    # for external annotation gaps and adjacent halves too, then map report frames back.
    gt_parts, pred_parts, frame_parts = [], [], []
    for start, stop in spec.periods:
        for a, b in pm.valid_runs(gt[start:stop]):
            a, b = a + start, b + start
            if gt_parts:
                gap = np.full(max(taus) + 1, pm.X, dtype=np.int8)
                gt_parts.append(gap)
                pred_parts.append(gap.copy())
                frame_parts.append(np.full(len(gap), -1, dtype=np.int64))
            gt_parts.append(gt[a:b])
            pred_parts.append(pred[a:b])
            frame_parts.append(np.arange(a, b))
    require(bool(gt_parts), "ground truth has no graded frames")
    mapped = np.concatenate(frame_parts)
    result = pm.evaluate(np.concatenate(gt_parts), np.concatenate(pred_parts), taus=taus, bootstrap_n=bootstrap)
    result["n_frames"] = spec.n_frames
    result["confusion"] = pm.confusion(gt, pred).tolist()
    for key in ("mean_spell_duration_s", "gt_mean_spell_duration_s"):
        result[key] *= pm.FPS / spec.fps
    for transition in result["transitions"].values():
        def original(event):
            return (int(mapped[event[0]]), event[1], event[2])
        transition["pairs"] = [(original(a), original(b)) for a, b in transition["pairs"]]
        transition["missed"] = [original(event) for event in transition["missed"]]
        transition["spurious"] = [original(event) for event in transition["spurious"]]
    return result


def metric_report(result):
    return dict(result, transitions={str(tau): {key: value for key, value in transition.items()
                                              if key not in ("pairs", "missed", "spurious")}
                                     for tau, transition in result["transitions"].items()})


def write_transitions(path, result):
    fields = ("tolerance_frames", "status", "gt_frame", "pred_frame", "latency_frames",
              "gt_from", "gt_to", "pred_from", "pred_to")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for tau, transitions in result["transitions"].items():
            rows = []
            for gt, pred in transitions["pairs"]:
                rows.append({"status": "matched", "gt_frame": gt[0], "pred_frame": pred[0],
                             "latency_frames": pred[0] - gt[0], "gt_from": pm.LABELS[gt[1]],
                             "gt_to": pm.LABELS[gt[2]], "pred_from": pm.LABELS[pred[1]], "pred_to": pm.LABELS[pred[2]]})
            for kind in ("missed", "spurious"):
                for event in transitions[kind]:
                    side = "gt" if kind == "missed" else "pred"
                    rows.append({"status": kind, side + "_frame": event[0], side + "_from": pm.LABELS[event[1]],
                                 side + "_to": pm.LABELS[event[2]]})
            rows.sort(key=lambda row: (row.get("gt_frame", row.get("pred_frame")), row["status"]))
            for row in rows:
                writer.writerow(dict(row, tolerance_frames=tau))


def summarize(entries):
    methods = sorted({entry["method"] for entry in entries})
    summary = []
    for method in methods:
        rows = [entry for entry in entries if entry["method"] == method]
        summary.append({"method": method, "matches": len(rows),
                        "spell_f1_mean": float(np.mean([row["metrics"]["spell_f1_mean"] for row in rows])),
                        "macro_f1": float(np.mean([row["metrics"]["macro_f1"] for row in rows])),
                        "mean_prediction_seconds": float(np.mean([row["prediction_seconds"] for row in rows]))})
    summary.sort(key=lambda row: (-row["spell_f1_mean"], -row["macro_f1"], row["method"]))
    for rank, row in enumerate(summary, 1):
        row["rank"] = rank
    return summary


def priorities(entries, method):
    scores = [entry["metrics"] for entry in entries if entry["method"] == method]
    headline = [score["spell_f1"][pm.SPELL_IOU_HEADLINE] for score in scores]
    missed = sum(score["n_missed"] for score in headline)
    spurious = sum(score["n_spurious"] for score in headline)
    n_gt = sum(score["n_gt_spells"] for score in scores)
    n_pred = sum(score["n_pred_spells"] for score in scores)
    cm = np.sum([np.asarray(score["confusion"]) for score in scores], axis=0)
    dead_errors = int(cm[pm.D, :].sum() - cm[pm.D, pm.D] + cm[:pm.X, pm.D].sum() - cm[pm.D, pm.D])
    graded = sum(score["n_graded"] for score in scores)
    candidates = [
        {"priority": "missed possessions at IoU 0.5", "count": missed, "denominator": n_gt,
         "rate": missed / n_gt if n_gt else 0.0},
        {"priority": "spurious possessions at IoU 0.5", "count": spurious, "denominator": n_pred,
         "rate": spurious / n_pred if n_pred else 0.0},
        {"priority": "dead-ball frame errors (missed or falsely predicted D)", "count": dead_errors,
         "denominator": graded, "rate": dead_errors / graded if graded else 0.0}]
    candidates.sort(key=lambda row: (-row["rate"], row["priority"]))
    return [row for row in candidates if row["count"] > 0][:2]


def write_comparison(out, report):
    columns = ("rank", "method", "matches", "spell_f1_mean", "macro_f1", "mean_prediction_seconds")
    with (out / "comparison.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["summary"])
    lines = ["# P4 comparison", "", "Evidence: **{}**. Split: **{}**. Accuracy evidence: **{}**.".format(
        report["evidence_kind"], report["split"], report["accuracy_evidence"]), "",
        "Each match has equal weight. Primary metric: official spellF1; tie-break: macroF1.", "",
        "| Method | Matches | spellF1 | macroF1 |", "|---|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append("| {} | {} | {:.6f} | {:.6f} |".format(row["method"], row["matches"], row["spell_f1_mean"], row["macro_f1"]))
    lines += ["", "## Error-analysis priorities", "",
              "For the leading method. Rates have different denominators; these are diagnostic",
              "triage signals, not causal estimates of the gains from fixing them.", ""]
    for priority in report["priorities"]:
        lines.append("- {}: {} / {} ({:.1%}).".format(priority["priority"], priority["count"],
                                                     priority["denominator"], priority["rate"]))
    if not report["priorities"]:
        lines.append("No errors in the measured diagnostics; do not infer robustness beyond these matches.")
    lines += ["", "Per-match JSON and transition CSVs retain the evidence. Bootstrap intervals are",
              "per match, when requested; no frame-independent or aggregate confidence interval is claimed.",
              "Synthetic scores establish integration behavior only. Public five-minute scores never select settings.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def benchmark(manifest_path, out, split="validation", bootstrap=0, freeze_path=None):
    require(split in ("validation", "test"), "benchmark split must be validation or test")
    require(integer(bootstrap) and bootstrap >= 0, "bootstrap must be a nonnegative integer")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    selected = [entry for entry in manifest["matches"] if entry["split"] == split]
    source, runtime = source_snapshot(), environment()
    manifest_record = file_record(manifest_path)
    if split == "test":
        from .holdout import authorize_test
        frozen, attempt_path = authorize_test(freeze_path, manifest_path, manifest, source, runtime)
        methods = {frozen["method"]}
    else:
        require(freeze_path is None, "--freeze is only used for the test split")
        methods = set(selected[0]["predictions"])
        require(REQUIRED_METHODS <= methods, "validation needs rules, tcn and tcn_viterbi on every match")
        require(all(set(entry["predictions"]) == methods for entry in selected),
                "all validation matches must have the same methods")
        attempt_path = None
    require(all(methods <= set(entry["predictions"]) for entry in selected), "selected method missing on a match")
    # Train labels detect duplicated training/validation data, but test bytes stay unopened here.
    splits = {"train", "validation", "test"} if split == "test" else {"train", "validation"}
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": SCHEMA_VERSION, "kind": "benchmark", "status": "running", "started_at": now(),
              "split": split, "manifest": manifest_record, "source": source, "environment": runtime,
              "bootstrap": bootstrap, "evidence_kind": manifest["kind"], "dataset_id": manifest["dataset_id"]}
    write_json(out / "run.json", record)
    started = time.perf_counter()
    reserved = False
    try:
        if attempt_path is not None:
            write_json(attempt_path, {"status": "started", "started_at": now(), "output": str(out),
                                      "freeze_id": frozen["freeze_id"]}, exclusive=True)
            reserved = True
        inputs = input_fingerprints(manifest, splits, methods=methods)
        record["inputs"] = inputs
        entries = []
        reports = out / "matches"
        reports.mkdir()
        for entry in selected:
            spec = MatchSpec.from_dict(entry["match"])
            gt = read_predictions(entry["ground_truth"], spec)
            if spec.n_frames == GAMES[GAME]["n_video_frames"]:
                public = read_predictions(PUBLIC_GT, MatchSpec(GAME, spec.n_frames, 25))
                require(not np.array_equal(gt.labels, public.labels),
                        "public BAR-ATH labels cannot be renamed as an external match")
            for method in sorted(methods):
                prediction_source = entry["predictions"][method]
                tick = time.perf_counter()
                if "config" in prediction_source:
                    prediction_run = run(prediction_source["config"], out / "predictions" / spec.match_id / method)
                    prediction_path = prediction_run["output"]["path"]
                else:
                    prediction_path = prediction_source["csv"]
                prediction_seconds = time.perf_counter() - tick
                prediction = read_predictions(prediction_path, spec)
                result = score_match(gt, prediction, spec, bootstrap=bootstrap)
                stem = name(spec.match_id) + "__" + method
                write_json(reports / (stem + ".json"), metric_report(result))
                write_transitions(reports / (stem + "_transitions.csv"), result)
                entries.append({"match_id": spec.match_id, "group_id": entry["group_id"], "fps": spec.fps,
                                "method": method, "prediction_source": prediction_source,
                                "prediction": file_record(prediction_path), "prediction_seconds": prediction_seconds,
                                "metrics": result})
        summary = summarize(entries)
        report = {"schema_version": SCHEMA_VERSION, "kind": "comparison", "status": "complete",
                  "split": split, "manifest": manifest_record, "dataset_id": manifest["dataset_id"],
                  "evidence_kind": manifest["kind"], "accuracy_evidence": manifest["kind"] == "external",
                  "source": source, "environment": runtime, "summary": summary,
                  "priorities": priorities(entries, summary[0]["method"]),
                  "entries": [dict(entry, metrics=metric_report(entry["metrics"])) for entry in entries]}
        require(input_fingerprints(manifest, splits, methods=methods) == inputs, "benchmark inputs changed during evaluation")
        require(file_record(manifest_path) == manifest_record, "manifest changed during evaluation")
        require(source_snapshot()["sha256"] == source["sha256"], "source changed during evaluation")
        require(environment()["sha256"] == runtime["sha256"], "environment changed during evaluation")
        write_comparison(out, report)
        write_json(out / "report.json", report)
        record.update(status="complete", report=file_record(out / "report.json"))
    except Exception as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        record.update(finished_at=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(out / "run.json", record)
        if reserved:
            write_json(attempt_path, {"status": record["status"], "finished_at": now(),
                                      "output": str(out), "freeze_id": frozen["freeze_id"]})
    return report


def smoke(prediction_path, out, bootstrap=0):
    require(integer(bootstrap) and bootstrap >= 0, "bootstrap must be a nonnegative integer")
    spec = MatchSpec(GAME, GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"])
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": SCHEMA_VERSION, "kind": "public_smoke", "status": "running", "started_at": now(),
              "purpose": "pipeline_alignment_and_label_interpretation_only", "accuracy_evidence": False,
              "source": source_snapshot(), "environment": environment()}
    write_json(out / "run.json", record)
    started = time.perf_counter()
    try:
        inputs = [file_record(prediction_path), file_record(PUBLIC_GT)]
        record["inputs"] = inputs
        gt = read_predictions(PUBLIC_GT, spec)
        prediction = read_predictions(prediction_path, spec)
        result = score_match(gt, prediction, spec, bootstrap)
        require([file_record(prediction_path), file_record(PUBLIC_GT)] == inputs, "smoke inputs changed")
        report = dict(record, status="complete", prediction=inputs[0], ground_truth=inputs[1], metrics=metric_report(result))
        write_json(out / "report.json", report)
        write_transitions(out / "transitions.csv", result)
        record.update(status="complete", report=file_record(out / "report.json"))
    except Exception as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        record.update(finished_at=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(out / "run.json", record)
    return report
