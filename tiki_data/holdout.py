"""Freeze validation-selected inference and guard accidental test-set reuse."""

from pathlib import Path

from .artifacts import read_json, write_json
from .benchmark import input_fingerprints, load_manifest
from .contracts import SCHEMA_VERSION, require
from .pipeline import load_config
from .provenance import digest, environment, file_record, now, source_snapshot


def freeze(manifest_path, validation, method, out):
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    validation = Path(validation).resolve()
    run = read_json(validation / "run.json")
    report = read_json(validation / "report.json")
    require(run.get("status") == "complete" and run.get("kind") == "benchmark"
            and report.get("split") == "validation" and report.get("kind") == "comparison",
            "freeze requires a completed external/synthetic validation comparison")
    require(file_record(validation / "report.json") == run["report"], "validation report was modified")
    require(report["manifest"] == file_record(manifest_path), "manifest changed since validation")
    source, runtime = source_snapshot(), environment()
    require(source["sha256"] == run["source"]["sha256"], "source changed since validation")
    require(runtime["sha256"] == run["environment"]["sha256"], "environment changed since validation")
    require(method in {row["method"] for row in report["summary"]}, "method was not compared on validation")
    require(input_fingerprints(manifest, {"train", "validation"},
                               {row["method"] for row in report["summary"]}) == run["inputs"],
            "validation inputs changed since comparison")
    for entry in manifest["matches"]:
        if entry["split"] == "train":
            continue
        require(method in entry["predictions"], "frozen method missing on a validation/test match")
        source_spec = entry["predictions"][method]
        if manifest["kind"] == "external":
            require("config" in source_spec and load_config(source_spec["config"])["predictor"]["kind"]
                    in ("baseline", "model", "adapter"),
                    "external freezes require runnable inference, not precomputed labels/probabilities")
    # No test ground truth is opened by freeze. Test model/feature inputs are fingerprinted.
    inputs = input_fingerprints(manifest, {"validation", "test"}, {method}, labels=False)
    core = {"schema_version": SCHEMA_VERSION, "kind": "algorithm_freeze", "method": method,
            "manifest": file_record(manifest_path), "validation": file_record(validation / "report.json"),
            "source_sha256": source["sha256"], "environment_sha256": runtime["sha256"],
            "prediction_inputs": inputs, "evidence_kind": manifest["kind"]}
    record = dict(core, freeze_id=digest(core), created_at=now())
    write_json(out, record, exclusive=True)
    return record


def authorize_test(freeze_path, manifest_path, manifest, source, runtime):
    require(freeze_path is not None, "test evaluation requires --freeze from a completed validation comparison")
    freeze_path = Path(freeze_path).resolve()
    frozen = read_json(freeze_path)
    require(frozen.get("kind") == "algorithm_freeze", "invalid freeze artifact")
    core = {key: value for key, value in frozen.items() if key not in ("freeze_id", "created_at")}
    require(digest(core) == frozen.get("freeze_id"), "freeze artifact was modified")
    require(frozen["manifest"] == file_record(manifest_path), "manifest differs from the frozen selection")
    require(frozen["source_sha256"] == source["sha256"], "source differs from the frozen algorithm")
    require(frozen["environment_sha256"] == runtime["sha256"], "environment differs from the freeze")
    require(file_record(frozen["validation"]["path"]) == frozen["validation"], "frozen validation evidence was modified")
    require(input_fingerprints(manifest, {"validation", "test"}, {frozen["method"]}, labels=False)
            == frozen["prediction_inputs"], "prediction inputs differ from the freeze")
    attempt = freeze_path.parent / "test_attempts" / (frozen["freeze_id"] + ".json")
    require(not attempt.exists(), "test was already attempted for this freeze; preserve the holdout record")
    return frozen, attempt
