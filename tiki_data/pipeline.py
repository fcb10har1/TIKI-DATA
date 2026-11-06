"""One-command inference, declared stage adapters, and byte-level reproduction."""

import importlib
import inspect
import json
from pathlib import Path
import random
import re
import time
from types import MappingProxyType

import numpy as np

from games import GAME, GAMES, asset_paths
from starter import baseline
from .artifacts import (read_batch, read_json, read_predictions, resolve_path,
                        write_batch, write_json, write_predictions)
from .contracts import (CLASS_ORDER, FeatureBatch, MatchSpec, PredictionBatch,
                        ProbabilityBatch, SCHEMA_VERSION, identifier, integer, require)
from .provenance import ROOT, digest, environment, file_record, now, source_snapshot


def name(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value),
            "names must contain 1-64 letters, digits, underscores or hyphens")
    return value


def object_fields(value, allowed, required=()):
    require(isinstance(value, dict), "configuration section must be an object")
    require(not set(value) - set(allowed), "unknown configuration fields: " + str(sorted(set(value) - set(allowed))))
    require(set(required) <= set(value), "missing configuration fields: " + str(sorted(set(required) - set(value))))


def stage_config(value, base, model=False):
    allowed = {"callable", "inputs", "settings"} | ({"expected_features"} if model else set())
    object_fields(value, allowed, {"callable"} | ({"expected_features"} if model else set()))
    target = value["callable"]
    require(isinstance(target, str) and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", target),
            "adapter callable must be module:function")
    inputs = value.get("inputs", {})
    settings = value.get("settings", {})
    require(isinstance(inputs, dict) and all(isinstance(key, str) for key in inputs), "inputs must be a path mapping")
    require(isinstance(settings, dict), "settings must be an object")
    out = {"callable": target, "inputs": {key: str(resolve_path(path, base)) for key, path in inputs.items()},
           "settings": settings}
    if model:
        names = value["expected_features"]
        require(isinstance(names, list) and bool(names) and all(isinstance(x, str) and x.strip() for x in names)
                and len(set(names)) == len(names), "expected_features must list the trained feature order")
        out["expected_features"] = names
    return out


def load_config(path):
    path = Path(path).resolve()
    value = read_json(path)
    object_fields(value, {"schema_version", "match", "method", "data_kind", "seed", "predictor", "decoder"},
                  {"schema_version", "match", "method", "data_kind", "seed", "predictor"})
    require(integer(value["schema_version"]) and value["schema_version"] == SCHEMA_VERSION,
            "unsupported configuration schema_version")
    spec = MatchSpec.from_dict(value["match"])
    name(value["method"])
    require(value["data_kind"] in ("supplied", "external", "synthetic"), "invalid data_kind")
    require((identifier(spec.match_id) == identifier(GAME)) == (value["data_kind"] == "supplied"),
            "the supplied match must be identified as supplied; other matches must be external or synthetic")
    require(integer(value["seed"]) and 0 <= value["seed"] < 2 ** 32, "seed must be an integer in [0,2**32)")
    predictor = value["predictor"]
    require(isinstance(predictor, dict), "predictor must be an object")
    kind = predictor.get("kind")
    base = path.parent
    if kind == "baseline":
        object_fields(predictor, {"kind", "window", "inputs", "pitch"}, {"kind", "window"})
        window = predictor["window"]
        require(integer(window) and window >= 0, "baseline window must be a nonnegative integer")
        inputs = predictor.get("inputs")
        if inputs is None:
            require(value["data_kind"] == "supplied", "external baselines need explicit ATD inputs and pitch")
            paths = asset_paths()
            inputs = {key: paths[key] for key in ("atd_meta", "atd_raw")}
        require(isinstance(inputs, dict) and set(inputs) == {"atd_meta", "atd_raw"},
                "baseline inputs must be atd_meta and atd_raw")
        pitch = predictor.get("pitch", list(GAMES[GAME]["pitch"]))
        require(value["data_kind"] == "supplied" or "pitch" in predictor, "external baselines need pitch dimensions")
        require(isinstance(pitch, list) and len(pitch) == 2
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) and x > 0 for x in pitch),
                "pitch must contain positive finite length and width in metres")
        resolved = {"kind": kind, "window": window, "pitch": pitch,
                    "inputs": {key: str(resolve_path(p, base)) for key, p in inputs.items()}}
    elif kind in ("labels", "probabilities"):
        object_fields(predictor, {"kind", "path"}, {"kind", "path"})
        resolved = {"kind": kind, "path": str(resolve_path(predictor["path"], base))}
    elif kind == "adapter":
        object_fields(predictor, {"kind", "stage"}, {"kind", "stage"})
        resolved = {"kind": kind, "stage": stage_config(predictor["stage"], base)}
    elif kind == "model":
        object_fields(predictor, {"kind", "features", "model"}, {"kind", "features", "model"})
        features = predictor["features"]
        if isinstance(features, dict) and "file" in features:
            object_fields(features, {"file"}, {"file"})
            features = {"file": str(resolve_path(features["file"], base))}
        else:
            features = stage_config(features, base)
        resolved = {"kind": kind, "features": features,
                    "model": stage_config(predictor["model"], base, model=True)}
    else:
        raise ValueError("predictor kind must be baseline, labels, probabilities, adapter or model")
    result = dict(value, match=spec.as_dict(), predictor=resolved)
    if "decoder" in value:
        require(kind not in ("labels", "baseline"), "a decoder requires probability output")
        result["decoder"] = stage_config(value["decoder"], base)
    def reject_fixture(stage):
        if isinstance(stage, dict):
            if str(stage.get("callable", "")).startswith("tiki_data.demo:"):
                require(value["data_kind"] == "synthetic", "example adapters require synthetic data_kind")
            for key, child in stage.items():
                if key != "settings":
                    reject_fixture(child)
    reject_fixture(result)
    return result


def config_inputs(config):
    found = set()

    def visit(value):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key == "inputs":
                found.update(item.values())
            elif key in ("path", "file"):
                found.add(item)
            elif key != "settings":
                visit(item)

    visit(config["predictor"])
    if "decoder" in config:
        visit(config["decoder"])
    return [file_record(path) for path in sorted(found)]


def semantic_config(config, inputs):
    """Use file contents rather than machine-specific paths when comparing setups."""
    hashes = {record["path"]: record["sha256"] for record in inputs}

    def replace(value):
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            if key == "inputs":
                result[key] = {name: {"sha256": hashes[path]} for name, path in item.items()}
            elif key in ("path", "file"):
                result[key] = {"sha256": hashes[item]}
            elif key == "settings":
                result[key] = item
            else:
                result[key] = replace(item)
        return result

    return digest(replace(config))


def call_stage(stage, batch, seed, source, timings, label):
    module_name, function_name = stage["callable"].split(":")
    function = getattr(importlib.import_module(module_name), function_name, None)
    require(callable(function), "adapter is not callable: " + stage["callable"])
    module_path = Path(inspect.getfile(function)).resolve()
    try:
        relative = module_path.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError("adapter source must be inside this repository")
    require(relative in source["files"], "adapter source is not covered by the source fingerprint")
    settings = json.loads(json.dumps(stage["settings"]))
    require("seed" not in settings or (integer(settings["seed"]) and settings["seed"] == seed),
            "stage seed disagrees with run seed")
    settings["seed"] = seed
    if isinstance(batch, (FeatureBatch, ProbabilityBatch)):
        for item in vars(batch).values():
            if isinstance(item, np.ndarray):
                item.setflags(write=False)
    started = time.perf_counter()
    result = function(batch, MappingProxyType({key: Path(path) for key, path in stage["inputs"].items()}), settings)
    timings[label] = time.perf_counter() - started
    return result


def predict(config, source):
    spec = MatchSpec.from_dict(config["match"])
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    producer = config["predictor"]
    kind = producer["kind"]
    timings = {}
    started = time.perf_counter()
    if kind == "baseline":
        inputs = producer["inputs"]
        count = 0
        with Path(inputs["atd_raw"]).open(encoding="utf-8") as handle:
            for count, line in enumerate(handle, 1):
                require(int(line.split(":", 1)[0]) == count, "ATD frame IDs must be contiguous and start at 1")
        require(count == spec.n_frames, "ATD record count disagrees with match length")
        codes = baseline.nearest_player_labels(inputs["atd_raw"], baseline.team_of_each_track(inputs["atd_meta"]),
                                               spec.n_frames, producer["pitch"])
        result = PredictionBatch(spec, np.arange(spec.n_frames), baseline.smooth(codes, producer["window"]))
    elif kind == "labels":
        result = read_predictions(producer["path"], spec)
    elif kind == "probabilities":
        result = read_batch(producer["path"], spec, "probabilities")
    elif kind == "adapter":
        result = call_stage(producer["stage"], spec, seed, source, timings, "predictor")
    else:
        features = producer["features"]
        features = (read_batch(features["file"], spec, "features") if "file" in features else
                    call_stage(features, spec, seed, source, timings, "features"))
        require(isinstance(features, FeatureBatch), "feature adapter must return FeatureBatch")
        features.validate(spec)
        require(tuple(producer["model"]["expected_features"]) == features.feature_names,
                "feature names/order differ from the trained model contract")
        result = call_stage(producer["model"], features, seed, source, timings, "model")
        require(isinstance(result, ProbabilityBatch), "model adapter must return ProbabilityBatch")
    timings["predictor_total"] = time.perf_counter() - started
    require(isinstance(result, (PredictionBatch, ProbabilityBatch)),
            "predictor must return PredictionBatch or ProbabilityBatch")
    result.validate(spec)
    probabilities = result if isinstance(result, ProbabilityBatch) else None
    if "decoder" in config:
        require(probabilities is not None, "decoder requires a ProbabilityBatch")
        result = call_stage(config["decoder"], probabilities, seed, source, timings, "decoder")
        require(isinstance(result, PredictionBatch), "decoder adapter must return PredictionBatch")
    elif probabilities is not None:
        result = PredictionBatch(spec, result.frames.copy(),
                                 np.asarray(CLASS_ORDER)[result.probabilities.argmax(axis=1)])
    return result.validate(spec), probabilities, timings


def run(config_path, out):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    out = Path(out).resolve()
    source = source_snapshot()
    runtime = environment()
    out.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": SCHEMA_VERSION, "kind": "inference", "status": "running",
              "started_at": now(), "config": config, "config_sha256": digest(config),
              "configuration_file": file_record(config_path), "source": source, "environment": runtime,
              "method": config["method"], "data_kind": config["data_kind"]}
    started = time.perf_counter()
    write_json(out / "run.json", record)
    write_json(out / "config.json", config)
    try:
        inputs = config_inputs(config)
        record["inputs"] = inputs
        record["semantic_config_sha256"] = semantic_config(config, inputs)
        predictions, probabilities, timings = predict(config, source)
        require(config_inputs(config) == inputs, "an input changed during inference")
        require(source_snapshot()["sha256"] == source["sha256"], "source changed during inference")
        require(environment()["sha256"] == runtime["sha256"], "environment changed during inference")
        require(file_record(config_path) == record["configuration_file"], "configuration changed during inference")
        write_predictions(out / "submission.csv", predictions)
        if probabilities is not None:
            write_batch(out / "probabilities.npz", probabilities)
            record["probabilities"] = file_record(out / "probabilities.npz")
        record.update(status="complete", output=file_record(out / "submission.csv"), timings=timings,
                      label_counts={str(label): int(count) for label, count in
                                    zip(*np.unique(predictions.labels, return_counts=True))})
    except Exception as error:
        record.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        record.update(finished_at=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(out / "run.json", record)
    return record


def reproduce(previous, out, config_path=None):
    previous = Path(previous).resolve()
    original = read_json(previous / "run.json")
    require(original.get("kind") == "inference" and original.get("status") == "complete",
            "reproduction requires a completed inference run")
    require(original["config"]["predictor"]["kind"] != "labels",
            "importing an existing label CSV is not inference reproduction")
    require(source_snapshot()["sha256"] == original["source"]["sha256"], "source changed since the original run")
    require(digest(original["config"]) == original["config_sha256"], "recorded configuration was modified")
    require(load_config(previous / "config.json") == original["config"], "configuration snapshot was modified")
    current_config_path = Path(config_path).resolve() if config_path else previous / "config.json"
    current_config = load_config(current_config_path)
    current_inputs = config_inputs(current_config)
    require(semantic_config(original["config"], original["inputs"]) == original["semantic_config_sha256"],
            "recorded input/configuration fingerprint was modified")
    require(semantic_config(current_config, current_inputs) == original["semantic_config_sha256"],
            "inputs changed or configuration differs from the original run")
    require(file_record(previous / "submission.csv")["sha256"] == original["output"]["sha256"],
            "original submission was modified")
    repeated = run(current_config_path, out)
    same = repeated["output"]["sha256"] == original["output"]["sha256"]
    result = {"kind": "reproduction", "status": "complete" if same else "failed",
              "original_run": str(previous), "identical_csv": same,
              "original_sha256": original["output"]["sha256"], "repeated_sha256": repeated["output"]["sha256"],
              "environment_matches": original["environment"]["sha256"] == repeated["environment"]["sha256"]}
    write_json(Path(out) / "reproduction.json", result)
    require(same, "reproduction produced different submission bytes")
    return result
