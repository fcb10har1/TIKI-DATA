"""Safe JSON/NPZ transport and official CSV validation."""

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from evaluation import possession_metrics as pm
from .contracts import (FeatureBatch, MatchSpec, PredictionBatch, ProbabilityBatch,
                        SCHEMA_VERSION, require)


def safe_path(path):
    path = Path(path).resolve()
    require(not any(part.lower() == ".env" or part.lower().startswith(".env.")
                    for part in path.parts), "credential/environment files are not pipeline inputs")
    return path


def read_json(path):
    def unique_object(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate JSON field: " + key)
            out[key] = value
        return out

    def finite_constant(value):
        raise ValueError("nonfinite JSON value: " + value)

    with safe_path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=finite_constant, object_pairs_hook=unique_object)


def write_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        return
    descriptor, name = tempfile.mkstemp(dir=str(path.parent), prefix=".p4-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def resolve_path(value, base):
    require(isinstance(value, str) and bool(value.strip()), "file path must be nonempty text")
    path = Path(value)
    return safe_path(path if path.is_absolute() else Path(base) / path)


def read_predictions(path, spec):
    table = pm.read_label_csv(safe_path(path))
    reason = pm.validate_submission(table, spec.n_frames)
    require(reason is None, "{}: {}".format(Path(path).name, reason))
    codes = pm.label_codes(table)
    return PredictionBatch(spec, np.arange(spec.n_frames), np.asarray(pm.LABELS)[codes]).validate(spec)


def write_predictions(path, batch):
    batch.validate(batch.match)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows(zip(batch.frames.tolist(), batch.labels.tolist()))
    read_predictions(path, batch.match)


def write_batch(path, batch):
    require(isinstance(batch, (FeatureBatch, ProbabilityBatch)),
            "NPZ supports FeatureBatch and ProbabilityBatch only")
    batch.validate(batch.match)
    data = {"schema_version": np.array(SCHEMA_VERSION),
            "match": np.array(json.dumps(batch.match.as_dict(), sort_keys=True)),
            "frames": batch.frames, "periods": batch.periods}
    if isinstance(batch, FeatureBatch):
        data.update(values=batch.values, observed=batch.observed,
                    feature_names=np.asarray(batch.feature_names, dtype=str))
    elif isinstance(batch, ProbabilityBatch):
        data.update(probabilities=batch.probabilities, class_order=np.asarray(batch.class_order, dtype=str))
    else:
        raise ValueError("NPZ supports FeatureBatch and ProbabilityBatch only")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **data)


def read_batch(path, spec, kind):
    require(kind in ("features", "probabilities"), "unknown NPZ artifact kind")
    with np.load(safe_path(path), allow_pickle=False) as data:
        common = {"schema_version", "match", "frames", "periods"}
        fields = {"values", "observed", "feature_names"} if kind == "features" else {"probabilities", "class_order"}
        require(set(data.files) == common | fields, "unexpected or missing NPZ fields")
        version = data["schema_version"]
        require(version.shape == () and version.dtype.kind in "iu" and version.item() == SCHEMA_VERSION,
                "unsupported artifact schema_version")
        match = MatchSpec.from_dict(json.loads(str(data["match"].item())))
        if kind == "features":
            batch = FeatureBatch(match, data["frames"], data["values"], data["observed"],
                                 tuple(data["feature_names"].tolist()), data["periods"])
        else:
            batch = ProbabilityBatch(match, data["frames"], data["probabilities"], data["periods"],
                                     tuple(data["class_order"].tolist()))
    return batch.validate(spec)
