"""P4-compatible inference adapter. All preprocessing is embedded in the checkpoint."""

import csv

import numpy as np
import torch

from tiki_data.contracts import CLASS_ORDER, ProbabilityBatch, require
from tiki_data.p1.features import apply_feature_policy
from .data import make_window, timeline_blocks, validate_schema
from .model import TCN, seed_all


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint["schema_version"] == 1 and tuple(checkpoint["class_order"]) == CLASS_ORDER,
            "unsupported checkpoint or class order")
    names = checkpoint["normalization"]["feature_names"]
    validate_schema(checkpoint["feature_schema"], names)
    model = TCN(len(names), **checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def predict_batch(features, model, normalization, core=512, device="cpu"):
    features.validate(features.match)
    require(tuple(normalization["feature_names"]) == features.feature_names, "checkpoint feature order mismatch")
    features = apply_feature_policy(features, normalization.get("mask_ball_speed", False))
    probabilities = np.empty((features.match.n_frames, 3), dtype=np.float64)
    model.eval()
    for left, right in timeline_blocks(features, include_outside=True):
        for start in range(left, right, core):
            stop = min(start + core, right)
            x = make_window(features, normalization, left, right, start, core, model.radius)
            logits = model(torch.from_numpy(x[None]).to(device))
            # float64 softmax ensures CSV/NPZ sums satisfy P4's strict tolerance.
            p = torch.softmax(logits[0, :, model.radius:model.radius + stop-start].double(), dim=0)
            probabilities[start:stop] = p.T.cpu().numpy()
    return ProbabilityBatch(features.match, features.frames.copy(), probabilities,
                            features.periods.copy()).validate(features.match)


def predict(features, inputs, settings):
    require(set(inputs) == {"checkpoint"}, "P2 inference needs exactly inputs.checkpoint")
    require(set(settings) <= {"seed", "device", "core_frames", "threads", "allow_synthetic", "mask_ball_speed"},
            "unknown P2 inference settings")
    seed_all(settings.get("seed", 42), settings.get("threads", 1))
    device = settings.get("device", "cpu")
    model, checkpoint = load_checkpoint(inputs["checkpoint"], device)
    require(checkpoint["data_kind"] != "synthetic" or settings.get("allow_synthetic") is True,
            "synthetic weights require explicit allow_synthetic=true; never use for competition")
    require(features.match.fps == checkpoint["fps"], "FPS differs from training; P1 must resample consistently")
    core = settings.get("core_frames", checkpoint["core_frames"])
    require(isinstance(core, int) and core > 0, "core_frames must be positive")
    # An inference override can hide speed for A; B's embedded policy always applies.
    features = apply_feature_policy(features, settings.get("mask_ball_speed", False))
    return predict_batch(features, model, checkpoint["normalization"], core, device)


def write_probability_csv(path, batch):
    batch.validate(batch.match)
    with open(path, "x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "p_H", "p_A", "p_D"))
        writer.writerows((int(frame), *(format(float(p), ".17g") for p in row))
                         for frame, row in zip(batch.frames, batch.probabilities))
