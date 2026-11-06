"""Small synthetic fixtures exercising adapter slots, never real accuracy evidence.

The example model is NOT a trained TCN. The example decoder is a tiny reference
Viterbi fixture, NOT P3's production possession algorithm.
"""

from pathlib import Path

import numpy as np

from .artifacts import read_batch, write_batch, write_json, write_predictions
from .contracts import CLASS_ORDER, FeatureBatch, MatchSpec, PredictionBatch, ProbabilityBatch

FEATURE_NAMES = ("toy_home_evidence", "toy_away_evidence", "toy_dead_evidence")


def example_features(match, inputs, settings):
    return read_batch(inputs["features"], match, "features")


def example_model(features, inputs, settings):
    values = features.values.copy()
    empty = ~features.observed.any(axis=1)
    values[empty] = 1 / 3
    return ProbabilityBatch(features.match, features.frames.copy(), values,
                            features.periods.copy())


def example_rules(match, inputs, settings):
    features = example_features(match, inputs, settings)
    labels = np.asarray(CLASS_ORDER)[features.values.argmax(axis=1)]
    indices = np.arange(7, match.n_frames, 11)
    labels[indices] = np.asarray(CLASS_ORDER)[(features.values[indices].argmax(axis=1) + 1) % 3]
    labels[features.periods == 0] = "X"
    return PredictionBatch(match, features.frames.copy(), labels)


def example_decoder(probabilities, inputs, settings):
    match = probabilities.match
    emissions = np.log(np.maximum(probabilities.probabilities, 1e-12))
    switch_cost = float(settings.get("switch_cost", 2.0))
    transitions = np.full((3, 3), -switch_cost)
    np.fill_diagonal(transitions, 0)
    labels = np.full(match.n_frames, "X", dtype="<U1")
    for start, stop in match.periods:
        score = emissions[start].copy()
        back = np.zeros((stop - start, 3), dtype=np.int8)
        for offset, frame in enumerate(range(start + 1, stop), 1):
            candidates = score[:, None] + transitions
            back[offset] = candidates.argmax(axis=0)
            score = candidates.max(axis=0) + emissions[frame]
        state = int(score.argmax())
        for frame in range(stop - 1, start - 1, -1):
            labels[frame] = CLASS_ORDER[state]
            state = int(back[frame - start, state])
    return PredictionBatch(match, probabilities.frames.copy(), labels)


def generate(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for index, split in enumerate(("train", "validation", "validation", "test")):
        match_id = "synthetic_{}".format(index)
        n_frames = 240 + 20 * index
        match = MatchSpec(match_id, n_frames, 10 if index == 2 else 25,
                          ((0, n_frames // 2), (n_frames // 2, n_frames)))
        frames = np.arange(n_frames)
        codes = (frames // (23 + 2 * index) + index) % 3
        labels = np.asarray(CLASS_ORDER)[codes]
        write_predictions(out / (match_id + "_gt.csv"), PredictionBatch(match, frames, labels))
        values = np.full((n_frames, 3), 0.05)
        values[frames, codes] = 0.90
        noisy = np.arange(9, n_frames, 19)
        values[noisy] = 0.05
        values[noisy, (codes[noisy] + 1) % 3] = 0.90
        observed = np.ones(values.shape, dtype=bool)
        observed[15::37] = False
        values[~observed] = 0
        batch = FeatureBatch(match, frames, values, observed, FEATURE_NAMES, match.period_ids())
        feature_path = out / (match_id + "_features.npz")
        write_batch(feature_path, batch)
        predictions = {}
        if split != "train":
            for method in ("rules", "tcn", "tcn_viterbi"):
                if method == "rules":
                    predictor = {"kind": "adapter", "stage": {"callable": "tiki_data.demo:example_rules",
                                 "inputs": {"features": feature_path.name}}}
                else:
                    predictor = {"kind": "model", "features": {"callable": "tiki_data.demo:example_features",
                                 "inputs": {"features": feature_path.name}},
                                 "model": {"callable": "tiki_data.demo:example_model",
                                           "expected_features": list(FEATURE_NAMES)}}
                config = {"schema_version": 1, "match": match.as_dict(), "method": method,
                          "data_kind": "synthetic", "seed": 42, "predictor": predictor}
                if method == "tcn_viterbi":
                    config["decoder"] = {"callable": "tiki_data.demo:example_decoder",
                                         "settings": {"switch_cost": 2.0}}
                config_name = match_id + "_" + method + ".json"
                write_json(out / config_name, config)
                predictions[method] = {"config": config_name}
        entries.append({"match": match.as_dict(), "group_id": match_id, "split": split,
                        "provenance": {"dataset": "P4 synthetic fixture", "source": "tiki_data/demo.py",
                                       "label_provenance": "algorithmically generated toy states; no match data"},
                        "ground_truth": match_id + "_gt.csv", "predictions": predictions})
    manifest = out / "manifest.json"
    write_json(manifest, {"schema_version": 1, "dataset_id": "synthetic_integration", "kind": "synthetic", "matches": entries})
    return manifest


def demonstrate(out):
    from .benchmark import benchmark
    from .holdout import freeze
    from .pipeline import reproduce

    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = generate(out / "inputs")
    benchmark(manifest, out / "validation", bootstrap=20)
    freeze(manifest, out / "validation", "tcn_viterbi", out / "freeze.json")
    benchmark(manifest, out / "test", split="test", freeze_path=out / "freeze.json", bootstrap=20)
    reproduce(out / "validation" / "predictions" / "synthetic_1" / "tcn_viterbi", out / "reproduction")
    return {"status": "complete", "evidence_kind": "synthetic", "accuracy_evidence": False, "output": str(out)}
