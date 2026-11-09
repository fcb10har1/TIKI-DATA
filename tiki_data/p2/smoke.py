"""Synthetic fixture for execution/shape/alignment checks, never accuracy evidence."""

from pathlib import Path

import numpy as np

from tiki_data.artifacts import read_json, write_batch, write_json, write_predictions
from tiki_data.contracts import FeatureBatch, MatchSpec, PredictionBatch
from tiki_data.provenance import ROOT


def prepare(out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = ("ball_speed_mps", "home_nearest_ball_m", "away_nearest_ball_m", "home_width_m")
    sources = (["ball.x", "ball.y"], ["ball.x", "ball.y", "home.positions"],
               ["ball.x", "ball.y", "away.positions"], ["home.positions"])
    schema = {"features": [{"name": name, "definition": "Synthetic numerical fixture; " + name,
                            "sources": source} for name, source in zip(names, sources)],
              "ball_dependent": list(names[:3]),
              "notes": "Synthetic 25 Hz features in metres; no real match or target columns used as inputs."}
    write_json(out / "feature_schema.json", schema)
    entries, features = [], {}
    for i, split in enumerate(("train", "validation", "test")):
        rng = np.random.default_rng(1000 + i)
        n = 401 + 42 * i
        match = MatchSpec("p2_synthetic_" + split, n, 25, ((3, n // 2), (n // 2 + 7, n - 4)))
        t = np.arange(n) / 25
        values = np.column_stack((2 + 2 * np.sin(t * 1.2 + i),
                                  5 + 4 * np.sin(t * .9 + i),
                                  5 - 4 * np.sin(t * .9 + i),
                                  35 + rng.normal(0, 3, n))).astype(np.float32)
        labels = np.where(values[:, 1] < values[:, 2], "H", "A")
        labels[values[:, 0] < 0.5] = "D"
        labels[match.period_ids() == 0] = "X"
        observed = rng.random(values.shape) > 0.06
        observed[55:110, :3] = False
        values[~observed] = 0
        batch = FeatureBatch(match, np.arange(n), values, observed, names, match.period_ids())
        write_batch(out / (match.match_id + "_features.npz"), batch)
        write_predictions(out / (match.match_id + "_gt.csv"), PredictionBatch(match, batch.frames, labels))
        write_json(out / (match.match_id + ".json"), match.as_dict())
        entries.append({"match": match.as_dict(), "group_id": match.match_id, "split": split,
                        "provenance": {"dataset": "P2 synthetic execution fixture", "source": "tiki_data/p2/smoke.py",
                                       "label_provenance": "algorithmically generated synthetic states; no football accuracy evidence"},
                        "ground_truth": match.match_id + "_gt.csv"})
        features[match.match_id] = match.match_id + "_features.npz"
    write_json(out / "manifest.json", {"schema_version": 1, "dataset_id": "p2_synthetic",
                                       "kind": "synthetic", "matches": entries})
    config = read_json(ROOT / "configs" / "p2_v1.json")
    config.update(evaluation_manifest="manifest.json", features=features, feature_schema="feature_schema.json")
    config["model"].update(width=8, dilations=[1, 2], dropout=0.0)
    config["training"].update(epochs=4, patience=2, batch_size=4, core_frames=64)
    write_json(out / "train.json", config)
    config["augmentation"].update(point_probability=0.05, ball_stretch_probability=0.75,
                                   ball_stretch_frames=[25, 75])
    write_json(out / "train_hidden.json", config)
    return {"status": "complete", "data_kind": "synthetic", "accuracy_evidence": False,
            "train_config": str(out / "train.json")}
