"""Whole-match external training; test partition is declared but never opened."""

from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from tiki_data.artifacts import read_batch, read_json, read_predictions, resolve_path, write_json
from tiki_data.benchmark import PUBLIC_GT, input_fingerprints, load_manifest, score_match
from tiki_data.contracts import CLASS_ORDER, MatchSpec, PredictionBatch, require
from tiki_data.provenance import environment, file_record, now, source_snapshot
from .data import Windows, fit_normalization, validate_schema
from .inference import load_checkpoint, predict_batch
from .model import TCN, seed_all


def load_training_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    require(set(config) == {"schema_version", "evaluation_manifest", "features", "feature_schema",
                            "seed", "fps", "model", "training", "augmentation"}, "invalid P2 config fields")
    require(config["schema_version"] == 1, "unsupported P2 config version")
    require(isinstance(config["seed"], int) and 0 <= config["seed"] < 2**32, "invalid seed")
    require(config["fps"] == 25, "V1 requires shared 25 Hz features; resample in P1")
    config["evaluation_manifest"] = str(resolve_path(config["evaluation_manifest"], path.parent))
    config["feature_schema"] = str(resolve_path(config["feature_schema"], path.parent))
    config["features"] = {key: str(resolve_path(value, path.parent)) for key, value in config["features"].items()}
    t = config["training"]
    require(set(t) == {"epochs", "patience", "batch_size", "core_frames", "learning_rate", "weight_decay",
                       "gradient_clip", "class_weights", "device", "threads"}, "invalid training settings")
    for key in ("epochs", "patience", "batch_size", "core_frames", "threads"):
        require(isinstance(t[key], int) and t[key] > 0, key + " must be a positive integer")
    for key in ("learning_rate", "gradient_clip"):
        require(isinstance(t[key], (int, float)) and np.isfinite(t[key]) and t[key] > 0, "invalid " + key)
    require(np.isfinite(t["weight_decay"]) and t["weight_decay"] >= 0, "invalid weight_decay")
    require(t["class_weights"] in ("none", "balanced"), "class_weights must be none or balanced")
    require(set(config["model"]) == {"width", "kernel", "dilations", "dropout"}, "invalid model settings")
    aug = config["augmentation"]
    require(set(aug) == {"point_probability", "ball_stretch_probability", "ball_stretch_frames"},
            "invalid augmentation settings")
    require(all(0 <= aug[key] <= 1 for key in ("point_probability", "ball_stretch_probability")),
            "invalid masking probability")
    limits = aug["ball_stretch_frames"]
    require(isinstance(limits, list) and len(limits) == 2
            and all(isinstance(x, int) for x in limits) and 1 <= limits[0] <= limits[1],
            "invalid ball_stretch_frames")
    return config


def load_data(config):
    manifest = load_manifest(config["evaluation_manifest"])
    require(set(config["features"]) == {entry["match"]["match_id"] for entry in manifest["matches"]},
            "feature map must declare every manifest match (test paths are not opened)")
    require(len(set(config["features"].values())) == len(config["features"]),
            "feature paths must be unique across underlying match groups")
    # P4 guards underlying match groups, known supplied identity, public/copy hashes,
    # and reused train/validation label bytes. No test label or feature file is read.
    records = input_fingerprints(manifest, {"train", "validation"}, methods=set())
    forbidden = {entry["ground_truth"] for entry in manifest["matches"]}
    schema = read_json(config["feature_schema"])
    data = {"train": [], "validation": []}
    names = None
    feature_hashes = set()
    for entry in manifest["matches"]:
        if entry["split"] == "test":
            continue
        spec = MatchSpec.from_dict(entry["match"])
        require(spec.fps == config["fps"], "all features must use the configured FPS")
        feature_path = config["features"][spec.match_id]
        require(feature_path not in forbidden, "features cannot be a ground-truth file")
        record = file_record(feature_path)
        require(record["sha256"] not in feature_hashes, "reused feature artifact across match groups")
        feature_hashes.add(record["sha256"])
        records.append(record)
        batch = read_batch(feature_path, spec, "features")
        names = names or batch.feature_names
        require(batch.feature_names == names, "feature order must match across train/validation")
        validate_schema(schema, names)
        labels = read_predictions(entry["ground_truth"], spec)
        require(np.all(labels.labels[batch.periods == 0] == "X"), "outside-period labels must be X")
        codes = np.full(spec.n_frames, -100, dtype=np.int64)
        for i, label in enumerate(CLASS_ORDER):
            codes[labels.labels == label] = i
        require((codes >= 0).any(), "match has no supervised labels")
        # Reject normalized/reformatted copies of the supplied public labels too.
        if spec.n_frames == 199319:
            public = read_predictions(PUBLIC_GT, MatchSpec("BAR-ATH", 199319, 25))
            public_mask = public.labels != "X"
            require(not np.array_equal(labels.labels[public_mask], public.labels[public_mask]),
                    "public labels cannot be renamed as an external match")
        data[entry["split"]].append((batch, codes))
    if config["augmentation"]["ball_stretch_probability"] > 0:
        require(bool(schema["ball_dependent"]), "ball masking needs declared ball-dependent channels")
    return manifest, schema, data, records


def validate(model, data, normalization, core, device):
    rows = []
    loss_sum, count = 0.0, 0
    for batch, labels in data:
        result = predict_batch(batch, model, normalization, core, device)
        truth = np.full(len(labels), "X", dtype="<U1")
        valid = labels >= 0
        truth[valid] = np.asarray(CLASS_ORDER)[labels[valid]]
        prediction = np.asarray(CLASS_ORDER)[result.probabilities.argmax(axis=1)]
        prediction[batch.periods == 0] = "X"
        metrics = score_match(PredictionBatch(batch.match, batch.frames, truth),
                              PredictionBatch(batch.match, batch.frames, prediction), batch.match)
        rows.append({"match_id": batch.match.match_id, "macro_f1": metrics["macro_f1"],
                     "spell_f1_mean": metrics["spell_f1_mean"]})
        loss_sum -= np.log(np.maximum(result.probabilities[valid, labels[valid]], 1e-300)).sum()
        count += int(valid.sum())
    return {"macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
            "spell_f1_mean": float(np.mean([row["spell_f1_mean"] for row in rows])),
            "cross_entropy": float(loss_sum / count), "matches": rows}


def train(config_path, out):
    config = load_training_config(config_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {"kind": "p2_training", "status": "running", "started_at": now(),
              "config": config, "source": source_snapshot(), "environment": environment(),
              "configuration_file": file_record(config_path), "test_accessed": False}
    started = time.perf_counter()
    write_json(out / "run.json", report)
    try:
        manifest, schema, data, inputs = load_data(config)
        report.update(data_kind=manifest["kind"], accuracy_evidence=manifest["kind"] == "external",
                      inputs=inputs + [file_record(config["evaluation_manifest"]),
                                       file_record(config["feature_schema"])],
                      provenance=[{key: entry[key] for key in ("match", "group_id", "split", "provenance")}
                                  for entry in manifest["matches"]])
        t = config["training"]
        seed_all(config["seed"], t["threads"])
        model = TCN(data["train"][0][0].values.shape[1], **config["model"]).to(t["device"])
        normalization = fit_normalization([batch for batch, _ in data["train"]])
        normalization["train_match_ids"] = [batch.match.match_id for batch, _ in data["train"]]
        write_json(out / "normalization.json", normalization)
        windows = Windows(data["train"], normalization, t["core_frames"], model.radius,
                          config["augmentation"], schema["ball_dependent"])
        require(len(windows) > 0, "no supervised training windows")
        loader = DataLoader(windows, batch_size=t["batch_size"], shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(config["seed"]))
        counts = sum((np.bincount(y[y >= 0], minlength=3) for _, y in data["train"]),
                     np.zeros(3, dtype=np.int64))
        require((counts > 0).all(), "training needs labels from all H,A,D classes")
        weights = counts.sum() / (3 * counts) if t["class_weights"] == "balanced" else np.ones(3)
        weights = torch.tensor(weights, dtype=torch.float32, device=t["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        report.update(parameters=sum(p.numel() for p in model.parameters()),
                      receptive_field_frames=2 * model.radius + 1, train_windows=len(windows),
                      train_class_counts=counts.tolist(), selection="validation mean spellF1, then macroF1")
        best, stale, best_epoch = None, 0, None
        for epoch in range(1, t["epochs"] + 1):
            model.train()
            loss_sum, weight_sum = 0.0, 0.0
            for x, y in loader:
                x, y = x.to(t["device"]), y.to(t["device"])
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)[:, :, model.radius:model.radius + t["core_frames"]]
                loss = F.cross_entropy(logits, y, weight=weights, ignore_index=-100, reduction="sum")
                denominator = weights[y[y >= 0]].sum()
                objective = loss / denominator
                require(bool(torch.isfinite(objective)), "nonfinite training loss")
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach().cpu())
                weight_sum += float(denominator.cpu())
            validation = validate(model, data["validation"], normalization, t["core_frames"], t["device"])
            key = (validation["spell_f1_mean"], validation["macro_f1"])
            improved = best is None or key > best
            row = {"epoch": epoch, "train_cross_entropy": loss_sum / weight_sum,
                   "validation": validation, "selected": improved, "data_kind": manifest["kind"],
                   "accuracy_evidence": manifest["kind"] == "external"}
            with (out / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                import json
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            print("epoch={} train_ce={:.6f} validation_ce={:.6f} selected={} kind={}".format(
                epoch, row["train_cross_entropy"], validation["cross_entropy"], improved, manifest["kind"]), flush=True)
            if improved:
                best, stale, best_epoch = key, 0, epoch
                checkpoint = {"schema_version": 1, "class_order": list(CLASS_ORDER),
                              "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                              "model_config": config["model"], "normalization": normalization,
                              "feature_schema": schema, "seed": config["seed"], "fps": config["fps"],
                              "core_frames": t["core_frames"], "data_kind": manifest["kind"],
                              "epoch": epoch, "validation": validation,
                              "training_source_sha256": report["source"]["sha256"]}
                torch.save(checkpoint, out / "best.pt")
            else:
                stale += 1
            if stale >= t["patience"]:
                break
        # Explicitly reload the serialized best model and validate its recorded score.
        reloaded, checkpoint = load_checkpoint(out / "best.pt", t["device"])
        repeated = validate(reloaded, data["validation"], normalization, t["core_frames"], t["device"])
        require(repeated == checkpoint["validation"], "checkpoint reload validation differs")
        require(source_snapshot()["sha256"] == report["source"]["sha256"], "source changed while training")
        for record in report["inputs"] + [report["configuration_file"]]:
            require(file_record(record["path"]) == record, "training input changed during run")
        report.update(status="complete", best_epoch=best_epoch, epochs_run=epoch,
                      checkpoint=file_record(out / "best.pt"), normalization=file_record(out / "normalization.json"),
                      checkpoint_reload_verified=True, best_validation=repeated)
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report.update(finished_at=now(), elapsed_seconds=time.perf_counter() - started)
        write_json(out / "run.json", report)
    return report
