"""Train-only scaling, explicit observation masking and boundary-safe windows."""

import re

import numpy as np
import torch
from torch.utils.data import Dataset

from tiki_data.contracts import require


def validate_schema(schema, names):
    require(set(schema) == {"features", "ball_dependent", "notes"}, "invalid feature schema")
    require(tuple(item["name"] for item in schema["features"]) == tuple(names),
            "schema must declare the exact feature order")
    require(bool(schema["notes"]), "document units, orientation and shared preprocessing")
    for item in schema["features"]:
        require(set(item) == {"name", "definition", "sources"} and item["definition"]
                and isinstance(item["sources"], list) and bool(item["sources"]),
                "each feature needs a definition and raw source fields")
        for text in [item["name"]] + item["sources"]:
            canonical = re.sub(r"[^a-z0-9]", "", text.lower())
            require(not any(token in canonical for token in
                            ("possession", "ballstatus", "ballstate", "target", "label",
                             "smarttag", "tacticaltag", "inplay", "matchstate", "ballalive",
                             "balldead", "isalive", "isdead", "teamincontrol")),
                    "target/tag-derived features are forbidden: " + text)
            require(not re.search(r"(?:track|slot|player)[_-]?\d", text.lower()),
                    "do not encode track slots as stable identities")
    require(len(set(schema["ball_dependent"])) == len(schema["ball_dependent"])
            and set(schema["ball_dependent"]) <= set(names), "unknown/duplicate ball-dependent feature")
    return schema


def fit_normalization(batches, allow_unobserved=()):
    """Population moments, observed training frames inside periods only, float64."""
    names = batches[0].feature_names
    count = np.zeros(len(names), dtype=np.int64)
    mean = np.zeros(len(names), dtype=np.float64)
    m2 = np.zeros(len(names), dtype=np.float64)
    for batch in batches:
        require(batch.feature_names == names, "feature order differs between matches")
        for j in range(len(names)):
            x = batch.values[(batch.periods > 0) & batch.observed[:, j], j].astype(np.float64)
            if not len(x):
                continue
            total = count[j] + len(x)
            delta = x.mean() - mean[j]
            m2[j] += ((x - x.mean()) ** 2).sum() + delta ** 2 * count[j] * len(x) / total
            mean[j] += delta * len(x) / total
            count[j] = total
    require(set(allow_unobserved) <= set(names), "unknown allowed unobserved feature")
    require(all(n > 0 or name in allow_unobserved for name, n in zip(names, count)),
            "every feature needs an observed training value unless explicitly allowed")
    # An explicitly disabled channel has neutral moments; no division by zero.
    scale = np.sqrt(np.divide(m2, count, out=np.ones_like(m2), where=count > 0))
    scale[scale < 1e-8] = 1.0
    return {"feature_names": list(names), "mean": mean.tolist(), "scale": scale.tolist(),
            "observed_count": count.tolist(), "fit_scope": "train_matches_in_period_observed_only"}


def encode(values, observed, normalization):
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    require(mean.shape == scale.shape == (values.shape[1],)
            and np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(),
            "invalid normalization")
    normalized = np.where(observed, (values.astype(np.float64) - mean) / scale, 0)
    result = np.concatenate((normalized, observed.astype(np.float64)), axis=1).T.astype(np.float32)
    require(np.isfinite(result).all(), "normalized features overflowed")
    return result


def timeline_blocks(batch, include_outside=False):
    # Adjacent halves retain distinct IDs. Each outside interval is independent too.
    edges = np.r_[0, np.flatnonzero(np.diff(batch.periods)) + 1, len(batch.frames)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])
            if include_outside or batch.periods[a] > 0]


def hide_observations(values, observed, rng, augmentation, ball_indices, speed_index=None):
    """Mask all ball-derived channels together, including motion/distance summaries.

    No target-dependent decisions; normalization remains fixed on unaugmented train data.
    P1 should recompute raw temporal features after occlusion for higher-fidelity V2.
    """
    values, observed = values.copy(), observed.copy()
    observed[rng.random(observed.shape) < augmentation.get("point_probability", 0)] = False
    if ball_indices and rng.random() < augmentation.get("ball_stretch_probability", 0):
        low, high = augmentation["ball_stretch_frames"]
        length = int(rng.integers(low, high + 1))
        start = int(rng.integers(0, max(1, len(values) - min(length, len(values)) + 1)))
        observed[start:start + length, ball_indices] = False
    probability = augmentation.get("speed_window_probability", 0)
    if probability > 0:
        require(speed_index is not None, "scalar-speed augmentation needs ball_speed_mps")
        if rng.random() < probability:
            observed[:, speed_index] = False
    values[~observed] = 0
    return values, observed


class Windows(Dataset):
    """Nonoverlapping supervised cores plus full receptive-field halos.

    Padding is never supervised. Original frame IDs are used for reconstruction.
    At true boundaries raw features/flags are zero-padded, never borrowed from a half.
    """
    def __init__(self, records, normalization, core, radius, augmentation=None, ball_names=()):
        require(isinstance(core, int) and core > 0, "core must be positive")
        self.records, self.normalization = records, normalization
        self.core, self.radius = core, radius
        self.augmentation = augmentation or {}
        self.ball_indices = [normalization["feature_names"].index(name) for name in ball_names]
        self.index = []
        for i, (batch, labels) in enumerate(records):
            for left, right in timeline_blocks(batch):
                for start in range(left, right, core):
                    stop = min(start + core, right)
                    if np.any(labels[start:stop] >= 0):
                        self.index.append((i, left, right, start, stop))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, left, right, start, stop = self.index[index]
        batch, labels = self.records[i]
        x = make_window(batch, self.normalization, left, right, start, self.core, self.radius,
                        self.augmentation, self.ball_indices)
        y = np.full(self.core, -100, dtype=np.int64)
        y[:stop-start] = labels[start:stop]
        y[y < 0] = -100
        return torch.from_numpy(x), torch.from_numpy(y)


def make_window(batch, normalization, left, right, start, core, radius,
                augmentation=None, ball_indices=()):
    a, b = max(left, start - radius), min(right, start + core + radius)
    values, observed = batch.values[a:b], batch.observed[a:b]
    if augmentation:
        # DataLoader uses num_workers=0; NumPy RNG is seeded and advances reproducibly.
        values, observed = hide_observations(values, observed, np.random.default_rng(
            int(np.random.randint(0, 2**32 - 1, dtype=np.int64))), augmentation, ball_indices,
            normalization["feature_names"].index("ball_speed_mps")
            if "ball_speed_mps" in normalization["feature_names"] else None)
    x = np.zeros((2 * values.shape[1], core + 2 * radius), dtype=np.float32)
    offset = a - (start - radius)
    x[:, offset:offset+b-a] = encode(values, observed, normalization)
    return x


def idsse_targets(possession, ball_status):
    """TARGET-ONLY DFL mapping. Unknown live possession stays X; never invent D."""
    possession, ball_status = np.asarray(possession), np.asarray(ball_status)
    require(possession.shape == ball_status.shape and possession.ndim == 1,
            "IDSSE target arrays must be aligned one-dimensional arrays")
    result = np.full(possession.shape, "X", dtype="<U1")
    result[ball_status == 0] = "D"
    result[(ball_status == 1) & (possession == 1)] = "H"
    result[(ball_status == 1) & (possession == 2)] = "A"
    return result
