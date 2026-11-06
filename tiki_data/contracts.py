"""Versioned, validated in-memory handoffs for P1, P2 and P3."""

from dataclasses import dataclass
import math
import re
from typing import Tuple

import numpy as np

from games import GAME, GAMES, half_video_blocks

SCHEMA_VERSION = 1
CLASS_ORDER = ("H", "A", "D")
LABELS = CLASS_ORDER + ("X",)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identifier(value):
    require(isinstance(value, str) and bool(value.strip()), "identifier must be nonempty text")
    canonical = re.sub(r"[^a-z0-9]", "", value.lower())
    require(bool(canonical), "identifier must include a letter or digit")
    return canonical


def integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    n_frames: int
    fps: float
    periods: Tuple[Tuple[int, int], ...] = ()

    def __post_init__(self):
        identifier(self.match_id)
        require(integer(self.n_frames) and self.n_frames > 0, "n_frames must be a positive integer")
        require(isinstance(self.fps, (int, float)) and not isinstance(self.fps, bool)
                and math.isfinite(self.fps) and self.fps >= 1, "fps must be finite and >= 1")
        require(isinstance(self.periods, (tuple, list)), "periods must be a list/tuple of boundary pairs")
        if identifier(self.match_id) == identifier(GAME):
            require(self.n_frames == GAMES[GAME]["n_video_frames"] and self.fps == 25,
                    "BAR-ATH requires 199319 frames at 25 fps")
            default = tuple((a, b + 1) for _, a, b in half_video_blocks(GAME))
        else:
            default = ((0, self.n_frames),)
        periods = tuple(tuple(pair) for pair in self.periods) if self.periods else default
        previous_stop = 0
        for pair in periods:
            require(len(pair) == 2 and all(integer(x) for x in pair),
                    "periods must contain integer [start, stop) pairs")
            start, stop = pair
            require(previous_stop <= start < stop <= self.n_frames,
                    "periods must be ordered, disjoint and within the match")
            previous_stop = stop
        if identifier(self.match_id) == identifier(GAME):
            require(periods == default, "BAR-ATH period boundaries must match games.py")
        object.__setattr__(self, "n_frames", int(self.n_frames))
        object.__setattr__(self, "fps", float(self.fps))
        object.__setattr__(self, "periods", periods)

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict), "match must be an object")
        require(set(value) <= {"match_id", "n_frames", "fps", "periods"}, "unknown match fields")
        require({"match_id", "n_frames", "fps"} <= set(value), "match needs match_id, n_frames, fps")
        return cls(**value)

    def as_dict(self):
        return {"match_id": self.match_id, "n_frames": self.n_frames,
                "fps": self.fps, "periods": [list(pair) for pair in self.periods]}

    def period_ids(self):
        out = np.zeros(self.n_frames, dtype=np.int32)
        for period, (start, stop) in enumerate(self.periods, 1):
            out[start:stop] = period
        return out


def validate_frames(frames, spec):
    require(isinstance(frames, np.ndarray) and frames.ndim == 1
            and np.issubdtype(frames.dtype, np.integer) and frames.dtype.kind != "b",
            "frames must be a one-dimensional integer array")
    require(np.array_equal(frames, np.arange(spec.n_frames)), "frames must be exactly 0..n_frames-1")


def validate_periods(periods, spec):
    require(isinstance(periods, np.ndarray) and np.issubdtype(periods.dtype, np.integer),
            "periods must be an integer array")
    require(np.array_equal(periods, spec.period_ids()), "period IDs disagree with match boundaries")


@dataclass(frozen=True)
class FeatureBatch:
    match: MatchSpec
    frames: np.ndarray
    values: np.ndarray
    observed: np.ndarray
    feature_names: Tuple[str, ...]
    periods: np.ndarray

    def validate(self, spec):
        require(self.match == spec, "features belong to a different match")
        validate_frames(self.frames, spec)
        validate_periods(self.periods, spec)
        require(isinstance(self.values, np.ndarray) and self.values.ndim == 2
                and self.values.shape[0] == spec.n_frames and self.values.shape[1] > 0
                and self.values.dtype.kind == "f", "features must be an N x F floating-point array")
        names = self.feature_names
        require(isinstance(names, tuple) and len(names) == self.values.shape[1]
                and all(isinstance(name, str) and bool(name.strip()) for name in names)
                and len(set(names)) == len(names), "feature names must be ordered, unique and match F")
        require(isinstance(self.observed, np.ndarray) and self.observed.dtype == np.bool_
                and self.observed.shape == self.values.shape, "observed must be an N x F boolean array")
        require(np.isfinite(self.values).all(), "features must be finite; use observation flags for missing values")
        require((self.values[~self.observed] == 0).all(), "unobserved features must have a zero placeholder")
        return self


@dataclass(frozen=True)
class ProbabilityBatch:
    match: MatchSpec
    frames: np.ndarray
    probabilities: np.ndarray
    periods: np.ndarray
    class_order: Tuple[str, ...] = CLASS_ORDER

    def validate(self, spec):
        require(self.match == spec, "probabilities belong to a different match")
        validate_frames(self.frames, spec)
        validate_periods(self.periods, spec)
        require(self.class_order == CLASS_ORDER, "probability columns must be H,A,D in that order")
        values = self.probabilities
        require(isinstance(values, np.ndarray) and values.shape == (spec.n_frames, 3)
                and values.dtype.kind == "f", "probabilities must be an N x 3 floating-point array")
        require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(),
                "probabilities must be finite and in [0,1]")
        require(np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0),
                "each probability row must sum to 1")
        return self


@dataclass(frozen=True)
class PredictionBatch:
    match: MatchSpec
    frames: np.ndarray
    labels: np.ndarray

    def validate(self, spec):
        require(self.match == spec, "predictions belong to a different match")
        validate_frames(self.frames, spec)
        require(isinstance(self.labels, np.ndarray) and self.labels.shape == (spec.n_frames,)
                and self.labels.dtype.kind in "UO" and all(label in LABELS for label in self.labels),
                "labels must be a one-dimensional H/A/D/X array")
        return self
