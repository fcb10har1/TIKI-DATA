"""Causal, position-only ball motion for D; recorded scalar speed stays missing.

Estimates are separate channels with their own validity flags. Original geometry
and its observation flags are never replaced by predicted positions.
"""
import json
from pathlib import Path

import numpy as np

from tiki_data.contracts import FeatureBatch, require
from .features import FEATURE_NAMES, apply_feature_policy, feature_schema

DEFAULT_SETTINGS = {
    "measurement_std_m": 0.5,
    "acceleration_std_mps2": 20.0,
    "initial_velocity_std_mps": 10.0,
    "max_velocity_std_mps": 10.0,
    "max_prediction_s": 0.5,
    "innovation_gate_squared": 25.0,
    "min_observations": 3,
}
MOTION_NAMES = ("ball_kalman_speed_mps", "ball_kalman_velocity_std_mps", "ball_kalman_age_s")


def validate_settings(settings):
    require(isinstance(settings, dict) and set(settings) == set(DEFAULT_SETTINGS), "invalid Kalman settings")
    for key, value in settings.items():
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and value > 0, "invalid Kalman " + key)
    require(isinstance(settings["min_observations"], int) and settings["min_observations"] >= 2,
            "Kalman min_observations must be an integer >=2")
    return dict(settings)


def motion_schema(settings):
    settings = validate_settings(settings)
    schema = feature_schema()
    definitions = (
        "Causal planar ball-speed estimate in m/s; flag means estimate valid, not directly observed",
        "Per-axis velocity standard deviation of the causal estimate in m/s",
        "Seconds since last observed ball position, capped at max_prediction_s; resets per period",
    )
    schema["features"] += [{"name": name, "definition": definition, "sources": ["ball.x", "ball.y", "time"]}
                           for name, definition in zip(MOTION_NAMES, definitions)]
    schema["ball_dependent"] += list(MOTION_NAMES)
    schema["notes"] += " D keeps recorded scalar speed hidden; separate causal estimates: " + json.dumps(settings, sort_keys=True)
    return schema


def estimate_motion(positions, match, settings=None):
    """Independent x/y constant-velocity filters with shared isotropic covariance.

    Predict-only support expires after a fixed short gap. Large innovations
    reinitialize the state and require a fresh warm-up. No future frame is read.
    """
    cfg = validate_settings(DEFAULT_SETTINGS if settings is None else settings)
    positions = np.asarray(positions, dtype=np.float64)
    require(positions.shape == (match.n_frames, 2), "ball positions must align with the complete timeline")
    values = np.zeros((match.n_frames, 3), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    dt = 1.0 / match.fps
    q = cfg["acceleration_std_mps2"] ** 2
    q00, q01, q11 = q * dt**4 / 4, q * dt**3 / 2, q * dt**2
    r = cfg["measurement_std_m"] ** 2
    for start, stop in match.periods:
        state = None
        last_seen = None
        seen = 0
        for i in range(start, stop):
            z = positions[i]
            observed = bool(np.isfinite(z).all())
            if state is not None:
                if (i - last_seen) * dt > cfg["max_prediction_s"]:
                    state = None
                    seen = 0
                else:
                    state[0] += state[1] * dt
                    p00, p01, p11 = (p00 + 2*dt*p01 + dt*dt*p11 + q00,
                                     p01 + dt*p11 + q01, p11 + q11)
            if observed:
                if state is not None:
                    innovation = z - state[0]
                    if float(innovation @ innovation) / (p00 + r) > cfg["innovation_gate_squared"]:
                        state = None
                        seen = 0
                if state is None:
                    state = np.array([z, np.zeros(2)], dtype=np.float64)
                    p00, p01, p11 = r, 0.0, cfg["initial_velocity_std_mps"] ** 2
                else:
                    k0, k1 = p00 / (p00 + r), p01 / (p00 + r)
                    state[0] += k0 * innovation
                    state[1] += k1 * innovation
                    p00, p01, p11 = (max(0., (1-k0)*p00), (1-k0)*p01, max(0., p11-k1*p01))
                seen += 1
                last_seen = i
            if last_seen is not None:
                values[i, 2] = min((i-last_seen)*dt, cfg["max_prediction_s"])
                valid[i, 2] = True
            if state is not None and seen >= cfg["min_observations"]:
                velocity_std = np.sqrt(p11)
                if velocity_std <= cfg["max_velocity_std_mps"]:
                    values[i, :2] = (np.linalg.norm(state[1]), velocity_std)
                    valid[i, :2] = True
    require(np.isfinite(values).all(), "Kalman output is nonfinite")
    return values, valid


def augment(batch, positions, settings=None):
    require(batch.feature_names == FEATURE_NAMES, "Kalman augmentation expects the original 19 P1 features")
    base = apply_feature_policy(batch, True)
    values, valid = estimate_motion(positions, batch.match, settings)
    return FeatureBatch(batch.match, batch.frames.copy(), np.column_stack((base.values, values)),
                        np.column_stack((base.observed, valid)), FEATURE_NAMES + MOTION_NAMES,
                        batch.periods.copy()).validate(batch.match)


def metrica_positions(match, raw_path, pitch):
    """Read only x/y; raw scalar speed and every target field are excluded."""
    result = np.full((match.n_frames, 2), np.nan, dtype=np.float64)
    count = 0
    with Path(raw_path).open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            require(count <= match.n_frames, "extra ATD rows")
            frame, _, ball = line.rstrip().split(":")
            require(int(frame) == count, "ATD frame IDs are not aligned")
            x, y = map(float, ball.split(",")[:2])
            if np.isfinite([x, y]).all():
                result[count-1] = (x*pitch[0], y*pitch[1])
    require(count == match.n_frames, "missing ATD rows")
    return result


def build(match, inputs, settings):
    """P4-compatible D feature adapter with explicit Kalman settings."""
    from games import GAME, GAMES
    from .metrica import build_metrica_features
    require(set(inputs) == {"atd_meta", "atd_raw"}, "D needs ATD metadata and raw input")
    require(set(settings) <= {"seed", "pitch", "kalman"} and "kalman" in settings, "D requires explicit Kalman settings")
    pitch = settings.get("pitch", GAMES[GAME]["pitch"])
    base = build_metrica_features(match, inputs["atd_meta"], inputs["atd_raw"], pitch, mask_ball_speed=True)
    return augment(base, metrica_positions(match, inputs["atd_raw"], pitch), settings["kalman"])
