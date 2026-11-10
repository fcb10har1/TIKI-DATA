"""Conservative tracking-only control and stoppage proxies, without label access."""

import numpy as np

from tiki_data.artifacts import read_batch
from tiki_data.contracts import CLASS_ORDER, PredictionBatch, ProbabilityBatch, require


DEFAULTS = {"control_distance_m": 2.5, "distance_margin_m": 0.75,
            "control_seconds": 0.2, "stationary_speed_mps": 0.3,
            "team_stationary_speed_mps": 1.2, "stoppage_seconds": 2.0}


def parameters(settings):
    require(not set(settings) - set(DEFAULTS) - {"seed"}, "unknown rules settings")
    result = dict(DEFAULTS, **{k: v for k, v in settings.items() if k != "seed"})
    for key, value in result.items():
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and value > 0, "invalid rules setting: " + key)
    return result


def predict_features(features, settings):
    features.validate(features.match)
    config = parameters(settings)
    names = {name: i for i, name in enumerate(features.feature_names)}
    needed = ("home_nearest_ball_m", "away_nearest_ball_m", "ball_speed_mps",
              "home_mean_speed_mps", "away_mean_speed_mps")
    require(set(needed) <= set(names), "rules require P1 distance and speed features")
    columns = [names[name] for name in needed]
    values, observed = features.values[:, columns], features.observed[:, columns]
    require(np.all(values[observed] >= 0), "distances and scalar speeds must be nonnegative")
    control_frames = max(1, int(config["control_seconds"] * features.match.fps + 0.5))
    stop_frames = max(1, int(config["stoppage_seconds"] * features.match.fps + 0.5))
    labels = np.full(features.match.n_frames, "X", dtype="<U1")
    for start, stop in features.match.periods:
        owner, state, candidate, count, stationary = "X", "X", "X", 0, 0
        for frame in range(start, stop):
            home, away, ball_speed, home_speed, away_speed = values[frame]
            valid = observed[frame]
            # Missing tracking is never evidence that the ball is dead.
            still = (valid[2:].all() and ball_speed < config["stationary_speed_mps"]
                     and max(home_speed, away_speed) < config["team_stationary_speed_mps"])
            stationary = stationary + 1 if still else 0
            if stationary >= stop_frames:
                state = "D"
                if stationary == stop_frames:
                    labels[frame - stationary + 1:frame + 1] = "D"
                candidate, count = "X", 0
            else:
                # Loose/contested/missing observations retain the last controller.
                next_owner = "X"
                if valid[0] and valid[1]:
                    if home <= config["control_distance_m"] and away - home >= config["distance_margin_m"]:
                        next_owner = "H"
                    elif away <= config["control_distance_m"] and home - away >= config["distance_margin_m"]:
                        next_owner = "A"
                if next_owner != "X":
                    count = count + 1 if next_owner == candidate else 1
                    candidate = next_owner
                    if count >= control_frames:
                        owner = candidate
                else:
                    candidate, count = "X", 0
                moving = valid[2] and ball_speed >= config["stationary_speed_mps"]
                if state != "D" or moving:
                    state = owner
            labels[frame] = state
    return PredictionBatch(features.match, features.frames.copy(), labels).validate(features.match)


def predict(match, inputs, settings):
    require(set(inputs) == {"features"}, "rules need exactly inputs.features")
    return predict_features(read_batch(inputs["features"], match, "features"), settings)


def emissions(features, settings):
    labels = predict_features(features, settings).labels
    probabilities = np.full((features.match.n_frames, 3), 1 / 3, dtype=np.float64)
    for index, label in enumerate(CLASS_ORDER):
        mask = labels == label
        probabilities[mask] = 0.05
        probabilities[mask, index] = 0.90
    return ProbabilityBatch(features.match, features.frames.copy(), probabilities, features.periods.copy())
