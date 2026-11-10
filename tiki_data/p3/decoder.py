"""Three-state log-space Viterbi with separate control/stoppage transition costs."""

import numpy as np

from tiki_data.artifacts import read_batch
from tiki_data.contracts import CLASS_ORDER, PredictionBatch, require
from .rules import emissions


def decode(probabilities, inputs, settings):
    probabilities.validate(probabilities.match)
    require(not set(settings) - {"seed", "switch_cost", "dead_cost", "rule_weight", "rules"},
            "unknown decoder settings")
    switch_cost = settings.get("switch_cost", 6.0)
    dead_cost = settings.get("dead_cost", 3.0)
    weight = settings.get("rule_weight", 0.0)
    for key, value in (("switch_cost", switch_cost), ("dead_cost", dead_cost), ("rule_weight", weight)):
        require(isinstance(value, (int, float)) and not isinstance(value, bool)
                and np.isfinite(value) and value >= 0, "invalid decoder setting: " + key)
    require(weight <= 1, "rule_weight must be in [0,1]")
    require(set(inputs) == ({"features"} if weight > 0 else set()),
            "decoder features input is required only for positive rule_weight")
    logp = np.log(np.maximum(probabilities.probabilities, 1e-12))
    if weight > 0:
        features = read_batch(inputs["features"], probabilities.match, "features")
        rule_p = emissions(features, settings.get("rules", {})).probabilities
        logp = (1 - weight) * logp + weight * np.log(rule_p)
    # Preserve physical-time smoothing strength for different sampling rates.
    logp = logp * (25.0 / probabilities.match.fps)
    transitions = np.array([[0, -switch_cost, -dead_cost],
                            [-switch_cost, 0, -dead_cost], [-dead_cost, -dead_cost, 0]])
    labels = np.full(probabilities.match.n_frames, "X", dtype="<U1")
    for start, stop in probabilities.match.periods:
        score = logp[start].copy()
        back = np.zeros((stop - start, 3), dtype=np.int8)
        for offset, frame in enumerate(range(start + 1, stop), 1):
            candidates = score[:, None] + transitions
            back[offset] = candidates.argmax(axis=0)
            score = candidates.max(axis=0) + logp[frame]
            score -= score.max()
        state = int(score.argmax())
        for frame in range(stop - 1, start - 1, -1):
            labels[frame] = CLASS_ORDER[state]
            state = int(back[frame - start, state])
    return PredictionBatch(probabilities.match, probabilities.frames.copy(), labels).validate(probabilities.match)
