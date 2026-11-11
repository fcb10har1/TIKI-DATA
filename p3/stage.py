"""P4 pipeline decoder stage.

Config (decoder section of a run config):

    "decoder": {
      "callable": "p3.stage:decode_stage",
      "inputs":   {"ball_observed": "ball_observed.npz"},     # optional
      "settings": {"temperature": 4.0, "fitted": []}          # DecoderConfig fields; seed is injected
    }

ball_observed: an .npz with key `ball_observed`, a bool array (n_frames,), True where the ORIGINAL
tracking has a finite ball position (not optical visibility). Without it every frame is treated as
observed. `fitted` lists the settings that were fitted on EXTERNAL data; anything not listed is an
unfitted prior and is recorded as such in the run's config.json.
"""
import dataclasses

import numpy as np

from tiki_data.contracts import PredictionBatch, require

from .decoder import DecoderConfig, decode_match

_FIELDS = {f.name for f in dataclasses.fields(DecoderConfig)} - {"fps"}  # fps comes from the match


def config_from_settings(settings, fps):
    unknown = set(settings) - _FIELDS - {"fitted"}
    require(not unknown, "unknown decoder settings: {}".format(sorted(unknown)))
    fitted = settings.get("fitted", [])
    require(isinstance(fitted, list) and set(fitted) <= _FIELDS - {"seed"},
            "'fitted' must list DecoderConfig setting names")
    values = {k: v for k, v in settings.items() if k != "fitted"}
    for key, value in values.items():
        want = type(getattr(DecoderConfig(), key))
        ok = isinstance(value, bool) if want is bool else (
            isinstance(value, (int, float)) and not isinstance(value, bool))
        require(ok, "decoder setting {} must be {}".format(key, want.__name__))
        if want is int:
            require(float(value).is_integer(), "decoder setting {} must be an integer".format(key))
            values[key] = int(value)
    cfg = DecoderConfig(fps=fps, **values)
    cfg.validate()
    return cfg


def load_ball_observed(path, n_frames):
    with np.load(path, allow_pickle=False) as data:
        require(set(data.files) == {"ball_observed"}, "ball_observed file must contain only 'ball_observed'")
        mask = data["ball_observed"]
    require(mask.dtype == np.bool_ and mask.shape == (n_frames,),
            "ball_observed must be a bool array of shape ({},)".format(n_frames))
    return mask


def decode_stage(batch, inputs, settings):
    match = batch.match
    cfg = config_from_settings(settings, match.fps)
    observed = load_ball_observed(inputs["ball_observed"], match.n_frames) if "ball_observed" in inputs else None
    labels = decode_match(batch.probabilities, observed, match.periods, cfg)
    return PredictionBatch(match, batch.frames.copy(), labels)
