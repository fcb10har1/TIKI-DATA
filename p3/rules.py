"""Rules-based emissions from automatic tracking (ATD): a fallback that needs no trained model.

`predict_emissions` is a P4 predictor adapter (docs/INTERFACES.md) returning an (N,3) H/A/D
ProbabilityBatch, to be decoded by `p3.stage:decode_stage`. Together they are the `rules_viterbi`
method.

The constants below are UNFITTED PRIORS chosen from football reasoning, not tuned on any match.

Evidence per frame (all in metres, from the original tracking):
  * team of the nearest players to the ball, weighted by how close that is: a ball in the middle of
    nowhere (in flight) says little about who has it, so the decoder keeps the previous state;
  * ball outside the pitch -> dead ball;
  * ball nearly stationary / players nearly stationary -> weak dead-ball evidence.
A frame with no ball observation gets exactly uniform probabilities (no evidence). A missing ball
is NOT dead-ball evidence by itself.
"""
import dataclasses
import warnings

import numpy as np

from starter.baseline import team_of_each_track
from tiki_data.contracts import ProbabilityBatch, require

HOME, AWAY = 0, 1


@dataclasses.dataclass(frozen=True)
class RulesConfig:
    pitch_length_m: float = 105.3
    pitch_width_m: float = 68.0
    control_radius_m: float = 3.0     # beyond this the ball is "loose": team evidence fades
    team_scale_m: float = 2.0         # distance advantage that gives ~1 nat of team evidence
    team_logit_max: float = 3.0
    out_margin_m: float = 0.5         # ball this far beyond a line counts as out of play
    out_logit: float = 4.0
    still_speed_mps: float = 0.5
    still_logit: float = 1.0
    players_slow_mps: float = 1.0
    slow_logit: float = 0.5
    dead_bias_logit: float = 0.0
    speed_window_s: float = 1.0

    def validate(self):
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            require(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v),
                    "{} must be a finite number".format(f.name))
        for name in ("pitch_length_m", "pitch_width_m", "control_radius_m", "team_scale_m",
                     "speed_window_s", "still_speed_mps", "players_slow_mps"):
            require(getattr(self, name) > 0, name + " must be > 0")
        for name in ("team_logit_max", "out_logit", "still_logit", "slow_logit", "out_margin_m"):
            require(getattr(self, name) >= 0, name + " must be >= 0")


def read_atd(raw_path, n_frames, n_slots):
    """(ball (N,2), players (N,S,2)) in normalised 0-1 coordinates, NaN where unobserved.

    Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...:<x,y,s>" with frame_id counting from 1.
    """
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    count = 0
    with open(raw_path, encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            parts = line.rstrip("\n").split(":")
            require(len(parts) == 3, "ATD line {} is not frame:players:ball".format(count))
            require(int(parts[0]) == count, "ATD frame IDs must be contiguous and start at 1")
            require(count <= n_frames, "ATD has more records than the match has frames")
            groups = parts[1].split(";")
            require(len(groups) == n_slots, "ATD line {} has {} slots, metadata has {}".format(
                count, len(groups), n_slots))
            for slot, group in enumerate(groups):
                if group[:1] == "N":
                    continue
                fields = group.split(",")
                players[count - 1, slot] = (float(fields[0]), float(fields[1]))
            if parts[2][:1] != "N":
                fields = parts[2].split(",")
                ball[count - 1] = (float(fields[0]), float(fields[1]))
    require(count == n_frames, "ATD record count {} disagrees with match length {}".format(count, n_frames))
    return ball, players


def _displacement_speed(xy, k, fps):
    """Speed (m/s) from positions k frames either side; NaN at edges or across gaps. xy in metres."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) > 2 * k:
        diff = xy[2 * k:] - xy[:-2 * k]
        out[k:-k] = np.sqrt((diff ** 2).sum(axis=-1)) / (2.0 * k / fps)
    return out


def emission_probabilities(ball_norm, players_norm, team_of_slot, fps, cfg=None):
    """(N,3) H,A,D probabilities from tracking arrays (normalised coordinates)."""
    cfg = cfg or RulesConfig()
    cfg.validate()
    scale = np.array([cfg.pitch_length_m, cfg.pitch_width_m], dtype=np.float32)
    ball = ball_norm * scale
    players = players_norm * scale
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)

    dist = np.sqrt(((players - ball[:, None, :]) ** 2).sum(axis=-1))          # (N,S), NaN if either missing
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d_home = np.where(team_of_slot[None, :] == HOME, dist, np.inf).min(axis=1)
    d_away = np.where(team_of_slot[None, :] == AWAY, dist, np.inf).min(axis=1)
    both = np.isfinite(d_home) & np.isfinite(d_away)
    near = np.minimum(d_home, d_away)
    control = np.where(np.isfinite(near), np.exp(-(near / cfg.control_radius_m) ** 2), 0.0)
    advantage = np.where(both, np.where(both, d_away, 0.0) - np.where(both, d_home, 0.0), 0.0)
    u = np.clip(control * advantage / cfg.team_scale_m, -cfg.team_logit_max, cfg.team_logit_max)

    k = max(1, int(round(cfg.speed_window_s * fps / 2.0)))
    ball_speed = _displacement_speed(ball, k, fps)
    still = np.nan_to_num(ball_speed, nan=np.inf) < cfg.still_speed_mps
    player_speed = _displacement_speed(players, k, fps)                        # (N,S)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)               # all-NaN rows
        typical = np.nanmedian(player_speed, axis=1)
    slow = np.nan_to_num(typical, nan=np.inf) < cfg.players_slow_mps

    bx, by = ball[:, 0], ball[:, 1]
    margin = cfg.out_margin_m
    out = ((bx < -margin) | (bx > cfg.pitch_length_m + margin) |
           (by < -margin) | (by > cfg.pitch_width_m + margin))

    logit_d = cfg.dead_bias_logit + cfg.out_logit * out + cfg.still_logit * still + cfg.slow_logit * slow
    logits = np.stack([u / 2.0, -u / 2.0, logit_d], axis=1).astype(np.float64)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    probs[~observed] = 1.0 / 3.0                                               # no ball -> no evidence
    return probs


def predict_emissions(match, inputs, settings):
    """P4 adapter. inputs: atd_raw, atd_meta. settings: any RulesConfig field, plus 'fitted'."""
    allowed = {f.name for f in dataclasses.fields(RulesConfig)} | {"fitted", "seed"}
    require(set(settings) <= allowed, "unknown rules settings: {}".format(sorted(set(settings) - allowed)))
    fitted = settings.get("fitted", [])
    require(isinstance(fitted, list) and set(fitted) <= allowed - {"fitted", "seed"},
            "'fitted' must list RulesConfig setting names")
    cfg = RulesConfig(**{k: v for k, v in settings.items() if k not in ("fitted", "seed")})
    team = team_of_each_track(str(inputs["atd_meta"]))
    ball, players = read_atd(inputs["atd_raw"], match.n_frames, len(team))
    probs = emission_probabilities(ball, players, team, match.fps, cfg)
    return ProbabilityBatch(match, np.arange(match.n_frames), probs, match.period_ids())
