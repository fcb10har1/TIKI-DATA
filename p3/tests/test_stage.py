"""The decoder as a P4 pipeline stage, using P4's own contracts, runner and reproduction check."""
import json

import numpy as np
import pytest

from evaluation import possession_metrics as pm
from p3 import synth
from p3.stage import config_from_settings, decode_stage
from tiki_data.artifacts import write_batch, write_json, write_predictions
from tiki_data.contracts import MatchSpec, PredictionBatch, ProbabilityBatch
from tiki_data.pipeline import reproduce, run

STAGE = "p3.stage:decode_stage"


def make_batch(n=5000, periods=((200, 2500), (3000, 4800)), seed=0, match_id="synthetic_p3", fps=25):
    spec = MatchSpec(match_id, n, fps, periods)
    rng = np.random.default_rng(seed)
    truth = synth.make_truth(n, rng, fps=int(fps), min_s=1.0)
    vis = synth.make_visibility(n, rng, fps=int(fps))
    P = synth.make_probs(truth, vis, rng)
    return ProbabilityBatch(spec, np.arange(n), P, spec.period_ids()).validate(spec), vis


def test_stage_returns_a_valid_prediction_batch_with_x_outside_periods():
    batch, vis = make_batch()
    out = decode_stage(batch, {}, {"seed": 0, "temperature": 4.0})
    assert isinstance(out, PredictionBatch)
    out.validate(batch.match)
    ids = batch.periods
    assert (out.labels[ids == 0] == "X").all() and set(out.labels[ids > 0]) <= {"H", "A", "D"}


def test_periods_are_decoded_independently():
    # all-H evidence in period 1 and all-A in period 2, adjacent: no carry-over across the boundary
    spec = MatchSpec("synthetic_adj", 1200, 25, ((0, 600), (600, 1200)))
    P = np.zeros((1200, 3)) + 0.02
    P[:600, 0] = 0.96
    P[600:, 1] = 0.96
    batch = ProbabilityBatch(spec, np.arange(1200), P, spec.period_ids()).validate(spec)
    out = decode_stage(batch, {}, {"seed": 0})
    assert set(out.labels[:600]) == {"H"} and set(out.labels[600:]) == {"A"}


def test_ball_observed_input_changes_the_result_as_intended(tmp_path):
    spec = MatchSpec("synthetic_vis", 800, 25)
    P = np.zeros((800, 3)) + 0.02
    P[:, 0] = 0.96
    P[300:500] = [0.02, 0.96, 0.02]                      # confident A while the ball is unobserved
    batch = ProbabilityBatch(spec, np.arange(800), P, spec.period_ids()).validate(spec)
    mask = np.ones(800, dtype=bool)
    mask[300:500] = False
    path = tmp_path / "ball.npz"
    np.savez(path, ball_observed=mask)
    with_mask = decode_stage(batch, {"ball_observed": path}, {"seed": 0})
    without = decode_stage(batch, {}, {"seed": 0})
    assert set(with_mask.labels) == {"H"} and "A" in set(without.labels)
    bad = tmp_path / "bad.npz"
    np.savez(bad, ball_observed=mask.astype(np.int8))
    with pytest.raises(ValueError):
        decode_stage(batch, {"ball_observed": bad}, {"seed": 0})


def test_settings_are_strict():
    for bad in ({"temprature": 4}, {"temperature": True}, {"use_mbr": 1}, {"n_samples": 3.5},
                {"fitted": ["nope"]}, {"temperature": -1}, {"fps": 50}):
        with pytest.raises(ValueError):
            config_from_settings(bad, 25.0)
    cfg = config_from_settings({"temperature": 4, "n_samples": 8, "seed": 5, "fitted": ["temperature"]}, 25.0)
    assert cfg.temperature == 4.0 and cfg.n_samples == 8 and cfg.seed == 5 and cfg.fps == 25.0


def test_match_fps_sets_durations_in_frames():
    from p3.decoder import _Chain
    assert _Chain(config_from_settings({}, 25.0)).lengths[0] == 25
    assert _Chain(config_from_settings({}, 10.0)).lengths[0] == 10


def test_full_size_bar_ath_output_passes_the_official_validator(tmp_path):
    spec = MatchSpec("BAR-ATH", 199319, 25)
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(3), size=spec.n_frames)
    batch = ProbabilityBatch(spec, np.arange(spec.n_frames), P, spec.period_ids()).validate(spec)
    out = decode_stage(batch, {}, {"seed": 0})
    write_predictions(tmp_path / "SYNTHETIC.csv", out)
    table = pm.read_label_csv(tmp_path / "SYNTHETIC.csv")
    assert pm.validate_submission(table, 199319) is None
    assert (out.labels[:11185] == "X").all() and (out.labels[90652:110808] == "X").all()
    assert (out.labels[11185:90652] != "X").all() and (out.labels[110808:190198] != "X").all()


def test_p4_runner_executes_and_reproduces_the_stage_byte_for_byte(tmp_path):
    batch, vis = make_batch()
    write_batch(tmp_path / "probs.npz", batch)
    np.savez(tmp_path / "ball.npz", ball_observed=vis)
    config = {"schema_version": 1, "match": batch.match.as_dict(), "method": "p3_rules_viterbi",
              "data_kind": "synthetic", "seed": 7,
              "predictor": {"kind": "probabilities", "path": "probs.npz"},
              "decoder": {"callable": STAGE, "inputs": {"ball_observed": "ball.npz"},
                          "settings": {"temperature": 4.0, "fitted": []}}}
    write_json(tmp_path / "config.json", config)
    record = run(tmp_path / "config.json", tmp_path / "run1")
    assert record["status"] == "complete"
    assert record["config"]["decoder"]["settings"]["fitted"] == []       # provenance is recorded
    result = reproduce(tmp_path / "run1", tmp_path / "run2")
    assert result["identical_csv"]
    # a different seed is injected into the stage and recorded
    assert json.loads((tmp_path / "run1" / "run.json").read_text())["config"]["seed"] == 7
