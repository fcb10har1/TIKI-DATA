import numpy as np
import pytest

from p3 import synth
from p3.rules import RulesConfig, emission_probabilities, predict_emissions, read_atd
from p3.spells import spell_f1
from p3.stage import decode_stage
from p3.synth_atd import make_world, write_atd
from tiki_data.artifacts import write_json
from tiki_data.contracts import MatchSpec
from tiki_data.pipeline import reproduce, run

FPS = 25
TEAM = np.array([0, 0, 1, 1])


def players_at(coords):
    """(N=len, S=4, 2) in metres converted to normalised, for a fixed arrangement."""
    return np.array(coords, dtype=np.float32) / np.array([105.3, 68.0], dtype=np.float32)


def one_frame(ball_m, home_m, away_m, n=60, speed_mps=3.0):
    """A live-play scene: everyone drifts together along x at speed_mps (distances stay fixed)."""
    scale = np.array([105.3, 68.0], dtype=np.float32)
    drift = (np.arange(n, dtype=np.float32) * speed_mps / FPS)[:, None]            # metres
    shift = np.concatenate([drift, np.zeros_like(drift)], axis=1)                  # (n,2)
    p = (np.array(home_m + away_m, dtype=np.float32)[None] + shift[:, None, :]) / scale
    b = (np.array(ball_m, dtype=np.float32)[None] + shift) / scale
    return b.astype(np.float32), p.astype(np.float32)


def test_nearest_team_gets_the_probability_and_loose_ball_is_weaker():
    # home player 1 m from ball, away 9 m: home favoured
    b, p = one_frame((50, 30), [(51, 30), (80, 10)], [(59, 30), (20, 50)])
    P = emission_probabilities(b, p, TEAM, FPS)
    mid = P[30]
    assert mid[0] > mid[1] + 0.1 and mid[0] > mid[2]
    # mirror: away favoured by the same margin
    b2, p2 = one_frame((50, 30), [(59, 30), (80, 10)], [(51, 30), (20, 50)])
    assert np.allclose(emission_probabilities(b2, p2, TEAM, FPS)[30][[0, 1]], mid[[1, 0]], atol=1e-9)
    # same advantage but the ball far from everyone (loose): much weaker team evidence
    b3, p3_ = one_frame((50, 30), [(70, 30), (80, 10)], [(79, 30), (20, 50)])
    far = emission_probabilities(b3, p3_, TEAM, FPS)[30]
    assert abs(far[0] - far[1]) < 0.5 * abs(mid[0] - mid[1])


def test_out_of_bounds_is_dead_and_missing_ball_is_exactly_uniform():
    b, p = one_frame((50, 30), [(51, 30), (80, 10)], [(59, 30), (20, 50)])
    b[:20] = np.array([1.02, 0.5], dtype=np.float32)               # ball beyond the end line
    b[40:] = np.nan
    P = emission_probabilities(b, p, TEAM, FPS)
    assert P[10].argmax() == 2 and P[10][2] > 0.9
    assert np.array_equal(P[45], np.full(3, 1 / 3))
    assert P[30].argmax() != 2


def test_probabilities_are_valid_even_with_all_players_missing():
    b, p = one_frame((50, 30), [(51, 30), (80, 10)], [(59, 30), (20, 50)])
    p[:] = np.nan
    P = emission_probabilities(b, p, TEAM, FPS)
    assert np.isfinite(P).all() and np.allclose(P.sum(1), 1) and (P >= 0).all()
    assert np.allclose(P[:, :2], P[:, [1, 0]])                     # no team evidence at all


def test_config_validation():
    with pytest.raises(ValueError):
        emission_probabilities(np.zeros((5, 2), np.float32), np.zeros((5, 4, 2), np.float32), TEAM, FPS,
                               RulesConfig(control_radius_m=0))
    with pytest.raises(ValueError):
        RulesConfig(out_logit=True).validate()


def _tracking_files(tmp_path, n, seed):
    rng = np.random.default_rng(seed)
    truth = synth.make_truth(n, rng, min_s=1.0)
    vis = synth.make_visibility(n, rng)
    ball, players, team = make_world(truth, rng)
    write_atd(tmp_path / "atd.txt", tmp_path / "atd.xml", ball, players, team, observed_ball=vis, rng=rng)
    return truth, vis, ball, players, team


def test_reader_roundtrips_and_rejects_bad_files(tmp_path):
    truth, vis, ball, players, team = _tracking_files(tmp_path, 300, 0)
    b, p = read_atd(tmp_path / "atd.txt", 300, len(team))
    assert np.array_equal(np.isfinite(b).all(1), vis)
    ok = np.isfinite(p).all(-1)
    assert np.allclose(p[ok], players.astype(np.float32)[ok], atol=2e-5)
    with pytest.raises(ValueError):
        read_atd(tmp_path / "atd.txt", 299, len(team))             # more records than frames
    with pytest.raises(ValueError):
        read_atd(tmp_path / "atd.txt", 301, len(team))             # fewer records than frames
    with pytest.raises(ValueError):
        read_atd(tmp_path / "atd.txt", 300, len(team) + 1)         # slot count disagrees with metadata
    bad = tmp_path / "bad.txt"
    lines = (tmp_path / "atd.txt").read_text().splitlines()
    lines[5] = "99" + lines[5][lines[5].index(":"):]
    bad.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        read_atd(bad, 300, len(team))


def test_rules_then_decoder_on_synthetic_tracking_beats_raw_and_smoothed_nearest_team(tmp_path):
    n = 12000
    truth, vis, ball, players, team = _tracking_files(tmp_path, n, 3)
    spec = MatchSpec("synthetic_rules", n, 25)
    inputs = {"atd_raw": tmp_path / "atd.txt", "atd_meta": tmp_path / "atd.xml"}
    batch = predict_emissions(spec, inputs, {"seed": 0})
    batch.validate(spec)
    raw = spec_argmax = batch.probabilities.argmax(1)

    def smooth(codes, hw=50):
        oh = np.stack([(codes == c).astype(float) for c in range(3)], 1)
        k = np.ones(2 * hw + 1)
        return np.stack([np.convolve(oh[:, c], k, mode="same") for c in range(3)], 1).argmax(1)

    decoded = decode_stage(batch, {}, {"seed": 0, "temperature": 2.0})
    codes = np.array([{"H": 0, "A": 1, "D": 2}[x] for x in decoded.labels])
    s_dec, s_raw, s_smooth = spell_f1(truth, codes), spell_f1(truth, raw), spell_f1(truth, smooth(raw))
    assert s_dec > s_smooth + 0.05 and s_dec > s_raw + 0.3, (s_dec, s_smooth, s_raw)
    # sanity: it labels a good share of frames correctly on this idealised world
    assert (codes == truth).mean() > 0.7


def test_unknown_or_bad_rules_settings_rejected(tmp_path):
    _tracking_files(tmp_path, 100, 1)
    spec = MatchSpec("synthetic_rules", 100, 25)
    inputs = {"atd_raw": tmp_path / "atd.txt", "atd_meta": tmp_path / "atd.xml"}
    for bad in ({"nonsense": 1}, {"fitted": ["nonsense"]}, {"out_logit": -1}):
        with pytest.raises(ValueError):
            predict_emissions(spec, inputs, bad)


def test_rules_viterbi_runs_end_to_end_through_p4_and_reproduces(tmp_path):
    n = 3000
    _tracking_files(tmp_path, n, 5)
    spec = MatchSpec("synthetic_rules_e2e", n, 25, ((100, 1500), (1600, 2900)))
    config = {"schema_version": 1, "match": spec.as_dict(), "method": "rules_viterbi",
              "data_kind": "synthetic", "seed": 1,
              "predictor": {"kind": "adapter", "stage": {
                  "callable": "p3.rules:predict_emissions",
                  "inputs": {"atd_raw": "atd.txt", "atd_meta": "atd.xml"}, "settings": {"fitted": []}}},
              "decoder": {"callable": "p3.stage:decode_stage", "settings": {"temperature": 2.0, "fitted": []}}}
    write_json(tmp_path / "config.json", config)
    record = run(tmp_path / "config.json", tmp_path / "r1")
    assert record["status"] == "complete" and record["label_counts"].get("X", 0) == 100 + 100 + 100
    assert reproduce(tmp_path / "r1", tmp_path / "r2")["identical_csv"]
