import itertools
from dataclasses import replace

import numpy as np
import pytest

from p3 import DecoderConfig, decode_half, decode_match, synth
from p3.decoder import (H, A, D, _Chain, _ffbs, _forward, _log_emission, _viterbi)
from p3.official import pm
from p3.spells import spell_f1

FPS = 25
ONE_SEC = FPS


def onehot_probs(codes, conf=0.96):
    P = np.full((len(codes), 3), (1 - conf) / 2)
    P[np.arange(len(codes)), codes] = conf
    return P


def seg(*parts):
    """seg((H, 100), (A, 80)) -> codes."""
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in parts])


def segments_of(codes):
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    b = np.concatenate(([0], breaks, [len(codes)]))
    return [(int(codes[b[i]]), int(b[i + 1] - b[i])) for i in range(len(b) - 1)]


VIT = DecoderConfig(use_mbr=False)


# ---------------------------------------------------------------- exactness on a tiny problem
def _tiny_cfg():
    # 2-frame minimum durations, so the expanded chain has 8 states and brute force is feasible.
    return DecoderConfig(fps=1, min_poss_s=2, min_dead_s=2, mean_poss_s=4, mean_dead_s=3,
                         use_mbr=False)


def _brute_force(chain, logE):
    n, K = logE.shape
    best, best_lp = None, -np.inf
    dist = {}
    for path in itertools.product(range(K), repeat=n):
        lp = chain.logpi[path[0]] + logE[0, path[0]]
        for t in range(1, n):
            lp += chain.logA[path[t - 1], path[t]] + logE[t, path[t]]
        if np.isfinite(lp):
            dist[path] = lp
            if lp > best_lp:
                best, best_lp = path, lp
    paths = list(dist)
    lps = np.array([dist[p] for p in paths])
    probs = np.exp(lps - lps.max())
    probs /= probs.sum()
    return best, dict(zip(paths, probs))


def test_viterbi_and_ffbs_match_brute_force():
    cfg = _tiny_cfg()
    chain = _Chain(cfg)
    assert chain.K == 8
    rng = np.random.default_rng(3)
    n = 5
    P = rng.dirichlet(np.ones(3), size=n)
    logE = _log_emission(P, None, cfg, chain)

    best, exact = _brute_force(chain, logE)
    assert tuple(_viterbi(chain.logpi, chain.logA, logE)) == best

    alpha = _forward(chain.pi, chain.A, logE)
    z = _ffbs(alpha, chain.A, 60000, np.random.default_rng(4))      # (n, M)
    counts = {}
    for col in map(tuple, z.T):
        counts[col] = counts.get(col, 0) + 1
    m = z.shape[1]
    assert set(counts) <= set(exact), "sampler produced a path of zero posterior probability"
    tv = 0.5 * sum(abs(counts.get(p, 0) / m - exact.get(p, 0.0)) for p in set(exact) | set(counts))
    assert tv < 0.02, tv


def test_transition_rows_sum_to_one_and_min_duration_enforced_by_structure():
    chain = _Chain(DecoderConfig())
    assert np.allclose(chain.A.sum(axis=1), 1.0)
    assert chain.K == 25 * 4
    # only sub-state 0 of each base state is enterable from outside the chain
    entry = set(chain.offsets[:-1].tolist())
    for j in range(chain.K):
        preds = set(np.flatnonzero(chain.A[:, j] > 0).tolist())
        if j not in entry:
            assert preds <= {j - 1, j}, (j, preds)


# ---------------------------------------------------------------- behaviour
def test_clean_input_is_reproduced_exactly():
    truth = seg((H, 400), (D, 200), (A, 500), (A, 0), (H, 350), (A, 300))
    out = decode_half(onehot_probs(truth), None, VIT)
    assert np.array_equal(out, truth)


def test_isolated_flicker_removed_but_real_turnover_kept():
    truth = seg((H, 500), (A, 500))
    P = onehot_probs(truth)
    P[200:205] = onehot_probs(np.full(5, A))      # 5-frame blip of A inside the H possession
    P[800:803] = onehot_probs(np.full(3, H))      # 3-frame blip of H inside the A possession
    out = decode_half(P, None, VIT)
    assert np.array_equal(out, truth)


def test_flicker_removed_even_with_weaker_evidence_and_mbr():
    truth = seg((H, 600), (A, 600))
    P = onehot_probs(truth, conf=0.7)
    P[300:310] = onehot_probs(np.full(10, A), conf=0.7)
    out = decode_half(P, None, DecoderConfig(use_mbr=True, n_samples=32, n_candidates=8))
    assert np.array_equal(out, truth)


def test_min_duration_holds_on_noisy_synthetic():
    rng = np.random.default_rng(5)
    truth = synth.make_truth(30000, rng)
    vis = synth.make_visibility(30000, rng)
    P = synth.make_probs(truth, vis, rng)
    for cfg in (VIT, DecoderConfig(min_poss_s=2.0, min_dead_s=1.5, use_mbr=False)):
        out = decode_half(P, vis, cfg)
        poss_min = round(cfg.min_poss_s * FPS)
        dead_min = round(cfg.min_dead_s * FPS)
        for (code, n) in segments_of(out)[:-1]:           # last segment may be cut by the block end
            assert n >= (dead_min if code == D else poss_min), (code, n)
        assert len(segments_of(out)) > 3


def test_missing_ball_carries_state_instead_of_trusting_garbage():
    truth = seg((H, 800))
    P = onehot_probs(truth)
    vis = np.ones(800, dtype=bool)
    vis[300:600] = False
    P[300:600] = onehot_probs(np.full(300, A))        # confident garbage while the ball is gone
    ignored = decode_half(P, vis, replace(VIT, missing_weight=0.0))
    trusted = decode_half(P, vis, replace(VIT, missing_weight=1.0))
    assert np.array_equal(ignored, truth)
    assert not np.array_equal(trusted, truth)         # shows the flag is what makes the difference


def test_nan_and_invalid_rows_are_uninformative():
    truth = seg((H, 600))
    P = onehot_probs(truth)
    P[100:300] = np.nan
    P[400:420] = -1.0
    out = decode_half(P, None, VIT)
    assert np.array_equal(out, truth)


def test_restart_prior_picks_the_other_team_after_dead_ball():
    # H possession, dead ball, then an alive stretch the evidence cannot split between H and A.
    flat_alive = np.tile([0.5, 0.5, 1e-6], (300, 1))
    P = np.vstack([onehot_probs(seg((H, 400), (D, 150))), flat_alive])
    P = P / P.sum(axis=1, keepdims=True)
    switch = decode_half(P, None, replace(VIT, restart_switch=0.8))
    same = decode_half(P, None, replace(VIT, restart_switch=0.2))
    assert switch[-1] == A and same[-1] == H
    # symmetric: dead ball after A
    P2 = np.vstack([onehot_probs(seg((A, 400), (D, 150))), flat_alive])
    P2 = P2 / P2.sum(axis=1, keepdims=True)
    assert decode_half(P2, None, replace(VIT, restart_switch=0.8))[-1] == H


def test_temperature_trades_off_trust():
    truth = seg((H, 400), (A, 400))
    P = onehot_probs(truth, conf=0.6)
    P[200:260] = onehot_probs(np.full(60, A), conf=0.6)   # 2.4 s of contrary evidence
    trusting = decode_half(P, None, replace(VIT, temperature=1.0))
    sceptical = decode_half(P, None, replace(VIT, temperature=8.0))
    assert (trusting[200:260] == A).any()                 # long run of evidence wins at T=1
    assert np.array_equal(sceptical, truth)               # tempered evidence cannot beat the prior


# ---------------------------------------------------------------- MBR
def test_mbr_is_deterministic_and_never_worse_than_viterbi_by_its_own_objective():
    rng = np.random.default_rng(6)
    truth = synth.make_truth(12000, rng)
    vis = synth.make_visibility(12000, rng)
    P = synth.make_probs(truth, vis, rng)
    cfg = DecoderConfig(temperature=4.0, use_mbr=True, n_samples=32, n_candidates=10)
    a, info = decode_half(P, vis, cfg, return_info=True)
    b = decode_half(P, vis, cfg)
    assert np.array_equal(a, b)
    u = info["mbr_utility"]
    assert u[info["mbr_choice"]] >= u[0] - 1e-12          # Viterbi is candidate 0


def test_decoder_beats_argmax_and_fixed_smoothing_on_synthetic():
    def smooth(P, hw=50):
        oh = np.zeros_like(P)
        oh[np.arange(len(P)), P.argmax(1)] = 1.0
        k = np.ones(2 * hw + 1)
        return np.stack([np.convolve(oh[:, c], k, mode="same") for c in range(3)], 1).argmax(1)

    scores = {"argmax": [], "smooth": [], "hsmm": []}
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        n = 25000
        truth = synth.make_truth(n, rng)
        vis = synth.make_visibility(n, rng)
        P = synth.make_probs(truth, vis, rng)
        scores["argmax"].append(spell_f1(truth, P.argmax(1)))
        scores["smooth"].append(spell_f1(truth, smooth(P)))
        out = decode_half(P, vis, DecoderConfig(use_mbr=False, temperature=4.0))
        scores["hsmm"].append(spell_f1(truth, out))
    m = {k: float(np.mean(v)) for k, v in scores.items()}
    assert m["hsmm"] > m["smooth"] + 0.03 and m["hsmm"] > m["argmax"] + 0.3, m


# ---------------------------------------------------------------- match-level plumbing
def test_decode_match_layout_and_determinism():
    n = 3000
    blocks = [(100, 1400), (1800, 2900)]
    rng = np.random.default_rng(7)
    truth = synth.make_truth(n, rng, min_s=1.0)
    P = onehot_probs(truth)
    cfg = DecoderConfig(use_mbr=True, n_samples=8, n_candidates=3)
    out = decode_match(P, None, blocks, cfg)
    assert out.shape == (n,) and out.dtype.kind == "U"
    inside = np.zeros(n, dtype=bool)
    for a, b in blocks:
        inside[a:b] = True
    assert (out[~inside] == "X").all()
    assert set(out[inside]) <= {"H", "A", "D"}
    assert np.array_equal(out, decode_match(P, None, blocks, cfg))


@pytest.mark.parametrize("blocks", [[(0, 10), (5, 20)], [(0, 99999)], [(10, 10)]])
def test_decode_match_rejects_bad_blocks(blocks):
    with pytest.raises(ValueError):
        decode_match(np.full((100, 3), 1 / 3), None, blocks, VIT)


def test_input_validation():
    with pytest.raises(ValueError):
        decode_half(np.zeros((10, 2)), None, VIT)
    with pytest.raises(ValueError):
        decode_half(np.zeros((10, 3)) + 1 / 3, np.ones(5, dtype=bool), VIT)
    with pytest.raises(ValueError):
        DecoderConfig(restart_switch=1.5).validate()
    assert len(decode_half(np.zeros((0, 3)), None, VIT)) == 0
    assert len(decode_half(np.full((1, 3), 1 / 3), None, DecoderConfig())) == 1
