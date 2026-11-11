"""Synthetic matches for testing the decoder. Illustrative only: not a model of real football.

Truth is a chain of possessions and dead-ball stretches. Probabilities are a noisy, autocorrelated
softmax around the truth, and the ball goes missing in runs, with uninformative probabilities
while it is missing (the failure mode the real tracking has).
"""
import numpy as np

H, A, D = 0, 1, 2


def _lognormal(rng, mean, sigma=0.7):
    mu = np.log(mean) - sigma ** 2 / 2.0
    return rng.lognormal(mu, sigma)


def make_truth(n, rng, fps=25, mean_poss_s=14.0, mean_dead_s=8.0, p_dead=0.5, p_switch=0.8,
               min_s=2.0):
    """int8 codes 0=H,1=A,2=D, length n. Every segment is >= min_s seconds except the last."""
    codes = np.empty(n, dtype=np.int8)
    pos, team = 0, int(rng.integers(0, 2))
    while pos < n:
        dur = max(int(min_s * fps), int(_lognormal(rng, mean_poss_s) * fps))
        codes[pos:pos + dur] = team
        pos += dur
        if pos >= n:
            break
        if rng.random() < p_dead:
            dur = max(int(min_s * fps), int(_lognormal(rng, mean_dead_s) * fps))
            codes[pos:pos + dur] = D
            pos += dur
            if rng.random() < p_switch:
                team = 1 - team
        else:
            team = 1 - team
    return codes


def make_visibility(n, rng, fps=25, mean_visible_s=4.0, mean_missing_s=3.0):
    """Boolean mask, ~43% missing by default, in runs (two-state Markov chain)."""
    p_leave_vis = 1.0 / (mean_visible_s * fps)
    p_leave_mis = 1.0 / (mean_missing_s * fps)
    vis = np.empty(n, dtype=bool)
    state = True
    u = rng.random(n)
    for t in range(n):
        vis[t] = state
        if state and u[t] < p_leave_vis:
            state = False
        elif (not state) and u[t] < p_leave_mis:
            state = True
    return vis


def make_probs(truth, visible, rng, signal=1.6, missing_signal=0.0, phi=0.97, noise=1.2):
    """(n,3) probabilities: softmax(signal*onehot + AR(1) noise). Weak/no signal where not visible."""
    n = len(truth)
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), truth] = 1.0
    strength = np.where(visible, signal, missing_signal)[:, None]
    eps = rng.normal(0.0, noise * np.sqrt(1 - phi ** 2), size=(n, 3))
    ar = np.empty((n, 3))
    ar[0] = rng.normal(0.0, noise, size=3)
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + eps[t]
    white = rng.normal(0.0, 0.4, size=(n, 3))
    logits = strength * onehot + ar + white
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)
