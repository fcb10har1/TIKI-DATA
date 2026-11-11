"""Metric-aware H/A/D decoder.

Input is an (N, 3) array of per-frame probabilities in the order H, A, D, from any source (the
rules baseline, the TCN, an ensemble). Output is one label per frame.

Model. A hidden semi-Markov chain over four base states:

    H, A      team in possession, ball alive
    Dh, Da    dead ball that FOLLOWED an H / an A possession

Dh and Da are both emitted as "D"; they exist only so the chain remembers who had the ball
before the stoppage, which lets it express restart logic (after a ball goes out, the other team
usually restarts) as a transition probability.

Minimum duration is exact: each base state is a chain of `min` sub-states that must be walked in
order, followed by a free sub-state with a geometric self-loop. Total time in a state is therefore
at least `min` frames, with mean `mean` frames.

Emissions. The probabilities are treated as likelihoods, tempered by `temperature` (>1 when the
upstream probabilities are over-confident or strongly autocorrelated, which frame-wise model
outputs are). Frames where the ball is not visible get weight `missing_weight` (default 0), so the
chain carries its state through occlusions instead of trusting a number the model made up.

Decoding.
  1. Viterbi gives the single most probable path.
  2. Minimum Bayes risk (optional): draw samples from the exact posterior with forward-filtering
     backward-sampling, and return the candidate with the highest mean spellF1 against the other
     samples. That is the sequence that maximises expected score under the model's own
     uncertainty, which the ranking metric (spellF1) rewards, rather than the most likely path.
"""
from dataclasses import dataclass, replace

import numpy as np

from .spells import spell_f1

H, A, D = 0, 1, 2
LABEL_CHARS = np.array(["H", "A", "D"])
# Base-state indices inside the chain.
S_H, S_A, S_DH, S_DA = 0, 1, 2, 3
_TEAM_OF_BASE = np.array([H, A, D, D], dtype=np.int8)


@dataclass(frozen=True)
class DecoderConfig:
    fps: int = 25
    min_poss_s: float = 1.0       # shortest allowed possession
    min_dead_s: float = 1.0       # shortest allowed dead-ball stretch
    mean_poss_s: float = 14.0     # mean possession length
    mean_dead_s: float = 10.0     # mean dead-ball stretch length
    turnover_frac: float = 0.5    # of possession exits, share that go straight to the other team
    restart_switch: float = 0.8   # after dead ball, probability the OTHER team gets the ball
    start_alive: float = 0.5      # prior that a graded block starts with the ball alive (neutral:
                                  # DATA.md says the 2nd graded range starts before kickoff)
    temperature: float = 1.0      # divides log-probabilities; >1 = trust the model less
    missing_weight: float = 0.0   # weight of evidence on frames where the ball is not visible
    prob_floor: float = 1e-4      # probabilities are clipped to [floor, 1] before the log
    use_mbr: bool = False         # off by default: did not beat tuned Viterbi on synthetic (README)
    n_samples: int = 64           # posterior samples used as pseudo-truth for MBR
    n_candidates: int = 24        # how many samples compete as MBR candidates (plus Viterbi)
    seed: int = 0

    def validate(self):
        for name in ("min_poss_s", "min_dead_s", "mean_poss_s", "mean_dead_s", "temperature"):
            if not getattr(self, name) > 0:
                raise ValueError("{} must be > 0".format(name))
        for name in ("turnover_frac", "restart_switch", "start_alive", "missing_weight"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("{} must be in [0, 1]".format(name))
        if not 0.0 < self.prob_floor < 1.0 / 3.0:
            raise ValueError("prob_floor must be in (0, 1/3)")
        if self.n_samples < 2 or self.n_candidates < 1:
            raise ValueError("n_samples >= 2 and n_candidates >= 1 required")


class _Chain:
    """Dense expanded-state transition structure for a config. K = sum of the four chain lengths."""

    def __init__(self, cfg):
        cfg.validate()
        fps = cfg.fps
        min_frames = [max(1, int(round(cfg.min_poss_s * fps)))] * 2 + \
                     [max(1, int(round(cfg.min_dead_s * fps)))] * 2
        mean_frames = [cfg.mean_poss_s * fps] * 2 + [cfg.mean_dead_s * fps] * 2
        self.lengths = min_frames
        self.offsets = np.concatenate(([0], np.cumsum(min_frames)))
        K = int(self.offsets[-1])
        self.K = K
        self.base = np.repeat(np.arange(4), min_frames).astype(np.int16)
        self.team = _TEAM_OF_BASE[self.base]

        def entry(s):
            return int(self.offsets[s])

        def free(s):
            return int(self.offsets[s] + self.lengths[s] - 1)

        # Leave rate of the free sub-state: time in state = (L - 1) + Geometric(leave), so
        # mean_frames = (L - 1) + 1 / leave.
        leave = [min(0.999, 1.0 / max(mean_frames[s] - (self.lengths[s] - 1), 1.0))
                 for s in range(4)]

        A_ = np.zeros((K, K))
        for s in range(4):
            for k in range(self.lengths[s] - 1):
                A_[entry(s) + k, entry(s) + k + 1] = 1.0
            A_[free(s), free(s)] += 1.0 - leave[s]

        t, r = cfg.turnover_frac, cfg.restart_switch
        # From a possession: straight to the other team, or into dead ball remembering who had it.
        A_[free(S_H), entry(S_A)] += leave[S_H] * t
        A_[free(S_H), entry(S_DH)] += leave[S_H] * (1.0 - t)
        A_[free(S_A), entry(S_H)] += leave[S_A] * t
        A_[free(S_A), entry(S_DA)] += leave[S_A] * (1.0 - t)
        # From dead ball: the other team restarts with probability r.
        A_[free(S_DH), entry(S_A)] += leave[S_DH] * r
        A_[free(S_DH), entry(S_H)] += leave[S_DH] * (1.0 - r)
        A_[free(S_DA), entry(S_H)] += leave[S_DA] * r
        A_[free(S_DA), entry(S_A)] += leave[S_DA] * (1.0 - r)

        if not np.allclose(A_.sum(axis=1), 1.0):
            raise AssertionError("transition rows must sum to 1")
        self.A = A_
        with np.errstate(divide="ignore"):
            self.logA = np.log(A_)

        pi = np.zeros(K)
        alive, dead = cfg.start_alive, 1.0 - cfg.start_alive
        pi[entry(S_H)] = alive / 2.0
        pi[entry(S_A)] = alive / 2.0
        pi[entry(S_DH)] = dead / 2.0
        pi[entry(S_DA)] = dead / 2.0
        self.pi = pi
        with np.errstate(divide="ignore"):
            self.logpi = np.log(pi)


def _log_emission(P, visible, cfg, chain):
    """(N, K) tempered log-likelihood per expanded state."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P must have shape (N, 3), got {}".format(P.shape))
    n = len(P)
    weight = np.ones(n)
    if visible is not None:
        visible = np.asarray(visible, dtype=bool)
        if visible.shape != (n,):
            raise ValueError("visible must have shape ({},), got {}".format(n, visible.shape))
        weight = np.where(visible, 1.0, cfg.missing_weight)
    bad = ~np.isfinite(P).all(axis=1) | (P < 0).any(axis=1)
    P = np.where(bad[:, None], 1.0 / 3.0, P)
    weight = np.where(bad, 0.0, weight)
    P = np.clip(P, cfg.prob_floor, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    logp = np.log(P) / cfg.temperature * weight[:, None]          # (N, 3)
    per_base = logp[:, [H, A, D, D]]                              # H, A, Dh, Da
    return per_base[:, chain.base]


def _viterbi(logpi, logA, logE):
    n, K = logE.shape
    delta = logpi + logE[0]
    back = np.zeros((n, K), dtype=np.int16)
    cols = np.arange(K)
    for t in range(1, n):
        cand = delta[:, None] + logA
        arg = cand.argmax(axis=0)
        back[t] = arg
        delta = cand[arg, cols] + logE[t]
    z = np.empty(n, dtype=np.int16)
    z[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        z[t - 1] = back[t, z[t]]
    return z


def _forward(pi, A, logE):
    """Scaled filtering distributions alpha[t] = P(z_t | y_0..t)."""
    n, K = logE.shape
    alpha = np.empty((n, K))
    e = np.exp(logE[0] - logE[0].max())
    a = pi * e
    alpha[0] = a / a.sum()
    for t in range(1, n):
        e = np.exp(logE[t] - logE[t].max())
        a = (alpha[t - 1] @ A) * e
        s = a.sum()
        if s <= 0.0:
            raise FloatingPointError("forward pass lost all probability mass at frame %d" % t)
        alpha[t] = a / s
    return alpha


def _ffbs(alpha, A, n_samples, rng):
    """(N, M) exact posterior samples of the expanded state path (no terminal constraint)."""
    n, K = alpha.shape
    z = np.empty((n, n_samples), dtype=np.int16)
    cdf = np.cumsum(alpha[-1])
    z[-1] = np.minimum((cdf[None, :] < rng.random(n_samples)[:, None] * cdf[-1]).sum(axis=1),
                       K - 1)
    for t in range(n - 2, -1, -1):
        w = alpha[t][:, None] * A[:, z[t + 1]]                    # (K, M)
        cs = np.cumsum(w, axis=0)
        tot = cs[-1]
        # tot == 0 cannot happen for a sampled z[t+1] (it has a predecessor with mass); guard
        # anyway so a numerical edge case degrades to the filtered mode instead of crashing.
        dead = tot <= 0.0
        u = rng.random(n_samples) * np.where(dead, 1.0, tot)
        idx = (cs < u[None, :]).sum(axis=0)
        idx = np.minimum(idx, K - 1)
        if dead.any():
            idx = np.where(dead, int(np.argmax(alpha[t])), idx)
        z[t] = idx
    return z


def decode_half(P, visible=None, cfg=None, return_info=False):
    """Decode one contiguous graded block. Returns int8 codes 0=H, 1=A, 2=D (length N).

    With return_info=True also returns a dict with the Viterbi path, the MBR choice and its
    expected utility, for diagnostics and tests.
    """
    cfg = cfg or DecoderConfig()
    chain = _Chain(cfg)
    P = np.asarray(P)
    n = len(P)
    if n == 0:
        out = np.zeros(0, dtype=np.int8)
        return (out, {}) if return_info else out
    logE = _log_emission(P, visible, cfg, chain)
    z_vit = _viterbi(chain.logpi, chain.logA, logE)
    vit = chain.team[z_vit]
    info = {"viterbi": vit}
    if not cfg.use_mbr or n < 2:
        return (vit, info) if return_info else vit

    rng = np.random.default_rng(cfg.seed)
    alpha = _forward(chain.pi, chain.A, logE)
    z = _ffbs(alpha, chain.A, cfg.n_samples, rng)                  # (N, M)
    samples = chain.team[z].T                                      # (M, N) int8
    n_cand = min(cfg.n_candidates, cfg.n_samples)
    cand_idx = rng.choice(cfg.n_samples, size=n_cand, replace=False)
    candidates = [vit] + [samples[i] for i in cand_idx]
    cand_ids = [-1] + [int(i) for i in cand_idx]

    utility = np.zeros(len(candidates))
    for c, (cand, cid) in enumerate(zip(candidates, cand_ids)):
        total, count = 0.0, 0
        for m in range(cfg.n_samples):
            if m == cid:
                continue  # never score a sample against itself
            total += spell_f1(samples[m], cand)
            count += 1
        utility[c] = total / count
    best = int(np.argmax(utility))                                 # ties -> earliest, Viterbi first
    out = candidates[best]
    info.update(mbr_utility=utility, mbr_choice=best, samples=samples)
    return (out, info) if return_info else out


def decode_match(P, visible, blocks, cfg=None):
    """Full-match decode. P is (n_frames, 3) over EVERY video frame; blocks are the graded halves.

    blocks: iterable of (first_frame, stop_frame), stop exclusive, e.g. for BAR-ATH
    [(11185, 90652), (110808, 190198)]. Each block is decoded independently (a possession cannot
    cross halftime) and every frame outside the blocks is written as 'X', which the scorer ignores.
    Returns an array of 'H' / 'A' / 'D' / 'X' strings, one per frame.
    """
    cfg = cfg or DecoderConfig()
    P = np.asarray(P)
    n_frames = len(P)
    if visible is not None and len(visible) != n_frames:
        raise ValueError("visible and P must have the same length")
    out = np.full(n_frames, "X", dtype="<U1")
    last_stop = 0
    for start, stop in sorted(blocks):
        if not (0 <= start < stop <= n_frames) or start < last_stop:
            raise ValueError("bad or overlapping block ({}, {}) for {} frames".format(
                start, stop, n_frames))
        last_stop = stop
        vis = None if visible is None else visible[start:stop]
        # Distinct, reproducible stream per block.
        codes = decode_half(P[start:stop], vis, replace(cfg, seed=cfg.seed + start))
        out[start:stop] = LABEL_CHARS[codes]
    return out
