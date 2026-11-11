"""Ablation on synthetic halves: what does each decoder component buy, by the official metrics?

    python scripts/ablation.py [--seeds 5] [--frames 40000]

Synthetic data is only a sanity/behaviour check. Real tuning needs external labelled matches.
"""
import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from p3 import DecoderConfig, decode_half  # noqa: E402
from p3 import synth  # noqa: E402
from p3.official import pm  # noqa: E402


def smooth_majority(P, half_width):
    onehot = np.zeros_like(P)
    onehot[np.arange(len(P)), P.argmax(1)] = 1.0
    kernel = np.ones(2 * half_width + 1)
    votes = np.stack([np.convolve(onehot[:, k], kernel, mode="same") for k in range(3)], axis=1)
    return votes.argmax(1).astype(np.int8)


def score(truth, pred):
    runs = [(0, len(truth))]
    s = pm.spell_scores(truth, pred, runs)
    cm3 = pm.scored_confusion(pm.confusion(truth, pred))
    return s["mean_f1"], pm.macro_f1(cm3), s["n_pred"], s["n_gt"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--frames", type=int, default=40000)
    args = ap.parse_args()

    base = DecoderConfig(use_mbr=False, temperature=4.0)
    methods = {
        "argmax": lambda P, v: P.argmax(1).astype(np.int8),
        "majority smooth (+-50)": lambda P, v: smooth_majority(P, 50),
        "HSMM, ball-missing trusted": lambda P, v: decode_half(P, None, base),
        "HSMM, missing ignored": lambda P, v: decode_half(P, v, base),
        "HSMM + MBR": lambda P, v: decode_half(P, v, replace(base, use_mbr=True)),
    }
    rows = {k: [] for k in methods}
    times = {k: 0.0 for k in methods}
    n_gt = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        truth = synth.make_truth(args.frames, rng)
        vis = synth.make_visibility(args.frames, rng)
        P = synth.make_probs(truth, vis, rng)
        for name, fn in methods.items():
            t0 = time.time()
            pred = fn(P, vis)
            times[name] += time.time() - t0
            sf1, mf1, npred, ngt = score(truth, pred)
            rows[name].append((sf1, mf1, npred))
        n_gt.append(ngt)
    print("synthetic halves: {} x {} frames, ~{:.0f} true spells each\n".format(
        args.seeds, args.frames, np.mean(n_gt)))
    print("{:<30} {:>9} {:>9} {:>12} {:>8}".format("method", "spellF1", "macroF1", "spells pred", "sec"))
    for name, vals in rows.items():
        v = np.array(vals)
        print("{:<30} {:>9.3f} {:>9.3f} {:>12.0f} {:>8.1f}".format(
            name, v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), times[name]))
