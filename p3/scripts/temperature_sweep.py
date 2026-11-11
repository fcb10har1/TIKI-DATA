"""Viterbi vs MBR across temperatures on synthetic halves (behaviour check, not tuning)."""
import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from p3 import DecoderConfig, decode_half, synth  # noqa: E402
from p3.spells import spell_f1  # noqa: E402

data = []
for seed in range(4):
    rng = np.random.default_rng(seed)
    n = 40000
    truth = synth.make_truth(n, rng)
    vis = synth.make_visibility(n, rng)
    data.append((truth, vis, synth.make_probs(truth, vis, rng)))
print("{:>5} {:>10} {:>10} {:>12} {:>12}".format("T", "vit F1", "mbr F1", "vit spells", "mbr spells"))
for T in (1, 2, 4, 8, 16, 32):
    row = []
    for use_mbr in (False, True):
        f, ns = [], []
        for truth, vis, P in data:
            out = decode_half(P, vis, DecoderConfig(temperature=T, use_mbr=use_mbr))
            f.append(spell_f1(truth, out))
            ns.append(len(np.flatnonzero(np.diff(out) != 0)) + 1)
        row += [np.mean(f), np.mean(ns)]
    print("{:>5} {:>10.3f} {:>10.3f} {:>12.0f} {:>12.0f}".format(T, row[0], row[2], row[1], row[3]))
