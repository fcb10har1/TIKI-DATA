"""Fast possession-spell F1, numerically identical to the official possession_metrics.spell_scores.

The decoder evaluates this thousands of times (minimum-Bayes-risk selection), where the official
pure-Python version would be too slow. tests/test_spells.py checks equality against the official
code on randomised inputs, so the two cannot drift apart silently.
"""
import numpy as np

H, A, D = 0, 1, 2
SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)  # same as possession_metrics.SPELL_IOUS (tested)


def spells(codes):
    """(starts, stops, teams): maximal runs of H or A in one contiguous block, stops exclusive."""
    codes = np.asarray(codes)
    n = len(codes)
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [n]))
    teams = codes[starts]
    keep = (teams == H) | (teams == A)
    return starts[keep].astype(np.int64), stops[keep].astype(np.int64), teams[keep].astype(np.int64)


def _sorted_overlaps(g, p):
    """[(iou, gi, pj)] same-team overlapping pairs, ordered exactly as the official scorer does."""
    gs, ge, gt = g
    ps, pe, pt = p
    inter = np.minimum(ge[:, None], pe[None, :]) - np.maximum(gs[:, None], ps[None, :])
    ok = (inter > 0) & (gt[:, None] == pt[None, :])
    gi, pj = np.nonzero(ok)
    if len(gi) == 0:
        return gi, pj, np.zeros(0)
    union = np.maximum(ge[gi], pe[pj]) - np.minimum(gs[gi], ps[pj])
    iou = inter[gi, pj] / union.astype(np.float64)
    order = np.lexsort((pj, gi, -iou))  # primary key last: -iou, then gt index, then pred index
    return gi[order], pj[order], iou[order]


def spell_f1(gt_codes, pred_codes, ious=SPELL_IOUS):
    """Mean over IoU thresholds of the F1 of one-to-one greedy spell matching."""
    g = spells(gt_codes)
    p = spells(pred_codes)
    n_gt, n_pred = len(g[0]), len(p[0])
    if n_gt == 0 or n_pred == 0:
        return 0.0
    gi, pj, iou = _sorted_overlaps(g, p)
    gi, pj, iou = gi.tolist(), pj.tolist(), iou.tolist()
    total = 0.0
    for thr in ious:
        used_g, used_p = set(), set()
        tp = 0
        for a, b, v in zip(gi, pj, iou):
            if v < thr:
                break
            if a in used_g or b in used_p:
                continue
            used_g.add(a)
            used_p.add(b)
            tp += 1
        if tp:
            precision = tp / float(n_pred)
            recall = tp / float(n_gt)
            total += 2.0 * precision * recall / (precision + recall)
    return total / len(ious)
