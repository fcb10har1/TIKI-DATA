"""Frozen correction proposals, features, guards and external-training targets."""
from dataclasses import dataclass
import numpy as np
from evaluation import possession_metrics as pm

def overlaps_linear(gs, ps):
    """Official IoU ordering, avoiding a quadratic loop over disjoint spells."""
    out = []
    j = 0
    for i, (a, b, c) in enumerate(gs):
        while j < len(ps) and ps[j][1] <= a:
            j += 1
        k = j
        while k < len(ps) and ps[k][0] < b:
            u, v, d = ps[k]
            if c == d:
                out.append(((min(b, v)-max(a, u))/(max(b, v)-min(a, u)), i, k))
            k += 1
    return sorted(out, key=lambda x: (-x[0], x[1], x[2]))


@dataclass(frozen=True)
class Edit:
    start: int
    stop: int
    label: int
    guard_start: int
    guard_stop: int


def codes(labels):
    if np.asarray(labels).dtype.kind in 'iu':
        return np.asarray(labels, dtype=np.int8)
    return pm.label_codes(labels)


def proposals(baseline, alternatives, spec, maximum_seconds=60.):
    """Unique full predicted spells; no GT-dependent proposals or target counts."""
    baseline = codes(baseline)
    alternatives = [codes(x) for x in alternatives]
    out = set()
    for left, right in spec.periods:
        runs = pm.segments(baseline, [(left, right)])
        starts = np.asarray([x[0] for x in runs])
        stops = np.asarray([x[1] for x in runs])
        for alternate in alternatives:
            for a, b, label in pm.segments(alternate, [(left, right)]):
                if label > 2 or b-a > maximum_seconds*spec.fps or np.all(baseline[a:b] == label):
                    continue
                i = int(np.searchsorted(stops, a, side='right'))
                j = int(np.searchsorted(starts, b, side='left'))-1
                # Reserve all baseline spells this edit changes, including any
                # neighbouring spell that merges with the new label.
                if a == starts[i] and i and runs[i-1][2] == label:
                    i -= 1
                if b == stops[j] and j+1 < len(runs) and runs[j+1][2] == label:
                    j += 1
                out.add(Edit(a, b, label, int(starts[i]), int(stops[j])))
    return sorted(out, key=lambda e: (e.start, e.stop, e.label))


def _probability_features(p, base, a, b, label, left, right, fps):
    # Canonicalise home/away relative to the proposed team. D has no team, so
    # sort its two live probabilities. There is no absolute team identity.
    def ordered(u, v):
        z = p[u:v]
        if label < 2:
            return z[:, [label, 1-label, 2]]
        return np.column_stack((z[:, 2], np.maximum(z[:, 0], z[:, 1]),
                                np.minimum(z[:, 0], z[:, 1])))
    v = ordered(a, b)
    rows = [v.mean(0), v.std(0), np.quantile(v, .1, axis=0), np.quantile(v, .9, axis=0)]
    logp = np.log(np.maximum(p[a:b], 1e-6))
    margin = logp[:, label]-logp[np.arange(b-a), base[a:b]]
    rows += [np.asarray([margin.mean(), np.mean(margin > 0),
                         np.mean(p[a:b].argmax(1) == label),
                         np.mean(-np.sum(p[a:b]*logp, axis=1))])]
    for u, v in ((max(left, a-int(fps)), a), (b, min(right, b+int(fps)))):
        rows.append(ordered(u, v).mean(0) if u < v else np.zeros(3))
    return np.concatenate(rows)


def feature_matrix(edits, baseline, alternatives, probabilities, boundaries,
                   values, observed, visible, spec, missing_weight):
    """Fixed, team-equivariant inference features; never receives ground truth."""
    base = codes(baseline)
    alts = [codes(x) for x in alternatives]
    rows = []
    for edit in edits:
        a, b, label = edit.start, edit.stop, edit.label
        left, right = next((l, r) for l, r in spec.periods if l <= a < b <= r)
        z = base[a:b]
        fraction = np.bincount(z, minlength=3)/(b-a)
        order = [label, 1-label, 2] if label < 2 else [2, 0, 1]
        fractions = fraction[order]
        if label == 2:
            fractions[1:] = np.sort(fractions[1:])[::-1]
        row = [np.asarray([np.log1p((b-a)/spec.fps), float(label == 2), missing_weight,
                           np.mean(z != label), np.sum(z[1:] != z[:-1]),
                           np.log1p((edit.guard_stop-edit.guard_start)/spec.fps),
                           float(a > left and base[a-1] == label),
                           float(b < right and base[b] == label)]), fractions]
        for alt in alts:
            row.append(np.asarray([np.mean(alt[a:b] == label),
                                   np.sum(alt[a+1:b] != alt[a:b-1])]))
        for p in probabilities:
            row.append(_probability_features(p, base, a, b, label, left, right, spec.fps))
        for boundary in boundaries:
            width = max(1, int(.2*spec.fps))
            local = boundary[a:b]
            row.append(np.asarray([boundary[a], boundary[min(b, right-1)],
                boundary[max(left,a-width):min(right,a+width+1)].max(),
                boundary[max(left,b-width):min(right,b+width+1)].max(),
                local.mean(), np.mean(local >= .3)]))
        row.append(np.asarray([visible[a:b].mean(), visible[max(left,a-int(spec.fps)):a+1].mean(),
                               visible[b-1:min(right,b+int(spec.fps))].mean()]))
        # Ball geometry and player motion, with masks and relative team order.
        for home, away in ((1,2),(3,4),(5,6),(7,8),(9,10),(15,16)):
            pair = []
            for col in (home, away):
                mask = observed[a:b, col]
                v = values[a:b, col][mask]
                pair.append([float(mask.mean()), float(v.mean()) if len(v) else 0.,
                             float(v.std()) if len(v) else 0.])
            if label == 1:
                pair.reverse()
            elif label == 2:
                pair.sort(key=lambda x: tuple(x))
            row.append(np.asarray(pair).ravel())
        rows.append(np.concatenate(row))
    if not rows:
        width = 50+2*len(alternatives)+22*len(probabilities)+6*len(boundaries)
        return np.empty((0, width), dtype=np.float32)
    result = np.asarray(rows, dtype=np.float32)
    assert np.isfinite(result).all()
    return result


def apply_edits(baseline, edits, gains, threshold):
    """Accept positive-quality edits greedily; protect their baseline spell spans."""
    out = codes(baseline).copy()
    taken = []
    accepted = []
    assert len(edits) == len(gains)
    for index in sorted(range(len(edits)), key=lambda i: (-gains[i], i)):
        if not np.isfinite(gains[index]) or gains[index] <= threshold:
            continue
        e = edits[index]
        if any(e.guard_start < b and a < e.guard_stop for a, b in taken):
            continue
        out[e.start:e.stop] = e.label
        taken.append((e.guard_start, e.guard_stop))
        accepted.append(index)
    return np.asarray(pm.LABELS)[out], accepted


def _spell_f1(gs, ps):
    overlap = overlaps_linear(gs, ps)
    denominator = len(gs)+len(ps)
    return float(np.mean([2*pm.match_spells(overlap, len(gs), len(ps), t)[0]/denominator
                          if denominator else 0. for t in pm.SPELL_IOUS]))


def training_targets(baseline, truth, edits):
    """Exact single-edit full-match SpellF1 deltas, normalized to event units.

    Unknown-labelled edits are omitted. These labels are used for training only;
    neither proposal generation nor inference calls this function.
    """
    base, gt = codes(baseline), codes(truth.labels)
    blocks = [(left+a, left+b) for left, right in truth.match.periods
              for a, b in pm.valid_runs(gt[left:right])]
    gs = pm.possession_spells(gt, blocks)
    ps = pm.possession_spells(base, blocks)
    original = _spell_f1(gs, ps)
    scale = (len(gs)+len(ps))/2.
    y = np.full(len(edits), np.nan)
    changed = base.copy()
    for i, edit in enumerate(edits):
        a, b, label = edit.start, edit.stop, edit.label
        if not any(left <= a < b <= right for left, right in blocks):
            continue
        changed[a:b] = label
        after = _spell_f1(gs, pm.possession_spells(changed, blocks))
        y[i] = (after-original)*scale
        changed[a:b] = base[a:b]
    return y
