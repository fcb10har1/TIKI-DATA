import numpy as np

from p3.official import pm
from p3.spells import SPELL_IOUS, spell_f1, spells


def _random_codes(rng, n, with_dead=True):
    codes = np.empty(n, dtype=np.int8)
    pos = 0
    while pos < n:
        dur = int(rng.integers(1, 120))
        codes[pos:pos + dur] = rng.integers(0, 3 if with_dead else 2)
        pos += dur
    return codes


def test_constants_match_official():
    assert tuple(SPELL_IOUS) == tuple(pm.SPELL_IOUS)


def test_spell_extraction_matches_official():
    rng = np.random.default_rng(1)
    for _ in range(50):
        c = _random_codes(rng, 2000)
        a, b, t = spells(c)
        official = pm.possession_spells(c, [(0, len(c))])
        assert list(zip(a.tolist(), b.tolist(), t.tolist())) == official


def test_spell_f1_equals_official_on_random_pairs():
    rng = np.random.default_rng(2)
    for i in range(300):
        n = 3000
        gt = _random_codes(rng, n)
        if i % 3 == 0:                      # near-duplicate prediction: boundaries jittered
            pred = gt.copy()
            for _ in range(10):
                p = int(rng.integers(0, n - 30))
                pred[p:p + int(rng.integers(1, 30))] = rng.integers(0, 3)
        else:
            pred = _random_codes(rng, n)
        ours = spell_f1(gt, pred)
        theirs = pm.spell_scores(gt, pred, [(0, n)])["mean_f1"]
        assert abs(ours - theirs) < 1e-12, (i, ours, theirs)


def test_edge_cases():
    n = 100
    allh = np.zeros(n, dtype=np.int8)
    alld = np.full(n, 2, dtype=np.int8)
    assert spell_f1(allh, allh) == 1.0
    assert spell_f1(allh, alld) == 0.0       # no predicted spells
    assert spell_f1(alld, allh) == 0.0       # no true spells
    assert spell_f1(alld, alld) == 0.0       # official convention: nothing to match -> 0
    assert pm.spell_scores(alld, alld, [(0, n)])["mean_f1"] == 0.0
    assert spell_f1(allh, np.ones(n, dtype=np.int8)) == 0.0   # wrong team never matches
