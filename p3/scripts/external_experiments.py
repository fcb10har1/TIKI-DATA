"""External-data-only experiments (IDSSE). Never touches BAR-ATH labels or the public window.

Run from the repo root. Work directory: runs/p3-exp (git-ignored), or set P3_EXP_DIR.
  python p3/scripts/external_experiments.py prep                       variants + training configs
  python -m tiki_data.p2 train --config runs/p3-exp/A_base.json --out runs/p3-exp/A_base   (also B_nospeed, C_nospeed_gaps)
  python p3/scripts/external_experiments.py eval A_base B_nospeed C_nospeed_gaps
  python p3/scripts/external_experiments.py combo                      decoder missing_weight / temperature grid
See docs/P3_EXPERIMENTS.md. The held-out test match (J03WR9) is never opened.
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
S = Path(os.environ.get("P3_EXP_DIR", ROOT / "runs" / "p3-exp"))
sys.path.insert(0, str(ROOT))

from tiki_data.contracts import MatchSpec
from tiki_data.artifacts import read_predictions
from tiki_data.p1.features import FEATURE_NAMES, BALL_DEPENDENT
from tiki_data.p2.inference import load_checkpoint, predict_batch
from tiki_data.artifacts import read_batch
from p3.decoder import DecoderConfig, decode_match
from p3.spells import spell_f1

EXT = ROOT / "data/external/idsse_processed"
MAN = json.loads((EXT / "manifest.json").read_text())
SPEED = FEATURE_NAMES.index("ball_speed_mps")
BALL_IDX = [FEATURE_NAMES.index(n) for n in BALL_DEPENDENT]
POS_IDX = FEATURE_NAMES.index("home_nearest_ball_m")


def entries(splits):
    return [e for e in MAN["matches"] if e["split"] in splits]


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def mask_speed(d):
    d = dict(d); v, o = d["values"].copy(), d["observed"].copy()
    v[:, SPEED] = 0.0; o[:, SPEED] = False
    d["values"], d["observed"] = v, o
    return d


def add_ball_gaps(d, seed, lo=25, hi=375, target=0.40):
    """Hide all ball-dependent channels in random contiguous stretches (BAR-ATH-like gaps)."""
    rng = np.random.default_rng(seed)
    d = dict(d); v, o = d["values"].copy(), d["observed"].copy()
    periods = d["periods"]
    for pid in np.unique(periods[periods > 0]):
        idx = np.flatnonzero(periods == pid)
        a, b = idx[0], idx[-1] + 1
        t = a
        while t < b:
            gap = int(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            keep = int(gap * (1 - target) / target * np.exp(rng.normal(0, 0.5)))
            t += keep
            e = min(t + gap, b)
            o[t:e, :][:, BALL_IDX] = False
            v[t:e, :][:, BALL_IDX] = 0.0
            t = e
    d["values"], d["observed"] = v, o
    return d


def drop_speed(d):
    d = dict(d)
    d["values"], d["observed"] = d["values"][:, 1:].copy(), d["observed"][:, 1:].copy()
    d["feature_names"] = d["feature_names"][1:]
    return d


def save(d, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)


def prep():
    base = json.loads((ROOT / "configs/p2_idsse_v1.json").read_text())
    abs_cfg = lambda feats, aug: dict(
        base, evaluation_manifest=str(EXT / "manifest.json"), feature_schema=str(EXT / "feature_schema.json"),
        features=feats, augmentation=aug)
    variants = {"clean": lambda d, i: d, "nospeed": lambda d, i: mask_speed(d),
                "barlike": lambda d, i: add_ball_gaps(mask_speed(d), seed=1000 + i)}
    for name, fn in variants.items():
        for i, e in enumerate(MAN["matches"]):
            mid = e["match"]["match_id"]
            p = S / "feat" / name / mid / "features.npz"
            if name == "clean" or e["split"] == "test":
                continue  # the held-out test match is never opened
            if not p.exists():
                save(fn(load_npz(EXT / "matches" / mid / "features.npz"), i), p)
    for i, e in enumerate(MAN["matches"]):
        mid = e["match"]["match_id"]
        if e["split"] == "test":
            continue
        raw = load_npz(EXT / "matches" / mid / "features.npz")
        save(drop_speed(raw), S / "feat/drop_clean" / mid / "features.npz")
        if e["split"] == "validation":
            save(drop_speed(add_ball_gaps(mask_speed(raw), seed=1000 + i)), S / "feat/drop_barlike" / mid / "features.npz")
    schema = json.loads((EXT / "feature_schema.json").read_text())
    schema["features"] = [f for f in schema["features"] if f["name"] != "ball_speed_mps"]
    schema["ball_dependent"] = [n for n in schema["ball_dependent"] if n != "ball_speed_mps"]
    (S / "schema18.json").write_text(json.dumps(schema, indent=1))
    clean = {e["match"]["match_id"]: str(EXT / "matches" / e["match"]["match_id"] / "features.npz") for e in MAN["matches"]}
    nosp = {e["match"]["match_id"]: (str(S / "feat/nospeed" / e["match"]["match_id"] / "features.npz")
            if e["split"] != "test" else clean[e["match"]["match_id"]]) for e in MAN["matches"]}  # test path declared, never read
    off = {"point_probability": 0.0, "ball_stretch_probability": 0.0, "ball_stretch_frames": [25, 250]}
    gaps = {"point_probability": 0.0, "ball_stretch_probability": 0.75, "ball_stretch_frames": [25, 375]}
    drop = {e["match"]["match_id"]: (str(S / "feat/drop_clean" / e["match"]["match_id"] / "features.npz")
            if e["split"] != "test" else clean[e["match"]["match_id"]]) for e in MAN["matches"]}
    cfgs = {"A_base": abs_cfg(clean, off), "B_nospeed": abs_cfg(drop, off), "C_nospeed_gaps": abs_cfg(drop, gaps)}
    for c in ("B_nospeed", "C_nospeed_gaps"):
        cfgs[c]["feature_schema"] = str(S / "schema18.json")
    for name, cfg in cfgs.items():
        (S / (name + ".json")).write_text(json.dumps(cfg, indent=1))
    # realised missing fractions of the BAR-like gaps (checks the simulation is in the right range)
    for e in entries({"validation"}):
        mid = e["match"]["match_id"]
        d = np.load(S / "feat/barlike" / mid / "features.npz")
        inside = d["periods"] > 0
        print(mid, "ball-gap fraction inside periods: %.3f" % (~d["observed"][inside][:, POS_IDX]).mean())


def to_codes(labels):
    out = np.full(len(labels), -1, dtype=np.int64)
    for i, c in enumerate("HAD"):
        out[labels == c] = i
    return out


def macro_acc(gt, pr):
    m = gt >= 0
    g, p = gt[m], pr[m]
    f1 = []
    for c in range(3):
        tp = np.sum((g == c) & (p == c)); fp = np.sum((g != c) & (p == c)); fn = np.sum((g == c) & (p != c))
        f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    drec = np.sum((g == 2) & (p == 2)) / max(1, np.sum(g == 2))
    return float(np.mean(f1)), float(np.mean(g == p)), float(drec)


def score(gt_letters, pred_letters, periods_blocks):
    gt, pr = to_codes(gt_letters), to_codes(pred_letters)
    parts_g, parts_p = [], []
    for a, b in periods_blocks:
        parts_g += [gt[a:b], [2]]; parts_p += [pr[a:b], [2]]   # D separator: no spell crosses halftime
    sg = spell_f1(np.concatenate(parts_g), np.concatenate(parts_p))
    mf, acc, drec = macro_acc(gt, pr)
    n_spells = lambda x: int(((np.diff(np.concatenate(([2], np.concatenate(parts_p)))) != 0) & (np.concatenate(parts_p) < 2)).sum())
    return dict(spellF1=sg, macroF1=mf, acc=acc, Drecall=drec, n_pred_spells=n_spells(0))


def decode(P, d, blocks, **kw):
    vis = d["observed"][:, list(d["feature_names"]).index("home_nearest_ball_m")]
    return decode_match(P, vis, blocks, DecoderConfig(**kw))


def evaluate(ckpt, condition, splits=("validation",), decoder_kw=None, probs_cache=None):
    decoder_kw = decoder_kw or dict(temperature=4.0)
    model, ck = load_checkpoint(ckpt)
    rows = []
    for e in entries(set(splits)):
        mid = e["match"]["match_id"]
        spec = MatchSpec.from_dict(e["match"])
        if Path(ckpt).parent.name.startswith("A"):
            path = {"clean": EXT / "matches" / mid / "features.npz",
                    "nospeed": S / "feat/nospeed" / mid / "features.npz",
                    "barlike": S / "feat/barlike" / mid / "features.npz"}[condition]
        else:
            path = {"clean": S / "feat/drop_clean" / mid / "features.npz",
                    "nospeed": S / "feat/drop_clean" / mid / "features.npz",
                    "barlike": S / "feat/drop_barlike" / mid / "features.npz"}[condition]
        fb = read_batch(str(path), spec, "features")
        P = predict_batch(fb, model, ck["normalization"]).probabilities
        d = load_npz(path)
        gt = read_predictions(str(EXT / e["ground_truth"]), spec).labels
        blocks = [tuple(b) for b in e["match"]["periods"]]
        rows.append((mid, P, d, gt, blocks))
        if probs_cache is not None:
            probs_cache[(str(ckpt), condition, mid)] = (P, d, gt, blocks)
    return rows


def summarize(rows, **decoder_kw):
    res = []
    for mid, P, d, gt, blocks in rows:
        pred = decode(P, d, blocks, **decoder_kw)
        res.append(score(gt, pred, blocks))
    return {k: float(np.mean([r[k] for r in res])) for k in res[0]}


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    elif sys.argv[1] == "combo":
        for model in ("A_base", "B_nospeed", "C_nospeed_gaps"):
            rows = evaluate(S / model / "best.pt", "barlike")
            print("\n%s / barlike (BAR-like ball gaps, speed absent)" % model)
            for label, kw in [("T4 mw0 (team now)", dict(temperature=4.0, missing_weight=0.0)),
                              ("T4 mw0.5", dict(temperature=4.0, missing_weight=0.5)),
                              ("T2 mw0.5", dict(temperature=2.0, missing_weight=0.5)),
                              ("T1 mw0.5", dict(temperature=1.0, missing_weight=0.5)),
                              ("T4 mw1", dict(temperature=4.0, missing_weight=1.0))]:
                r = summarize(rows, **kw)
                print("  %-18s spellF1 %.3f  macroF1 %.3f  acc %.3f  Drecall %.3f  spells %3.0f" % (
                    label, r["spellF1"], r["macroF1"], r["acc"], r["Drecall"], r["n_pred_spells"]))
    elif sys.argv[1] == "eval":
        print("model, condition: mean over validation matches (J03WPY, J03WQQ); decoder T=4 (team setting)")
        for name in sys.argv[2:]:
            for cond in ("clean", "nospeed", "barlike"):
                rows = evaluate(S / name / "best.pt", cond)
                r = summarize(rows, temperature=4.0)
                print("%-16s %-8s spellF1 %.3f  macroF1 %.3f  acc %.3f  Drecall %.3f  predSpells %.0f" % (
                    name, cond, r["spellF1"], r["macroF1"], r["acc"], r["Drecall"], r["n_pred_spells"]))
