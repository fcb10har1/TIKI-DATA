"""Where do dead-ball errors come from? Prints AGGREGATES ONLY. Run it locally.

    python p3/scripts/dead_ball_diagnosis.py --pred evaluation/results/<your_file>.csv

Inputs (all local, none are uploaded anywhere):
  --pred   a full-length prediction CSV (frame,label), the same file you would submit
  --gt     the public 5-minute labels (default: the one shipped in the repo)
  --atd    the tactical ATD raw tracking file (default: data/tactical)

What it does NOT do: it never tunes anything and it never prints per-frame or per-event rows.
It is a diagnosis for the team to discuss; do not use its numbers to choose settings on the
public window (RULES.md / EVALUATION.md).

Please share only the printed totals, and only if you are comfortable doing so.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

N_FRAMES, FPS = 199319, 25
LABELS = ("H", "A", "D", "X")


def runs(mask):
    """(start, stop) half-open runs of True in a boolean array."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2], edges[1::2]))


def run_summary(mask):
    lengths = np.array([b - a for a, b in runs(mask)], dtype=float) / FPS
    if lengths.size == 0:
        return "0 runs"
    return "{} runs; seconds min {:.1f} / median {:.1f} / max {:.1f}".format(
        lengths.size, lengths.min(), np.median(lengths), lengths.max())


def confusion(truth, pred):
    """3x4 counts: rows true H/A/D, columns predicted H/A/D/X."""
    table = np.zeros((3, 4), dtype=np.int64)
    for i, t in enumerate(LABELS[:3]):
        for j, p in enumerate(LABELS):
            table[i, j] = int(np.sum((truth == t) & (pred == p)))
    return table


def load_labels(path):
    frame = pd.read_csv(path)
    if len(frame) != N_FRAMES or not np.array_equal(frame["frame"].to_numpy(), np.arange(N_FRAMES)):
        raise SystemExit("{}: expected {} rows with frame 0..{}".format(path, N_FRAMES, N_FRAMES - 1))
    return frame["label"].to_numpy().astype("<U1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", default=str(ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"))
    parser.add_argument("--atd", default=str(ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"))
    parser.add_argument("--meta", default=str(ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"))
    args = parser.parse_args(argv)

    from p3.rules import read_atd
    from starter.baseline import team_of_each_track

    gt, pred = load_labels(args.gt), load_labels(args.pred)
    scored = gt != "X"
    window = np.flatnonzero(scored)
    print("scored frames: {} (video {}..{})".format(window.size, window[0], window[-1]))

    ball, _ = read_atd(args.atd, N_FRAMES, len(team_of_each_track(args.meta)))
    missing = np.isnan(ball[:, 0])

    table = confusion(gt, pred)
    print("\nconfusion (frames). rows = truth, columns = prediction")
    print("        pred H   pred A   pred D   pred X")
    for name, row in zip(LABELS[:3], table):
        print("true {}  {:7d}  {:7d}  {:7d}  {:7d}".format(name, *row))
    wrong = int(table.sum() - np.trace(table[:, :3]))
    dead_wrong = int(table[2].sum() - table[2, 2] + table[:2, 2].sum())
    print("\nframe accuracy {:.2%}".format(np.trace(table[:, :3]) / table.sum()))
    if wrong:
        print("wrong frames involving D (either side): {:.1%} of all wrong frames".format(
            dead_wrong / wrong))
        print("  true D predicted as H/A/X: {}   H/A predicted as D: {}".format(
            int(table[2].sum() - table[2, 2]), int(table[:2, 2].sum())))

    print("\nball missing: {:.1%} of scored frames; {:.1%} of TRUE-D frames; {:.1%} of PREDICTED-D frames".format(
        missing[scored].mean(),
        missing[scored & (gt == "D")].mean() if np.any(scored & (gt == "D")) else float("nan"),
        missing[scored & (pred == "D")].mean() if np.any(scored & (pred == "D")) else float("nan")))

    lo, hi = window[0], window[-1] + 1
    print("\nrun lengths inside the scored window")
    print("  true D       :", run_summary(gt[lo:hi] == "D"))
    print("  predicted D  :", run_summary(pred[lo:hi] == "D"))
    print("  ball missing :", run_summary(missing[lo:hi]))
    print("  note: the window holds only ~17 possessions, so every figure above is a small sample.")


if __name__ == "__main__":
    main()
