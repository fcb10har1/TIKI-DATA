"""Write ball_observed.npz for a supplied match: True where the ORIGINAL tracking has a finite ball x,y.

    python p3/scripts/make_ball_observed.py [--out data/tactical/BAR-ATH_ball_observed.npz]

Reads data/tactical/BAR-ATH_tactical_atd.{txt,xml} locally. Prints one aggregate (the share of frames
with a ball). The output is derived from supplied tracking: keep it local, never commit it (data/ is ignored).
"""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from p3.rules import read_atd
from starter.baseline import team_of_each_track

N_FRAMES = 199319


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--atd", default=str(ROOT / "data/tactical/BAR-ATH_tactical_atd.txt"))
    p.add_argument("--meta", default=str(ROOT / "data/tactical/BAR-ATH_tactical_atd.xml"))
    p.add_argument("--out", default=str(ROOT / "data/tactical/BAR-ATH_ball_observed.npz"))
    a = p.parse_args(argv)
    ball, _ = read_atd(a.atd, N_FRAMES, len(team_of_each_track(a.meta)))
    observed = np.isfinite(ball).all(axis=1)
    out = Path(a.out)
    if out.exists():
        raise SystemExit("refusing to overwrite " + str(out))
    np.savez(out, ball_observed=observed)
    print("wrote {}; ball observed on {:.1%} of all frames".format(out, observed.mean()))


if __name__ == "__main__":
    main()
