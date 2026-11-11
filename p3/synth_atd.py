"""Synthetic tracking in the ATD text format, for tests only (not football, not match data)."""
import numpy as np

from .synth import make_visibility

HOME, AWAY = 0, 1


def make_world(truth, rng, fps=25, n_per_team=6, length=105.3, width=68.0):
    """Normalised ball (N,2) and players (N,S,2) consistent with truth codes 0=H,1=A,2=D.

    Alive play: the ball sits at the possessing team's carrier, who changes every few seconds.
    Dead ball: the ball is stationary outside the pitch and everyone slows down.
    """
    n = len(truth)
    S = 2 * n_per_team
    team = np.array([HOME] * n_per_team + [AWAY] * n_per_team)
    pos = rng.uniform([0.1, 0.1], [0.9, 0.9], size=(S, 2))
    players = np.empty((n, S, 2))
    speed = np.where(truth[:, None] == 2, 0.0002, 0.0015)
    vel = np.zeros((S, 2))
    for t in range(n):
        if t % 50 == 0:
            vel = rng.normal(0, 1, size=(S, 2))
            vel /= np.maximum(np.linalg.norm(vel, axis=1, keepdims=True), 1e-9)
        pos = np.clip(pos + vel * speed[t, 0] * (1 + 0.1 * rng.normal(size=(S, 1))), 0.02, 0.98)
        players[t] = pos
    ball = np.empty((n, 2))
    carrier = 0
    for t in range(n):
        c = truth[t]
        if c == 2:
            ball[t] = (1.01, 0.5)
            continue
        if t == 0 or truth[t - 1] != c or t % 75 == 0:
            members = np.flatnonzero(team == (HOME if c == 0 else AWAY))
            carrier = int(rng.choice(members))
        ball[t] = players[t, carrier] + rng.normal(0, 0.0005, size=2)
    return ball, players, team


def write_atd(raw_path, meta_path, ball, players, team, observed_ball=None, drop_player=0.02, rng=None):
    rng = rng or np.random.default_rng(0)
    n, S, _ = players.shape
    with open(raw_path, "w", encoding="utf-8") as f:
        for t in range(n):
            groups = []
            for s in range(S):
                if rng.random() < drop_player:
                    groups.append("NaN,NaN,NaN")
                else:
                    groups.append("{:.5f},{:.5f},0.0".format(*players[t, s]))
            if observed_ball is not None and not observed_ball[t]:
                b = "NaN,NaN,NaN"
            else:
                b = "{:.5f},{:.5f},0.0".format(*ball[t])
            f.write("{}:{}:{}\n".format(t + 1, ";".join(groups), b))
    teams = ['<Team id="T0"><Name>Home</Name></Team>', '<Team id="T1"><Name>Away</Name></Team>']
    plist = "".join('<Player id="P{}" teamId="T{}"/>'.format(i, int(tm)) for i, tm in enumerate(team))
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("<Root>{}{}</Root>".format("".join(teams), plist))
