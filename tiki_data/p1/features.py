"""Provider-neutral, permutation-invariant P1 feature definitions.

Every calculation uses only player/ball tracking observations. Possession labels,
ball status and tactical tags are deliberately excluded from the feature path.
"""

import numpy as np

from tiki_data.contracts import require


FEATURES = (
    ("ball_speed_mps", "Observed scalar ball speed in metres per second",
     ["ball.speed"]),
    ("home_nearest_ball_m", "Minimum ball-to-observed-home-player distance in metres",
     ["ball.x", "ball.y", "home.positions"]),
    ("away_nearest_ball_m", "Minimum ball-to-observed-away-player distance in metres",
     ["ball.x", "ball.y", "away.positions"]),
    ("home_second_nearest_ball_m", "Second-smallest ball-to-home-player distance in metres",
     ["ball.x", "ball.y", "home.positions"]),
    ("away_second_nearest_ball_m", "Second-smallest ball-to-away-player distance in metres",
     ["ball.x", "ball.y", "away.positions"]),
    ("home_players_within_3m", "Observed home players within three metres of the ball",
     ["ball.x", "ball.y", "home.positions"]),
    ("away_players_within_3m", "Observed away players within three metres of the ball",
     ["ball.x", "ball.y", "away.positions"]),
    ("home_players_within_5m", "Observed home players within five metres of the ball",
     ["ball.x", "ball.y", "home.positions"]),
    ("away_players_within_5m", "Observed away players within five metres of the ball",
     ["ball.x", "ball.y", "away.positions"]),
    ("ball_to_home_centroid_m", "Distance from the ball to the observed home-team centroid in metres",
     ["ball.x", "ball.y", "home.positions"]),
    ("ball_to_away_centroid_m", "Distance from the ball to the observed away-team centroid in metres",
     ["ball.x", "ball.y", "away.positions"]),
    ("home_width_m", "Observed home-team lateral y range in metres; requires two players",
     ["home.positions"]),
    ("away_width_m", "Observed away-team lateral y range in metres; requires two players",
     ["away.positions"]),
    ("home_length_m", "Observed home-team longitudinal x range in metres; requires two players",
     ["home.positions"]),
    ("away_length_m", "Observed away-team longitudinal x range in metres; requires two players",
     ["away.positions"]),
    ("home_mean_speed_mps", "Mean scalar speed of observed home players in metres per second",
     ["home.speeds"]),
    ("away_mean_speed_mps", "Mean scalar speed of observed away players in metres per second",
     ["away.speeds"]),
    ("home_players_observed", "Number of observed home-player tracklets in the frame",
     ["home.positions"]),
    ("away_players_observed", "Number of observed away-player tracklets in the frame",
     ["away.positions"]),
)

FEATURE_NAMES = tuple(item[0] for item in FEATURES)
BALL_DEPENDENT = FEATURE_NAMES[:11]
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def feature_schema():
    """P2-reviewed schema payload; names exactly match FeatureBatch column order."""
    return {
        "features": [
            {"name": name, "definition": definition, "sources": list(sources)}
            for name, definition, sources in FEATURES
        ],
        "ball_dependent": list(BALL_DEPENDENT),
        "notes": (
            "V1 at 25 Hz. Coordinates are converted to pitch metres before feature calculation. "
            "Home/away are match-relative sides. Features are permutation-invariant over temporary "
            "track slots. Missing values use zero placeholders with observed=False. The same numerical "
            "definitions must be used for external and supplied matches. No labels, ball-status fields "
            "or tactical tags enter the feature path."
        ),
    }


def _positions(value):
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == 2 and result.shape[1] == 2, "positions must have shape N x 2")
    require(np.isfinite(result).all(), "observed positions must be finite")
    return result


def _speeds(value):
    result = np.asarray(value, dtype=np.float64)
    require(result.ndim == 1, "speeds must be one-dimensional")
    require(np.isfinite(result).all(), "observed speeds must be finite")
    return result


def compute_frame(home_positions, away_positions, home_speeds, away_speeds,
                  ball_position=None, ball_speed=None):
    """Return one `(values, observed)` feature row from metric observations.

    Player order has no meaning. Speeds may contain fewer observations than positions
    when a provider supplies a valid position but no valid scalar speed.
    """
    home = _positions(home_positions)
    away = _positions(away_positions)
    home_speed = _speeds(home_speeds)
    away_speed = _speeds(away_speeds)
    values = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    observed = np.zeros(len(FEATURE_NAMES), dtype=np.bool_)

    def put(name, value):
        require(np.isfinite(value), "feature values must be finite")
        values[INDEX[name]] = np.float32(value)
        observed[INDEX[name]] = True

    # Counts are actual observations even when no tracklet is currently visible.
    put("home_players_observed", len(home))
    put("away_players_observed", len(away))

    for side, positions, speeds in (("home", home, home_speed), ("away", away, away_speed)):
        if len(positions) >= 2:
            put(side + "_width_m", np.ptp(positions[:, 1]))
            put(side + "_length_m", np.ptp(positions[:, 0]))
        if len(speeds):
            put(side + "_mean_speed_mps", speeds.mean())

    ball = None if ball_position is None else np.asarray(ball_position, dtype=np.float64)
    if ball is not None:
        require(ball.shape == (2,) and np.isfinite(ball).all(),
                "observed ball position must contain two finite values")
        if ball_speed is not None and np.isfinite(ball_speed):
            put("ball_speed_mps", float(ball_speed))
        for side, positions in (("home", home), ("away", away)):
            if not len(positions):
                continue
            distances = np.sort(np.linalg.norm(positions - ball, axis=1))
            put(side + "_nearest_ball_m", distances[0])
            if len(distances) >= 2:
                put(side + "_second_nearest_ball_m", distances[1])
            put(side + "_players_within_3m", np.count_nonzero(distances <= 3.0))
            put(side + "_players_within_5m", np.count_nonzero(distances <= 5.0))
            put("ball_to_" + side + "_centroid_m",
                np.linalg.norm(positions.mean(axis=0) - ball))

    return values, observed
