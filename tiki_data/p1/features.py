"""Provider-neutral, permutation-invariant P1 feature definitions.

Every calculation uses only player/ball tracking observations. Possession labels,
ball status and tactical tags are deliberately excluded from the feature path.
"""

import numpy as np

from tiki_data.contracts import FeatureBatch, require


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


def apply_feature_policy(batch, mask_ball_speed=False):
    """Apply the same scalar-speed policy to either provider or cached P1 batches.

    Preserve the timeline and all geometry/player channels. Never mutate a cached
    batch: A and B can safely share the original artifact. False is baseline A.
    """
    require(isinstance(mask_ball_speed, bool), "mask_ball_speed must be boolean")
    batch.validate(batch.match)
    if not mask_ball_speed:
        return batch
    require("ball_speed_mps" in batch.feature_names, "missing ball_speed_mps feature")
    column = batch.feature_names.index("ball_speed_mps")
    values, observed = batch.values.copy(), batch.observed.copy()
    values[:, column] = 0
    observed[:, column] = False
    return FeatureBatch(batch.match, batch.frames.copy(), values, observed,
                        batch.feature_names, batch.periods.copy()).validate(batch.match)


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


def compute_frames(home_positions, away_positions, home_speeds, away_speeds,
                   ball_positions, ball_speeds, chunk_size=8192):
    """Vectorized equivalent of :func:`compute_frame` for complete matches.

    Player slots remain anonymous and may be absent (NaN) on any frame. Processing
    in chunks bounds temporary memory for full 25 Hz matches while preserving the
    exact provider-neutral feature definitions used by the supplied-match adapter.
    """
    home_positions = np.asarray(home_positions)
    away_positions = np.asarray(away_positions)
    home_speeds = np.asarray(home_speeds)
    away_speeds = np.asarray(away_speeds)
    ball_positions = np.asarray(ball_positions)
    ball_speeds = np.asarray(ball_speeds)
    require(home_positions.ndim == away_positions.ndim == 3
            and home_positions.shape[2] == away_positions.shape[2] == 2,
            "batched positions must have shape frames x players x 2")
    n_frames = home_positions.shape[0]
    require(away_positions.shape[0] == n_frames
            and home_speeds.shape == home_positions.shape[:2]
            and away_speeds.shape == away_positions.shape[:2]
            and ball_positions.shape == (n_frames, 2)
            and ball_speeds.shape == (n_frames,), "batched observations are not frame-aligned")
    require(isinstance(chunk_size, int) and chunk_size > 0, "chunk_size must be positive")

    values = np.zeros((n_frames, len(FEATURE_NAMES)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=np.bool_)

    for start in range(0, n_frames, chunk_size):
        stop = min(start + chunk_size, n_frames)
        ball = ball_positions[start:stop].astype(np.float64, copy=False)
        ball_speed = ball_speeds[start:stop].astype(np.float64, copy=False)
        ball_valid = np.isfinite(ball).all(axis=1)
        speed_valid = ball_valid & np.isfinite(ball_speed)
        values[start:stop, INDEX["ball_speed_mps"]][speed_valid] = ball_speed[speed_valid]
        observed[start:stop, INDEX["ball_speed_mps"]][speed_valid] = True

        for side, positions_source, speeds_source in (
                ("home", home_positions, home_speeds),
                ("away", away_positions, away_speeds)):
            positions = positions_source[start:stop].astype(np.float64, copy=False)
            speeds = speeds_source[start:stop].astype(np.float64, copy=False)
            position_valid = np.isfinite(positions).all(axis=2)
            counts = position_valid.sum(axis=1)
            count_name = side + "_players_observed"
            values[start:stop, INDEX[count_name]] = counts
            observed[start:stop, INDEX[count_name]] = True

            enough = counts >= 2
            for coordinate, suffix in ((1, "_width_m"), (0, "_length_m")):
                coordinate_values = positions[:, :, coordinate]
                low = np.min(np.where(position_valid, coordinate_values, np.inf), axis=1)
                high = np.max(np.where(position_valid, coordinate_values, -np.inf), axis=1)
                name = side + suffix
                values[start:stop, INDEX[name]][enough] = high[enough] - low[enough]
                observed[start:stop, INDEX[name]][enough] = True

            scalar_valid = np.isfinite(speeds) & position_valid
            scalar_counts = scalar_valid.sum(axis=1)
            has_speed = scalar_counts > 0
            speed_mean = np.divide(np.where(scalar_valid, speeds, 0).sum(axis=1),
                                   scalar_counts, out=np.zeros(len(counts)), where=has_speed)
            speed_name = side + "_mean_speed_mps"
            values[start:stop, INDEX[speed_name]][has_speed] = speed_mean[has_speed]
            observed[start:stop, INDEX[speed_name]][has_speed] = True

            has_position = ball_valid & (counts > 0)
            differences = positions - ball[:, None, :]
            distances = np.sqrt(np.sum(differences * differences, axis=2))
            distances[~position_valid | ~ball_valid[:, None]] = np.inf
            nearest = np.min(distances, axis=1)
            nearest_name = side + "_nearest_ball_m"
            values[start:stop, INDEX[nearest_name]][has_position] = nearest[has_position]
            observed[start:stop, INDEX[nearest_name]][has_position] = True

            has_second = ball_valid & (counts >= 2)
            second = np.partition(distances, 1, axis=1)[:, 1]
            second_name = side + "_second_nearest_ball_m"
            values[start:stop, INDEX[second_name]][has_second] = second[has_second]
            observed[start:stop, INDEX[second_name]][has_second] = True

            for radius in (3, 5):
                radius_name = "{}_players_within_{}m".format(side, radius)
                inside = np.count_nonzero(distances <= radius, axis=1)
                values[start:stop, INDEX[radius_name]][has_position] = inside[has_position]
                observed[start:stop, INDEX[radius_name]][has_position] = True

            sums = np.where(position_valid[:, :, None], positions, 0).sum(axis=1)
            centroids = np.divide(sums, counts[:, None], out=np.zeros_like(sums),
                                  where=counts[:, None] > 0)
            centroid_distance = np.linalg.norm(centroids - ball, axis=1)
            centroid_name = "ball_to_" + side + "_centroid_m"
            values[start:stop, INDEX[centroid_name]][has_position] = centroid_distance[has_position]
            observed[start:stop, INDEX[centroid_name]][has_position] = True

    require(np.isfinite(values).all(), "computed feature matrix contains nonfinite values")
    require((values[~observed] == 0).all(), "unobserved features must retain zero placeholders")
    return values, observed
