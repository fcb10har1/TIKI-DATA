"""P4-compatible P1 adapter."""

from games import GAME, GAMES
from tiki_data.contracts import MatchSpec, identifier, require

from .metrica import build_metrica_features


def build(match, inputs, settings):
    """Build Metrica ATD features without reading labels, tags or video."""
    require(isinstance(match, MatchSpec), "P1 build expects a MatchSpec")
    require(set(inputs) == {"atd_meta", "atd_raw"},
            "P1 inputs must be exactly atd_meta and atd_raw")
    require(set(settings) <= {"seed", "pitch", "mask_ball_speed"}, "unknown P1 settings")
    pitch = settings.get("pitch")
    if pitch is None:
        require(identifier(match.match_id) == identifier(GAME),
                "external Metrica inputs require settings.pitch")
        pitch = GAMES[GAME]["pitch"]
    require(isinstance(pitch, (list, tuple)) and len(pitch) == 2
            and all(isinstance(x, (int, float)) and x > 0 for x in pitch),
            "pitch must be [length_m, width_m]")
    return build_metrica_features(match, inputs["atd_meta"], inputs["atd_raw"], pitch,
                                  mask_ball_speed=settings.get("mask_ball_speed", False))
