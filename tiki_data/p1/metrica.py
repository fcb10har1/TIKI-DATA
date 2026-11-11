"""Streaming Metrica FIFA/EPTS ATD conversion into the shared FeatureBatch."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from tiki_data.contracts import FeatureBatch, MatchSpec, require

from .features import FEATURE_NAMES, apply_feature_policy, compute_frame


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def team_slots(path):
    """Return `(home_team_id, slot_team_ids)` in raw ATD group order."""
    root = ET.parse(path).getroot()
    teams = [element for element in root.iter() if _tag(element) == "Team"]
    players = [element for element in root.iter() if _tag(element) == "Player"]
    require(len(teams) == 2, "ATD metadata must contain exactly two teams")
    require(bool(players), "ATD metadata contains no player track slots")
    team_ids = [team.get("id") for team in teams]
    require(all(team_ids) and len(set(team_ids)) == 2, "ATD team IDs must be distinct and nonempty")
    slots = [player.get("teamId") for player in players]
    require(all(team in team_ids for team in slots), "ATD player slot has an unknown team")
    return team_ids[0], tuple(slots)


def parse_observation(text):
    """Parse `x,y,speed`; position may be present when scalar speed is missing."""
    text = text.strip()
    if not text or text[0].upper() == "N":
        return None
    fields = text.split(",")
    require(len(fields) >= 3, "ATD observation must contain x,y,speed")
    x, y, speed = (float(value) for value in fields[:3])
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y, speed if np.isfinite(speed) else None


def build_metrica_features(match, atd_meta, atd_raw, pitch, mask_ball_speed=False):
    """Read one complete ATD timeline and return a P2-compatible FeatureBatch."""
    require(isinstance(match, MatchSpec), "match must be a MatchSpec")
    require(match.fps == 25, "P1 V1 and P2 V1 require 25 Hz")
    length, width = (float(pitch[0]), float(pitch[1]))
    require(np.isfinite([length, width]).all() and length > 0 and width > 0,
            "pitch dimensions must be positive and finite")
    home_id, slots = team_slots(Path(atd_meta))
    values = np.zeros((match.n_frames, len(FEATURE_NAMES)), dtype=np.float32)
    observed = np.zeros_like(values, dtype=np.bool_)
    count = 0

    with Path(atd_raw).open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            require(count <= match.n_frames, "ATD contains more rows than the match timeline")
            parts = line.rstrip().split(":")
            require(len(parts) == 3, "ATD row must have frame, players and ball sections")
            frame_id = int(parts[0])
            require(frame_id == count, "ATD frame IDs must be contiguous and start at 1")
            groups = parts[1].split(";")
            require(len(groups) == len(slots), "ATD player-group count differs from metadata slots")

            home_positions, away_positions = [], []
            home_speeds, away_speeds = [], []
            for slot, group in zip(slots, groups):
                item = parse_observation(group)
                if item is None:
                    continue
                x, y, speed = item
                position = (x * length, y * width)
                if slot == home_id:
                    home_positions.append(position)
                    if speed is not None:
                        home_speeds.append(speed)
                else:
                    away_positions.append(position)
                    if speed is not None:
                        away_speeds.append(speed)

            ball_item = parse_observation(parts[2])
            if ball_item is None:
                ball_position = ball_speed = None
            else:
                ball_x, ball_y, ball_speed = ball_item
                ball_position = (ball_x * length, ball_y * width)

            row_values, row_observed = compute_frame(
                np.asarray(home_positions, dtype=np.float64).reshape(-1, 2),
                np.asarray(away_positions, dtype=np.float64).reshape(-1, 2),
                np.asarray(home_speeds, dtype=np.float64),
                np.asarray(away_speeds, dtype=np.float64),
                ball_position, ball_speed,
            )
            values[count - 1] = row_values
            observed[count - 1] = row_observed

    require(count == match.n_frames, "ATD row count disagrees with the match timeline")
    frames = np.arange(match.n_frames, dtype=np.int64)
    batch = FeatureBatch(match, frames, values, observed, FEATURE_NAMES, match.period_ids())
    return apply_feature_policy(batch, mask_ball_speed)
