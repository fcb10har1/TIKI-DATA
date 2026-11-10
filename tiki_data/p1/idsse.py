"""Streaming IDSSE XML conversion into P1 features and P2 training artifacts."""

from collections import Counter
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from tiki_data.artifacts import (read_batch, read_json, read_predictions, write_batch,
                                 write_json, write_predictions)
from tiki_data.benchmark import load_manifest
from tiki_data.contracts import FeatureBatch, MatchSpec, PredictionBatch, require
from tiki_data.p2.data import idsse_targets

from .features import FEATURE_NAMES, compute_frames, feature_schema


PERIODS = ("firstHalf", "secondHalf")
SOURCE_URL = "https://doi.org/10.6084/m9.figshare.28196177"
PUBLICATION_URL = "https://doi.org/10.1038/s41597-025-04505-y"
FILE_ID = re.compile(r"DFL-MAT-([A-Za-z0-9]+)\.xml$")


def _tag(element):
    return element.tag.rsplit("}", 1)[-1]


def _match_id(path):
    match = FILE_ID.search(Path(path).name)
    require(match is not None, "IDSSE filename has no DFL match ID: " + Path(path).name)
    return match.group(1)


def discover(raw_dir):
    """Return complete match-information/event/position triplets."""
    raw_dir = Path(raw_dir)
    require(raw_dir.is_dir(), "IDSSE raw directory does not exist: " + str(raw_dir))
    kinds = {
        "matchinformation": "match_information",
        "events_raw": "events",
        "positions_raw": "positions",
    }
    records = {}
    for path in sorted(raw_dir.glob("*.xml")):
        matching = [value for token, value in kinds.items() if token in path.name]
        require(len(matching) == 1, "unrecognized or ambiguous IDSSE XML filename: " + path.name)
        match_id = _match_id(path)
        record = records.setdefault(match_id, {})
        require(matching[0] not in record, "duplicate IDSSE file kind for " + match_id)
        record[matching[0]] = path.resolve()
    require(bool(records), "IDSSE raw directory contains no XML matches")
    for match_id, record in records.items():
        require(set(record) == set(kinds.values()), "IDSSE match {} does not have all three XML files".format(match_id))
    return [(match_id, records[match_id]) for match_id in sorted(records)]


def read_match_information(path):
    """Read the small IDSSE match-information XML and home/away roster IDs."""
    root = ET.parse(path).getroot()
    general = next((element for element in root.iter() if _tag(element) == "General"), None)
    environment = next((element for element in root.iter() if _tag(element) == "Environment"), None)
    require(general is not None and environment is not None, "IDSSE match information is incomplete")
    match_id = general.get("MatchId", "").removeprefix("DFL-MAT-")
    home_id, away_id = general.get("HomeTeamId"), general.get("GuestTeamId")
    require(match_id and home_id and away_id and home_id != away_id, "invalid IDSSE match/team IDs")
    teams = {}
    for element in root.iter():
        if _tag(element) != "Team":
            continue
        team_id = element.get("TeamId")
        if team_id not in (home_id, away_id):
            continue
        players = tuple(player.get("PersonId") for player in element.iter()
                        if _tag(player) == "Player" and player.get("PersonId"))
        require(players and len(players) == len(set(players)), "duplicate or missing IDSSE player IDs")
        teams[team_id] = {"name": element.get("TeamName", team_id), "players": players}
    require(set(teams) == {home_id, away_id}, "IDSSE roster does not contain both teams")
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    require(np.isfinite(pitch).all() and min(pitch) > 0, "invalid IDSSE pitch dimensions")
    return {
        "match_id": match_id,
        "home_id": home_id,
        "away_id": away_id,
        "home_name": general.get("HomeTeamName", teams[home_id]["name"]),
        "away_name": general.get("GuestTeamName", teams[away_id]["name"]),
        "home_players": teams[home_id]["players"],
        "away_players": teams[away_id]["players"],
        "pitch": pitch,
    }


def _float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def read_ball_timeline(path):
    """First streaming pass: establish the official half timelines and targets."""
    records = {period: [] for period in PERIODS}
    active_period = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = _tag(element)
        if event == "start" and tag == "FrameSet":
            period = element.get("GameSection")
            active_period = period if (period in records and element.get("TeamId", "").upper() == "BALL") else None
        elif event == "end" and tag == "Frame" and active_period is not None:
            records[active_period].append((
                int(element.get("N")),
                _float(element.get("X")),
                _float(element.get("Y")),
                _float(element.get("S")) / 3.6,
                int(element.get("BallPossession", "-1")),
                int(element.get("BallStatus", "-1")),
            ))
        elif event == "end" and tag == "FrameSet":
            active_period = None
        if event == "end":
            element.clear()

    arrays, boundaries, raw_periods = [], [], []
    offset = 0
    for period in PERIODS:
        require(bool(records[period]), "IDSSE position XML has no {} ball timeline".format(period))
        array = np.asarray(records[period], dtype=np.float64)
        raw_frames = array[:, 0].astype(np.int64)
        require(np.all(np.diff(raw_frames) == 1), "IDSSE {} ball frames are not contiguous".format(period))
        require(len(np.unique(raw_frames)) == len(raw_frames), "duplicate IDSSE ball frames")
        arrays.append(array)
        boundaries.append((offset, offset + len(array)))
        raw_periods.append({"name": period, "start": int(raw_frames[0]),
                            "stop_inclusive": int(raw_frames[-1]), "n_frames": len(raw_frames)})
        offset += len(array)
    combined = np.concatenate(arrays)
    return {
        "raw_frames": combined[:, 0].astype(np.int64),
        "positions": combined[:, 1:3].astype(np.float32),
        "speeds": combined[:, 3].astype(np.float32),
        "possession": combined[:, 4].astype(np.int8),
        "status": combined[:, 5].astype(np.int8),
        "periods": tuple(boundaries),
        "raw_periods": raw_periods,
    }


def read_player_tracks(path, information, timeline):
    """Second streaming pass: align anonymous home/away observations to ball frames."""
    n_frames = len(timeline["raw_frames"])
    rosters = (information["home_players"], information["away_players"])
    positions = [np.full((n_frames, len(roster), 2), np.nan, dtype=np.float32) for roster in rosters]
    speeds = [np.full((n_frames, len(roster)), np.nan, dtype=np.float32) for roster in rosters]
    slots = [{person: i for i, person in enumerate(roster)} for roster in rosters]
    team_side = {information["home_id"]: 0, information["away_id"]: 1}
    indices = {}
    for period, (start, stop) in zip(PERIODS, timeline["periods"]):
        raw = timeline["raw_frames"][start:stop]
        indices[period] = {int(frame): start + i for i, frame in enumerate(raw)}

    active = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = _tag(element)
        if event == "start" and tag == "FrameSet":
            period, team_id, person = (element.get("GameSection"), element.get("TeamId"),
                                       element.get("PersonId"))
            if period in indices and team_id in team_side:
                side = team_side[team_id]
                require(person in slots[side], "tracked IDSSE player is absent from match roster: " + str(person))
                active = (period, side, slots[side][person])
            else:
                active = None
        elif event == "end" and tag == "Frame" and active is not None:
            period, side, slot = active
            index = indices[period].get(int(element.get("N")))
            if index is not None:
                x, y = _float(element.get("X")), _float(element.get("Y"))
                if np.isfinite(x) and np.isfinite(y):
                    require(not np.isfinite(positions[side][index, slot]).any(),
                            "duplicate IDSSE player observation")
                    positions[side][index, slot] = (x, y)
                    speed = _float(element.get("S"))
                    if np.isfinite(speed):
                        speeds[side][index, slot] = speed / 3.6
        elif event == "end" and tag == "FrameSet":
            active = None
        if event == "end":
            element.clear()
    return positions[0], positions[1], speeds[0], speeds[1]


def convert_match(match_id, files, out):
    """Convert one complete IDSSE match into the shared P1/P2 artifacts."""
    information = read_match_information(files["match_information"])
    require(information["match_id"] == match_id, "IDSSE filename and match-information IDs disagree")
    timeline = read_ball_timeline(files["positions"])
    match = MatchSpec(match_id, len(timeline["raw_frames"]), 25, timeline["periods"])
    home_positions, away_positions, home_speeds, away_speeds = read_player_tracks(
        files["positions"], information, timeline)
    values, observed = compute_frames(
        home_positions, away_positions, home_speeds, away_speeds,
        timeline["positions"], timeline["speeds"])
    batch = FeatureBatch(match, np.arange(match.n_frames, dtype=np.int64), values, observed,
                         FEATURE_NAMES, match.period_ids()).validate(match)
    labels = idsse_targets(timeline["possession"], timeline["status"])
    target = PredictionBatch(match, np.arange(match.n_frames, dtype=np.int64), labels).validate(match)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_batch(out / "features.npz", batch)
    write_predictions(out / "labels.csv", target)
    write_json(out / "match.json", match.as_dict(), exclusive=True)
    label_counts = Counter(labels.tolist())
    conversion = {
        "schema_version": 1,
        "match_id": match_id,
        "source": SOURCE_URL,
        "source_files": {kind: path.name for kind, path in files.items()},
        "teams": {"home": {"id": information["home_id"], "name": information["home_name"]},
                  "away": {"id": information["away_id"], "name": information["away_name"]}},
        "pitch_metres": list(information["pitch"]),
        "period_source_frames": timeline["raw_periods"],
        "coordinate_conversion": "none; IDSSE X/Y are already pitch-centred metres",
        "speed_conversion": "IDSSE S divided by 3.6 (km/h to m/s)",
        "target_mapping": {"BallStatus=0": "D", "BallStatus=1 and BallPossession=1": "H",
                           "BallStatus=1 and BallPossession=2": "A", "otherwise": "X"},
        "label_counts": {label: int(label_counts.get(label, 0)) for label in ("H", "A", "D", "X")},
        "feature_observed_fraction": {
            name: float(observed[:, index].mean()) for index, name in enumerate(FEATURE_NAMES)},
    }
    write_json(out / "conversion.json", conversion, exclusive=True)
    return match


def _readme():
    return """# IDSSE V1 processed handoff

Generated from the seven-match IDSSE dataset for P2's TCN. Raw XML is not included.

Each match folder contains `features.npz` (the 19 P1 inputs), `labels.csv`
(`frame,label`), `match.json`, and `conversion.json`. Labels come only from the
IDSSE `BallPossession` and `BallStatus` fields; those fields never enter features.

From the repository root, P2 should run:

```powershell
git lfs install
git lfs pull
pip install -r requirements-p2.txt
python -m pytest tests/test_p1.py tests/test_p2.py
python -m tiki_data.p2 train --config configs/p2_idsse_v1.json --out artifacts/p2-idsse-v1
```

The split is fixed by whole match before model fitting: four train matches, two
validation matches, and one untouched test match. Do not use the test labels for
model, feature, decoder, or hyperparameter selection.

## Attribution and licence

IDSSE is licensed CC BY 4.0 with authorization from Deutsche Fussball Liga (DFL).
Credit DFL and cite Bassek, Rein, Weber and Memmert (2025), *Scientific Data*
12:195, DOI 10.1038/s41597-025-04505-y. Dataset DOI:
10.6084/m9.figshare.28196177. These artifacts are transformations of that dataset.
"""


def build_dataset(raw_dir, out, template_config, config_out, resume=False):
    """Convert all seven matches and emit a fixed whole-match P2 package."""
    matches = discover(raw_dir)
    require(len(matches) == 7, "IDSSE V1 expects exactly seven complete matches")
    ids = [match_id for match_id, _ in matches]
    split = {match_id: ("train" if index < 4 else "validation" if index < 6 else "test")
             for index, match_id in enumerate(ids)}
    out = Path(out).resolve()
    require(not out.exists() or resume, "processed IDSSE output already exists: " + str(out))
    out.mkdir(parents=True, exist_ok=resume)
    (out / "matches").mkdir(exist_ok=resume)
    require(not (out / "manifest.json").exists(), "processed IDSSE manifest already exists")
    require(not Path(config_out).exists(), "P2 IDSSE config already exists")
    converted = {}
    for match_id, files in matches:
        match_out = out / "matches" / match_id
        if resume and match_out.exists():
            match = MatchSpec.from_dict(read_json(match_out / "match.json"))
            read_batch(match_out / "features.npz", match, "features")
            read_predictions(match_out / "labels.csv", match)
            require((match_out / "conversion.json").is_file(), "resumed match lacks conversion metadata")
            converted[match_id] = match
            print("Reusing validated IDSSE match {} ({})".format(match_id, split[match_id]), flush=True)
        else:
            print("Converting IDSSE match {} ({})".format(match_id, split[match_id]), flush=True)
            converted[match_id] = convert_match(match_id, files, match_out)

    write_json(out / "feature_schema.json", feature_schema(), exclusive=True)
    manifest = {
        "schema_version": 1,
        "dataset_id": "idsse_v1",
        "kind": "external",
        "matches": [{
            "match": converted[match_id].as_dict(),
            "group_id": "DFL-MAT-" + match_id,
            "split": split[match_id],
            "provenance": {
                "dataset": "IDSSE (Bassek et al. 2025), CC BY 4.0, DFL-authorized",
                "source": SOURCE_URL,
                "label_provenance": "DFL BallPossession/BallStatus mapped by tiki_data.p2.data.idsse_targets",
            },
            "ground_truth": "matches/{}/labels.csv".format(match_id),
        } for match_id in ids],
    }
    write_json(out / "manifest.json", manifest, exclusive=True)
    (out / "README.md").write_text(_readme(), encoding="utf-8", newline="\n")
    load_manifest(out / "manifest.json")

    template = read_json(template_config)
    template["evaluation_manifest"] = "../data/external/idsse_processed/manifest.json"
    template["feature_schema"] = "../data/external/idsse_processed/feature_schema.json"
    template["features"] = {
        match_id: "../data/external/idsse_processed/matches/{}/features.npz".format(match_id)
        for match_id in ids
    }
    write_json(config_out, template, exclusive=True)
    return {"matches": len(ids), "splits": dict(Counter(split.values())),
            "output": str(out), "config": str(Path(config_out).resolve())}
