"""Observed ball coordinates from the supplied tactical tracking only."""
from pathlib import Path
import numpy as np
from tiki_data.contracts import require

def metrica_positions(match, raw_path, pitch):
    """Read only x/y; raw scalar speed and every target field are excluded."""
    result = np.full((match.n_frames, 2), np.nan, dtype=np.float64)
    count = 0
    with Path(raw_path).open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, 1):
            require(count <= match.n_frames, "extra ATD rows")
            frame, _, ball = line.rstrip().split(":")
            require(int(frame) == count, "ATD frame IDs are not aligned")
            x, y = map(float, ball.split(",")[:2])
            if np.isfinite([x, y]).all():
                result[count-1] = (x*pitch[0], y*pitch[1])
    require(count == match.n_frames, "missing ATD rows")
    return result
