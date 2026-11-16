"""Observation-level occlusion followed by the shared P1 feature calculation."""
import numpy as np
import torch
from torch.utils.data import Dataset
from tiki_data.contracts import FeatureBatch
from tiki_data.p1.features import FEATURE_NAMES, compute_frames, feature_schema
from tiki_data.p2.data import encode, timeline_blocks

AGE_NAME = 'ball_seconds_since_observed'
AGE_CAP = 10.0
RAW_KEYS = ('home_positions', 'away_positions', 'home_speeds', 'away_speeds', 'ball_positions', 'ball_speeds')


def age_seconds(visible, fps, initial=0.0):
    """Causal age; initial means elapsed time just before the first input frame.

    Before any observation, age counts from the start of the period. No future
    observation is used. Capping at 10 seconds is a fixed, unfitted definition.
    """
    t = np.arange(len(visible), dtype=np.float64)
    last = np.maximum.accumulate(np.where(visible, t, -1.0 - initial * fps))
    return np.minimum((t - last) / fps, AGE_CAP).astype(np.float32)


def schema_with_age():
    schema = feature_schema()
    schema['features'].append(dict(name=AGE_NAME,
        definition='Seconds since the last finite planar ball observation, causal, period-reset, capped at 10 seconds',
        sources=['ball.positions', 'frame_time']))
    schema['notes'] += ' Age is calculated identically for training and inference after raw occlusion.'
    return schema


def raw_slice(raw, start, stop):
    return {key: value[start:stop] for key, value in raw.items() if key in RAW_KEYS}


def features(raw, fps=25, missing=None, hide_speed=False, with_age=False, initial_age=0.0):
    """Copy/hide raw measurements BEFORE distances or any derived features exist.

    This V1 experiment has no velocity/Kalman channels. Future temporal features
    must consume these occluded positions, never the original measurements.
    """
    ball = np.array(raw['ball_positions'], copy=True)
    speed = np.array(raw['ball_speeds'], copy=True)
    if missing is not None:
        ball[missing] = np.nan
        speed[missing] = np.nan
    if hide_speed: speed[:] = np.nan
    values, observed = compute_frames(raw['home_positions'], raw['away_positions'],
        raw['home_speeds'], raw['away_speeds'], ball, speed)
    visible = np.isfinite(ball).all(axis=1)
    if with_age:
        age = age_seconds(visible, fps, initial_age)
        values = np.column_stack((values, age)).astype(np.float32)
        observed = np.column_stack((observed, np.ones(len(age), dtype=bool)))
    return values, observed, visible


def batch_from_raw(raw, spec, missing=None, hide_speed=False, with_age=False):
    names = FEATURE_NAMES + ((AGE_NAME,) if with_age else ())
    values = np.zeros((spec.n_frames, len(names)), dtype=np.float32)
    observed = np.zeros(values.shape, dtype=bool)
    visible = np.zeros(spec.n_frames, dtype=bool)
    for a, b in spec.periods:
        values[a:b], observed[a:b], visible[a:b] = features(raw_slice(raw,a,b), spec.fps,
            None if missing is None else missing[a:b], hide_speed, with_age)
    batch = FeatureBatch(spec, np.arange(spec.n_frames), values, observed, names, spec.period_ids())
    return batch.validate(spec), visible


def sample_occlusion(n, rng, config):
    missing = rng.random(n) < config['point_probability']
    if rng.random() < config['ball_stretch_probability']:
        lo, hi = config['ball_stretch_frames']
        length = min(n, int(rng.integers(lo, hi+1)))
        start = int(rng.integers(n-length+1))
        missing[start:start+length] = True
    hide_speed = rng.random() < config['speed_window_probability']
    return missing, hide_speed


class RawWindows(Dataset):
    """Whole-match disjoint supervised cores; same halos/boundaries as baseline."""
    def __init__(self, records, raw, normalization, core, radius, augmentation, seed=42):
        self.records, self.raw = records, raw
        self.normalization, self.core, self.radius = normalization, core, radius
        self.augmentation, self.seed, self.epoch = augmentation, seed, 0
        self.index = []
        self.clean_age = {}
        for i, (batch, labels) in enumerate(records):
            age = np.zeros(batch.match.n_frames, dtype=np.float32)
            for left, right in timeline_blocks(batch):
                visible = np.isfinite(raw[i]['ball_positions'][left:right]).all(axis=1)
                age[left:right] = age_seconds(visible, batch.match.fps)
                for start in range(left, right, core):
                    stop = min(start+core, right)
                    if np.any(labels[start:stop] >= 0):
                        self.index.append((i,left,right,start,stop))
            self.clean_age[i] = age

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        i,left,right,start,stop = self.index[index]
        batch, labels = self.records[i]
        a,b = max(left,start-self.radius), min(right,start+self.core+self.radius)
        # Dedicated RNG means augmentation does not change model/dropout RNG.
        rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
        missing, hide_speed = sample_occlusion(b-a,rng,self.augmentation)
        initial = float(self.clean_age[i][a-1]) if a > left else 0.0
        values, observed, _ = features(raw_slice(self.raw[i],a,b),batch.match.fps,
            missing,hide_speed,True,initial)
        x = np.zeros((2*values.shape[1],self.core+2*self.radius),dtype=np.float32)
        offset = a-(start-self.radius)
        x[:,offset:offset+b-a] = encode(values,observed,self.normalization)
        y = np.full(self.core,-100,dtype=np.int64)
        y[:stop-start] = labels[start:stop]
        return torch.from_numpy(x),torch.from_numpy(y)
