"""Inference for a frozen continued-verifier bundle; no label access."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch

from tiki_data.contracts import CLASS_ORDER, PredictionBatch, ProbabilityBatch, require
from tiki_data.p1.features import apply_feature_policy
from tiki_data.p2.inference import load_checkpoint, predict_batch
from tiki_data.p2.model import seed_all
from . import verifier_edits as verify
from . import verifier_segments as segment
from .verifier_decoder import decode


def _asset(folder, record):
    path = (folder / record['file']).resolve()
    require(path.parent == folder, 'model asset must be inside its bundle')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'],
            'verifier bundle asset hash mismatch')
    return path


def load_bundle(configuration):
    path = Path(configuration).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    require(config['schema_version'] == 1 and config['kind'] == 'correction_verifier',
            'unsupported verifier configuration')
    require(tuple(config['class_order']) == CLASS_ORDER, 'class order mismatch')
    require(config['environment']['sklearn'] == sklearn.__version__,
            'use the sklearn version recorded with the verifier bundle')
    for record in config['source_files']:
        source = Path(__file__).resolve().parents[2] / record['file']
        require(hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256'],
                'frozen verifier implementation changed')
    historical, saved = load_checkpoint(_asset(path.parent, config['historical']))
    members = []
    for record in config['proposers']:
        checkpoint = torch.load(_asset(path.parent, record), map_location='cpu', weights_only=True)
        model = segment.SegmentTCN(checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        members.append((model.eval(), checkpoint))
    require(len(members) == 2, 'expected the two frozen segment proposers')
    # This is our own saved forest, with its content hash verified before loading.
    forest = joblib.load(_asset(path.parent, config['verifier']))
    return config, (historical, saved), members, forest


def predict(base, ball_positions, configuration):
    """Return historical frame probabilities and verifier-corrected labels.

    Inputs are the shared 19 tracking features and aligned observed planar ball
    coordinates. No ground truth, public-window bounds or match-specific rules
    are accepted. The historical probabilities are not calibrated confidences
    for the corrected output labels.
    """
    seed_all(42, 1)
    config, original, proposers, forest = load_bundle(configuration)
    require(base.match.fps == config['fps'], 'frame-rate mismatch')
    positions = np.asarray(ball_positions)
    require(positions.shape == (base.match.n_frames, 2), 'expected aligned N x 2 ball positions')
    features = apply_feature_policy(base, True)
    visible = np.isfinite(positions).all(1)
    model, saved = original
    probabilities = predict_batch(features, model, saved['normalization'], saved['core_frames'])
    p = probabilities.probabilities
    baseline = decode(p, visible, base.match, config['historical_decoder'])
    proposal_base = decode(p, visible, base.match, config['proposal_baseline_decoder'])
    alternatives, evidence, boundaries = [], [p], []
    for model, checkpoint in proposers:
        cache = segment.infer(model, features, checkpoint['normalization'])
        labels, _ = segment.segment_decode(model, cache, proposal_base, base.match,
                                           **config['proposal_decoder'])
        alternatives.append(labels)
        evidence.append(torch.from_numpy(cache[1]).softmax(1).numpy())
        boundaries.append(cache[2])
    edits = verify.proposals(baseline, alternatives, base.match,
                             maximum_seconds=config['maximum_edit_seconds'])
    x = verify.feature_matrix(edits, baseline, alternatives, evidence, boundaries,
        features.values, features.observed, visible, base.match,
        config['historical_decoder']['missing_weight'])
    gains = forest.predict(x) if len(x) else np.empty(0)
    labels, _ = verify.apply_edits(baseline, edits, gains, config['acceptance_threshold'])
    return probabilities, PredictionBatch(base.match, base.frames.copy(), labels).validate(base.match)
