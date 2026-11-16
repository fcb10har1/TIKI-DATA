# Saved possession model — 16 November 2026

Current selection: `ensemble_new_speed_kalman__combo_top_3`. Selection prioritises
external validation **spellF1**, with macroF1 secondary. The two included CPU TCN
checkpoints (seeds 42 and 43) are byte-identical to the evaluated checkpoints.
Each contains its own training-only normalization and feature schema. Their
SHA256 hashes and relative paths are in `configuration.json`.

| External mean metric | Reference | Selected ensemble |
|---|---:|---:|
| SpellF1 | 0.529976 | **0.592973** |
| MacroF1 | 0.872482 | 0.908327 |
| Accuracy | 88.3871% | 92.0244% |
| Dead-ball recall | 0.914535 | 0.966084 |

Scores equally average J03WPY and J03WQQ, and equally average clean, point dropout,
2-second gaps, 10-second gaps, missing supplied speed and combined corruption
within each match. Selected per-match mean spellF1 is 0.580004 and 0.605942.
The exact decoder improves spellF1 at both seeds versus the same-seed reference:
0.585054 versus 0.529976 at seed 42; 0.582620 versus 0.498986 at seed 43.
The previously incomplete paired-seed confirmation has been completed.

These two games have been repeatedly used for development. They are not fresh
tests, and six corruptions do not create six independent matches. J03WR9 has
historical experiment artifacts and cannot be certified untouched. No new public
five-minute evaluation was used for this selection. See `external_metrics.json`
for all match/condition aggregates. Small differences between leading decoder
variants are not evidence of statistical significance.

## Inference

From the repository root, install `requirements-selected.txt` in a suitable
environment. The validated setup uses Python 3.13 and CPU Torch 2.10.0. Then:

```python
from tiki_data.p2.selected import predict

# base: the shared 19-feature P1 FeatureBatch in pitch metres, with half boundaries.
# ball_positions: aligned N x 2 metric coordinates; missing positions are NaN.
probabilities, predictions = predict(base, ball_positions)
```

Or use saved local inputs (outputs must use a new path in an ignored directory):

```bash
python -m tiki_data.p2.selected --match MATCH_SPEC.json --features P1_FEATURES.npz --ball-positions BALL_XY.npy --out runs/selected-inference/predictions.csv
```

The input timelines must align at 25 Hz. Resampling and coordinate conversion
belong to the provider's P1 adapter. Corrupt raw observations before computing
geometry, age or filtered motion. Do not feed possession, live/dead or tactical
labels into features. Missing scalar speed differs from measured zero through
the observation masks. Derived speed replaces supplied ball speed; no extrapolated
velocity is marked observed. Kalman state and age reset at each half boundary.

The model uses 21 features plus their masks: team/ball geometry, team movement,
causal Kalman speed, velocity uncertainty and observation age. The offline TCN
uses +/-62 frames of context, with no context across halves. Two probability
vectors are averaged 50:50 before sparse Viterbi decoding. The frozen decoder
uses temperature 2, one-second live/dead minima, full missing-ball evidence and
separate turnover/stoppage edge adjustments. The signed adjustments in the config
are subtracted from transition log-scores, preserving the evaluated convention.

Inference refuses to overwrite an existing output. It prints no tracking rows,
ground-truth rows or per-frame predictions. This release contains weights,
configuration, code and external aggregate results only; competition assets,
submission CSVs and private comparison files stay local.

## External training provenance

Training games: J03WMX, J03WN1, J03WOH and J03WOY, grouped by whole match. Training
uses raw missing-observation augmentation, disjoint supervised cores and
training-only normalization. Speed-field dropout affects the supplied field;
coordinate-derived speed remains available when its required observations survive.
Checkpoint selection uses clean external raw-TCN spellF1, then macroF1, with up to
60 epochs and patience 8. No SkillCorner data trained these checkpoints.

Data credit: **Deutsche Fussball Liga (DFL)**. IDSSE data are CC BY 4.0:
[dataset](https://doi.org/10.6084/m9.figshare.28196177).
Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). *An integrated dataset of
spatiotemporal and event data in elite soccer*. Scientific Data 12, 195.
[Publication](https://doi.org/10.1038/s41597-025-04505-y).

Tests: `python -m pytest tests/test_selected.py -q`.
