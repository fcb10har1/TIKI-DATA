# TIKIDATA: continued correction verifier

## Approach and selected model

We infer home possession (H), away possession (A), and dead ball (D) from
automatic player/ball tracking. A small noncausal residual temporal convolutional
network (TCN) provides frame probabilities. A duration decoder provides an initial
sequence. Two learned segment proposers suggest complete possession intervals;
a random-forest verifier accepts only sufficiently promising, non-conflicting
corrections. X is reserved for frames outside the organiser-provided half ranges,
not an additional learned possession class.

The submitted candidate is `continued_verifier_control_seed42`. It uses the
historical speed-masked TCN, the continued seed 42 proposer from epoch 10, the
original seed 43 proposer, and the original seed 42 depth 10 verification forest.
The continued model completed 15 epochs, stopping on patience. The selected
checkpoint and all decoder/verifier thresholds were frozen before its public
diagnostic. This release packages that exact candidate without further training.

The TCN uses 32 hidden channels, residual kernel 3 convolutions at dilations
1,2,4,8,16, and a 125-frame receptive field. Inputs are 19 tracking features plus
their 19 observation masks. Features cover nearest and second-nearest player
distances, nearby-player counts, ball-to-team-centroid distances, team width and
length, mean player speed, and observed-player counts. Ball scalar speed is
always masked, identically during training and inference. Training-only means
and scales are embedded in each checkpoint. Missing entries become normalized
zero with a false observation mask. Halves and outside-play blocks are processed
independently; overlapping context windows do not create duplicate output rows.

Segment networks add a boundary head and an interval classifier using pooled
interiors, endpoints, frame evidence and duration. The external-training
objective combines frame cross-entropy, 0.2 times boundary loss and 0.5 times
interval classification loss. The continued control has no added CDF loss.
Proposals use learned boundary peaks and the initial sequence's cuts; the
segment decoder uses threshold 0.3, interval evidence weight 3 and transition cost 1.

The verifier is a 128-tree RandomForestRegressor with depth 10, minimum leaf 20,
feature fraction 0.75, bootstrap sample fraction 0.85 and seed 42. It uses 132
tracking/prediction-derived features. Its training target is the exact
single-edit change in official multi-IoU SpellF1, scaled to possession-event
units. It accepts predicted gains above 0.3 and reserves all affected baseline
spells to prevent conflicting edits. Proposed edits last at most 60 seconds.

## Assets and training labels

Inference uses only the supplied tactical ATD metadata XML and raw tracking TXT.
The XML provides track-to-team assignment; positions are converted into pitch
metres. The tracking's one-based frame index is converted to zero-based video
frames. We use all available tracking context and the organiser's existing
timeline/half definitions from `games.py`. No new match-specific transition
timestamps are introduced. Videos, broadcast tracking, homographies and Smart
Tagging are not inputs to this selected pipeline. Ball-speed and Kalman/model
alternatives investigated earlier are absent from this release.

The only labelled training dataset is IDSSE, the seven-match Bundesliga dataset
released by Bassek, Rein, Weber and Memmert with DFL authorisation under
CC BY 4.0. Labels are the provider's `BallPossession` and `BallStatus` fields,
not annotations created by our team. The published definition is player control
for possession and referee interruption for inactive play. We map inactive play
to D and active home/away possession to H/A; unknown targets are excluded from
the training loss. The publication describes the provider's collection process
but does not identify the individual annotator of each frame.
[Dataset](https://doi.org/10.6084/m9.figshare.28196177),
[data description and attribution](https://www.nature.com/articles/s41597-025-04505-y).

| Split | Whole matches | Frames |
|---|---|---:|
| Training | J03WMX, J03WN1, J03WOH, J03WOY | 567,278 |
| Validation | J03WPY, J03WQQ | 288,556 |
| Reserved external test | J03WR9 | 146,810 |

All 567,278 training frames have mapped H/A/D targets, approximately 378.2 minutes
at 25 Hz. Splits were assigned by whole match before window construction;
normalization uses only observed training frames. The continued model and
verification forest use only these training matches. The forest's original
mistake mining uses in-sample frozen detector predictions, which can overfit.
The reserved external test was not accessed to choose this continuation or
during release preparation. No labels from the supplied match train any component.

## Possession, stoppage and restart rules

The initial hidden-state decoder has H, A, and two dead-ball states that remember
the preceding possessing team. It uses 1 second hard minimum durations, mean live
duration 14 seconds and mean dead duration 10 seconds, a 0.5 direct-turnover share,
and a 0.8 prior for the other team restarting after a stoppage. Frame probabilities
use temperature 4 and floor 0.0001. Initial decoding gives missing-ball frames
zero emission weight; the proposal-baseline decoder uses weight 0.5. The verifier
can replace intervals after this initial decoding.

Loose balls, challenges and brief touches are resolved through learned temporal
evidence and these generic duration/transition priors. We do not treat every
nearest-player change as a turnover, or missing tracking as a stoppage. We have
no separately hand-coded, event-specific loose-ball or challenge rule. Restarts
are statistical transitions rather than supplied match events. The provider's
control definition may differ from the competition's possession convention.

## Evaluation and limitations

| Evaluation | Accuracy | macroF1 | SpellF1 | D recall |
|---|---:|---:|---:|---:|
| J03WPY, six-condition mean | 87.7535% | 0.860450 | 0.522245 | 0.908828 |
| J03WQQ, six-condition mean | 84.4029% | 0.840325 | 0.544387 | 0.811308 |
| External mean | 86.0782% | 0.850387 | 0.533316 | 0.860068 |
| External clean mean | 89.5823% | 0.885145 | 0.597009 | 0.917884 |
| Public five-minute diagnostic | 88.3200% | 0.865117 | 0.650000 | 0.694350 |

The six fixed conditions are clean, 10% missing individual observations,
2 second gaps,10 second gaps, missing ball speed, and combined corruption.
Corruption is applied to raw ball observations before dependent features are
computed. Clean and missing-speed conditions duplicate one another for this
speed-masked model. External means weight matches and conditions equally.
Official SpellF1 averages possession matching at IoU thresholds 0.3,0.4,0.5,0.6,0.7.

The continuation was selected within its experiment using external mean
per-match SpellF1. Compared with its uncontinued verifier reference it improves
external SpellF1 by 0.007033 and accuracy by 0.214507 percentage points; public
SpellF1 and accuracy are unchanged. These results are not claims about full-match
competition performance. Both validation matches and the public window have
been repeatedly diagnosed across experiments, so the public result is not an
independent test. The team explicitly chose this frozen candidate after those
diagnostics. No further public scoring, training or tuning was done for packaging.

Missing observations, few labelled matches, control-definition differences,
in-sample verifier mining, hard minimum durations and missed proposal intervals
remain limitations, especially for brief possessions. Higher frame accuracy
does not guarantee recovery of more complete possessions. More labelled match
diversity and independently held-out validation would be the next priorities.

## Reproduction and deliverables

Use CPython 3.13.1, Windows 64-bit and the pinned CPU packages in `requirements.txt`.
All required learned assets and source hashes are in `models/continued-verifier/`.
The public repository contains no licensed match data or generated prediction CSV.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m tiki_data --atd-meta data/tactical/BAR-ATH_tactical_atd.xml --atd-raw data/tactical/BAR-ATH_tactical_atd.txt --out runs/submission/TIKIDATA.csv
.venv\Scripts\python.exe evaluation/validate_submission.py runs/submission/TIKIDATA.csv
```

The command regenerates all 199,319 rows from raw tracking, checks the official
format, and checks byte-for-byte identity using the frozen reference's SHA-256.
It prints only progress and aggregate validation metadata. The release test
suite checks asset integrity, guarded corrections, half separation and
deterministic inference. Release validation is recorded in
`models/continued-verifier/release_validation.json`.

Submit `TIKIDATA.csv`, the public repository URL and this `TIKIDATA.md` document
through the organiser's form. Installing the same pins on other operating
systems is possible, but byte-identical output there has not been independently
verified. Existing research checkpoints are preserved locally and in earlier
Git history; only this candidate is included in the current release tree.
