# P2: TCN training and inference V1

P2 uses the existing [team contracts](INTERFACES.md) without changing P1/P3/P4
or official code. The adapter is `tiki_data.p2.inference:predict`. Its only file
input is `checkpoint`; normalization, feature order, architecture, FPS, seed,
class order and feature schema are embedded in that checkpoint. Model output is
`ProbabilityBatch`, columns **H,A,D**. `X` is an ignored target/decoder abstention,
never a fourth neural class. The optional CSV is `frame,p_H,p_A,p_D`.

**Current status:** implementation and synthetic execution verification only.
No external football training, real possession accuracy, real-model robustness
gain or competition submission is claimed. See [P2_STATUS.md](P2_STATUS.md) for
actual run evidence. Synthetic checkpoints are explicitly marked and require
`allow_synthetic=true` to load through the adapter.

## Install and verify

Run from the repository root. Tested: Windows, CPython 3.13.1, PyTorch 2.10.0
CPU, NumPy 2.4.6, pandas 3.0.3. P4's separate Python 3.12 dependency file is
preserved; the P2 environment runs both suites. Do not install the two sets of
conflicting NumPy/pandas pins into one environment.

```powershell
py -3.13 -m venv .venv-p2
.\.venv-p2\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-p2\Scripts\python.exe -m pip install -r requirements-p2.txt
.\.venv-p2\Scripts\python.exe -m unittest discover -s tests -v
```

Use `.venv-p2/bin/python` on Linux/macOS with Python 3.13. CPU, one thread and
deterministic PyTorch algorithms are the reproducibility defaults. CUDA is
configurable but was not tested; deterministic kernels may reject unsupported
operations. Cross-device/version bitwise equality is not promised. Training
starts from the seed each time; interrupted runs are recorded as failed and
should be restarted in a new directory. Optimizer resume is not implemented.

## Working synthetic execution example

All output directories must be new. The commands below use `python` to mean
the interpreter in the environment above. The fixture creates separate train,
validation and test match groups and two explicit halves per match.

```powershell
python -m tiki_data.p2 prepare-smoke --out runs/p2-smoke-inputs
python -m tiki_data.p2 train --config runs/p2-smoke-inputs/train.json --out runs/p2-v1-smoke
python -m tiki_data.p2 infer --features runs/p2-smoke-inputs/p2_synthetic_validation_features.npz --match runs/p2-smoke-inputs/p2_synthetic_validation.json --checkpoint runs/p2-v1-smoke/best.pt --allow-synthetic --out runs/p2-smoke-probabilities
python -m tiki_data run --config runs/p2-smoke-probabilities/p4_model.json --out runs/p2-p4-smoke
python -m tiki_data reproduce --run runs/p2-p4-smoke --out runs/p2-p4-repeat
```

The training run saves `best.pt`, `normalization.json`, `epochs.jsonl`, and
`run.json`. It reloads the best checkpoint and checks that validation results
match the saved results exactly. Logs contain synthetic metric values to test
the selection mechanism; those values measure no football accuracy. Inference
exports probabilities in NPZ and CSV and creates a runnable P4 `p4_model.json`.
P4's argmax CSV at this stage is a diagnostic TCN output, not P3 decoding.

After the unaugmented baseline runs, exercise observation hiding separately:

```powershell
python -m tiki_data.p2 train --config runs/p2-smoke-inputs/train_hidden.json --out runs/p2-hidden-smoke
```

This checks the augmentation path, not its value on real tracking.

## External training input: P1 and team handoff

The current checkout contains no P1 feature code, feature NPZs or external raw
tracking/label exports. `configs/p2_v1.json` is an explicit template and fails
on missing inputs. It never silently substitutes synthetic data. Supply:

1. A P4-compatible external manifest as described in [P4.md](P4.md), with
   `schema_version`, `dataset_id`, `kind=external`, and `matches`. Each match
   needs its actual `MatchSpec`, underlying `group_id`, split, provenance and
   `ground_truth` CSV path. Declare all train/validation/test groups **before**
   creating windows. Different exports of one match must share a group and
   cannot be split. At least one independent match per partition is required;
   the team should reserve enough validation matches to assess variance.
2. One P1 `FeatureBatch` NPZ per match. Use `tiki_data.artifacts.write_batch`.
   Add each path to the P2 config's `features` map keyed by exact match ID.
   Frame arrays are complete `0..N-1`; half-open periods reset context.
   Zero placeholders plus `observed=False` represent missing values, including
   unavailable distances/derivatives. Observed physical zero remains valid.
3. Frame-aligned `frame,label` external targets with H/A/D or X when unknown;
   outside-period labels must be X. Preserve ambiguous/unknown annotation gaps.
   Do not create D labels merely because the ball is absent.
4. An ordered feature schema JSON, shared by all datasets, like:

```json
{
  "features": [
    {"name": "home_nearest_ball_m", "definition": "Minimum Euclidean ball-to-home-player distance in metres; missing if ball or all home players absent", "sources": ["ball.x", "ball.y", "home.positions"]},
    {"name": "home_width_m", "definition": "Max minus min lateral coordinate over observed home players in metres", "sources": ["home.positions"]}
  ],
  "ball_dependent": ["home_nearest_ball_m"],
  "notes": "25 Hz; common pitch coordinates and orientation; document actual P1 preprocessing version, units, availability policy and derivative support here"
}
```

Schema names must exactly match NPZ column order. Include every ball-derived
distance, position, motion and shape-relative-to-ball feature in `ball_dependent`.
The schema is a reviewed allowlist, not an automatic proof of feature lineage.
Code rejects common possession/status/tag field names and stable slot identities
in names or source fields. P1/team must review actual feature derivations to
exclude disguised or indirectly derived target information. Target arrays never
enter the model's data path. Training labels are loaded separately from features.

Use identical numerical computations for external and supplied matches. Good
candidates are permutation-invariant nearest-distance summaries, team centroids,
widths/spreads, movement summaries, ball motion and their observation flags.
Player slot numbers must never enter the neural vector as identities. P1 must
declare which derivatives need both endpoints observed; reset derivative and
interpolation state at halves. Do not perform target-aware feature extraction,
team assignment, preprocessing or interpolation. Tactical-tag fusion belongs
to P3; V1 does not accept tactical tags as neural features. Observation flags
are automatically appended by P2; avoid redundant independently computed flag
columns that would become inconsistent during feature masking.

V1 requires 25 Hz for train/validation/inference. P1 owns any resampling and
must document frame/time mapping and preserve stoppages, not load alive-only
frames. Never infer external half boundaries from a fixed 45-minute duration.
For supplied tactical ATD, follow P4's documented `frame_id - 1` alignment;
the model neither applies offsets nor repairs missing timeline rows.

## IDSSE target semantics and limitations

Candidate source: the [IDSSE authors' repository](https://github.com/spoho-datascience/idsse-data)
and [dataset DOI](https://doi.org/10.6084/m9.figshare.28196177).
There are seven complete Bundesliga/2. Bundesliga matches, with TRACAB tracking
and synchronized events. Data are CC BY 4.0: credit Deutsche Fußball Liga (DFL)
and cite Bassek, Rein, Weber and Memmert (2025), *Scientific Data* 12:195,
[paper DOI](https://doi.org/10.1038/s41597-025-04505-y).

The [paper's position-data description](https://www.nature.com/articles/s41597-025-04505-y.pdf)
defines `BallPossession=1/2` as home/away possession, based on the player in
control, and `BallStatus=0/1` as inactive/active. Inactive includes referee
interruptions. The [floodlight DFL parser](https://floodlight.readthedocs.io/en/latest/_modules/floodlight/io/dfl.html)
independently exposes these exact code dictionaries and extracts both fields
from the ball frames. `idsse_targets(possession, ball_status)` is a tested,
target-only conversion helper:

| Raw condition | Target |
|---|---|
| BallStatus 0, any possession code | D |
| BallStatus 1 and BallPossession 1 | H |
| BallStatus 1 and BallPossession 2 | A |
| Any other/unknown combination | X, excluded from loss |

P1 must align these raw arrays by actual frame IDs and game section, use match
metadata for home/away mapping, and retain original frame IDs in conversion
metadata even if rebasing each external timeline to zero. `BallPossession` and
`BallStatus` are **targets only**. This V1 does not implement or claim a verified
raw IDSSE XML-to-P1 feature converter; the helper takes already aligned arrays.

The competition [control definition](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md)
retains the previous team during loose/contested live play and excludes uncontrolled
touches/deflections from possession changes. IDSSE's published description does
not establish equivalent handling or identical transition times for those cases.
Treat its targets as a related provider annotation, not a verified reproduction
of competition ground truth. Report both per-frame and segment scores as external
proxy results. Raw code frequencies, unknown rates, frame continuity and half
coverage still need auditing on actual files. Optical-tracking quality and
missingness can also differ from automatic tactical-video tracking.

No external matches have been downloaded or converted in this implementation.
Downloading raw data alone would not provide P1's shared feature computation.
The [official rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md)
permit accessible external data and prohibit hand labels on the supplied match;
the public five-minute labels are never used to fit weights, normalization,
features, decoder settings or hyperparameters.

## Real training and inference

Update the external paths/IDs in a copy of `configs/p2_v1.json`. Paths resolve
relative to that config; manifest label paths resolve relative to the manifest.
P2's companion config keeps P4's strict manifest schema intact.

```powershell
python -m tiki_data.p2 train --config configs/p2_v1.json --out artifacts/p2-v1
python -m tiki_data.p2 infer --features data/features/BAR-ATH_features.npz --match data/features/BAR-ATH_match.json --checkpoint artifacts/p2-v1/best.pt --out runs/p2-bar-ath
```

`BAR-ATH_match.json` may contain
`{"match_id":"BAR-ATH","n_frames":199319,"fps":25}`; `MatchSpec` supplies the
official half boundaries. Real inference must produce exactly 199,319 probability
rows with finite entries summing to one, including outside periods and fully
missing frames. Outside-period spans are independent model sequences; P3 should
emit X there using period IDs. Probabilities there are well formed but untrained.

Default architecture: 32 hidden channels, five residual blocks, two kernel-3
convolutions per block with dilations 1/2/4/8/16, dropout 0.1. The receptive field
is 125 frames (about five seconds), symmetrically using future and past context;
this is an offline model. No batch normalization mixes padded or other-match
statistics. Core windows are 512 frames, with 62-frame halos on either side.
Only real core labels contribute to cross-entropy; X and padding are ignored.
Inference writes each core once to its original indices. Short halves, remainders,
adjacent halves, all-missing inputs and outside-period spans keep their alignment.

Training uses AdamW, optional train-only balanced class weights, gradient clipping,
and early stopping. Each epoch evaluates unaugmented validation matches with P4's
official-metric wrapper, resetting at periods. Checkpoints are selected by mean
per-match spellF1, then macroF1; exact ties keep the earlier checkpoint. Cross-entropy
is logged but does not override that rule. Final test files are never opened by
training, including during normalization or checkpoint reload. Config/feature/
label/schema hashes, source fingerprint, environment, seeds, class counts,
selected epoch and all epochs' losses/metrics are recorded. File or source changes
during a run cause failure. Load uses `torch.load(..., weights_only=True)` and a
plain state dictionary, not a pickled model object.

## Missing-observation experiment after V1

Create a separate config/run, keeping the same split and selection procedure.
For example set `point_probability=0.05`, `ball_stretch_probability=0.5`, and
`ball_stretch_frames=[25,250]`. The first independently hides observed feature
entries; the second masks all declared ball-dependent channels over a randomly
located contiguous 1–10 second stretch per window. Values and flags change
together, original arrays stay unchanged, normalization remains fixed, and
validation is unaugmented. Padding, periods and targets remain unchanged.

This is **feature-level** robustness training. It does not recompute derivatives,
interpolation, team summaries or cached flags from corrupted raw tracking, and
context outside a masked interval can retain information from those computations.
P1 should later supply dependency-aware/raw-observation masking followed by feature
recomputation. Evaluate realistic missing-ball duration bins and fixed corruption
seeds on validation only; compare clean and corrupted macroF1/spellF1 for baseline
and augmented models. A successful synthetic execution provides no evidence of
an accuracy benefit. Actual augmentation settings require external validation.

## P3 and P4 handoff

- **P3:** consume `ProbabilityBatch` directly or `probabilities.npz` using
  `read_batch(path, match, "probabilities")`. CSV is available if preferred.
  Columns are H,A,D, never re-ordered; do not pass X as a fourth neural state.
  Reset Viterbi at each period; use the existing rules/tags outside the neural
  feature path. P2 does not supply production Viterbi or calibrated probabilities.
- **P4:** use the generated `p4_model.json`, or configure
  `callable="tiki_data.p2.inference:predict"`, `inputs={"checkpoint":".../best.pt"}`,
  and `expected_features` equal to checkpoint feature order. Embedded normalization
  adds no hidden file dependency. The sibling normalization JSON is for inspection.
  Add P3's decoder stanza only when its actual adapter is available. Run with
  `python -m tiki_data run --config <config> --out <new-directory>`.
- **Evaluation:** compare rules, TCN and TCN+Viterbi on identical whole validation
  matches via P4's benchmark; select with mean spellF1 then macroF1, freeze the
  full pipeline, and use P4's one-attempt final external test. P2's epoch metrics
  are raw argmax TCN diagnostics. They do not replace the team comparison or tune
  P3's decoder. Do not inspect final-test results to choose another checkpoint.

Use P4's reproduction and official validator on the final decoded CSV. Keep
restricted data, derived frame-level exports, logs and weights under ignored
`data/`, `runs/` or `artifacts/`; share allowed weights with hashes and the
training command through the team's chosen channel. No messages were sent to
teammates or external services by this implementation.
