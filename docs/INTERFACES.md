# P1–P4 handoff contract, version 1

The executable definitions live in `tiki_data/contracts.py`; NPZ transport is in
`tiki_data/artifacts.py`. This is P4's proposed integration contract. P1–P3 must
confirm their production feature names, preprocessing and module entry points.

## Timeline and common metadata

`MatchSpec(match_id, n_frames, fps, periods)` describes one complete timeline.
Frames are integers `0..n_frames-1`, ordered, complete, and duplicate-free.
Periods are ordered, disjoint half-open `[start, stop)` ranges. Their IDs are
1, 2, ...; frames outside them have ID 0. Independent periods reset model and
decoder state. Default external period: the entire timeline.

BAR-ATH has exactly 199,319 frames at 25 fps and periods `[11185,90652)` and
`[110808,190198)`, from `games.py`. Tactical ATD `frame_id - 1` gives a submission
frame. The unrelated corrected-tracking offsets in `games.py` must not be
applied to the provided ATD. Broadcast data needs its own alignment first.

Home/away are match-relative sides. Every external converter must document team
mapping, coordinate units/orientation, resampling, and ground-truth label meaning.
Do not treat vendor team tags alone as independently verified possession/spell
ground truth. A dead ball is `D`; contested or loose live play is still `H` or
`A` according to the competition's control definition.

## P1 output: `FeatureBatch`

| Field | Contract |
|---|---|
| `match` | Exact `MatchSpec` supplied to the adapter |
| `frames` | Integer array `(N,)`, exactly `0..N-1` |
| `values` | Floating array `(N,F)`, finite, at least one feature |
| `observed` | Boolean array `(N,F)`; false means missing, never inferred from a numeric sentinel |
| `feature_names` | Tuple of unique, nonempty names in the exact model input order |
| `periods` | Integer array `(N,)`, exactly `match.period_ids()` |

Missing feature values have a finite zero placeholder **and** `observed=False`.
An observed physical zero is distinct. Feature units should be encoded in names
and documented by P1. P4 supplies no normalization fitted to validation/test;
P1/P2 must fit preprocessing only on training matches and declare saved state as
a model input. P4 checks P2's `expected_features` list against the actual order.

## P2 output / P3 emissions: `ProbabilityBatch`

`match`, `frames`, `periods` follow the same contract. `probabilities` is a
floating `(N,3)` array. Columns are **H,A,D**, each value finite and in `[0,1]`,
and each row sums to 1 with absolute tolerance `1e-6`. `class_order` is exactly
`("H","A","D")`. There is no automatic normalization or column permutation.

P2 must supply probabilities for all frames, including missing-observation
frames. Outside periods, decoders can emit `X`; probability rows are still
well formed. A short segment must be explicitly expanded to the complete
timeline by its producer, rather than silently padding frames in P4.

## P3 output: `PredictionBatch`

`match`, `frames`, and a one-dimensional `(N,)` string array `labels`, containing
only H/A/D/X. Every generated CSV is canonical UTF-8 `frame,label`, LF newlines,
one row per frame, no extra index. The official validator is used before a run
can be marked complete. Imported CSVs receive exactly the official whitespace/
case normalization and structural checks.

## Adapter signatures

Every callable is explicitly configured as `python_module:function`; the module
must be in this repository and included in its source fingerprint. No shell
commands, dynamic `eval`, or implicit module discovery are used.

```python
def build_features(match: MatchSpec, inputs, settings) -> FeatureBatch: ...
def predict_model(features: FeatureBatch, inputs, settings) -> ProbabilityBatch: ...
def predict_rules(match: MatchSpec, inputs, settings) -> PredictionBatch: ...
def predict_emissions(match: MatchSpec, inputs, settings) -> ProbabilityBatch: ...
def decode(probabilities: ProbabilityBatch, inputs, settings) -> PredictionBatch: ...
```

`inputs` is a read-only mapping of declared file names to absolute `Path` values.
`settings` is a private copy of JSON settings with the run's `seed` injected.
Explicit stage seeds must agree. Feature and probability input arrays are marked
read-only before handing them to the next adapter. Return a new batch with
the same timeline; do not modify inputs, read labels, or hide file dependencies.
P2 is responsible for its framework's seeding, checkpoint loading, device policy,
and evaluation mode. P3 owns control/stoppage rules and decoder parameters.

An emissions-producing rules adapter plus a decoder can implement the fallback
`rules_viterbi` method without involving P2. Until P3 supplies that adapter,
`configs/baseline.json` remains the available baseline; it must not be relabelled
as rules + Viterbi.

## NPZ transport

Use `write_batch(path, batch)` / `read_batch(path, match, kind)` from
`tiki_data.artifacts`. Files contain these exact keys:

- Common: scalar integer `schema_version=1`, scalar JSON string `match`, `frames`, `periods`.
- Features: `values`, `observed`, Unicode `feature_names`.
- Probabilities: `probabilities`, Unicode `class_order`.

Loading uses `allow_pickle=False`. Object arrays, stale schema versions,
unexpected fields, invalid values, and timeline mismatches fail. NPZ inputs and
weights are fingerprinted and ignored by Git. Changes to these contracts require
a schema version update and coordinated producer/consumer changes.
