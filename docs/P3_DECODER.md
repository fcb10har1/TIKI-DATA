# P3 possession decoder

Plug-in decoder stage for the P4 pipeline: `p3.stage:decode_stage`. Takes a validated
`ProbabilityBatch` (H, A, D order) and returns a `PredictionBatch`. Each period in `match.periods`
is decoded independently; frames outside the periods are `X`.

```json
"decoder": {
  "callable": "p3.stage:decode_stage",
  "inputs":   {"ball_observed": "ball_observed.npz"},
  "settings": {"temperature": 4.0, "fitted": []}
}
```

- `ball_observed.npz` (optional): key `ball_observed`, bool `(n_frames,)`, True where the ORIGINAL
  tracking has a finite ball x and y. Frames where it is False carry the previous state
  (`missing_weight=0`) instead of trusting the model. P4's `FeatureBatch`/`ProbabilityBatch` do not
  carry this mask, so P1 must write it as a separate file; without it all frames count as observed.
- `settings`: any `DecoderConfig` field except `fps` (taken from the match) and `seed` (injected from
  the run seed). Unknown names are errors. `fitted` lists settings fitted on EXTERNAL data; every
  other setting is an unfitted prior. The list is recorded in the run's `config.json`.

## Method
Four-state semi-Markov chain (H, A, D-after-H, D-after-A): exact minimum durations, geometric tails,
a restart prior (after dead ball the other team usually gets the ball), Viterbi decoding. Optional
minimum-Bayes-risk selection over exact posterior samples (`use_mbr`, off by default).

## Evidence so far (synthetic only; says nothing about the real match)
`python p3/scripts/ablation.py`, `python p3/scripts/temperature_sweep.py`:
- decoder with ball-missing handling: spellF1 0.54 vs 0.24 for +-50 frame majority smoothing
- ignoring ball visibility: 0.30
- MBR did not beat a well-tuned Viterbi (0.534 vs 0.571) and is ~4x slower; it is less sensitive to
  temperature. Off by default; re-test on external labelled matches.

## Rules baseline (`rules_viterbi` fallback, no model needed)
`p3.rules:predict_emissions` reads the ATD (`atd_raw`, `atd_meta`) and emits H/A/D probabilities from
nearest-team-to-ball (faded when the ball is loose), ball outside the pitch, and ball/players nearly
stationary. A frame with no ball observation gets exactly uniform probabilities (no evidence, and
a missing ball is never dead-ball evidence by itself). All constants are UNFITTED priors.

```json
"predictor": {"kind": "adapter", "stage": {"callable": "p3.rules:predict_emissions",
    "inputs": {"atd_raw": "<path>", "atd_meta": "<path>"}, "settings": {"fitted": []}}},
"decoder":   {"callable": "p3.stage:decode_stage", "settings": {"temperature": 2.0, "fitted": []}}
```
Only exercised on synthetic tracking in the ATD text format (public schema from `starter/baseline.py`).
It has NOT been run on the supplied ATD; do that locally and check it only with the pass/fail
messages of `python -m tiki_data run`.

## Not done
Smart Tagging evidence (team-tag semantics unverified), calibration, ensembling, any fitting on
external data. All defaults (temperature, durations, restart probability, rules constants) are
UNFITTED priors.

## Tests
`pip install -r requirements-p3.txt && python -m pytest` (36 tests: brute-force exactness of Viterbi
and the sampler, equality with the official spellF1, behaviour, and P4 integration including
`run` + `reproduce`). P4's own suite: `python -m unittest discover -s tests`.

No restricted data was read to write or test this. Real inputs must be processed locally, with
no rows, plots or per-frame outputs shared with hosted AI tools.
