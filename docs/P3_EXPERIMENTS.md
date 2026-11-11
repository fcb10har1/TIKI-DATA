# P3 experiments on external matches (no BAR-ATH labels used)

Follows the evaluation protocol in the 11 Nov team review: BAR-ATH's public window is not used for
any choice here. Everything below uses the IDSSE matches only: train J03WMX/WN1/WOH/WOY, validate
J03WPY/WQQ. The held-out test match J03WR9 is never opened. Reproduce with
`p3/scripts/external_experiments.py` (commands in its docstring). Local retrain, seed 42.

## What was tested

- **A_base**: team TCN as configured (19 features, ball speed always observed in training).
- **B_nospeed**: ball speed column removed (18 features, schema written by the script).
- **C_nospeed_gaps**: B plus ball-gap augmentation (probability 0.75, stretches 25-375 frames).
- Inference conditions on the validation matches: clean; speed absent (as on BAR-ATH); and
  "barlike": speed absent plus random ball-position gaps hiding about 36% of frames (BAR-ATH full
  match: 40.7% missing). The gaps are simulated at random, so real gap structure may differ.
- Decoder: `p3.decoder`, mean of the two validation matches.

## Results (mean of 2 validation matches; differences under about 0.04 spellF1 are noise)

Input mismatch (decoder T=4, missing_weight=0, the team's current setting):

| Model | clean spellF1 | speed absent | barlike |
|---|---:|---:|---:|
| A_base | 0.590 | 0.531 | 0.397 |
| B_nospeed | 0.546 | 0.546 | 0.403 |
| C_nospeed_gaps | 0.521 | 0.521 | 0.397 |

Ball gaps cost about 0.15 spellF1 for every model. Feeding the speed-trained model an absent speed
costs about 0.06 spellF1 and 0.05 macroF1.

Decoder `missing_weight` under barlike gaps (T=4):

| Model | mw=0 (now) spellF1 / acc / D recall | mw=0.5 spellF1 / acc / D recall |
|---|---|---|
| A_base | 0.397 / 0.799 / 0.829 | 0.427 / 0.819 / 0.921 |
| B_nospeed | 0.403 / 0.815 / 0.805 | 0.463 / 0.851 / 0.919 |
| C_nospeed_gaps | 0.397 / 0.827 / 0.817 | 0.470 / 0.870 / 0.899 |

`missing_weight=0.5` helped both matches for all three models (per-match gains 0.005 to 0.077).
With C, temperature 2 or 1 added 0.03-0.04 on top (0.500 / 0.509), but that is within noise.

One-factor ablation (B, barlike): minimum possession or dead duration (1, 2, 4 s) and
`restart_switch` (0.5, 0.8, 0.95) changed spellF1 by 0.01 or less. `mean_poss_s=8` gave +0.03,
within noise. `min_poss_s=4` and temperature 8 were worse.

## Reading

1. The largest decoder lever is `missing_weight`, not the duration or restart priors. At 0 the
   decoder discards the model on frames without a ball; the model still has useful evidence there.
2. Training without ball speed removes the speed mismatch with no real loss. Gap augmentation adds
   accuracy under gaps (0.851 to 0.870 with mw 0.5) but not clearly spellF1 on its own.
3. Not established: real BAR-ATH gains. Missing ball frames cluster in stoppages and restarts, which
   random gaps do not capture; two validation matches are a small sample; a team case study showed
   a gap where discarding evidence helped. Do not read these numbers as BAR-ATH performance.

## What to do next

- Decoder only, no retrain: set `"missing_weight": 0.5` in the decoder settings.
- Speed fix: P1 must export an 18-feature BAR-ATH batch without ball speed and P2 must train and
  load the 18-feature checkpoint. This changes the feature contract, so it needs P1 and P2 sign-off.
- Before relying on either, rerun on a fresh external match, not J03WR9.
