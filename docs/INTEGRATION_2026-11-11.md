# P1-P4 integration: 11 November 2026

All available remote team work was merged into `main`. Remote `P1` contributed
the seven-match IDSSE processed handoff and training configuration. P2's TCN
implementation and remote P4's integration implementation were already ancestors
of main. The fetched P4 branch contained P3 interfaces and a synthetic Viterbi
example, but no production rules/decoder. Tracking-only rules, exact Viterbi, and
the `python -m tiki_data full` workflow were added in this integration.

## Executed evidence

Real TCN training used the unchanged `configs/p2_idsse_v1.json`: four training
matches, two validation matches, one held-out match, CPU, seed 42. Early stopping
completed after 25 epochs; epoch 17 was selected using validation spellF1 then
macroF1. Checkpoint reload reproduced validation exactly. Training and
normalization never opened the final test match.

Checkpoint SHA-256:
`63acfdda0320939af2c99463ddfc62a51596be58e60c6bb00fb7a7732b6cfe5c`.

Equal-weight per-match validation results, using fixed decoder costs 6/3 and no
rule fusion:

| Method | Matches | spellF1 | macroF1 |
|---|---:|---:|---:|
| TCN + Viterbi | 2 | 0.573429 | 0.887421 |
| TCN argmax | 2 | 0.330502 | 0.885164 |
| Tracking rules | 2 | 0.322681 | 0.432583 |

TCN + Viterbi was frozen before opening final test labels. Its single held-out
match result was **spellF1 0.580556, macroF1 0.892579**. No settings were changed
after inspecting that result. Both validation predictions reproduced with
identical CSV bytes. All generated CSVs passed the official structural contract.
Per-match reports include 1,000-sample bootstrap intervals and transition traces.

The leading method's validation diagnostics found 383/822 missed possessions
(46.6%) and 256/695 spurious possessions (36.8%) at IoU 0.5. These remain concrete
error-analysis priorities, not estimates of achievable improvement. IDSSE label
semantics and tracking quality differ from the challenge; these are external
proxy results, not BAR-ATH competition accuracy.

The original 60-test suite passed before adding P3. The synthetic integration
demo completed validation, freeze, selected-method test and reproduction. All
four official baseline public scores reproduced, and the 199,319-row committed
baseline CSV passed the official validator. Final regression: **67 tests passed**,
including seven new P3 tests for global optimality, flicker suppression, period
resets, missing observations, control/stoppage behavior and optional rule fusion.
`pip check`, compilation and `git diff --check` passed.

## Local evidence and commands

```powershell
python -m tiki_data.p2 train --config configs/p2_idsse_v1.json --out artifacts/p2-idsse-20261111
python -m tiki_data full --config configs/p2_idsse_v1.json --checkpoint artifacts/p2-idsse-20261111/best.pt --out runs/full-20261111 --bootstrap 1000
python -m unittest discover -s tests -v
```

These output directories already exist; inspect them, do not overwrite them or
rerun the held-out test. Training evidence is in `artifacts/p2-idsse-20261111/`.
Comparison, freeze, one-attempt test record, and reproduction evidence are in
`runs/full-20261111/`. Weights and frame-level outputs remain local and ignored.
The source revision recorded during execution is the merge revision, with dirty
source fingerprints covering the subsequently added P3/workflow files.

## Remaining external inputs

The supplied-match step records `external_complete`, with these missing assets:

- `data/tactical/BAR-ATH_tactical_atd.xml`
- `data/tactical/BAR-ATH_tactical_atd.txt`

No BAR-ATH assets or feature exports were found in this checkout, Downloads, or
the desktop project search. Install the licensed originals or supply the P1
BAR-ATH NPZ. A 199,319-row candidate competition prediction cannot be generated
from the public labels or fabricated tracking. The full command already supports
those assets and exact-byte reproduction when available. Independent teammate
reproduction and an actual competition submission remain separate acceptance steps.
