# P2 delivery evidence and remaining inputs

Verified locally on 9 November 2026 (Asia/Singapore). Repository inspection found
P4 integration code/contracts, no P1/P3 production modules, and only `data/README.md`
under `data/`. No applicable AGENTS.md was found. Existing teammates' source,
contracts, configs and official evaluation files were preserved. The only edit
to an existing tracked file adds `.venv-p2/` to `.gitignore`; P2 lives in new files.

## Experiments actually run

All training below uses the explicit synthetic fixture in `tiki_data/p2/smoke.py`.
It verifies execution, tensor/frame contracts and checkpoint transport only.
**No football accuracy or real robustness benefit is established.** Final
external test data, public five-minute labels and supplied match assets were
not used for model training, selection or normalization.

| Execution | Evidence | Result |
|---|---|---|
| Unaugmented synthetic training | `runs/p2-v1-smoke/epochs.jsonl`, `run.json`, `best.pt`, `normalization.json` | Completed; 3 epochs, early stopping, epoch 1 selected by configured validation rule; saved checkpoint reloaded and validation exactly reproduced |
| Hidden-observation synthetic training | `runs/p2-hidden-smoke/epochs.jsonl`, `run.json`, `best.pt`, `normalization.json` | Completed; 3 epochs, epoch 1 selected; point masking 0.05, ball-stretch probability 0.75 and 25–75 frame stretches; reload verified |
| Probability CLI, 443-frame validation fixture | `runs/p2-smoke-probabilities/` | H,A,D NPZ and `frame,p_H,p_A,p_D` CSV exported, runnable P4 config generated |
| Actual P4 adapter execution and reproduction | `runs/p2-p4-smoke/`, `runs/p2-p4-repeat/reproduction.json` | Identical CSV bytes, matching environment |
| Full-length synthetic alignment | `runs/p2-full-alignment-inputs/`, `runs/p2-full-probabilities/`, `runs/p2-full-p4/` | 199,319 finite probability rows; includes 5,000 consecutive all-missing frames; official CSV validator accepts frames 0..199318 |
| Full-length P4 reproduction | `runs/p2-full-p4-repeat/reproduction.json` | Identical 199,319-row CSV bytes; SHA-256 `356b45d837097c46f037f5019475da59087f5d552de1f04e388364c73d7bb6e1` |
| Full regression suite | `runs/p2-unittest.log` | **55 passed** (8 P2, 47 existing P4), 153.232 seconds in the tested environment |
| Syntax, whitespace and artifact checks | `compileall`, `git diff --check`, `git check-ignore` | Passed; generated weights, logs, exports and P2 environment ignored |

Synthetic training used width 8, kernel 3, dilations [1,2], dropout 0, core size
64, batch size 4, CPU, one thread and seed 42. This is a deliberately small
execution fixture; the production template is the configurable width-32 TCN.
The full-length synthetic diagnostic CSV predicts A on every frame. Its format
validity and reproduction say nothing about possession accuracy; it is not a
candidate competition submission. The official validator's suggestion to score
against public labels was not followed for this fabricated timeline.

The first P2 test run exposed a Windows NumPy `randint` default-int32 limit.
The seed draw now requests int64; all 8 P2 checks then passed. No existing test
was edited or weakened. The 55-test run also exercises P4's existing public
metric parity tests; those tests do not enter P2 model selection.

Local baseline checkpoint SHA-256:
`3d0dd890b7a2f57f206a7c078b05d7b0de39c49e2799d63a3a51f1190e4f890c`.
Augmented checkpoint SHA-256:
`6c497a1337c0ca2d46ef211d687d0e4bfb615876985bf666840f468e3c166dfc`.
Both are **synthetic weights**, stored locally in ignored run directories.
Full environment/source/input fingerprints and synthetic per-epoch losses and
metrics are in each run JSON/log, not presented as football results here.

## Required next inputs and owners

- **P1:** real external and supplied-match `FeatureBatch` exports using one
  reviewed ordered feature schema, observation masks, common 25 Hz computations,
  coordinate convention, explicit halves, and documented source dependencies.
  No possession/status target fields, tactical tags or stable tracker-slot IDs
  in the feature path. The raw IDSSE-to-P1 feature converter is still absent.
- **P1 + team:** accessible external tracking and aligned targets, annotation
  provenance, IDSSE code/continuity audit, home/away mapping, original-frame
  conversion metadata and a fixed underlying-match train/validation/test manifest.
  No external data files have been obtained or converted by P2 in this delivery.
- **P2:** run real training after those inputs arrive; compare unaugmented and
  observation-hiding variants on validation only; publish permitted weights or
  their reproducible training command with input versions/hashes. Coordinate raw
  masking/recomputed derivatives with P1 for a later robustness experiment.
- **P3:** consume H,A,D emissions through the unchanged contract and provide real
  rule/tag fusion and period-reset Viterbi adapter. Decide abstention downstream.
- **P4:** benchmark rules/TCN/TCN+Viterbi on identical external validation matches,
  retain per-match macroF1 and spellF1, freeze before final external test, then
  verify the real complete 199,319-frame CSV and teammate reproduction.

The official challenge documents and IDSSE primary sources were read. Confirmed
label encodings, unresolved loose-ball/control definition differences, dependency
instructions, exact input examples and run commands are in [P2.md](P2.md).
The technical-document insert is [P2_TECHNICAL.md](P2_TECHNICAL.md).
P4's older status document remains an unchanged record of its delivery; this file
records the subsequently added P2 component. No external messages, commits,
uploads, publishing or competition submission were performed.
