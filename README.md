# TIKIDATA — continued verifier

This is the submission release for **TIKIDATA**. It contains one selected
pipeline: the frozen historical TCN, the continued seed42 segment proposer,
the original seed43 proposer, and the correction-verification forest.
All four checkpoints are required components of this pipeline.

The team-named technical document is prepared locally at
`local_submission_docs/TIKIDATA.md` and is intentionally not tracked.
The possession file is generated locally as `runs/submission/TIKIDATA.csv`.
Competition tracking, labels, predictions, other model checkpoints and research
outputs are excluded from this release. The prior work remains in Git history
and in the author's local archive.

## Install

Use **CPython 3.13.1, 64-bit, CPU**. This release is validated on Windows.
From a fresh clone, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The pinned CPU PyTorch wheel comes from the official PyTorch wheel index.
The model bundle is included in `models/continued-verifier/`; training and the
external dataset are not required to regenerate the submission.
The files are ordinary Git files, not LFS pointers or private downloads.
Verify their checksums with `python tools/verify_release.py`.
Attribution and licence boundaries are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Regenerate the submission

Obtain the organisers' licensed tactical tracking files separately. From the
repository root, with those files under `data/tactical/`, run:

```powershell
.venv\Scripts\python.exe -m tiki_data --out runs/submission/TIKIDATA.csv
.venv\Scripts\python.exe evaluation/validate_submission.py runs/submission/TIKIDATA.csv
```

For an authorised input bundle outside Git, pass
`--atd-meta PATH_TO_XML --atd-raw PATH_TO_TXT` to that same command.
`H` means home possession, `A` away possession, `D` dead ball, and `X` an
outside-period output. The neural class order is H/A/D; X is not a trained
class. The neural probabilities are not averaged: two independent segment
proposers and a verification forest correct the historical baseline.
The feature order, normalization, decoder settings and checkpoint hashes
are recorded in `models/continued-verifier/inference_manifest.json`.
The command reads the complete tactical tracking timeline, writes exactly
199,319 rows, runs the official format checks, and verifies the complete CSV's
SHA-256 against the frozen reference. It refuses to overwrite an existing file.
It does not read any ground-truth labels, video, tactical tags or cached predictions.
If an exact-match check fails, confirm the input files and pinned CPU environment;
do not edit prediction rows.
The measured runtime and fresh-environment reproduction checks are recorded in
`models/continued-verifier/release_validation.json`. CPU execution was tested;
GPU execution and other operating systems are not claimed as verified.

## Data-use policy for this release

The newer organiser-email clarification supplied by the team takes precedence
over conflicting older wording in `RULES.md`: all training, fitting,
calibration and tuning use external data only. No adaptation, including
self-supervised adaptation, is performed on the supplied game. No predictions
are checked against its video, no manual labels or corrections are made, and
supplied data is not uploaded to AI services. Local automatic format and
reproduction checks are permitted. This packaging task performs no new model
training, tuning, scoring or selection.

## Recorded evaluation

| Evaluation | Accuracy | macroF1 | SpellF1 | Dead-ball recall |
|---|---:|---:|---:|---:|
| External, mean of two validation matches and six fixed conditions | 86.0782% | 0.850387 | 0.533316 | 0.860068 |
| External, clean only, mean of two matches | 89.5823% | 0.885145 | 0.597009 | 0.917884 |
| Supplied public five minutes, full tracking context | 88.3200% | 0.865117 | 0.650000 | 0.694350 |

These are existing frozen-run results, not full-match competition scores.
The public window has been inspected through repeated aggregate diagnostics;
it is not an independent test estimate. No training or threshold tuning used
competition labels. The final release is the user's explicit choice following
those diagnostics. The seven external matches have a historical development
record; the reserved split name does not establish a fresh independent test.
This release reports no new external-test result. Dataset and model provenance
are recorded in `models/continued-verifier/training_provenance.json`.

## Tests and files

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests/test_submission_release.py -q
```

- `tiki_data/submission.py`: raw tracking to validated CSV.
- `tiki_data/p1/`: feature extraction and observed-ball availability.
- `tiki_data/p2/`: frozen temporal networks, correction features and verifier.
- `p3/`: the duration-chain implementation used by this model.
- `models/continued-verifier/`: only the selected model's required assets, hashes and provenance.
- `evaluation/`: the organisers' scorer and format validator, unchanged.
- `tools/verify_release.py`: source/model checksums and exact-CSV verification.
- `SUBMISSION.md`, `RULES.md`, `DATA.md`, `EVALUATION.md`: organiser requirements.

Optional research utilities retained for inspection are train-only scaling and
window construction in `tiki_data/p2/data.py`, the segment-training objective
in `verifier_segments.py`, and external correction-target generation in
`verifier_edits.py`. These are not an end-to-end retraining command. Judges
load the frozen assets directly. Archived experimental training orchestration
is not needed for inference. Local automatic scoring uses
`evaluation/run_evaluation.py` with separately authorised labels and is optional.
Negative-result summaries are preserved in the local technical document.

Upload `TIKIDATA.csv`, provide this public repository URL, and upload
`TIKIDATA.md` through the organisers' form using the registered team name
**TIKIDATA**. The form URL comes from the registration email; this repository
does not submit the form automatically.
