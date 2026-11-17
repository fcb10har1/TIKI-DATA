# TIKIDATA — continued verifier

This is the submission release for **TIKIDATA**. It contains one selected
pipeline: the frozen historical TCN, the continued seed42 segment proposer,
the original seed43 proposer, and the correction-verification forest.
All four checkpoints are required components of this pipeline.

The technical document to upload is [TIKIDATA.md](TIKIDATA.md).
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

## Regenerate the submission

Obtain the organisers' licensed tactical tracking files separately. From the
repository root, with those files under `data/tactical/`, run:

```powershell
.venv\Scripts\python.exe -m tiki_data --out runs/submission/TIKIDATA.csv
.venv\Scripts\python.exe evaluation/validate_submission.py runs/submission/TIKIDATA.csv
```

For files elsewhere, pass `--atd-meta PATH_TO_XML --atd-raw PATH_TO_TXT`.
The command reads the complete tactical tracking timeline, writes exactly
199,319 rows, runs the official format checks, and verifies the complete CSV's
SHA-256 against the frozen reference. It refuses to overwrite an existing file.
It does not read any ground-truth labels, video, tactical tags or cached predictions.
If an exact-match check fails, confirm the input files and pinned CPU environment;
do not edit prediction rows.

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
those diagnostics. See the technical document for limitations and provenance.

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
- `SUBMISSION.md`, `RULES.md`, `DATA.md`, `EVALUATION.md`: organiser requirements.

Upload `TIKIDATA.csv`, provide this public repository URL, and upload
`TIKIDATA.md` through the organisers' form using the registered team name
**TIKIDATA**. The form URL comes from the registration email; this repository
does not submit the form automatically.
