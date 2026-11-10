# IDSSE V1 processed handoff

Generated from the seven-match IDSSE dataset for P2's TCN. Raw XML is not included.

Each match folder contains `features.npz` (the 19 P1 inputs), `labels.csv`
(`frame,label`), `match.json`, and `conversion.json`. Labels come only from the
IDSSE `BallPossession` and `BallStatus` fields; those fields never enter features.

From the repository root, P2 should run:

```powershell
pip install -r requirements-p2.txt
python -m pytest tests/test_p1.py tests/test_p2.py
python -m tiki_data.p2 train --config configs/p2_idsse_v1.json --out artifacts/p2-idsse-v1
```

The split is fixed by whole match before model fitting: four train matches, two
validation matches, and one untouched test match. Do not use the test labels for
model, feature, decoder, or hyperparameter selection.

## Attribution and licence

IDSSE is licensed CC BY 4.0 with authorization from Deutsche Fussball Liga (DFL).
Credit DFL and cite Bassek, Rein, Weber and Memmert (2025), *Scientific Data*
12:195, DOI 10.1038/s41597-025-04505-y. Dataset DOI:
10.6084/m9.figshare.28196177. These artifacts are transformations of that dataset.
