# P4 task board and acceptance status

P4 tooling is implemented and locally verified on `P4`.
The user confirmed that inputs absent from the repository are not currently
available. Real V1 acceptance remains **blocked**; a synthetic score is not a
substitute for actual external validation.

## Ownership and handoff

| Owner | Deliverable | Status / next evidence |
|---|---|---|
| P1 | Restricted-data loader and real feature artifact | Blocked: supply assets and confirm ordered feature names, units, missing flags, periods |
| P1 + team | External dataset provenance and match-level train/validation/test manifest | Blocked: agree actual independent match groups and ground-truth meaning |
| P2 | Trained TCN inference adapter, preprocessing state, weights and dependency pins | Blocked: supply runnable production inference and accessible, checksum-pinned weights or a reproduction training command |
| P3 | Production rules/emissions and Viterbi adapters | Blocked: supply actual rules, decoder settings and period-reset behavior; existing baseline is retained |
| P4 | Runner, handoff validation, CSV validation, hashes/timing | Verified; final checks recorded below |
| P4 | Ablation benchmark, transition reports, diagnostic priorities, experiment index | Implemented; synthetic workflow and official metric parity verified |
| P4 | Validation freeze, single-method test guard, exact-byte reproduction | Implemented; failure, tamper and relocation checks verified |
| P4 | Runbook, interface contract, technical component report | Implemented; see `P4.md`, `INTERFACES.md`, `P4_TECHNICAL.md` |
| Teammate + P4 | Clean setup and actual full-match CSV reproduction | Fresh local environment/isolated checkout and relocated-input reproduction passed on a full-size synthetic input; actual teammate/real-match check blocked |
| Team | Final public reproduction package, final CSV, full technical document, submission receipt | Pending real inputs and team review; no competition submission performed |

## Refinement loops

1. Initial synthetic workflow and official starter succeeded. Initial regression
   run: 34 passed, one invalid assertion failed because it demanded two diagnostic
   findings when only one category had errors. The replacement requires measured
   nonzero findings only; a separate controlled case verifies two categories.
2. Hardened holdout reservation/failure logging, ground-truth isolation, synthetic
   labelling, environment/input checks, reproduction with relocated assets and
   explicit failure on nondeterminism. Expanded suite: **43 passed**.
3. Final review added index overwrite protection, original-frame/period checks,
   normalized-public-label rejection, and final acceptance verification. A new
   fixture initially recreated the original Windows newline bytes, so the earlier
   public-file hash guard correctly rejected it before the intended normalization
   check. Distinct header/case formatting now exercises the intended branch.
   Final suite: **46 passed**. No valid test was removed or weakened.
4. Publication review identified Git's LF/CRLF conversion as a source-fingerprint
   portability issue. Source digests now normalize line endings; data and weight
   hashes remain byte-exact. A regression verifies equivalent checkouts match
   while real source changes and differing input bytes remain detectable.
   Publication suite: **47 passed**.

The loops are bounded by `P4_WORKFLOW.json`; reviews were sequential self-review
plus automated checks. No independent human approval is implied.

## Verification record

| Command / check | Result |
|---|---|
| `.venv/Scripts/python.exe -m unittest discover -s tests -v` | **47 passed**, including contracts, failures, official parity, FPS/period handling, leakage, freeze, nondeterminism, relocated inputs and source newline portability |
| `python evaluation/run_evaluation.py --no-detail --out runs/official-starter` | All four committed baselines scored; published public scores reproduced |
| `python evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv` | **199,319 rows valid**, frame sequence 0–199318 |
| `python -m tiki_data demo --out runs/verified-demo` | Complete synthetic validation → freeze → selected-method test → identical-byte reproduction |
| `python -m tiki_data check-manifest runs/verified-demo/inputs/manifest.json` | Valid: one train, two validation, one test match |
| `python -m tiki_data smoke --csv evaluation/results/baseline_smooth50.csv --out runs/final-public-smoke` | 7,500 graded frames; pipeline check only, no accuracy selection |
| Full-size synthetic ATD → baseline → official validator | **199,319 rows valid**; generated geometry is synthetic, not the provided match |
| Fresh venv + isolated local checkout with exact working source + relocated inputs → `reproduce` | **Identical CSV bytes**, SHA-256 `3eadbdf753d527e47f3f61abf01fe30de3e34f6b8ccba50a46289f414389e987` |
| Workflow JSON validator | Zero errors, zero warnings |
| `pip check`, Python syntax, Markdown relative links, Git ignore checks, `git diff --check` | Passed |

The clean setup used a separate checkout and a fresh install of
`requirements-p4.txt`; it did not rely on the original virtual environment.
Proof is in `runs/clean-setup-verification.json`, with archived reproduction
files in `runs/clean-reproduction-evidence/`. Synthetic comparison evidence is
in `runs/verified-demo/`. These are local, ignored artifacts. Restricted assets
and production predictions have not been added to Git.

Automatic approval review rejected removal of the generated
`runs/clean-setup-temp` directory with the reason `blocked by policy`. The
temporary checkout/environment remains ignored under `runs/`. Verification and
its preserved evidence succeeded before the cleanup rejection.

## Real V1 gates still open

- Full supplied-match generation and official validation using the actual ATD.
- Rules, TCN, and TCN + Viterbi comparison on agreed external validation matches.
- Two consequential real-model weaknesses supported by that benchmark.
- A teammate's independent reproduction using the real pipeline and pinned inputs.
- Final weights/configuration selection, untouched external test, and complete
  team submission package. The public five minutes cannot close these gaps.
