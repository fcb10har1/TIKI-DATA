# P1 V1 status

Implemented on 9–10 November 2026 on branch `P1`.

## Delivered

- P4-compatible `tiki_data.p1.adapter:build`.
- Streaming Metrica ATD XML/TXT converter with exact frame-continuity checks.
- Nineteen ordered, metric, permutation-invariant tracking features.
- Explicit `(values, observed)` missing-data handling.
- P2-compatible committed feature schema and ball-dependent allowlist.
- CLI export of `features.npz`, `match.json` and `feature_schema.json`.
- Synthetic geometry, missingness, NPZ round-trip and actual P2 TCN contract tests.
- Streaming IDSSE XML converter with original-frame and half alignment.
- Seven external match feature batches, labels, manifest and P2 training config.

## Verification evidence

- Real supplied ATD conversion completed locally: 199,319 frames × 19 features.
- Exported `FeatureBatch` reloaded through `read_batch` and passed the strict
  frame, period, finite-value, feature-order and zero-placeholder contracts.
- P2 schema validation accepted the exact committed feature order and
  ball-dependent allowlist.
- Five P1 tests passed, including construction of the actual P2 `TCN` and a
  validated `(N,3)` `ProbabilityBatch` from the P1 schema.
- P2's loader accepted four train and two validation matches, all with the exact
  ordered 19-feature schema; the test match was declared but not opened.
- IDSSE conversion produced 1,002,644 frames and all H/A/D classes in every match.
- Complete regression suite passed: 60 tests across P1, P2 and P4.
- `compileall` and `git diff --check` passed.

The successful ignored local handoff is
`data/features/p1-v1-bar-ath-r1/{features.npz,match.json,feature_schema.json}`.

The exploratory notebooks were run locally. Saved execution outputs and
execution metadata were cleared when P1 was merged into main; notebook cell
sources were preserved. Re-run them locally with the restricted assets, and
clear regenerated match-derived outputs before committing notebook changes.

## Not claimed

- No possession accuracy improvement is claimed.
- No supplied-match labels or public ground truth were used.
- No real TCN checkpoint or external accuracy result is claimed yet.
- Raw IDSSE XML is not committed; only the processed CC BY 4.0 handoff is tracked.
- No Smart Tagging, homography or video feature enters P2 V1.

The immediate handoff to P2 is `configs/p2_idsse_v1.json` plus the validated
processed dataset under `data/external/idsse_processed/`. P2 can now begin real
external TCN training without the raw IDSSE XML.
