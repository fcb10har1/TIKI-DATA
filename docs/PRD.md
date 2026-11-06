# P4 integration and evaluation

## Intent and scope

P4 connects independently developed data/features (P1), neural inference (P2), and
rules/decoding (P3) into reproducible possession predictions. The competition
requirements in `CHALLENGE.md`, `RULES.md`, `SUBMISSION.md`, and `EVALUATION.md`
remain authoritative. The official evaluator and baseline are preserved.

P4 owns the command-line runner, versioned handoffs, CSV validation, external
match comparisons, experiment records, algorithm freeze, reproduction checks,
and integration documentation. Training a TCN, inventing a production decoder,
obtaining restricted data, publishing, and submitting are outside P4's scope.

## Requirements

1. One documented command creates a complete, validated `frame,label` CSV from
   the match assets using the existing baseline, or from configured P1–P3 adapters
   or their exported artifacts. Missing stages fail; they never silently fall back.
2. Handoffs specify match identity, frame numbering, FPS, period boundaries,
   feature order, observation flags, and probability order `H,A,D`. Invalid or
   misaligned artifacts fail before producing a successful run.
3. Supplied-match outputs have 199,319 rows, frame sequence 0–199318, and labels
   H/A/D/X. Every generated CSV uses the official validator.
4. Compare rules, TCN, and TCN + Viterbi on identical external validation matches.
   Rank by the unweighted mean of per-match official `spell_f1_mean`, then
   `macro_f1`. Include per-match scores, transition diagnostics, uncertainty when
   requested, and two evidence-backed error-analysis priorities.
5. Split by underlying match. Reject overlapping match groups, reused label
   files, and known supplied-match identities/public labels in external evaluation.
   The public five minutes have a separate smoke-check command and never select
   settings. Synthetic fixtures are always identified as synthetic.
6. Freeze a method using a completed validation comparison before accessing the
   test split. Verify code/environment/configuration/input fingerprints. An
   exclusive local attempt record prevents accidental repeated test evaluation
   for the same freeze, including failed attempts after evaluation starts.
7. Each run records configuration, code revision and dirty state, source digest,
   environment, input and output SHA-256, elapsed time, and success/failure.
   Refuse to overwrite an existing experiment. Reproduction reruns inference and
   compares the CSV bytes; copying an earlier output is not reproduction.
8. Generated predictions, feature/probability artifacts, restricted assets, and
   local experiment outputs stay out of version control. No credentials are used.

## Acceptance and evidence

| Criterion | Required evidence |
|---|---|
| Runner and handoffs | Unit and CLI integration tests, including invalid inputs |
| Official compatibility | All four committed baselines validate and reproduce their documented public scores |
| Reproduction | Two independent executions produce identical submission bytes; altered inputs/code are rejected |
| Evaluation protocol | Synthetic multi-match ablation plus split-leakage, holdout, FPS, and period-boundary regression tests |
| Real V1 | Full supplied-match inference, actual rules/TCN/TCN+Viterbi external validation, teammate reproduction |
| Documentation | Setup, adapter contracts, commands, experiment interpretation, limitations, task board, technical report |

## Decisions and constraints

- JSON configurations and NumPy NPZ artifacts avoid adding a workflow framework.
- P4 uses the existing NumPy/pandas metric library, not a reimplementation.
- Source fingerprints normalize LF/CRLF checkout differences; input data,
  weights and output fingerprints remain byte-exact.
- External FPS may differ: possession IoU is unchanged; transition tolerances are
  scaled from 12/25 frames at 25 Hz and durations use the actual FPS.
- Periods and ungraded gaps are independent sequences; temporal scoring cannot
  match across them. All external methods use exactly the same grading mask.
- Paths in JSON resolve relative to that JSON file. Adapter file dependencies
  must be declared as inputs so their bytes enter the experiment fingerprint.
- P1–P3 adapters are explicit Python functions in the repository. Their module
  source is covered by the source fingerprint. P4 does not claim a fixture is a
  trained TCN or a production Viterbi decoder.
- Local holdout guards prevent accidental reuse, not deliberate circumvention.
  Data provenance and truly untouched test selection remain team responsibilities.

## Status

P4 tooling is implemented and locally verified: 47 regression tests pass, all four
official baseline metrics agree exactly, and a fresh environment/isolated checkout
reproduces a 199,319-row synthetic baseline CSV byte for byte. Real V1 acceptance is blocked
by missing restricted match assets, P1–P3 production outputs/adapters, and an
agreed external dataset and split. These dependencies cannot be replaced by
synthetic scores. See `docs/P4_STATUS.md` for final verification and ownership.
