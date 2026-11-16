# Experiment source guide

See [the dated audit](../../docs/EXPERIMENT_AUDIT_2026-11-16.md) for the complete chronology, results, failures, scope and corrections. The original `README.txt` is preserved as documentation of the earlier capped boundary/motion screen; its claims about a single public entrypoint and four tests are not descriptions of the entire later directory.

## Scope

- Original boundary/relation screen: `run.py`, `continue.py`, `train_model.py`, `boundary.py`, `feature_engine.py`, `decoder.py`, `public_ab.py`, `public_full.py`.
- Follow-ups: `combined.py`, `spell_experiment.py`, `spell_weighting.py`, `full_context_comparison.py`, `extended_a_training.py`, `historical_error_audit.py`.
- Final factorial: `historical_factorial.py`, `historical_factorial_training.py`, `historical_factorial_report.py`, `historical_factorial_continue.py`, `skillcorner_expansion.py`.
- Shared feature/model/decoder helpers and reporting/control scripts are preserved alongside the entrypoints. Executed Python sources have not been rewritten for publication.

## Execution prerequisites and caveats

Install repository `requirements-selected.txt` and pytest in an appropriate Python environment. Original runs used CPython 3.13 on CPU and a locally installed dependency directory. `common.py` refers to ignored historical manifests, processed features, raw arrays, checkpoints and caches. Supplied inference additionally needs the user's separately licensed local tracking assets. A fresh clone cannot reproduce the whole research programme without those prerequisites; the published default model has its separate portable inference workflow.

Run scripts from the repository root. `run.py` and `continue.py` execute work on import; do not import them as libraries. Continuation scripts can launch public evaluation automatically. Several later public/full-context/factorial entrypoints access supplied inputs, rather than only `public_full.py`.

Earlier budgeted scripts retain their original wall-time limits. The later extended/factorial trainers use epoch/patience rules. Historical output markers intentionally prevent overwriting old experiments. The factorial's `train` resumes matching configurations with optimizer/RNG state. Frozen `finish`/report phases reuse committed outputs and verify provenance. The separate continuation driver uses exclusive lock/log files and is not itself restart-idempotent: check that the prior process has stopped and use direct phase recovery where appropriate; never delete evidence merely to rerun it.

J03WR9 is held out from the latest study's training/tuning but was evaluated historically. Legacy filenames containing `untouched_test_confirmation` remain unchanged; the audit corrects that terminology. Public scores are five-minute diagnostics using full tracking context, not private full-match scores.

## Synthetic validation

From PowerShell at the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\experiments\boundary_motion"
python -m pytest experiments/boundary_motion/test_controls.py experiments/boundary_motion/test_spell_weighting.py -q -p no:cacheprovider
python -c "from historical_factorial_training import synthetic_window_checks; from skillcorner_expansion import self_test; print(synthetic_window_checks()); self_test()"
```

These checks use synthetic inputs. They cover boundary targets/gradients, disabled-decoder parity, permutation/missing-data behavior, spell weighting, raw dropout/target invariance, context size/parameter parity, sampler resume and the expansion adapter. Do not run a training or public entrypoint just to validate installation.

All new research outputs and checkpoints stay under Git-ignored `runs/`. Do not force-add supplied data, predictions, caches, dependency copies or private reference CSVs. Small aggregate-only publication snapshots are explicitly named in the audit.

Frozen hashes identify the original local file bytes. Git may normalize line endings on checkout; a reconstructed checkout is not automatically byte-identical to those archived sources. Preserve original local evidence when resuming a frozen run.
