Controlled boundary and relative-motion experiments

All training, normalization, checkpoint and decoder selection use external IDSSE
matches. The supplied game's inputs and labels are opened only by public_full.py,
after freeze.json records the selected assets and external rationale. That script
prints aggregate metrics only, checks official CSV validity, and repeats inference.

Required packages: repository requirements-selected.txt plus pytest for the four
synthetic controls. This run reused already installed packages; commands.txt in
the ignored output directory records the experiment-local dependency path.

Entry points, run sequentially from the repository root:
  python -m pytest experiments/boundary_motion/test_controls.py -q -p no:cacheprovider
  python experiments/boundary_motion/run.py
  python experiments/boundary_motion/continue.py

run.py reproduces the frozen reference, then attempts A42 and B42. continue.py
waits for those attempts, scores time-capped saved checkpoints descriptively,
screens only completed runs, repeats eligible finalists at seed43, freezes the
external selection, performs the one public batch, and writes the report.
The predeclared budgets and eligibility gates are never relaxed after results.
Output markers intentionally reject a second run into the same directory.

Outputs: runs/boundary-motion-20261116/ (Git-ignored). Never commit these data,
predictions, reports or caches. New checkpoints, if produced, remain local there.
The preserved reference checkpoints and portable inference implementation remain
in models/possession-kalman-ensemble-20261116 and tiki_data/p2/selected*.py.
Each checkpoint embeds its feature schema and training-only normalization.
freeze.json records hashes of code, checkpoints and configurations.

For an incomplete training attempt, training_ledger.json is authoritative over a
child status.json left at RUNNING by process termination. Its best saved checkpoint
can be scored, but descriptive results cannot qualify it for model selection.
Neither seed43 nor A+B is claimed to have run unless its artifacts and ledger show
completion. The seventh IDSSE match was scored historically and is not sealed.
