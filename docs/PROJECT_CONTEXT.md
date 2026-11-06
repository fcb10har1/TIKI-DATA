# TIKI-DATA project context

P4 integration source of truth: [PRD.md](PRD.md). The competition rules and file
contract remain in the original top-level documents. P4's runbook is [P4.md](P4.md);
the executable P1–P3 handoff contract is [INTERFACES.md](INTERFACES.md).

The official starter and evaluator have not been edited. The new `tiki_data/`
package provides inference, validation, external comparisons, frozen holdout
evaluation, reproduction, and experiment indexing. `configs/baseline.json` is
the currently runnable production entry point once the restricted ATD arrives.
Other production stages must be supplied through the documented adapters or
artifacts. `tiki_data/demo.py` is explicitly synthetic test support.

Run `python -m unittest discover -s tests -v` in the pinned P4 environment.
Generated evidence lives in ignored `runs/` and `artifacts/`. P4's status and
outstanding team dependencies are recorded in [P4_STATUS.md](P4_STATUS.md).
