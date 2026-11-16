# Football possession experiment audit - 16 November 2026

This audit records the repository history and local experiment evidence reviewed before the authorized GitHub publication. Times below are Singapore time (UTC+08:00). It publishes source, methodology and aggregate results only. Supplied tracking, video, ground truth, private comparison CSVs, per-frame predictions, new checkpoints and original experiment records remain local.

## Outcome and important corrections

- The latest historical-backbone factorial completed **19 training runs**: 16 combinations at seed 42 and three confirmations at seed 43. **810 frozen recipes** were publicly scored, representing 672 distinct complete prediction files. No planned factorial run is incomplete.
- The **0.800 public SpellF1 target was not reached**. The highest public result remains the historical model's **0.657143**. The latest externally selected ensemble scores **0.583094 externally / 0.593103 publicly**.
- **J03WR9 is not a historically untouched test match.** `runs/full-20261111/run.json` and `test/report.json` show an earlier evaluation, with SpellF1 0.580556. In the latest study it was excluded from training and tuning, and evaluated after selection was frozen. Its confirmation result is valid as a study-held-out comparison, but not as a fresh sealed test. Earlier wording in the conversation and generated reports calling it "untouched" was incorrect. Frozen filenames and records are retained as evidence; this audit corrects their interpretation.
- **0.583094 is the winner within the latest historical-backbone study, not the highest external result across every earlier study.** The completed boundary-model ensemble previously achieved 0.604529 externally, with weaker public performance (0.482051).
- Full tracking context means using all available tracking for inference. It does **not** mean access to the competition's private full-match ground truth. Public scores cover only the supplied five-minute labelled window.
- The saved default submission/model remains the Kalman ensemble from commit `d7dfc09`. The later experiment winners were saved locally and were not promoted to that default.

## Repository foundation

| Singapore date | Commit | Work already saved |
|---|---|---|
| 7 Nov | `9cc83a3` | P4 integration and evaluation workflow |
| 9 Nov | `f5c40de`, `87c00ae`, `a222caa` | TCN training/inference and tracking-feature integration |
| 10-11 Nov | `e048323`, `e2b33dd`, `7e3c19e`, `de10e02` | Converted external matches and P1/P2 handoff |
| 11 Nov | `5ecdc2c`, `cd715a3` | End-to-end P1-P4 evaluation, semi-Markov decoder and synthetic tests |
| 11 Nov | `59b7276`, `d3ec37a` | External decoder/feature ablations, dead-ball diagnostic and observation-mask builder |
| 12 Nov | `6653d0d` | Speed masking and Kalman TCN comparisons |
| 16 Nov, 10:59 | `d7dfc09` | Saved Kalman ensemble and portable inference |

The publication containing this audit adds the subsequent research source without replacing those checkpoints or changing the default model.

## Experiment chronology

Scores belong to each study's stated protocol. Earlier cropped-context public results must not be mixed with later full-context comparisons. Evidence paths below are local and Git-ignored; they are identifiers, not files included in this publication.

| Study / Singapore time | What actually ran | Outcome and local evidence |
|---|---|---|
| Initial missingness/decoder study, 16 Nov 02:22 | Raw missing-ball augmentation, missing speed, decoder duration/transition/evidence experiments; four frozen variants | Both improvements selected externally. `runs/missingness-decoder-20261116/selection.json`, `external_comparison.json`, `public/comparison.json` |
| Private reference agreement | Compared historical and both-improvement predictions with the user-supplied reference, using aligned frame denominators | Agreement only, not ground-truth accuracy. The private CSV was explicitly ignored. `runs/majority-reference-comparison-20261116/agreement.json`, `verification.json` |
| Controlled follow-up, 06:23 | Nine completed runs: two-head, two boundary-weight variants, consistency, speed removal, speed differencing, Kalman speed, reference seed 43, Kalman seed 43; decoder and seed comparisons | No candidate passed all original acceptance/regression guards, so the reference was retained. `runs/possession-controlled-study-20261116/training_ledger.json`, `recommendation.json` |
| Broader programme, 10:26 | Cached decoder/ensemble and symmetry comparisons; ball-free/geometry-free attempts; external-data pilot | Saved Kalman ensemble external SpellF1 0.592973. Ball-free and geometry-free attempts failed; the normalization exception for intentionally unobserved channels was documented. Further programme work was cancelled by the user, not completed. `runs/broad-study-20261116/{recommendation,training_ledger,correction,interruption}.json` |
| Boundary / relative-motion screen, 11:08 | Reproduced saved reference; A42 boundary model and B42 relative-motion model attempted | A42 timed out after 30 epochs (best 25); B42 after 21 (best 19). They are incomplete descriptive screens. The parent ledger overrides stale child `RUNNING` status files. `runs/boundary-motion-20261116/training_ledger.json`, `training_summary.csv` |
| Boundary + relation combination, 12:33 | AB42 completed 34 epochs (best 26), with enabled/disabled boundary decoding controls | Enabled strength 2: external 0.583486 / public 0.452632. Mixed per-match changes failed the seed-43 repetition gate. Submission unchanged. `combined_followup/{repeat_decision,freeze}.json` and `models/AB42/status.json` under the boundary study |
| Short-spell weighting, 14:06 | SW42 completed 44 epochs (best 36) | External 0.596902 / public 0.475000. No seed-43 repeat: the original allocation was exhausted. `spell_weighting_followup/` under the boundary study |
| Full-context comparison, 14:49 | Inference-only rerun of 21 saved checkpoints / 42 recipe entries, yielding 39 distinct full prediction files | Historical missing-evidence weight 0: public SpellF1 0.657143, accuracy 84.3333%. Weight 0.5: 0.634483 / 90.12%. `runs/full-context-model-comparison-20261116/{freeze,results,verification,distinct_models}.json` |
| Completed A training, 15:07 | Removed the wall-time cutoff. Replayed seed 42 because its old best-only checkpoint lacked optimizer state; epoch-25 weights matched exactly. Added optimizer/RNG resume support | A42 completed 44 epochs, best 36; A43 completed 35, best 27. A42 enabled: external 0.600129 / public 0.518919. Ensemble: 0.604529 / 0.482051. `runs/a42-completed-training-20261116/{plan,resume_check,external_results,improvement_summary,verification}.json` |
| Historical-reference diagnosis | Aggregate-only diagnosis of the weight-0 and weight-0.5 references | Under-segmentation, short-possession misses, boundary errors and sensitivity to missing ball observations. No training. `runs/historical-two-reference-diagnosis-20261116/` |
| Historical factorial, protocol 15:55; final freeze 21:51 | Full declared factorial, second-seed checks, paired ensembles, external selection, one frozen public comparison batch | 19 complete training runs; 45 checkpoint/ensemble groups x 18 decoders = 810 recipes. External winner 0.583094; public target not achieved. `runs/historical-factorial-20261116/{protocol,training_matrix,confirmation_plan,freeze,summary}.json` |

### Initial four-variant study

External values average the two validation matches and six predefined conditions. Public values here use the **same five-minute tracking crop**, unlike the later full-context table.

| Variant | External accuracy | External macroF1 | External SpellF1 | Public accuracy | Public macroF1 | Public SpellF1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 83.7295% | 0.825861 | 0.446225 | 83.3333% | 0.810235 | 0.445714 |
| Augmentation only | 84.4117% | 0.832968 | 0.467326 | 89.0267% | 0.876272 | 0.551724 |
| Decoder only | 86.2191% | 0.848786 | 0.487262 | 83.0800% | 0.812288 | 0.468571 |
| Both | 88.3871% | 0.872482 | 0.529976 | 88.3467% | 0.870157 | 0.554839 |

A display-delta bug caused by sorted JSON keys choosing the wrong first row was corrected by explicitly selecting `baseline`. Raw first-pass results and `public/reporting_correction.json` were preserved; this was a reporting correction, not model retuning.

The separate private-reference comparison found within-half agreement of 84.5817% for historical 90.12 and 83.6331% for both improvements. Existing five-minute CSV agreement was 90.0000% and 86.5467%. These percentages measure agreement with the private reference, not accuracy against public labels. The reference file and comparison CSVs remain local.

## Latest factorial design

The current research source is in `experiments/boundary_motion/`. Its original `README.txt` describes the earlier capped study; see [the current source guide](../experiments/boundary_motion/AUDIT_GUIDE.md) for scope and execution caveats.

1. **Decoder comparison:** hard 1.0-second minimum, soft 1.0-second duration penalty, or hard 0.2-second minimum; transition penalties -1/0/+1; missing-ball model-evidence weights 0/0.5. Temperature remains 4. This gives 18 decoder settings.
2. **Checkpoint selection:** retain raw-argmax and decoded-SpellF1 checkpoints, selected on clean external validation. Compare both using the full external decoder grid.
3. **Boundary weighting:** double training loss within +/-0.4 seconds of valid external transitions. Possession targets remain unchanged.
4. **Missing-data/context factors:** remove raw ball observations before dependent features; point dropout 5%, interval probability 30%, interval length 25-250 frames. Compare 125-frame versus 253-frame receptive fields without increasing parameter count. All variants use the historical 19-feature, permanently speed-masked definition and observation masks; no observation-age/Kalman extension is silently added.
5. **External expansion:** four original training matches versus those four plus two SkillCorner matches. Added labels are partial provider-derived H/A supervision, admitted only when possession and explicit live-phase information agree. Unknown targets are ignored; no D labels are fabricated. Source: [SkillCorner Open Data](https://github.com/SkillCorner/opendata), [MIT licence](https://github.com/SkillCorner/opendata/blob/master/LICENSE). Downloads were verified against Git-LFS hashes and remain local.

Four binary training factors give 16 seed-42 arms. Raw/decoded selection and 18 decoders yield 576 main configurations. The historical reference, seed-43 runs for the two strongest recipes plus control, and paired ensembles bring the total to 810.

Whole-match split: four original training matches, two validation matches (J03WPY/J03WQQ), one study-held-out but historically evaluated match (J03WR9). Normalization is fitted on training matches only. Expanded arms sample 1,112 distinct windows / 70 optimizer steps per epoch, matching the controls. Maximum 60 epochs; stop once both checkpoint selectors are stale for eight epochs. There is no wall-time termination in this factorial. Matching windows does not equalize labelled-frame counts or total early-stopped updates.

All checkpoint hashes, recipes and the external selection rationale were frozen before this public batch. Full tracking was used to generate each prediction CSV before public labels were read. Public metrics did not change the selected checkpoint, decoder or training plan.

## Latest aggregate results

External values below are equal-match/equal-condition validation means under the latest study's six-condition objective. The public columns always score the same supplied five-minute labels using full tracking context.

| Recipe | External accuracy | External macroF1 | External SpellF1 | External D recall | Public accuracy | Public macroF1 | Public SpellF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Historical SpellF1 reference, missing weight 0 | 83.7789% | 0.826994 | 0.456889 | 0.824042 | 84.3333% | 0.786656 | **0.657143** |
| Historical 90.12% reference, missing weight 0.5 | 86.7674% | 0.855377 | 0.491945 | 0.893466 | **90.1200%** | 0.885878 | 0.634483 |
| Best new intervention, excluding exact replay: augmentation + longer context, raw checkpoint | 84.1336% | 0.830548 | 0.459044 | 0.831795 | 90.0933% | 0.889052 | 0.651852 |
| External-selected boundary + augmentation + expanded-data ensemble | **90.3237%** | **0.892856** | **0.583094** | **0.931218** | 88.1600% | 0.859484 | 0.593103 |

The best new-intervention row excludes the exact historical replay control, which ties 0.657143. This is a public diagnostic ranking, not the externally selected candidate. Machine-readable aggregates in [the snapshot directory](experiment_audit_2026-11-16/) provide authoritative full-precision values and one externally ranked setting per candidate group.

Selected candidate: `b1m1c0e1_ensemble_decoded_best`, seeds 42/43, original 125-frame context, hard 0.2-second minimum, transition penalty -1, missing evidence weight 0.5, temperature 4. Its validation SpellF1 values are 0.558939 and 0.607250. Compared with the historical 90.12% reference:

- External SpellF1 **+0.091149**, accuracy **+3.5563 percentage points**; mean missed possessions 243.75 -> 207.25 and extras 98.67 -> 74.00 at IoU 0.5.
- Public SpellF1 **-0.041379**, accuracy **-1.9600 points**. It still misses 8 of 17 public possessions at IoU 0.5.
- Study-held-out J03WR9: SpellF1 **0.628923 vs 0.586107**, accuracy **92.4590% vs 90.6839%**. Misses decrease 190 -> 166, but extras increase 69 -> 76. This is not a fresh-test claim.

A diagnostic recipe improving both validation matches relative to the 90.12% reference reaches external/public SpellF1 0.549655/0.642857, with public accuracy 87.7733%. It also does not beat the historical public SpellF1 leader, and it did not replace the frozen external selection.

### What helped, and what remains uncertain

Matched contrasts over the 16-arm factorial give mean SpellF1 changes of +0.01721 for decoded checkpoint selection, +0.01491 for boundary weighting, +0.01102 for missing augmentation, +0.00234 for longer context and +0.00220 for data expansion. Context and expansion help in roughly half of contrasts; expansion is not uniformly beneficial. These correlated comparisons on two repeatedly used validation matches are descriptive, not independent significance tests.

Clean and `speed_missing` are exact duplicates because speed is already masked. All 1,620 checked match/recipe pairs agree, so the original six-condition objective double-weights clean. A diagnostic five-condition calculation, without reselection, gives selected/reference SpellF1 **0.572737/0.481612**. The repeated public window and partially weak extra labels limit conclusions about unseen competition footage. Completed runs hitting the epoch maximum are complete under the plan, not proof of full convergence.

## Failures, recovery and verification

- Preserve the original A42/B42 timeout records separately from the later completed A runs. Their old child status markers are not completion evidence.
- Preserve the failed ball-free/geometry-free attempts and cancelled broad-programme stages; do not count them as successful experiments.
- A Windows atomic-replace failure interrupted `b1m0c1e1s42` after epoch 51. Its optimizer and all RNG states were intact. Training resumed at epoch 52 with unchanged mathematics/configuration after bounded rename retries were added.
- Public execution now uses atomic writes, immutable freeze checks, provenance/hash-checked probability caches, committed CSV hashes, and reuse of completed aggregate results. Synthetic interruption/resume tests passed. The separate continuation driver still has exclusive lock/log files; it must not be blindly restarted or cleaned up.
- Checkpoint reload validation passed for completed training. Compact scoring was checked against the official scorer. Expansion adapter, target invariance, masking, context parameter counts and sampler-resume checks passed.
- Before this publication, **7 synthetic pytest tests passed**, all **31 Python sources parsed**, factorial window/sampler checks passed, and the SkillCorner adapter self-test passed. No new training or public scoring was run for publication. An installed pytest-plugin deprecation warning did not affect the seven passing tests.
- A source/literal review found no embedded competition records, per-frame outputs, credentials, personal absolute paths or copied confidential documents in the 32 original experiment source/text files.
- Final local verification checked 2,646 generated files: ignored and untracked; protected checkpoints and frozen sources unchanged. The new publication audit/snapshots are deliberately separate from those original ignored records.

## Commands and publication boundary

The commands used by the final study, run from the repository root with its local prerequisites, are:

```text
python experiments/boundary_motion/historical_factorial.py baseline
python experiments/boundary_motion/skillcorner_expansion.py
python experiments/boundary_motion/historical_factorial.py train
python experiments/boundary_motion/historical_factorial.py finish
python experiments/boundary_motion/historical_factorial_report.py
```

The external adapter was validated before its readiness marker was created. `train` was resumed after the recorded epoch-51 I/O failure. The hidden continuation driver invoked `finish` and the report after all 16 arms completed. Exact study command files and frozen hashes remain under `runs/`; they must not be force-added to Git. The current source guide explains why a fresh clone alone cannot reproduce every historical run.

Publication scope: `experiments/boundary_motion/` source/tests, this audit, a small aggregate-only snapshot with hashes of its local evidence, and documentation links. No new weights, supplied assets, full original reports/manifests, caches, prediction CSVs, or `BAR-ATH_majority_FINAL.csv` are published. Original runtime flags saying "no commit/push" describe those completed experiment executions; this later publication is separately authorized by the user's audit-and-push request.
