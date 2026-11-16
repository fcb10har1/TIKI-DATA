"""Format completed reports without changing any model, prediction or selection."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/boundary-motion-20261116'


def main():
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['csv_validated'] and verification['repeat_identical']
    with (OUT/'results.csv').open(newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
    ledger={r['id']:r for r in json.loads((OUT/'training_ledger.json').read_text())}
    # The initial report put failed-job wall time in its evaluation column.
    # Correct the presentation from the preserved authoritative training ledger.
    for r in rows:
        if r['experiment'] in ledger and r['status']=='FAILED':
            job=ledger[r['experiment']]
            r['training_seconds']=job['training_seconds']
            r['evaluation_seconds']=job['evaluation_seconds']
    with (OUT/'results.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['Reporting addendum: aggregate comparison and interpretation',
        'This reporting-only step corrected failed-run runtime column placement from training_ledger.json. Frozen model/code/configuration assets, predictions and selection were not changed.',
        'Experiment | Status | spellF1 | macroF1 | Accuracy % | Dead-ball recall']
    for r in rows:
        if r['spellF1']:
            lines.append(' | '.join([r['experiment'],r['status'],f"{float(r['spellF1']):.6f}",f"{float(r['macroF1']):.6f}",f"{float(r['accuracy_pct']):.4f}",f"{float(r['dead_ball_recall']):.6f}"]))
    lines += [
        'INCOMPLETE scores are descriptive; no new training attempt completed its original stopping rule. A finished 30 epochs (best25), B21 (best19); both reached the 900-second allocation plus20-second supervisor allowance. Seed43 and A+B were not run.',
        'Against the matching seed42 reference, auxiliary-head training with boundary switching disabled improved spellF1 by0.009288 and accuracy by0.5213 percentage points. SpellF1 gains occurred on both matches (+0.007580/+0.010996) and five of six conditions. Enabling boundary switching reduced this checkpoint score: strengths0.5/1/2 yielded0.592287/0.592403/0.586546 versus0.594342 disabled. No paired-seed or completed-training benefit is established.',
        'Relative motion reduced spellF1 by0.025089 and accuracy by0.0479 percentage points against matching seed42. It regressed on both matches (-0.043086/-0.007092) and all six condition means.',
        'External error diagnostics (means over12 cases, not independent samples): reference seed42 had8.67 split possessions,45.00 merges,199.08 missed and82.67 extra possessions. A-disabled had13.33 splits,44.50 merges,193.33 missed and85.92 extras: fewer missed spells but more fragmentation. B had17.42 splits,45.00 merges,205.50 missed and102.92 extras, with wrong-team frames rising from8447.75 to9175.58. Matched boundary start/end MAE rose from1.025/1.263 seconds to1.071/1.431 seconds for B. These use IoU0.5 diagnostics, not the multi-threshold headline metric.',
        'Public reference and selected model are identical: accuracy81.00%, macroF1 0.756492, spellF1 0.448780, dead-ball recall0.434309. Changes are zero. This post-freeze diagnostic exposes an external-to-public generalization gap; it does not authorize retuning or reselection.',
        'Recommendation: retain the preserved two-seed reference. A future separately budgeted study could complete auxiliary-head training and both seeds; boundary modulation itself did not help in this run. Only two repeatedly used validation matches remain available, and synthetic corruptions are not independent games.',
        'Detailed cells and clean/corrupted comparisons: per_match_condition.csv, per_match_means.csv and condition_means.csv. Live/dead confusion and per-class precision/recall: live_dead_confusion.csv. Commands, configurations, normalization, source/weight hashes and provenance are saved alongside these tables.'
    ]
    with (OUT/'REPORT.txt').open('a') as handle:handle.write('\n\n'+'\n\n'.join(lines)+'\n')
    with (OUT/'commands.txt').open('a') as handle:handle.write('\npython experiments/boundary_motion/report_addendum.py\n')
    freeze=json.loads((OUT/'freeze.json').read_text())
    assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in freeze['source_hashes']+freeze['configuration_hashes']+freeze['selected_checkpoints']+freeze['reference_checkpoints'])
    audit=OUT/'final_reporting_audit.json'
    paths={p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    checked=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',text=True,capture_output=True,check=True)
    assert set(checked.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(OUT.relative_to(ROOT))],cwd=ROOT)
    audit.write_text(json.dumps(dict(reporting_only=True,frozen_assets_unchanged=True,generated_files_ignored=True,
        tracked_generated_files=0,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('Reporting addendum saved; frozen assets unchanged; all generated outputs ignored and untracked.')


if __name__=='__main__':main()
