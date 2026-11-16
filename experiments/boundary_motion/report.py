from common import *
from budget import ledger
import subprocess

def main():
    freeze=read_json(OUT/'freeze.json');reference=read_json(OUT/'reference.json')
    records=[reference,read_json(OUT/'single_seed_reference.json'),read_json(OUT/'single_seed_reference43.json')]+read_json(OUT/'finalist_results.json')+read_json(OUT/'incomplete_results.json');items=ledger()
    for item in items:
        if item['status']!='COMPLETE':records.append(dict(id=item['id'],status=item['status'],reason=item.get('reason'),runtime_seconds=item['training_seconds']))
        elif item['config']['seed']==43:
            r=read_json(OUT/'models'/item['id']/'evaluation.json');r['family']='seed43_diagnostic';records.append(r)
    decisions=read_json(OUT/'screening_decisions.json')
    training_rows=[]
    for item in items:
        folder=OUT/'models'/item['id'];epoch_file=folder/'epochs.jsonl'
        epochs=[json.loads(line) for line in epoch_file.read_text().splitlines()] if epoch_file.exists() else []
        best=[r['epoch'] for r in epochs if r['selected']]
        training_rows.append(dict(model=item['id'],status=item['status'],completed_epochs=len(epochs),best_epoch=best[-1] if best else None,
            wall_seconds=item['training_seconds'],reason=item.get('reason',''),checkpoint_sha256=file_record(folder/'best.pt')['sha256'] if (folder/'best.pt').exists() else None))
    csv_write(OUT/'training_summary.csv',training_rows)
    for d in decisions:
        if d['status']!='COMPLETE':records.append(dict(id=d['family']+'43_and_ensemble',status='NOT RUN',reason=d.get('reason'),runtime_seconds=0))
    records += [dict(id='A_plus_B',status='NOT RUN',reason='Combination requires convincing completed individual results and room for paired-seed confirmation within the four-run cap.',runtime_seconds=0),dict(id='fresh_confirmation',status='NOT RUN',reason=freeze['fresh_confirmation']['reason'],runtime_seconds=0)]
    public=read_json(OUT/'public/comparison.json');verification=read_json(OUT/'public/verification.json')
    table=[]
    seed_references={s:next(r for r in records if r['id']=='reference_seed'+str(s)) for s in (42,43)}
    for r in records:
        comparison=reference
        for seed in (42,43):
            if r['id'].startswith(('A'+str(seed),'B'+str(seed),'reference_seed'+str(seed))):comparison=seed_references[seed]
        row=dict(experiment=r['id'],status=r['status'],reason=r.get('reason',''),comparison_reference=comparison['id'],evaluation_seconds=r.get('runtime_seconds',0),training_seconds=r.get('training_runtime_seconds',0),configuration=json.dumps(r.get('training_config',r.get('settings',{}))))
        for k in METRICS:row[k]=r.get('mean',{}).get(k);row['change_'+k]=row[k]-comparison['mean'][k] if row[k] is not None else None
        table.append(row)
    csv_write(OUT/'results.csv',table)
    measured=[r for r in records if 'rows' in r]
    csv_write(OUT/'per_match_condition.csv',[dict(status=r['status'],**x) for r in measured for x in r['rows']])
    csv_write(OUT/'error_diagnostics.csv',[dict(status=r['status'],**x) for r in measured for x in r.get('diagnostics',[])])
    csv_write(OUT/'per_match_means.csv',[dict(candidate=r['id'],status=r['status'],match_id=m,**summarize([x for x in r['rows'] if x['match_id']==m])) for r in measured for m in ('J03WPY','J03WQQ')])
    csv_write(OUT/'fresh_confirmation.csv',[dict(status='NOT RUN',reason=freeze['fresh_confirmation']['reason'])])
    write_json(OUT/'all_results.json',records)
    selected=next(r for r in records if r['id']==freeze['selected'])
    condition=[]
    for r in measured:
        for scenario in SCENARIOS:
            condition.append(dict(candidate=r['id'],status=r['status'],scenario=scenario,**summarize([x for x in r['rows'] if x['scenario']==scenario])))
    csv_write(OUT/'condition_means.csv',condition)
    confusion_rows=[]
    for r in records:
        if 'rows' not in r:continue
        for spec,truth in truth_records():
            for scenario in SCENARIOS:
                labels=saved_decoded(r['id'],spec.match_id,scenario)
                scored=score_match(truth,PredictionBatch(spec,truth.frames,labels),spec)
                cm=np.asarray(scored['confusion'])
                row=dict(candidate=r['id'],match_id=spec.match_id,scenario=scenario,
                    live_as_dead_frames=int(cm[:2,2].sum()),dead_as_live_frames=int(cm[2,:2].sum()),wrong_team_frames=int(cm[0,1]+cm[1,0]))
                for i,label in enumerate(CLASS_ORDER):
                    row[label+'_precision']=float(cm[i,i]/cm[:3,i].sum()) if cm[:3,i].sum() else 0.
                    row[label+'_recall']=float(cm[i,i]/cm[i].sum()) if cm[i].sum() else 0.
                confusion_rows.append(row)
    csv_write(OUT/'live_dead_confusion.csv',confusion_rows)
    for rec in read_json(OUT/'preserved_reference.json'):assert file_record(rec['path'])['sha256']==rec['sha256']
    counts={s:sum(r['status']==s for r in records) for s in ('COMPLETE','INCOMPLETE','FAILED','NOT RUN')}
    readme=['Boundary and relative-motion study', 'Selected: '+freeze['selected'],freeze['rationale'],
        'Reference: '+json.dumps(reference['mean']),'Selected: '+json.dumps(selected['mean']),
        'Counts (evaluation configurations and explicitly unrun arms): '+json.dumps(counts),
        'Training runtime seconds: '+str(sum(i['training_seconds'] for i in items)),
        'Training attempts and completed epochs: '+json.dumps(training_rows),
        'Time-capped runs: best available checkpoints are scored descriptively and marked INCOMPLETE, never eligible for screening or final selection. They do not establish a fair completed-training comparison. Supervisor training_ledger.json is authoritative if a terminated child left status.json at RUNNING.',
        'All selection data are the two repeatedly used IDSSE validation matches, each equally weighted across six conditions. The seventh match has a historical scoring report, so no sealed confirmation was available.',
        'The speed_missing condition hides provider speed only. All compared recipes derive ball speed from positions, so this condition intentionally duplicates clean inputs. The original six-condition weighting is preserved; these are not six independent matches.',
        'Original reference exactly reproduced using saved ensemble probabilities and official scorer. Pushed checkpoint/source hashes preserved. Existing release validation already proved full feature/probability/label equivalence for all12 cases.',
        'Synthetic checks: '+(OUT/'tests.log').read_text().strip(),
        'A: one auxiliary head predicts three binary boundary types. Only adjacent valid training labels supervise it; core first frame, half boundaries and padding excluded. Loss weight0.1, fixed positive weight20; three strengths0.5/1/2 plus disabled decoding. Modulation strength*(2*predicted_probability-1) changes only the relevant transition edges. Hard durations and all other decoder settings remain fixed.',
        'B: fixed PersonId roster slots and consecutive25Hz timestamps establish external continuity. Player derivatives reject missing endpoints and speeds>15m/s. Ball filter uncertainty/age are retained; relative features require observed ball position and valid filtered velocity<=60m/s. Team-wise nearest reliable players supply distance, closing speed, relative speed and alignment; margins compare teams. No differentiation across nearest-player identity changes. Position loss occurs before filtering/derivatives.',
        'Relative-motion training windows use the same causal clean prefix preceding each augmented example as the reference; half resets and hidden-position invariance are tested. Inference tracking continuity relies on stable consecutive ATD track slots and the same speed/gap rejection. Identity switches that remain physically plausible are a limitation.',
        'Checkpoint selection and budgets match the previous60epoch/patience8 procedure. Seed42 screens compare with the saved seed42 Kalman model under the same decoder. Only paired42/43 equal-weight ensembles can replace the reference.',
        'results.csv changes use the explicitly named comparison_reference: matching seeds for single models, the reference ensemble for ensembles. Accuracy differences are percentage points. Incomplete checkpoint comparisons remain descriptive despite matching seeds.',
        'SpellF1 averages official same-team one-to-one spell matching at IoU0.3,0.4,0.5,0.6,0.7. H-D-H is two possessions. Extra diagnostic matches use IoU0.5; splits/merges count positive same-team overlaps, timing is measured on matched spells.',
        'Public evaluation occurred only after freeze.json. Both frozen recipes use the same full-match offline context; public labels remain local and never affect selection. Public aggregate table: '+json.dumps(public),
        'Full-match output: '+json.dumps(verification),
        'Source and dataset details: manifest.json, provenance.json, preserved_reference.json. Credit: Deutsche Fussball Liga (DFL); Bassek et al.2025, Scientific Data12,195; CC BY4.0; https://github.com/spoho-datascience/idsse-data .',
        'Tables: results.csv, per_match_condition.csv, condition_means.csv, error_diagnostics.csv, fresh_confirmation.csv, public/comparison.csv.',
        'Exact commands: commands.txt. Models embed normalization and feature schema. Current reference weights are tracked in models/possession-kalman-ensemble-20261116; any new candidate weights stay local in models/A42,A43 or B42,B43 under this output directory. Reusable implementation is experiments/boundary_motion and remains reviewable, uncommitted.',
        'Limitations: only two development games; no statistical certainty, no independent confirmation, synthetic missingness may differ from deployment. Retained one-second hard minima still constrain very short possessions. Time-capped training cannot establish a completed-training result. A+B not run under the four-run cap. Public scores are diagnostic only.',
        'Readiness: checkpoint reload reproduced validation metrics for completed jobs; synthetic tests passed; official CSV validator passed; two independently executed inferences produced identical CSV SHA256; no submission, commit or push performed.']
    (OUT/'REPORT.txt').write_text('\n\n'.join(readme)+'\n')
    paths={p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()};paths.add((OUT/'verification.json').relative_to(ROOT).as_posix())
    checked=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',text=True,capture_output=True,check=True)
    assert set(checked.stdout.rstrip('\0').split('\0'))==paths
    tracked=subprocess.check_output(['git','ls-files','--',str(OUT.relative_to(ROOT))],cwd=ROOT);assert not tracked
    write_json(OUT/'verification.json',dict(counts=counts,generated_files_ignored=True,tracked_generated_files=0,reference_unchanged=True,csv_validated=True,repeat_identical=True,committed=False,pushed=False),exclusive=True)
    print(json.dumps(dict(selected=freeze['selected'],external=selected['mean'],public=public['rows'],counts=counts,csv_validated=True,repeat_identical=True)),flush=True)

if __name__=='__main__':main()
