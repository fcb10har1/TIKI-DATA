"""One controlled loss-weighting run, training-fit audit, external and public tests."""
from common import *
import subprocess
from budget import ledger
from feature_engine import Factory
from models import load_model
from decoder import BASE,decode
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.model import seed_all
from tiki_data.artifacts import write_predictions

DEST=OUT/'spell_weighting_followup'
MODEL='SW42'


def train_and_evaluate(config):
    items=ledger();limit=read_json(OUT/'manifest.json')['training_budget']
    assert len([r for r in items if r['status']!='NOT RUN'])<limit['max_new_runs'],'Training run cap exhausted'
    assert sum(r['training_seconds'] for r in items)+limit['per_run_seconds']<=limit['cumulative_seconds']
    assert not any(r['id']==MODEL for r in items)
    config_path=OUT/'run_configs'/(MODEL+'.json');write_json(config_path,config,exclusive=True)
    result=dict(id=MODEL,config=config,status='FAILED',training_seconds=0.,evaluation_seconds=0.)
    started=time.perf_counter()
    try:
        with (OUT/'logs'/f'{MODEL}_train.log').open('x') as handle:
            subprocess.run([sys.executable,str(SOURCE/'spell_weighting.py'),str(config_path)],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=limit['per_run_seconds']+20)
        result['status']='COMPLETE'
    except Exception as error:
        result['reason']=type(error).__name__+': '+str(error)
    result['training_seconds']=time.perf_counter()-started
    folder=OUT/'models'/MODEL
    if (folder/'status.json').exists():result['training_status']=read_json(folder/'status.json')
    items.append(result);write_json(OUT/'training_ledger.json',items)
    if not (folder/'best.pt').exists():raise RuntimeError('No saved checkpoint; see preserved training log')
    if result['status']!='COMPLETE':
        assert 'Timeout' in (result.get('reason','')+' '+result.get('training_status',{}).get('error','')),'Training failed for a non-budget reason'
    tick=time.perf_counter()
    try:
        with (OUT/'logs'/f'{MODEL}_evaluate.log').open('x') as handle:
            subprocess.run([sys.executable,str(SOURCE/'train_model.py'),'evaluate',MODEL],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=360)
        result['evaluation_seconds']=time.perf_counter()-tick
        result['evaluation_status']='COMPLETE'
    except Exception as error:
        result['evaluation_status']='FAILED';result['evaluation_error']=str(error)
        items[-1]=result;write_json(OUT/'training_ledger.json',items);raise
    items[-1]=result;write_json(OUT/'training_ledger.json',items)
    return result


def predict(features,checkpoint,visible):
    model,saved=load_model(checkpoint)
    probability=predict_batch(features,model,saved['normalization'],saved['core_frames']).probabilities
    return decode(probability,visible,features.match,BASE)


def training_fit(checkpoints):
    seed_all(42,1);data=external_data()['train'];rows=[]
    truth_by_id={e['match']['match_id']:e for e in external_entries(('train',))}
    for base,_ in data:
        raw=raw_load(base.match.match_id);features=Factory(base,raw,'kalman').clean()
        truth=read_predictions(truth_by_id[base.match.match_id]['ground_truth'],base.match)
        for name,path in checkpoints.items():
            labels=predict(features,path,np.isfinite(raw['ball_positions']).all(1))
            rows.append(dict(candidate=name,match_id=base.match.match_id,scenario='clean',**metrics(truth,labels)))
    csv_write(DEST/'training_per_match.csv',rows)
    summary=[dict(candidate=name,**summarize([r for r in rows if r['candidate']==name])) for name in checkpoints]
    write_json(DEST/'training_fit.json',dict(regime='Clean in-sample training matches; optimistic fit diagnostic, never used for selection',rows=rows,summary=summary),exclusive=True)
    print(json.dumps(dict(training_fit_aggregate=summary)),flush=True)
    return summary


def public(checkpoints,frozen):
    from games import GAME,GAMES
    from tiki_data.p1.metrica import build_metrica_features
    from tiki_data.p1.kalman import metrica_positions
    from tiki_data.p2.selected_features import build_features
    directory=DEST/'public';directory.mkdir(exist_ok=False)
    write_json(directory/'started.json',dict(utc=now(),freeze=file_record(DEST/'freeze.json')),exclusive=True)
    for record in frozen['checkpoints'].values():assert file_record(record['path'])['sha256']==record['sha256']
    seed_all(42,1);spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25);pitch=GAMES[GAME]['pitch']
    meta=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.xml'
    tracking=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.txt'
    base=build_metrica_features(spec,meta,tracking,pitch);ball=metrica_positions(spec,tracking,pitch)
    features=build_features(base,ball);visible=np.isfinite(ball).all(1)
    predictions={name:predict(features,path,visible) for name,path in checkpoints.items()}
    for name,labels in predictions.items():write_predictions(directory/(name+'.csv'),PredictionBatch(spec,base.frames,labels).validate(spec))
    repeated=predict(features,checkpoints['spell_weighted_seed42'],visible)
    assert np.array_equal(predictions['spell_weighted_seed42'],repeated),'Repeated inference changed labels'
    csv_path=directory/'spell_weighted_seed42.csv'
    check=subprocess.run([sys.executable,'evaluation/validate_submission.py',str(csv_path)],cwd=ROOT,capture_output=True,text=True)
    (directory/'validator.log').write_text(check.stdout+check.stderr);assert check.returncode==0,'CSV validation failed; see local log'
    # Prediction choices are complete and frozen before reading public labels.
    truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
    rows=[dict(candidate=name,**metrics(truth,labels)) for name,labels in predictions.items()]
    old=read_json(OUT/'public/comparison.json')['rows'][0]
    rows.append(dict(candidate='reference_ensemble',**{k:old[k] for k in METRICS}))
    write_json(directory/'comparison.json',dict(rows=rows,diagnostic_only=True,selected_unchanged='reference'),exclusive=True)
    csv_write(directory/'comparison.csv',rows)
    write_json(directory/'verification.json',dict(official_validator_passed=True,rows=spec.n_frames,repeat_identical=True,
        csv=file_record(csv_path)),exclusive=True)
    print(json.dumps(dict(public_aggregate_metrics=rows)),flush=True)
    return rows


def main():
    DEST.mkdir(exist_ok=False)
    prior=read_json(OUT/'freeze.json')
    protected=prior['source_hashes']+prior['configuration_hashes']+prior['selected_checkpoints']+prior['reference_checkpoints']
    protected += [file_record(p) for p in [OUT/'freeze.json',OUT/'REPORT.txt',OUT/'public/selected_full_match.csv',
        OUT/'models/A42/best.pt',OUT/'models/B42/best.pt',OUT/'models/AB42/best.pt']]
    active=ROOT/'runs/active_possession_candidate.json'
    if active.exists():protected.append(file_record(active))
    for rec in protected:assert file_record(rec['path'])['sha256']==rec['sha256']
    cfg=dict(id=MODEL,architecture='tcn',speed='kalman',boundary_radius=0,boundary_weight=1.,consistency=0.,seed=42,spell_weight_cap=4.)
    manifest=read_json(OUT/'manifest.json')
    protocol=dict(created_utc=now(),configuration=cfg,decoder=BASE,training_budget=manifest['training_budget'],
        prior_ledger=ledger(),split=manifest['splits'],scenarios=SCENARIOS,corruption_seed=manifest['corruption_seed'],
        weighting='Full uncensored H/A spells: clip(sqrt(training-only median complete live spell duration / duration),1,4). D/censored=1, unknown=0. Computed before windowing.',
        training_fit='Clean training matches scored only after checkpoint selection; not generalization evidence',
        selection='Unchanged clean external validation raw-TCN spellF1 then macroF1. Fixed decoder; no parameter search.',
        confirmation='Final remaining run slot: seed43 unavailable under original four-run cap. Reference remains selected.',
        public='One post-freeze diagnostic batch; no tuning or model switch',
        source=[file_record(SOURCE/name) for name in ('spell_weighting.py','spell_experiment.py','test_spell_weighting.py')],
        preserved=protected)
    write_json(DEST/'protocol.json',protocol,exclusive=True)
    (DEST/'commands.txt').write_text('python -m pytest experiments/boundary_motion/test_spell_weighting.py -q -p no:cacheprovider\npython experiments/boundary_motion/spell_experiment.py\nPYTHONPATH includes existing runs/missingness-decoder-20261116/deps\n')
    (DEST/'tests.log').write_text((OUT/'spell_weighting_tests.log').read_text())
    print('Starting fixed capped spell-weighting seed42 experiment; final allocated training slot.',flush=True)
    job=train_and_evaluate(cfg);write_json(DEST/'training_run.json',job,exclusive=True)
    result=read_json(OUT/'models'/MODEL/'evaluation.json')
    result['status']='COMPLETE' if job['status']=='COMPLETE' else 'INCOMPLETE'
    baseline=read_json(OUT/'single_seed_reference.json')
    write_json(DEST/'external_results.json',dict(reference_seed42=baseline,candidate=result),exclusive=True)
    csv_write(DEST/'external_per_match_condition.csv',baseline['rows']+result['rows'])
    external=[dict(candidate=name,**r['mean']) for name,r in [('reference_seed42',baseline),('spell_weighted_seed42',result)]]
    csv_write(DEST/'external_summary.csv',external)
    checkpoints=dict(reference_seed42=ROOT/'models/possession-kalman-ensemble-20261116/model_seed42.pt',spell_weighted_seed42=OUT/'models'/MODEL/'best.pt')
    fit=training_fit(checkpoints)
    frozen=dict(created_utc=now(),checkpoints={name:file_record(path) for name,path in checkpoints.items()},
        decoder=BASE,weighting=file_record(OUT/'models'/MODEL/'spell_weighting.json'),protocol=file_record(DEST/'protocol.json'),
        external_results=file_record(DEST/'external_results.json'),training_status=result['status'],selected_unchanged='reference',
        reason='Single seed only; original run cap exhausted. Public and training-fit scores cannot select a replacement.')
    write_json(DEST/'freeze.json',frozen,exclusive=True)
    print(json.dumps(dict(external_aggregate_metrics=external,training_status=result['status'],public_settings_frozen=True)),flush=True)
    pub=public(checkpoints,frozen)
    table=[]
    for regime,rows in [('training_clean_in_sample',fit),('validation_six_conditions',external),('public_five_minutes',pub)]:
        table += [dict(regime=regime,**r) for r in rows]
    for name,r in [('reference_seed42',baseline),('spell_weighted_seed42',result)]:
        table.append(dict(regime='validation_clean',candidate=name,**summarize([x for x in r['rows'] if x['scenario']=='clean'])))
    csv_write(DEST/'comparison.csv',table)
    report=dict(method=protocol,training=job,weighting=read_json(OUT/'models'/MODEL/'spell_weighting.json'),comparison=table,
        external_spell_change=result['mean']['spellF1']-baseline['mean']['spellF1'],
        external_accuracy_change_pp=result['mean']['accuracy_pct']-baseline['mean']['accuracy_pct'],
        limitations='Training-fit scores are optimistic. Only two reused validation matches; no independent test; one seed. Weighting may increase false live spells. Incomplete training, if present, is descriptive only.',
        selected_unchanged='reference',committed=False,pushed=False)
    (DEST/'REPORT.txt').write_text(json.dumps(report,indent=2))
    for rec in protected+protocol['source']:assert file_record(rec['path'])['sha256']==rec['sha256']
    audit=DEST/'verification.json'
    paths={p.relative_to(ROOT).as_posix() for directory in (DEST,OUT/'models'/MODEL,OUT/'model_predictions'/MODEL) for p in directory.rglob('*') if p.is_file()}
    paths.add(audit.relative_to(ROOT).as_posix())
    checked=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',text=True,capture_output=True,check=True)
    assert set(checked.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(OUT.relative_to(ROOT))],cwd=ROOT)
    write_json(audit,dict(generated_files_ignored=True,tracked_generated_files=0,previous_assets_unchanged=True,
        selected_unchanged='reference',committed=False,pushed=False),exclusive=True)


if __name__=='__main__':main()
