"""Explicitly requested A+B follow-up; external selection, local diagnostics only."""
from common import *
import subprocess
from budget import run_one,ledger
from train_model import evaluate_cached
from boundary import evaluate_boundary,boundary_probabilities,decode_boundary
from decoder import BASE,decode
from models import load_model,fresh_model,classification_loss
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.model import seed_all
from tiki_data.artifacts import write_batch,write_predictions
from tiki_data.contracts import ProbabilityBatch

DEST=OUT/'combined_followup'


def attempt(name,seed):
    result=run_one(dict(id=name,architecture='boundary',speed='relative',boundary_radius=0,
        boundary_weight=1.,consistency=0.,seed=seed))
    folder=OUT/'models'/name
    if result['status']!='COMPLETE' and (folder/'best.pt').exists():
        child=read_json(folder/'status.json')
        if 'Timeout' not in (result.get('reason','')+' '+child.get('error','')):
            return result,[]
        with (OUT/'logs'/(name+'_partial_evaluate.log')).open('x') as handle:
            subprocess.run([sys.executable,str(SOURCE/'train_model.py'),'evaluate',name],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=360)
    if not (folder/'evaluation.json').exists():return result,[]
    rows=[read_json(folder/'evaluation.json')]+read_json(folder/'boundary_grid.json')
    for r in rows:
        r['training_status']=result['status']
        r['status']='COMPLETE' if result['status']=='COMPLETE' else 'INCOMPLETE'
        r['seed_count']=1;r['training_seconds']=result['training_seconds']
    return result,rows


def public(frozen):
    from public_full import raw_players
    from games import GAME,GAMES
    from tiki_data.p1.metrica import build_metrica_features
    from tiki_data.p1.kalman import metrica_positions
    from feature_engine import Factory
    directory=DEST/'public';directory.mkdir(exist_ok=False)
    write_json(directory/'started.json',dict(utc=now(),freeze=file_record(DEST/'freeze.json')),exclusive=True)
    seed_all(42,1)
    spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25);pitch=GAMES[GAME]['pitch']
    metadata=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.xml'
    tracking=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.txt'
    base=build_metrica_features(spec,metadata,tracking,pitch)
    ball=metrica_positions(spec,tracking,pitch)
    features=Factory(base,raw_players(spec,metadata,tracking,pitch,ball),'relative').clean()
    probabilities=[];boundaries=[]
    for rec in frozen['checkpoints']:
        assert file_record(rec['path'])['sha256']==rec['sha256']
        model,saved=load_model(rec['path'])
        probabilities.append(predict_batch(features,model,saved['normalization'],saved['core_frames']).probabilities)
        boundaries.append(boundary_probabilities(features,model,saved['normalization'],saved['core_frames']))
    p=np.mean(probabilities,axis=0);b=np.mean(boundaries,axis=0);visible=np.isfinite(ball).all(1)
    labels={'AB_boundary_disabled':decode(p,visible,spec,BASE),
        'AB_boundary_enabled':decode_boundary(p,visible,spec,BASE,b,frozen['enabled_strength'])}
    for name,values in labels.items():
        write_predictions(directory/(name+'.csv'),PredictionBatch(spec,base.frames,values).validate(spec))
    # All outputs, weights and decoder settings are fixed before this read.
    truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
    prior=read_json(OUT/'public_ab_diagnostic/comparison.json')['rows']
    rows=[dict(candidate=name,**metrics(truth,values)) for name,values in labels.items()]
    for row in rows:
        for key in METRICS:row['change_'+key]=row[key]-prior[0][key]
    write_json(directory/'comparison.json',dict(rows=rows,previous_frozen_comparison=prior,
        diagnostic_only=True,selected_unchanged='reference'),exclusive=True)
    csv_write(directory/'comparison.csv',prior+rows)
    print(json.dumps(dict(public_aggregate_metrics=rows)),flush=True)
    return rows


def main():
    DEST.mkdir(exist_ok=False)
    previous=read_json(OUT/'freeze.json')
    protected=previous['source_hashes']+previous['configuration_hashes']+previous['selected_checkpoints']+previous['reference_checkpoints']
    protected += [file_record(p) for p in [OUT/'models/A42/best.pt',OUT/'models/B42/best.pt',OUT/'REPORT.txt',
        OUT/'public/selected_full_match.csv',OUT/'public_ab_diagnostic/comparison.json',OUT/'freeze.json']]
    active=ROOT/'runs/active_possession_candidate.json'
    if active.exists():protected.append(file_record(active))
    for rec in protected:assert file_record(rec['path'])['sha256']==rec['sha256']
    manifest=read_json(OUT/'manifest.json');old_ledger=ledger()
    protocol=dict(created_utc=now(),authorization='User explicitly requested combined boundary switching and relative-motion model',
        status='Exploratory follow-up; previously planned A+B arm now explicitly requested after public diagnostics',
        training_budget=manifest['training_budget'],prior_ledger=old_ledger,splits=manifest['splits'],scenarios=SCENARIOS,
        corruption_seed=manifest['corruption_seed'],loss_weight=.1,strengths=[0.,.5,1.,2.],
        seed42_gate='Completed run; positive mean spellF1 against matched seed42 and no per-match regression over0.005',
        seed43='Only after passing seed42 gate; same settings, average probabilities equally; no further decoder search',
        public='Freeze external-best enabled strength and report it plus disabled ablation; no public-based tuning or adoption',
        preserved=protected,source=file_record(__file__),command='python experiments/boundary_motion/combined.py')
    write_json(DEST/'protocol.json',protocol,exclusive=True)
    (DEST/'command.txt').write_text(protocol['command']+'\nExisting experiment dependencies use PYTHONPATH=runs/missingness-decoder-20261116/deps\n')
    # Verify the joint feature width/head combination before starting its timed worker.
    import torch
    seed_all(42,1);cfg=read_json(OUT/'baseline/training_config.json')['model']
    model=fresh_model(34,cfg,'boundary','relative');core=32
    x=torch.randn(2,68,core+2*model.radius);y=torch.zeros((2,core),dtype=torch.long);y[:,16:]=2
    loss,_=classification_loss(model,x,y,model.radius,core);loss.mean().backward()
    assert model.boundary.weight.grad.abs().sum()>0
    assert model.network[0].weight.grad[:,21:34].abs().sum()>0
    write_json(DEST/'smoke_test.json',dict(combined_forward_backward=True,boundary_and_motion_gradients=True),exclusive=True)
    del model,x,y,loss
    print('Combined architecture verified; starting external-only AB42 training.',flush=True)
    job,results=attempt('AB42',42)
    if not results:
        write_json(DEST/'failure.json',dict(training=job,reason='No evaluable combined checkpoint'),exclusive=True)
        raise RuntimeError('Combined run produced no evaluable checkpoint; preserved logs identify failure')
    write_json(DEST/'seed42_results.json',results,exclusive=True)
    baseline=read_json(OUT/'single_seed_reference.json')
    best=max(results,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
    enabled=max([r for r in results if r.get('strength',0)>0],key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
    changes={m:summarize([r for r in best['rows'] if r['match_id']==m])['spellF1']-summarize([r for r in baseline['rows'] if r['match_id']==m])['spellF1'] for m in ('J03WPY','J03WQQ')}
    repeat=job['status']=='COMPLETE' and best['mean']['spellF1']>baseline['mean']['spellF1'] and min(changes.values())>=-.005
    decision=dict(repeat_seed43=repeat,seed42_training_status=job['status'],overall_strength=best.get('strength',0.),
        enabled_strength=enabled['strength'],per_match_spell_changes=changes,reason='Predeclared external-only completion and improvement gate')
    write_json(DEST/'repeat_decision.json',decision,exclusive=True)
    checkpoints=[OUT/'models/AB42/best.pt'];public_results=results
    if repeat:
        job43,rows43=attempt('AB43',43);results+=rows43
        if job43['status']=='COMPLETE':
            root=OUT/'ensemble_predictions/AB'
            for spec,_ in truth_records():
                for scenario in SCENARIOS:
                    paths=[OUT/'model_predictions'/name/spec.match_id/scenario for name in ('AB42','AB43')]
                    members=[read_batch(path/'probabilities.npz',spec,'probabilities') for path in paths]
                    write_batch(root/spec.match_id/scenario/'probabilities.npz',ProbabilityBatch(spec,members[0].frames,
                        (members[0].probabilities+members[1].probabilities)/2,members[0].periods))
                    np.save(root/spec.match_id/scenario/'boundary.npy',(np.load(paths[0]/'boundary.npy')+np.load(paths[1]/'boundary.npy'))/2)
            public_results=[evaluate_cached(root,'AB_ensemble_disabled',BASE),evaluate_boundary(root,'AB_ensemble_enabled',BASE,enabled['strength'])]
            for r in public_results:r['seed_count']=2;r['training_status']='COMPLETE'
            results+=public_results;checkpoints.append(OUT/'models/AB43/best.pt')
    write_json(DEST/'external_results.json',results,exclusive=True)
    csv_write(DEST/'external_per_match_condition.csv',[dict(status=r['status'],**x) for r in results for x in r['rows']])
    csv_write(DEST/'external_summary.csv',[dict(candidate=r['id'],status=r['status'],**r['mean']) for r in results])
    frozen=dict(created_utc=now(),checkpoints=[file_record(p) for p in checkpoints],enabled_strength=enabled['strength'],
        decoder=BASE,public_seed_count=len(checkpoints),external_results=file_record(DEST/'external_results.json'),
        rationale=decision,source=file_record(__file__),selected_unchanged='reference')
    write_json(DEST/'freeze.json',frozen,exclusive=True)
    print(json.dumps(dict(external_summary=[dict(candidate=r['id'],status=r['status'],**r['mean']) for r in public_results],
        frozen_enabled_strength=enabled['strength'],seed_count=len(checkpoints))),flush=True)
    public_rows=public(frozen)
    for rec in protected:assert file_record(rec['path'])['sha256']==rec['sha256']
    (DEST/'REPORT.txt').write_text('Combined boundary and relative-motion follow-up\n\n'+json.dumps(dict(
        protocol=protocol,repeat_decision=decision,external=[dict(candidate=r['id'],status=r['status'],**r['mean']) for r in results],
        public=public_rows,limitations='Exploratory follow-up, two reused validation matches, no untouched confirmation. Incomplete training is descriptive only; single-seed comparisons are not ensemble improvements.',
        selected_unchanged='reference',committed=False,pushed=False),indent=2))
    audit=DEST/'verification.json';paths={p.relative_to(ROOT).as_posix() for p in DEST.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    for name in ('AB42','AB43'):
        paths.update(p.relative_to(ROOT).as_posix() for directory in (OUT/'models'/name,OUT/'model_predictions'/name) for p in directory.rglob('*') if p.is_file())
    checked=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',text=True,capture_output=True,check=True)
    assert set(checked.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(OUT.relative_to(ROOT))],cwd=ROOT)
    write_json(audit,dict(generated_files_ignored=True,tracked_generated_files=0,previous_assets_unchanged=True,
        selected_unchanged='reference',committed=False,pushed=False),exclusive=True)


if __name__=='__main__':main()
