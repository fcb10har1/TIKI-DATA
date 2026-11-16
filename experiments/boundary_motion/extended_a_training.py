"""Complete the frozen A experiment at seeds 42/43 without a wall-time cutoff.

The original best-only checkpoint cannot resume AdamW exactly. Replay seed42
deterministically, verify its saved prefix, then finish the unchanged stopping
rule. Every new epoch includes an atomic full optimizer/RNG resume checkpoint.
"""
from common import *
import gc
import random
import shutil
import subprocess
import torch
from torch.utils.data import DataLoader
from tiki_data.p2.model import seed_all
from tiki_data.p2.data import fit_normalization
from tiki_data.p2.train import validate
from tiki_data.p2.inference import predict_batch
from tiki_data.artifacts import write_batch, write_predictions
from tiki_data.contracts import ProbabilityBatch
from models import fresh_model, load_model, classification_loss
from feature_engine import Factory, TrainingWindows, schema
from boundary import boundary_probabilities, decode_boundary
from decoder import BASE, decode

DEST=ROOT/'runs/a42-completed-training-20261116'
ORIGINAL=OUT


def atomic_save(value,path):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(value,temporary);temporary.replace(path)


def prepare():
    plan_path=DEST/'plan.json'
    if plan_path.exists():
        plan=read_json(plan_path)
        assert file_record(__file__)['sha256']==plan['source']['sha256']
        for r in plan['protected']:assert file_record(r['path'])['sha256']==r['sha256']
        return plan
    DEST.mkdir(exist_ok=False)
    old=read_json(ORIGINAL/'manifest.json')
    prior=read_json(ORIGINAL/'baseline/training_config.json')
    protected=[file_record(p) for p in [ORIGINAL/'models/A42/best.pt',ORIGINAL/'models/A42/epochs.jsonl',
        ORIGINAL/'models/A42/status.json',ORIGINAL/'manifest.json',ROOT/'runs/active_possession_candidate.json']]
    protected += [file_record(p) for p in (ROOT/'models/possession-kalman-ensemble-20261116').glob('*') if p.is_file()]
    protected += [file_record(p) for p in Path(__file__).parent.glob('*.py') if p.name!=Path(__file__).name]
    plan=dict(created_utc=now(),source=file_record(__file__),protected=protected,
        authorization='User requested more A42 training after discussion of removing the 15-minute cutoff and repeating seed43.',
        seeds=[42,43],training=prior,split=old['splits'],scenarios=list(SCENARIOS),
        decoder=BASE,boundary_strengths=[0.,1.],
        checkpoint_selection='Unchanged external clean validation raw spellF1, then macroF1; patience8, maximum60 epochs.',
        resource_policy='Two sequential single-thread runs; no wall-time cutoff. Natural stopping rule bounds work.',
        replay='Original best.pt lacks optimizer/RNG. Replay seed42 from original seed; verify historical epochs and epoch25 tensors.',
        ensemble='Equal seed42/43 class probabilities and boundary probabilities; fixed original enabled strength1 and disabled ablation.',
        public='One diagnostic batch after completed external results and checkpoint freeze; full tracking input; same public five-minute scorer. No tuning or adoption from public.',
        selected_candidate='Existing saved submission remains unchanged.')
    write_json(plan_path,plan,exclusive=True)
    (DEST/'command.txt').write_text('python experiments/boundary_motion/extended_a_training.py\n')
    return plan


def train(seed,prior,data,factories,clean,norm):
    name='A'+str(seed)+'_completed';folder=DEST/'models'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists() and read_json(folder/'status.json').get('status')=='COMPLETE':
        print(name+' already completed; preserving checkpoint.',flush=True);return name
    started=time.perf_counter();t=prior['training']
    config=dict(id=name,architecture='boundary',speed='kalman',boundary_radius=0,boundary_weight=1.,consistency=0.,seed=seed)
    write_json(folder/'config.json',config);write_json(folder/'normalization.json',norm)
    seed_all(seed,t['threads']);model=fresh_model(len(norm['feature_names']),prior['model'],'boundary','kalman')
    windows=TrainingWindows(data['train'],factories['train'],norm,t['core_frames'],model.radius,prior['augmentation'],seed,0,1.)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(windows,batch_size=t['batch_size'],shuffle=True,num_workers=0,generator=generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    best=None;stale=0;first_epoch=1;prior_seconds=0.
    old_rows={r['epoch']:r for r in [json.loads(line) for line in (ORIGINAL/'models/A42/epochs.jsonl').read_text().splitlines()]} if seed==42 else {}
    replay_verified=False
    resume_path=folder/'last_resume.pt'
    if resume_path.exists():
        saved=torch.load(resume_path,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);generator.set_state(saved['loader_rng'])
        random.setstate(saved['python_rng'])
        n=saved['numpy_rng'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        best=tuple(saved['best']);stale=saved['stale'];first_epoch=saved['epoch']+1
        prior_seconds=saved['seconds'];replay_verified=saved['replay_verified']
        # Drop a log append from an interrupted epoch whose full checkpoint did not persist.
        rows=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines()]
        (folder/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<first_epoch))
        print(f'{name}: exact resume at epoch {first_epoch}.',flush=True)
    write_json(folder/'status.json',dict(status='RUNNING',next_epoch=first_epoch))
    try:
        epoch=first_epoch-1
        if stale<t['patience']:
            for epoch in range(first_epoch,t['epochs']+1):
                epoch_started=time.perf_counter();windows.epoch=epoch;model.train();total=0.;steps=0
                for x,y,w in loader:
                    optimizer.zero_grad(set_to_none=True)
                    per_frame,_=classification_loss(model,x,y,model.radius,t['core_frames'])
                    weights=w*(y>=0);loss=(per_frame*weights).sum()/weights.sum()
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();total+=float(loss.detach());steps+=1
                val=validate(model,clean['validation'],norm,t['core_frames'],'cpu')
                key=(val['spell_f1_mean'],val['macro_f1']);improved=best is None or key>best
                if seed==42 and epoch in old_rows:
                    assert val==old_rows[epoch]['validation'],'Historical external validation replay differs'
                    assert improved==old_rows[epoch]['selected'],'Historical checkpoint selection differs'
                    assert abs(total/steps-old_rows[epoch]['training_objective'])<1e-10,'Training replay differs'
                if seed==42 and epoch==25:
                    old=torch.load(ORIGINAL/'models/A42/best.pt',map_location='cpu',weights_only=True)
                    assert all(torch.equal(v,old['state_dict'][k]) for k,v in model.state_dict().items()),'Epoch25 weights differ'
                    replay_verified=True;del old
                    print('Seed42 historical epoch25 weights reproduced exactly.',flush=True)
                if improved:
                    best=key;stale=0
                    checkpoint=dict(schema_version=1,class_order=list(CLASS_ORDER),state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
                        model_config=prior['model'],normalization=norm,feature_schema=schema('kalman'),seed=seed,fps=25,
                        core_frames=t['core_frames'],data_kind='external',epoch=epoch,validation=val,architecture='boundary',ablation=config,augmentation=prior['augmentation'])
                    atomic_save(checkpoint,folder/'best.pt')
                else:stale+=1
                row=dict(epoch=epoch,training_objective=total/steps,validation=val,selected=improved,stale=stale,seconds=time.perf_counter()-epoch_started)
                with (folder/'epochs.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
                n=np.random.get_state()
                atomic_save(dict(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),python_rng=random.getstate(),
                    numpy_rng=(n[0],n[1].tolist(),n[2],n[3],n[4]),best=best,stale=stale,
                    seconds=prior_seconds+time.perf_counter()-started,replay_verified=replay_verified),resume_path)
                write_json(folder/'status.json',dict(status='RUNNING',epoch=epoch,stale=stale,best_raw_validation_spellF1=best[0]))
                print(json.dumps(dict(model=name,epoch=epoch,raw_validation_spellF1=key[0],best_raw_validation_spellF1=best[0],stale=stale,epoch_seconds=row['seconds'])),flush=True)
                if stale>=t['patience']:break
        reloaded,saved=load_model(folder/'best.pt')
        assert validate(reloaded,clean['validation'],norm,t['core_frames'],'cpu')==saved['validation']
        if seed==42:assert replay_verified
        status=dict(status='COMPLETE',epochs_run=epoch,best_epoch=saved['epoch'],seconds=prior_seconds+time.perf_counter()-started,
            stopping_reason='patience8' if stale>=t['patience'] else 'maximum60epochs',reload_verified=True,
            original_epoch25_identical=replay_verified if seed==42 else None,checkpoint=file_record(folder/'best.pt'))
        write_json(folder/'status.json',status);print(json.dumps(dict(model=name,**status)),flush=True)
    except Exception as error:
        write_json(folder/'status.json',dict(status='FAILED',error=type(error).__name__+': '+str(error)))
        raise
    return name


def cache(name,data):
    folder=DEST/'models'/name;root=DEST/'predictions'/name
    if (root/'complete.json').exists():return root
    model,saved=load_model(folder/'best.pt');seed_all(saved['seed'],1)
    for base,_ in data['validation']:
        factory=Factory(base,raw_load(base.match.match_id),'kalman')
        for scenario in SCENARIOS:
            missing=np.load(cache_folder(base.match.match_id,scenario)/'missing.npy')
            f=factory.corrupted(missing,scenario in ('speed_missing','combined'))
            path=root/base.match.match_id/scenario;path.mkdir(parents=True,exist_ok=True)
            write_batch(path/'probabilities.npz',predict_batch(f,model,saved['normalization'],saved['core_frames']))
            np.save(path/'boundary.npy',boundary_probabilities(f,model,saved['normalization'],saved['core_frames']))
        print(name+': external inference complete for '+base.match.match_id,flush=True)
    write_json(root/'complete.json',dict(checkpoint=file_record(folder/'best.pt')))
    return root


def evaluate(root,name,strength):
    rows=[]
    for spec,truth in truth_records():
        for scenario in SCENARIOS:
            path=root/spec.match_id/scenario
            p=read_batch(path/'probabilities.npz',spec,'probabilities').probabilities
            visible=np.load(cache_folder(spec.match_id,scenario)/'visible.npy')
            labels=decode_boundary(p,visible,spec,BASE,np.load(path/'boundary.npy'),strength) if strength else decode(p,visible,spec,BASE)
            rows.append(dict(candidate=name,match_id=spec.match_id,scenario=scenario,**metrics(truth,labels)))
    return dict(id=name,strength=strength,mean=summarize(rows),
        per_match={m:summarize([r for r in rows if r['match_id']==m]) for m in sorted({r['match_id'] for r in rows})},rows=rows)


def external(names,data):
    roots=[cache(name,data) for name in names]
    ensemble=DEST/'predictions/A_ensemble_completed'
    for base,_ in data['validation']:
        spec=base.match
        for scenario in SCENARIOS:
            paths=[root/spec.match_id/scenario for root in roots];out=ensemble/spec.match_id/scenario;out.mkdir(parents=True,exist_ok=True)
            batches=[read_batch(p/'probabilities.npz',spec,'probabilities') for p in paths]
            write_batch(out/'probabilities.npz',ProbabilityBatch(spec,batches[0].frames,np.mean([p.probabilities for p in batches],axis=0),batches[0].periods).validate(spec))
            np.save(out/'boundary.npy',np.mean([np.load(p/'boundary.npy') for p in paths],axis=0))
    results=[]
    for name,root in list(zip(names,roots))+[('A_ensemble_completed',ensemble)]:
        for strength in (0.,1.):
            results.append(evaluate(root,name+('_enabled' if strength else '_disabled'),strength))
    references=[]
    for r in read_json(PREVIOUS/'all_results.json'):
        if r['id']=='ensemble_new_speed_kalman__combo_top_3':references.append(dict(r,comparison='saved ensemble reference'))
    for r in read_json(ROOT/'runs/broad-study-20261116/cached_results.json'):
        if r['id'].startswith('top_recipe_seed_'):references.append(dict(r,comparison='matched-seed plain TCN'))
    old_disabled=read_json(ORIGINAL/'models/A42/evaluation.json')
    old_enabled=next(r for r in read_json(ORIGINAL/'models/A42/boundary_grid.json') if r['strength']==1.)
    write_json(DEST/'external_results.json',dict(results=results,references=references,old_a42=[old_disabled,old_enabled]))
    csv_write(DEST/'external_summary.csv',[dict(candidate=r['id'],**r['mean']) for r in results+references+[old_disabled,old_enabled]])
    print(json.dumps(dict(external_summary=[dict(candidate=r['id'],**r['mean']) for r in results])),flush=True)
    return results,references


def public(names):
    from games import GAME,GAMES
    from tiki_data.p1.metrica import build_metrica_features
    from tiki_data.p1.kalman import metrica_positions
    from tiki_data.p2.selected_features import build_features
    out=DEST/'public';out.mkdir(exist_ok=False)
    spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25);pitch=GAMES[GAME]['pitch']
    meta=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.xml';tracking=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.txt'
    base=build_metrica_features(spec,meta,tracking,pitch);ball=metrica_positions(spec,tracking,pitch);f=build_features(base,ball)
    visible=np.isfinite(ball).all(1);probabilities=[];boundaries=[]
    for name in names:
        model,saved=load_model(DEST/'models'/name/'best.pt')
        probabilities.append(predict_batch(f,model,saved['normalization'],saved['core_frames']).probabilities)
        boundaries.append(boundary_probabilities(f,model,saved['normalization'],saved['core_frames']))
    predictions={}
    for i,name in enumerate(names+['A_ensemble_completed']):
        p=probabilities[i] if i<len(names) else np.mean(probabilities,axis=0)
        b=boundaries[i] if i<len(names) else np.mean(boundaries,axis=0)
        for strength in (0.,1.):
            candidate=name+('_enabled' if strength else '_disabled')
            labels=decode_boundary(p,visible,spec,BASE,b,strength) if strength else decode(p,visible,spec,BASE)
            predictions[candidate]=labels
            write_predictions(out/(candidate+'.csv'),PredictionBatch(spec,base.frames,labels).validate(spec))
    truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
    rows=[dict(candidate=name,**metrics(truth,labels)) for name,labels in predictions.items()]
    previous=read_json(ORIGINAL/'public_ab_diagnostic/comparison.json')['rows']
    write_json(out/'comparison.json',dict(rows=rows,previous=previous,diagnostic_only=True,selected_unchanged=True))
    csv_write(out/'comparison.csv',rows)
    for name in predictions:
        result=subprocess.run([sys.executable,'evaluation/validate_submission.py',str(out/(name+'.csv'))],cwd=ROOT,capture_output=True,text=True)
        (out/(name+'_validator.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0
    print(json.dumps(dict(public_aggregate_metrics=rows)),flush=True)
    return rows


def finish(plan,names,results,references,public_rows):
    for r in plan['protected']:assert file_record(r['path'])['sha256']==r['sha256']
    report=['Completed A42 follow-up','No wall-time cutoff; unchanged maximum60 epochs and patience8. External-only training and checkpoint selection.',
        'Seed42 replay verifies original historical metrics and epoch25 tensors exactly. Full optimizer and RNG states now saved each epoch.',
        'Fixed decoder and boundary strengths0/1; no tuning based on public results. Existing submission unchanged.']
    for name in names:
        status=read_json(DEST/'models'/name/'status.json')
        report.append(f"{name}: {status['epochs_run']} epochs; best {status['best_epoch']}; stop {status['stopping_reason']}.")
    report.append('External mean over same two matches and six scenarios:')
    report += [f"{r['id']}: "+json.dumps(r['mean']) for r in results+references]
    report.append('Public diagnostic, full tracking context, same five-minute labels:')
    report += [json.dumps(r) for r in public_rows]
    report += ['Per-match and per-scenario external details: external_results.json. Frozen settings and checkpoint hashes: freeze.json.',
        'Limitations: only two validation matches; small repeatedly inspected public window; no untouched external test evaluated. No adoption from public scores.',
        'Command: python experiments/boundary_motion/extended_a_training.py']
    (DEST/'REPORT.txt').write_text('\n'.join(report)+'\n')
    audit=DEST/'verification.json';paths={p.relative_to(ROOT).as_posix() for p in DEST.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    check=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',capture_output=True,text=True,check=True)
    assert set(check.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(DEST.relative_to(ROOT))],cwd=ROOT)
    write_json(audit,dict(completed=True,all_outputs_ignored=True,tracked_outputs=0,old_assets_unchanged=True,
        no_commit_or_push=True,public_diagnostic_only=True,submission_unchanged=True))
    print('Completed training, external/public diagnostics and preservation checks.',flush=True)


def main():
    plan=prepare();prior=plan['training']
    # Route new feature caches to this run while retaining original external inputs.
    import feature_engine
    feature_engine.OUT=DEST
    print('Preparing external-only features; seeds42/43, max60 epochs, patience8, no wall-time cutoff.',flush=True)
    data=external_data();factories={};clean={}
    for split in data:
        factories[split]=[Factory(base,raw_load(base.match.match_id),'kalman') for base,_ in data[split]]
        clean[split]=[(f.clean(),y) for f,(_,y) in zip(factories[split],data[split])]
    norm=fit_normalization([b for b,_ in clean['train']])
    norm.update(mask_ball_speed=False,train_match_ids=[b.match.match_id for b,_ in clean['train']],speed_policy='kalman')
    assert norm==read_json(ORIGINAL/'models/A42/normalization.json')
    train_groups={b.match.match_id for b,_ in data['train']};valid_groups={b.match.match_id for b,_ in data['validation']}
    assert train_groups.isdisjoint(valid_groups)
    names=[train(seed,prior,data,factories,clean,norm) for seed in plan['seeds']]
    del factories,clean;gc.collect()
    results,references=external(names,data)
    freeze=dict(created_utc=now(),plan=file_record(DEST/'plan.json'),external_results=file_record(DEST/'external_results.json'),
        checkpoints=[file_record(DEST/'models'/name/'best.pt') for name in names],decoder=BASE,strengths=[0.,1.],
        selected_candidate='Existing submission remains unchanged; additional diagnostics only',public_scores_accessed=False)
    write_json(DEST/'freeze.json',freeze,exclusive=True)
    print('External evaluation complete; checkpoints/settings frozen before public diagnostic batch.',flush=True)
    rows=public(names)
    finish(plan,names,results,references,rows)


if __name__=='__main__':main()
