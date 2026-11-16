"""External-only factorial development, followed by one frozen public audit."""
from common import *
import argparse, itertools, hashlib, subprocess, gc, io, os, tempfile, uuid
import torch
from tiki_data.p1.features import apply_feature_policy
from tiki_data.p2.data import fit_normalization,timeline_blocks
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.model import seed_all
from tiki_data.artifacts import write_batch,write_predictions
from models import load_model
from raw_features import batch_from_raw
from temporal_decoder import transitions,sparse_viterbi
from p3.decoder import DecoderConfig,_log_emission

DEST=ROOT/'runs/historical-factorial-20261116'
HISTORICAL=ROOT/'artifacts/p2-idsse-no-ball-speed-20261111/best.pt'

# I/O-only recovery. Frozen model/decoder/scorer mathematics are unchanged.
_original_write_json=write_json
_io_retries=[]


def _retry_io(operation,description):
    for attempt in range(10):
        try:return operation()
        except PermissionError:
            if attempt==9:raise
            _io_retries.append(dict(operation=description,attempt=attempt+1))
            time.sleep(min(2.,.1*2**attempt))


def write_json(path,value,exclusive=False):
    # The continuation driver provides one writer. Atomic replacement avoids an
    # interrupted exclusive create exposing a half-written freeze/manifest.
    path=Path(path)
    if exclusive and path.exists():raise FileExistsError(str(path))
    return _retry_io(lambda:_original_write_json(path,value,exclusive=False),'atomic JSON '+str(path))


def _atomic_bytes(path,contents):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(dir=path.parent,prefix='.factorial-',suffix='.tmp')
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'wb') as handle:
            handle.write(contents);handle.flush();os.fsync(handle.fileno())
        _retry_io(lambda:os.replace(temporary,path),'replace '+str(path))
    finally:
        if temporary.exists():_retry_io(lambda:temporary.unlink(),'remove own temporary file')


def _atomic_text(path,text):_atomic_bytes(path,text.encode('utf-8'))


def _atomic_csv(path,rows):
    handle=io.StringIO(newline='');writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);_atomic_text(path,handle.getvalue())


def _hash(path):return _retry_io(lambda:file_record(path)['sha256'],'hash '+str(path))


def _read_json_retry(path):return _retry_io(lambda:read_json(path),'read JSON '+str(path))


def _load_or_commit_json(path,value):
    path=Path(path)
    if path.exists():
        saved=_read_json_retry(path)
        assert saved==value,'Existing committed JSON differs: '+str(path)
        return saved
    write_json(path,value,exclusive=True);return value


def _valid_probabilities(value,spec):
    assert value.shape==(spec.n_frames,3) and np.isfinite(value).all()
    assert (value>=0).all() and np.allclose(value.sum(1),1.,rtol=0,atol=1e-12)
    return value


def _cached_public_probabilities(path,metadata,compute,spec):
    path=Path(path);record_path=path.with_suffix('.json')
    record=_read_json_retry(record_path) if record_path.exists() else None
    if record is not None:
        assert record['metadata']==metadata,'Public probability-cache provenance changed'
        assert record['shape']==[spec.n_frames,3] and record['dtype']=='float64','Public probability-cache schema changed'
        if path.exists():
            assert _hash(path)==record['sha256'],'Public probability-cache bytes changed'
            return _valid_probabilities(_retry_io(lambda:np.load(path,allow_pickle=False),'read probability cache'),spec)
    probabilities=_valid_probabilities(compute(),spec)
    buffer=io.BytesIO();np.save(buffer,probabilities,allow_pickle=False);encoded=buffer.getvalue()
    digest=hashlib.sha256(encoded).hexdigest()
    if record is not None:assert digest==record['sha256'],'Recomputed cache differs from committed cache'
    if path.exists():
        # An interruption may leave an atomic array without its metadata. Prove
        # it is the same deterministic output before adopting it.
        assert _hash(path)==digest,'Unmanifested probability cache differs from frozen inference'
    else:_atomic_bytes(path,encoded)
    _load_or_commit_json(record_path,dict(metadata=metadata,sha256=digest,shape=list(probabilities.shape),dtype=str(probabilities.dtype)))
    return probabilities


def _committed_prediction(path,spec,frames,labels=None,record=None):
    path=Path(path)
    if path.exists() and record is not None:
        assert _hash(path)==record['sha256'],'Committed prediction CSV changed'
        return record
    if path.exists():
        # Pre-recovery interrupted direct writes can be incomplete. Preserve the
        # partial artifact; valid but different predictions are never overwritten.
        try:existing=_retry_io(lambda:read_predictions(path,spec),'validate unmanifested CSV')
        except (ValueError,AssertionError):
            partial=path.with_name(path.name+'.incomplete-'+uuid.uuid4().hex)
            _retry_io(lambda:path.rename(partial),'preserve incomplete CSV')
        else:
            assert labels is not None and np.array_equal(existing.labels,labels),'Unmanifested CSV differs from frozen decoding'
            return file_record(path)
    assert labels is not None,'Missing committed CSV must be reconstructed from frozen probabilities'
    def commit():
        temporary=path.with_name('.prediction-'+uuid.uuid4().hex+'.csv')
        try:
            write_predictions(temporary,PredictionBatch(spec,frames,labels).validate(spec))
            digest=_hash(temporary)
            if record is not None:assert digest==record['sha256'],'Reconstructed prediction differs from committed CSV'
            _retry_io(lambda:os.replace(temporary,path),'commit prediction CSV')
        finally:
            if temporary.exists():_retry_io(lambda:temporary.unlink(),'remove own CSV temporary')
    _retry_io(commit,'atomic prediction CSV '+str(path))
    return file_record(path)


def decode(p,visible,spec,s):
    cfg=DecoderConfig(fps=spec.fps,temperature=4.,missing_weight=s['missing_weight'],
        min_poss_s=s['minimum'],min_dead_s=s['minimum'],mean_poss_s=14.,mean_dead_s=10.,
        turnover_frac=.5,restart_switch=.8,start_alive=.5,prob_floor=1e-4,use_mbr=False,seed=42)
    chain,pred,weights=transitions(cfg,s['duration'],s['transition_penalty'],1.)
    out=np.full(spec.n_frames,'X',dtype='<U1')
    for a,b in spec.periods:
        path=sparse_viterbi(chain.logpi,pred,weights,_log_emission(p[a:b],visible[a:b],cfg,chain))
        out[a:b]=np.asarray(CLASS_ORDER)[chain.team[path]]
    return out


def grid():
    return [dict(id=f'{duration}_{minimum}_t{penalty}_w{weight}',duration=duration,minimum=minimum,
                 transition_penalty=penalty,missing_weight=weight)
            for (duration,minimum),penalty,weight in itertools.product([('hard',1.),('soft',1.),('hard',.2)],[-1.,0.,1.],[0.,.5])]


def protocol():
    DEST.mkdir(exist_ok=True)
    path=DEST/'protocol.json'
    if path.exists():return read_json(path)
    protected=[file_record(HISTORICAL),file_record(ROOT/'runs/active_possession_candidate.json')]
    protected += [file_record(p) for p in (ROOT/'models/possession-kalman-ensemble-20261116').glob('*') if p.is_file()]
    d=dict(created_utc=now(),protected=protected,decoder_grid=grid(),
        factors=dict(boundary_weighting=[False,True],augmentation=[False,True],long_context=[False,True],external_expansion=[False,True]),
        seeds=dict(screen=42,confirmation=43),training='60epochs maximum, patience8 for both raw and decoded selectors; no wall-time termination.',
        selection='External equal-match/equal-scenario mean spellF1; macroF1 tie-break. Report missed/extra and per-match consistency. Paired seed confirmations of top two externally ranked recipes plus original-feature control.',
        checkpoint_selection='Retain raw-best and fixed-decoder-best selected on two clean external validation matches. Compare both under full external decoder grid after training.',
        scenarios=list(SCENARIOS),splits=read_json(ORIGINAL_MANIFEST())['splits'],
        expansion='Two deterministically chosen licensed external SkillCorner matches if adapter passes checks; H/A weak labels only where tracking agrees with explicit live phases; other targets X. IDSSE test never used for development.',
        public='Freeze every evaluated recipe, selected candidate and all checkpoint hashes before a single public diagnostic batch. Score all frozen combinations; report public winner without public-directed retraining or switching active submission.',
        scope='All combinations of these declared factors, not an unbounded hyperparameter search; 20s context and additional external downloads are outside this grid.',
        source=file_record(__file__))
    write_json(path,d,exclusive=True)
    (DEST/'commands.txt').write_text('python experiments/boundary_motion/historical_factorial.py baseline\npython experiments/boundary_motion/historical_factorial.py train\npython experiments/boundary_motion/historical_factorial.py finish\n')
    return d


def ORIGINAL_MANIFEST():return OUT/'manifest.json'


def case_inputs():
    cases=[]
    for spec,truth in truth_records():
        raw=raw_load(spec.match_id)
        for scenario in SCENARIOS:
            path=DEST/'feature_cache'/spec.match_id/scenario
            feature=path/'features.npz'
            if not feature.exists():
                missing=np.load(cache_folder(spec.match_id,scenario)/'missing.npy')
                f,visible=batch_from_raw(raw,spec,missing,True,False)
                path.mkdir(parents=True,exist_ok=True);write_batch(feature,f);np.save(path/'visible.npy',visible)
            cases.append((read_batch(feature,spec,'features'),truth,np.load(path/'visible.npy'),scenario))
    return cases


def compact_metrics(truth,labels):
    # Same official functions, bypassing unrelated transition reports. Half/gap
    # boundaries are kept independent; all predictions within periods are H/A/D.
    gt=pm.label_codes(truth.labels);pred=pm.label_codes(labels)
    cm=pm.confusion(gt,pred);gs=[];ps=[];offset=0
    for a,b in truth.match.periods:
        runs=pm.valid_runs(gt[a:b]);g=pm.possession_spells(gt[a:b],runs);p=pm.possession_spells(pred[a:b],runs)
        gs.extend((s+offset,t+offset,c) for s,t,c in g);ps.extend((s+offset,t+offset,c) for s,t,c in p);offset+=b-a+100
    overlaps=pm.spell_overlaps(gs,ps);scores=[];half=None
    for threshold in pm.SPELL_IOUS:
        tp,fp,fn=pm.match_spells(overlaps,len(gs),len(ps),threshold)
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        if threshold==.5:half=(tp,fp,fn)
    tp,fp,fn=half
    return dict(accuracy_pct=float(100*np.trace(cm[:3,:3])/cm[:3].sum()),macroF1=float(pm.macro_f1(cm[:3])),
        spellF1=float(np.mean(scores)),dead_ball_recall=float(cm[2,2]/cm[2].sum()),
        spell_precision_05=tp/(tp+fp) if tp+fp else 0.,spell_recall_05=tp/(tp+fn) if tp+fn else 0.,
        missed_possessions=fn,extra_possessions=fp,gt_possessions=len(gs),pred_possessions=len(ps))


def evaluate_probs(probs,cases,settings):
    rows=[]
    for p,(f,truth,visible,scenario) in zip(probs,cases):
        labels=decode(p,visible,f.match,settings)
        rows.append(dict(match_id=f.match.match_id,scenario=scenario,**compact_metrics(truth,labels)))
    means=summarize(rows)
    means.update({k:float(np.mean([r[k] for r in rows])) for k in ['spell_precision_05','spell_recall_05','missed_possessions','extra_possessions']})
    return dict(mean=means,per_match={m:summarize([r for r in rows if r['match_id']==m]) for m in sorted({r['match_id'] for r in rows})},rows=rows)


def probabilities(checkpoint,name,cases):
    folder=DEST/'probabilities'/name;folder.mkdir(parents=True,exist_ok=True)
    model,saved=load_model(checkpoint);seed_all(saved['seed'],1)
    result=[]
    for f,_,_,scenario in cases:
        path=folder/(f.match.match_id+'_'+scenario+'.npy')
        if not path.exists():np.save(path,predict_batch(f,model,saved['normalization'],saved['core_frames']).probabilities)
        result.append(np.load(path))
    return result


def evaluate_checkpoint(name,path,cases,settings=None):
    output=DEST/'evaluations'/(name+'.json')
    if output.exists():return read_json(output)
    p=probabilities(path,name,cases);rows=[]
    for s in settings or grid():rows.append(dict(candidate=name,checkpoint=file_record(path),decoder=s,**evaluate_probs(p,cases,s)))
    write_json(output,rows);print(json.dumps(dict(evaluated=name,best_external=max(r['mean']['spellF1'] for r in rows))),flush=True)
    return rows


def baseline():
    protocol();seed_all(42,1);cases=case_inputs()
    rows=evaluate_checkpoint('historical',HISTORICAL,cases)
    best=max(rows,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
    # Verify compact scorer against the complete official pipeline once.
    p=probabilities(HISTORICAL,'historical',cases)[0];f,truth,visible,_=cases[0]
    labels=decode(p,visible,f.match,best['decoder']);a=compact_metrics(truth,labels);b=metrics(truth,labels)
    assert all(abs(a[k]-b[k])<1e-12 for k in METRICS)
    write_json(DEST/'fixed_training_decoder.json',best['decoder'])
    write_json(DEST/'baseline_selection.json',dict(best=best,scorer_verified=True,public_accessed=False))
    print(json.dumps(dict(baseline_external_best=best['mean'],fixed_training_decoder=best['decoder'])),flush=True)


def load_training(expansion=False):
    data=external_data();data={k:[(apply_feature_policy(b,True),y) for b,y in rows] for k,rows in data.items()}
    raw={b.match.match_id:raw_load(b.match.match_id) for rows in data.values() for b,_ in rows}
    if expansion:
        from skillcorner_expansion import load_expansion
        extra,extra_raw=load_expansion();data['train']+=extra;raw.update(extra_raw)
    norm=fit_normalization([b for b,_ in data['train']],allow_unobserved=('ball_speed_mps',))
    norm.update(mask_ball_speed=True,train_match_ids=[b.match.match_id for b,_ in data['train']])
    assert not set(norm['train_match_ids'])&{b.match.match_id for b,_ in data['validation']}
    return data,raw,norm


def train():
    from historical_factorial_training import train_arm
    plan=protocol();cases=case_inputs();s=read_json(DEST/'fixed_training_decoder.json')
    validation=[(f,t,v) for f,t,v,scenario in cases if scenario=='clean']
    config=read_json(ROOT/'configs/p2_idsse_no_ball_speed.json')
    factor_order=[(False,False,False),(True,False,False),(False,True,False),(False,False,True),
                  (True,True,False),(True,False,True),(False,True,True),(True,True,True)]
    configs=[dict(id=f'b{int(b)}m{int(m)}c{int(c)}e{int(e)}s42',seed=42,boundary_weighting=b,augmentation=m,long_context=c,external_expansion=e)
             for e in (False,True) for b,m,c in factor_order]
    core=config['training']['core_frames']
    matched_windows=sum(1 for batch,target in external_data()['train']
        for left,right in timeline_blocks(batch) for start in range(left,right,core)
        if np.any(target[start:min(start+core,right)]>=0))
    for arm in configs:
        if arm['external_expansion']:arm['epoch_windows']=matched_windows
    amendment=DEST/'expansion_compute_control.json'
    if not amendment.exists():
        write_json(amendment,dict(created_utc=now(),epoch_windows=matched_windows,
            policy='Expanded-data arms sample without replacement across six training matches each epoch, with the same optimizer-step count as four-match controls.',
            public_accessed=False,reason='Isolate additional match diversity from simply adding optimizer updates; declared before any expanded-data arm.'),exclusive=True)
    write_json(DEST/'training_matrix.pending.json',configs)
    (DEST/'training_matrix.pending.json').replace(DEST/'training_matrix.json')
    for expansion in (False,True):
        if expansion:
            ready=DEST/'external_expansion/ready.json'
            if not ready.exists():
                print('Expansion dataset not ready; base factorial completed. Rerun train after readiness.',flush=True);return
        data,raw,norm=load_training(expansion)
        for arm in [c for c in configs if c['external_expansion']==expansion]:
            status=train_arm(arm,DEST/'models'/arm['id'],data,raw,norm,config,s,validation,decode_fn=decode)
            write_json(DEST/'training_status'/(arm['id']+'.json'),status)
            for selector in ('raw_best','decoded_best'):
                evaluate_checkpoint(arm['id']+'_'+selector,DEST/'models'/arm['id']/(selector+'.pt'),cases)
        del data,raw;gc.collect()


def collected():
    return [r for p in sorted((DEST/'evaluations').glob('*.json')) for r in read_json(p)]


def confirmation():
    from historical_factorial_training import train_arm
    intended=read_json(DEST/'training_matrix.json')
    incomplete=[c['id'] for c in intended if not (DEST/'training_status'/(c['id']+'.json')).exists()
                or read_json(DEST/'training_status'/(c['id']+'.json')).get('status')!='COMPLETE']
    assert not incomplete,'Factorial training incomplete: '+', '.join(incomplete)
    cases=case_inputs();s=read_json(DEST/'fixed_training_decoder.json');validation=[(f,t,v) for f,t,v,sc in cases if sc=='clean']
    rows=collected();configs=read_json(DEST/'training_matrix.json');ranks=[]
    for arm in configs:
        group=[r for r in rows if r['candidate'].startswith(arm['id']+'_')]
        if group:ranks.append((max(group,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1'])),arm))
    ranks.sort(key=lambda pair:(-pair[0]['mean']['spellF1'],-pair[0]['mean']['macroF1']))
    chosen=[pair[1] for pair in ranks[:2]]
    control=next(c for c in configs if c['id']=='b0m0c0e0s42')
    if control not in chosen:chosen.append(control)
    choice=DEST/'confirmation_plan.json'
    if choice.exists():chosen=read_json(choice)
    else:write_json(choice,chosen,exclusive=True)
    config=read_json(ROOT/'configs/p2_idsse_no_ball_speed.json')
    for arm in chosen:
        second=dict(arm,seed=43,id=arm['id'][:-2]+'43')
        data,raw,norm=load_training(arm['external_expansion'])
        status=train_arm(second,DEST/'models'/second['id'],data,raw,norm,config,s,validation,decode_fn=decode)
        write_json(DEST/'training_status'/(second['id']+'.json'),status)
        for selector in ('raw_best','decoded_best'):
            name=second['id']+'_'+selector;evaluate_checkpoint(name,DEST/'models'/second['id']/(selector+'.pt'),cases)
            first=arm['id']+'_'+selector;ensemble=arm['id'][:-3]+'_ensemble_'+selector
            member_probs=[probabilities(DEST/'models'/a['id']/(selector+'.pt'),a['id']+'_'+selector,cases) for a in (arm,second)]
            p=[(member_probs[0][i]+member_probs[1][i])/2 for i in range(len(cases))]
            output=DEST/'evaluations'/(ensemble+'.json')
            if not output.exists():
                results=[dict(candidate=ensemble,members=[first,name],decoder=setting,**evaluate_probs(p,cases,setting)) for setting in grid()]
                write_json(output,results)
        del data,raw;gc.collect()


def _public_freeze(plan):
    path=DEST/'freeze.json';current=collected()
    if path.exists():
        frozen=_read_json_retry(path)
        assert frozen['recipes']==current,'External recipes changed after public freeze'
        assert frozen['preserved']==plan['protected'],'Protected-asset protocol changed'
        if 'protocol_record' in frozen:
            assert _hash(DEST/'protocol.json')==frozen['protocol_record']['sha256'],'Experiment protocol changed'
    else:
        selected=max(current,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
        frozen=dict(created_utc=now(),external_selected=selected,recipes=current,
            reason='Maximum external equal-match/six-scenario mean spellF1; no public scores accessed in this study.',
            preserved=plan['protected'],protocol_record=file_record(DEST/'protocol.json'),
            source_hashes=[file_record(p) for p in Path(__file__).parent.glob('historical_factorial*.py')],
            recovery='Atomic outputs; bounded transient-file-lock retries; frozen selection reused; completed heldout/public aggregates never recomputed; interrupted scoring resumes the same frozen batch without feedback.')
        write_json(path,frozen,exclusive=True)
    for record in frozen['preserved']+frozen['source_hashes']:
        assert _hash(record['path'])==record['sha256'],'Frozen asset or inference source changed'
    for row in frozen['recipes']:
        if 'checkpoint' in row:assert _hash(row['checkpoint']['path'])==row['checkpoint']['sha256'],'Frozen checkpoint changed'
    assert frozen['external_selected'] in frozen['recipes']
    return frozen


def _heldout_once(rows,selected):
    path=DEST/'untouched_test_confirmation.json'
    if path.exists():
        saved=_read_json_retry(path)
        names={row['candidate'] for row in saved['rows']}
        assert selected['candidate'] in names and {'historical_weight_0.0','historical_weight_0.5'}<=names
        return saved
    # One untouched external-test confirmation after selection. Never feeds back
    # into training, decoder choice, checkpoint selection or seed selection. A
    # completed aggregate is reused on recovery, without reopening test labels.
    heldout=[]
    for entry in external_entries(('test',)):
        test_spec=MatchSpec.from_dict(entry['match'])
        base_test=read_batch(ROOT/'data/external/idsse_processed/matches'/test_spec.match_id/'features.npz',test_spec,'features')
        test_features=apply_feature_policy(base_test,True)
        test_truth=read_predictions(entry['ground_truth'],test_spec)
        observed=base_test.observed[:,1]|base_test.observed[:,2]
        chosen_members=[selected] if 'checkpoint' in selected else [next(r for r in rows if r['candidate']==n) for n in selected['members']]
        probabilities_test=[]
        for member in chosen_members:
            model,saved=load_model(member['checkpoint']['path']);seed_all(saved['seed'],1)
            probabilities_test.append(predict_batch(test_features,model,saved['normalization'],saved['core_frames']).probabilities)
        label=decode(np.mean(probabilities_test,axis=0),observed,test_spec,selected['decoder'])
        heldout.append(dict(candidate=selected['candidate'],match_id=test_spec.match_id,**compact_metrics(test_truth,label)))
        model,saved=load_model(HISTORICAL)
        prob=predict_batch(test_features,model,saved['normalization'],saved['core_frames']).probabilities
        for weight in (0.,.5):
            setting=dict(duration='hard',minimum=1.,transition_penalty=0.,missing_weight=weight)
            heldout.append(dict(candidate='historical_weight_'+str(weight),match_id=test_spec.match_id,
                **compact_metrics(test_truth,decode(prob,observed,test_spec,setting))))
    result=dict(rows=heldout,regime='clean only; one final confirmation, not used for selection')
    write_json(path,result,exclusive=True);return result


def public():
    from games import GAME,GAMES
    from tiki_data.p1.metrica import build_metrica_features
    from tiki_data.p1.kalman import metrica_positions
    plan=protocol();frozen=_public_freeze(plan);rows=frozen['recipes'];selected=frozen['external_selected']
    _heldout_once(rows,selected)
    out=DEST/'public';out.mkdir(exist_ok=True);(out/'predictions').mkdir(exist_ok=True)
    spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25);pitch=GAMES[GAME]['pitch']
    expected=[]
    for i,row in enumerate(rows):
        name=f'{i:04d}_'+row['candidate']+'_'+row['decoder']['id']
        expected.append(dict(name=name,candidate=row['candidate'],decoder=row['decoder'],external=row['mean'],path=str(out/'predictions'/(name+'.csv'))))
    _load_or_commit_json(out/'prediction_plan.json',expected)
    manifest_path=out/'predictions_frozen.json';manifest=None
    if manifest_path.exists():
        manifest=_read_json_retry(manifest_path)
        assert [{k:v for k,v in row.items() if k!='prediction_file'} for row in manifest]==expected,'Frozen prediction manifest changed'
        if all('prediction_file' in row and Path(row['path']).exists() for row in manifest):
            for row in manifest:_committed_prediction(row['path'],spec,None,record=row['prediction_file'])
        else:manifest=None
    if manifest is None:
        meta=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.xml';tracking=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.txt'
        inputs=dict(metadata=file_record(meta),tracking=file_record(tracking),match=spec.as_dict(),pitch=list(pitch))
        _load_or_commit_json(out/'inference_inputs.json',inputs)
        f=build_metrica_features(spec,meta,tracking,pitch,mask_ball_speed=True);ball=metrica_positions(spec,tracking,pitch);visible=np.isfinite(ball).all(1)
        ps={};unique={}
        for row in rows:
            if 'checkpoint' in row:unique[row['candidate']]=row['checkpoint']
        cache=out/'probabilities';cache.mkdir(exist_ok=True)
        for name,record in unique.items():
            model,saved=load_model(record['path']);seed_all(saved['seed'],1)
            metadata=dict(checkpoint=record,inputs=inputs,features=list(f.feature_names),normalization=saved['normalization'],core=saved['core_frames'])
            ps[name]=_cached_public_probabilities(cache/(name+'.npy'),metadata,
                lambda:predict_batch(f,model,saved['normalization'],saved['core_frames']).probabilities,spec)
            print('Public inference ready: '+name,flush=True)
        progress_path=out/'predictions_progress.json'
        progress=_read_json_retry(progress_path) if progress_path.exists() else {}
        if manifest_path.exists():
            for committed in _read_json_retry(manifest_path):
                if 'prediction_file' in committed:progress.setdefault(committed['name'],committed['prediction_file'])
        manifest=[]
        for row,entry in zip(rows,expected):
            path=Path(entry['path']);record=progress.get(entry['name'])
            if record is not None and path.exists():
                record=_committed_prediction(path,spec,None,record=record)
            else:
                p=ps[row['candidate']] if 'checkpoint' in row else np.mean([ps[n] for n in row['members']],axis=0)
                labels=decode(p,visible,spec,row['decoder'])
                record=_committed_prediction(path,spec,f.frames,labels,record)
            manifest.append(dict(entry,prediction_file=record));progress[entry['name']]=record
            write_json(progress_path,progress)
        # Same deterministic recipe order; each CSV now has a verified full hash.
        if manifest_path.exists():
            prior=_read_json_retry(manifest_path)
            assert [{k:v for k,v in row.items() if k!='prediction_file'} for row in prior]==expected
        write_json(manifest_path,manifest)
    result_path=out/'results.json'
    if result_path.exists():
        scores=_read_json_retry(result_path)
        assert len(scores)==len(manifest) and {r['name'] for r in scores}=={r['name'] for r in manifest},'Incomplete public aggregate results'
        expected_by_name={r['name']:r for r in manifest}
        for row in scores:
            saved=expected_by_name[row['name']]
            assert all(row[k]==saved[k] for k in ('candidate','decoder','external','path')),'Public aggregate recipe changed'
    else:
        # Every prediction and its hash are committed before any public labels are
        # opened. If interrupted here, repeat only this frozen diagnostic scoring.
        truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
        scores=[]
        for r in manifest:
            labels=_retry_io(lambda:read_predictions(r['path'],spec),'read committed prediction CSV').labels
            scores.append(dict(**r,public=compact_metrics(truth,labels)))
        scores.sort(key=lambda r:(-r['public']['spellF1'],-r['public']['macroF1']))
        write_json(result_path,scores,exclusive=True)
    _atomic_csv(out/'leaderboard.csv',[dict(candidate=r['candidate'],decoder=r['decoder']['id'],external_spellF1=r['external']['spellF1'],**r['public']) for r in scores])
    for r in scores[:3]:
        check=subprocess.run([sys.executable,'evaluation/validate_submission.py',r['path']],cwd=ROOT,capture_output=True,text=True)
        _atomic_text(out/(r['name']+'_validator.log'),check.stdout+check.stderr);assert check.returncode==0
    for r in plan['protected']:assert file_record(r['path'])['sha256']==r['sha256']
    chosen=next(r for r in scores if r['candidate']==selected['candidate'] and r['decoder']==selected['decoder'])
    summary=dict(external_selected=chosen,public_top_three=scores[:3],total_frozen_recipes=len(scores),active_submission_unchanged=True)
    write_json(DEST/'summary.json',summary)
    _atomic_text(DEST/'REPORT.txt',json.dumps(summary,indent=2)+'\nFull external per-match/scenario results: evaluations/. Public results: public/results.json. No public-directed retraining.\nRecovery: atomic writes, bounded transient file-lock retries, immutable freeze, committed cache/CSV hashes, and reuse of completed heldout/public aggregates.\n')
    write_json(DEST/'io_recovery_audit.json',dict(retries=_io_retries,semantics=frozen.get('recovery','Pre-recovery freeze preserved'),
        prediction_count=len(manifest),heldout_aggregate_reused_or_committed=True,public_aggregate_reused_or_committed=True))
    audit=DEST/'verification.json';paths={p.relative_to(ROOT).as_posix() for p in DEST.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    check=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',capture_output=True,text=True,check=True)
    assert set(check.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(DEST.relative_to(ROOT))],cwd=ROOT)
    write_json(audit,dict(outputs_ignored=True,tracked_outputs=0,reference_unchanged=True,no_commit_push=True))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['baseline','train','finish']);a=parser.parse_args()
    if a.stage=='baseline':baseline()
    elif a.stage=='train':train()
    else:confirmation();public()
