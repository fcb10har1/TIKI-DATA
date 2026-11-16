"""External-only 19-feature historical TCN factorial training.

API: train_arm(config, dest, data, raw_by_id, normalization,
               training_config, decoder_settings, validation_inputs,
               decode_fn=None).

config: id, seed, boundary_weighting, augmentation, long_context. The three
boolean interventions form a 2**3 factorial. The no-augmentation branch consumes
the original stored features; the augmentation branch hides raw ball observations
before recomputing the same 19 feature definitions. Recorded ball speed is always
unobserved. No observation-age or Kalman feature is silently introduced.

Expanded-data arms may specify epoch_windows to sample that many distinct
training windows per epoch. This matches the base arm's optimizer-step budget
without silently increasing updates just because more matches were added.

Long context changes only the final dilation from16 to48: the model retains five
blocks and the same parameter count, while receptive field grows125->253frames.
The earlier blocks preserve fine-resolution coverage; no extra block is added.

All writes are under dest. The caller supplies external records and a decoder
fixed using external data before training. No competition files are accessed.
"""
from pathlib import Path
import copy
import json
import random
import shutil
import time

from common import np, CLASS_ORDER, PredictionBatch, score_match
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tiki_data.p1.features import FEATURE_NAMES, apply_feature_policy, feature_schema
from tiki_data.p2.model import TCN, seed_all
from tiki_data.p2.data import encode, timeline_blocks
from tiki_data.p2.inference import predict_batch
from raw_features import features as raw_features, raw_slice, sample_occlusion


MODERATE = dict(point_probability=.05, ball_stretch_probability=.3,
                ball_stretch_frames=[25, 250], speed_window_probability=1.)


def _replace(temporary, target):
    """Keep atomic writes while tolerating transient Windows scanner locks."""
    for attempt in range(12):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.1 * 2**attempt, 2.))


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    _replace(temporary, path)


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temporary)
    _replace(temporary, path)


def _copy(source, target):
    target = Path(target)
    temporary = target.with_suffix(target.suffix+'.tmp')
    shutil.copyfile(source, temporary)
    _replace(temporary, target)


class HistoricalWindows(Dataset):
    """Nonoverlapping supervised cores with boundary-safe raw observation halos."""
    def __init__(self, records, raw_by_id, normalization, core, radius, config):
        self.records = [(apply_feature_policy(batch, True), labels) for batch, labels in records]
        self.raw = raw_by_id
        self.norm, self.core, self.radius = normalization, core, radius
        self.config, self.seed, self.epoch = dict(config), config['seed'], 0
        self.index, self.weights = [], []
        assert tuple(normalization['feature_names']) == FEATURE_NAMES
        assert normalization.get('mask_ball_speed') is True
        for i, (batch, labels) in enumerate(self.records):
            weight = np.ones(len(labels), np.float32)
            assert batch.feature_names == FEATURE_NAMES
            assert not batch.observed[:, 0].any()
            for left, right in timeline_blocks(batch):
                if config['boundary_weighting']:
                    boundary = np.flatnonzero((labels[left+1:right] != labels[left:right-1]) &
                        (labels[left+1:right] >= 0) & (labels[left:right-1] >= 0))+left+1
                    radius_frames = int(round(.4*batch.match.fps))
                    for frame in boundary:
                        weight[max(left,frame-radius_frames):min(right,frame+radius_frames)] = 2.
                for start in range(left, right, core):
                    stop = min(start+core, right)
                    if np.any(labels[start:stop] >= 0):
                        self.index.append((i,left,right,start,stop))
            self.weights.append(weight)
        if not self.index:
            raise ValueError('No external training windows')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i,left,right,start,stop = self.index[index]
        batch,labels = self.records[i]
        a,b = max(left,start-self.radius),min(right,start+self.core+self.radius)
        if self.config['augmentation']:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
            missing,_ = sample_occlusion(b-a,rng,MODERATE)
            values,observed,_ = raw_features(raw_slice(self.raw[batch.match.match_id],a,b),
                batch.match.fps,missing,hide_speed=True,with_age=False)
        else:
            values,observed = batch.values[a:b],batch.observed[a:b]
        assert values.shape[1] == 19 and not observed[:,0].any()
        x = np.zeros((38,self.core+2*self.radius),np.float32)
        offset = a-(start-self.radius)
        x[:,offset:offset+b-a] = encode(values,observed,self.norm)
        y = np.full(self.core,-100,np.int64)
        y[:stop-start] = labels[start:stop]
        y[y<0] = -100
        weight = np.zeros(self.core,np.float32)
        weight[:stop-start] = self.weights[i][start:stop]
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(weight)


def _validation(model, norm, core, validation_inputs, settings, decode_fn):
    raw_rows, decoded_rows = [], []
    loss_sum, count = 0., 0
    for features,truth,visible in validation_inputs:
        assert features.feature_names == FEATURE_NAMES
        p = predict_batch(features,model,norm,core).probabilities
        raw = np.asarray(CLASS_ORDER)[p.argmax(axis=1)]
        raw[features.periods==0] = 'X'
        raw_result = score_match(truth,PredictionBatch(features.match,features.frames,raw),features.match)
        raw_rows.append(dict(match_id=features.match.match_id,macro_f1=raw_result['macro_f1'],
                             spell_f1_mean=raw_result['spell_f1_mean']))
        labels = decode_fn(p,visible,features.match,settings)
        decoded_result = score_match(truth,PredictionBatch(features.match,features.frames,labels),features.match)
        confusion = np.asarray(decoded_result['confusion'])
        decoded_rows.append(dict(match_id=features.match.match_id,macro_f1=decoded_result['macro_f1'],
            spell_f1_mean=decoded_result['spell_f1_mean'],
            accuracy_pct=float(100*np.trace(confusion[:3,:3])/confusion[:3].sum())))
        codes = np.full(features.match.n_frames,-100,np.int64)
        for i,label in enumerate(CLASS_ORDER):codes[truth.labels==label] = i
        valid = codes>=0
        loss_sum -= np.log(np.maximum(p[valid,codes[valid]],1e-300)).sum()
        count += int(valid.sum())
    def mean(rows):
        return dict(macro_f1=float(np.mean([r['macro_f1'] for r in rows])),
                    spell_f1_mean=float(np.mean([r['spell_f1_mean'] for r in rows])),matches=rows)
    raw = mean(raw_rows)
    raw['cross_entropy'] = float(loss_sum/count)
    return raw,mean(decoded_rows)


def train_arm(config, dest, data, raw_by_id, normalization, training_config,
              decoder_settings, validation_inputs, decode_fn=None):
    """Run/resume one arm; return status with raw_best/decoded_best checkpoint paths.

    data['train'] contains (FeatureBatch, encoded_labels) records. Validation
    inputs are (FeatureBatch, PredictionBatch truth, visible_bool_vector) tuples.
    Callers must use the same whole-match split for every arm and fit normalization
    on external training matches only. Stopping requires BOTH selectors to have
    eight stale epochs, bounded by the unchanged maximum60 epochs.
    """
    if decode_fn is None:
        from temporal_decoder import decode as decode_fn
    required = {'id','seed','boundary_weighting','augmentation','long_context'}
    assert required <= set(config)
    for key in ('boundary_weighting','augmentation','long_context'):
        assert isinstance(config[key],bool)
    dest = Path(dest).resolve()
    dest.mkdir(parents=True,exist_ok=True)
    t = training_config['training']
    assert t['class_weights']=='none' and t['device']=='cpu'
    assert tuple(normalization['feature_names'])==FEATURE_NAMES
    assert normalization.get('mask_ball_speed') is True
    assert normalization['observed_count'][0]==0
    assert normalization['mean'][0]==0 and normalization['scale'][0]==1
    train_ids = {batch.match.match_id for batch,_ in data['train']}
    val_ids = {features.match.match_id for features,_,_ in validation_inputs}
    assert train_ids.isdisjoint(val_ids), 'Training/validation match overlap'
    assert set(normalization['train_match_ids'])==train_ids
    setup = dict(config=config,normalization=normalization,training=training_config,
                 decoder=decoder_settings,train_ids=sorted(train_ids),validation_ids=sorted(val_ids),
                 augmentation=MODERATE if config['augmentation'] else None,
                 boundary_weighting='2x loss within +/-0.4s of valid external transitions' if config['boundary_weighting'] else 'uniform',
                 stopping_rule='Both raw and decoded selectors stale >= patience, or maximum epochs')
    config_path = dest/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==setup, 'Refusing changed resume configuration'
    else:
        _json(config_path,setup)
    status_path = dest/'status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('status')=='COMPLETE':return status
    started = time.perf_counter()
    model_config = copy.deepcopy(training_config['model'])
    if config['long_context']:
        assert model_config['dilations']==[1,2,4,8,16]
        model_config['dilations'][-1]=48
    seed_all(config['seed'],t['threads'])
    model = TCN(19,**model_config)
    windows = HistoricalWindows(data['train'],raw_by_id,normalization,t['core_frames'],model.radius,config)
    generator = torch.Generator().manual_seed(config['seed'])
    epoch_windows=config.get('epoch_windows',len(windows))
    if 'epoch_windows' in config:
        assert config.get('external_expansion') is True,'Window caps are restricted to expanded-data arms'
        assert isinstance(epoch_windows,int) and not isinstance(epoch_windows,bool) and 1<=epoch_windows<=len(windows)
        sampler=RandomSampler(windows,replacement=False,num_samples=epoch_windows,generator=generator)
        loader=DataLoader(windows,batch_size=t['batch_size'],shuffle=False,sampler=sampler,num_workers=0,generator=generator)
    else:
        # Preserve the original base-arm RNG consumption and shuffle path exactly.
        loader=DataLoader(windows,batch_size=t['batch_size'],shuffle=True,num_workers=0,generator=generator)
    optimizer_steps_per_epoch=len(loader)
    optimizer = torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
    class_weights = torch.ones(3,dtype=torch.float32)
    best = {'raw':None,'decoded':None}
    stale = {'raw':0,'decoded':0}
    best_epochs = {'raw':None,'decoded':None}
    first_epoch, prior_seconds = 1,0.
    resume_path = dest/'last_resume.pt'
    if resume_path.exists():
        saved = torch.load(resume_path,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['torch_rng']);generator.set_state(saved['loader_rng'])
        random.setstate(saved['python_rng'])
        n=saved['numpy_rng'];np.random.set_state((n[0],np.asarray(n[1],np.uint32),n[2],n[3],n[4]))
        best={k:tuple(v) if v is not None else None for k,v in saved['best'].items()}
        stale=saved['stale'];best_epochs=saved['best_epochs']
        first_epoch=saved['epoch']+1;prior_seconds=saved['seconds']
        if (dest/'epochs.jsonl').exists():
            rows=[json.loads(line) for line in (dest/'epochs.jsonl').read_text().splitlines()]
            (dest/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['epoch']<first_epoch))
    _json(status_path,dict(status='RUNNING',next_epoch=first_epoch,epoch_windows=epoch_windows,
                           optimizer_steps_per_epoch=optimizer_steps_per_epoch))
    (dest/'epochs').mkdir(exist_ok=True)
    try:
        epoch=first_epoch-1
        if not all(value>=t['patience'] for value in stale.values()):
            for epoch in range(first_epoch,t['epochs']+1):
                before=time.perf_counter();windows.epoch=epoch;model.train()
                total,count=0.,0.
                for x,y,w in loader:
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(x)[:,:,model.radius:model.radius+t['core_frames']]
                    if config['boundary_weighting']:
                        losses=F.cross_entropy(logits,y,ignore_index=-100,reduction='none')
                        weights=w*(y>=0)
                        numerator=(losses*weights).sum();denominator=weights.sum()
                    else:
                        # Match the historical trainer's exact reduction order.
                        numerator=F.cross_entropy(logits,y,weight=class_weights,ignore_index=-100,reduction='sum')
                        denominator=class_weights[y[y>=0]].sum()
                    loss=numerator/denominator
                    assert torch.isfinite(loss),'Nonfinite training objective'
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();total+=float(numerator.detach());count+=float(denominator)
                raw_val,decoded_val=_validation(model,normalization,t['core_frames'],validation_inputs,decoder_settings,decode_fn)
                selected={}
                for kind,val in [('raw',raw_val),('decoded',decoded_val)]:
                    key=(val['spell_f1_mean'],val['macro_f1'])
                    selected[kind]=best[kind] is None or key>best[kind]
                    if selected[kind]:best[kind]=key;stale[kind]=0;best_epochs[kind]=epoch
                    else:stale[kind]+=1
                checkpoint=dict(schema_version=1,class_order=list(CLASS_ORDER),
                    state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
                    model_config=model_config,normalization=normalization,feature_schema=feature_schema(),
                    seed=config['seed'],fps=25,core_frames=t['core_frames'],data_kind='external',epoch=epoch,
                    validation=raw_val,decoded_validation=decoded_val,decoder_settings=decoder_settings,
                    architecture='tcn',ablation=config,augmentation=MODERATE if config['augmentation'] else None)
                epoch_path=dest/'epochs'/f'epoch_{epoch:03d}.pt'
                _save(epoch_path,checkpoint)
                for kind in ('raw','decoded'):
                    if selected[kind]:_copy(epoch_path,dest/(kind+'_best.pt'))
                row=dict(epoch=epoch,train_cross_entropy=total/count,raw=raw_val,decoded=decoded_val,
                         selected=selected,stale=stale.copy(),epoch_seconds=time.perf_counter()-before)
                with (dest/'epochs.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
                n=np.random.get_state()
                _save(resume_path,dict(epoch=epoch,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(),loader_rng=generator.get_state(),python_rng=random.getstate(),
                    numpy_rng=(n[0],n[1].tolist(),n[2],n[3],n[4]),best=best,stale=stale,best_epochs=best_epochs,
                    seconds=prior_seconds+time.perf_counter()-started))
                _json(status_path,dict(status='RUNNING',epoch=epoch,best_epochs=best_epochs,stale=stale,
                    epoch_windows=epoch_windows,optimizer_steps_per_epoch=optimizer_steps_per_epoch))
                print(json.dumps(dict(model=config['id'],epoch=epoch,raw_spellF1=raw_val['spell_f1_mean'],
                    decoded_spellF1=decoded_val['spell_f1_mean'],selected=selected,stale=stale)),flush=True)
                if all(value>=t['patience'] for value in stale.values()):break
        for kind in ('raw','decoded'):
            saved=torch.load(dest/(kind+'_best.pt'),map_location='cpu',weights_only=True)
            check=TCN(19,**saved['model_config']);check.load_state_dict(saved['state_dict']);check.eval()
            raw_val,decoded_val=_validation(check,normalization,t['core_frames'],validation_inputs,decoder_settings,decode_fn)
            assert raw_val==saved['validation'] and decoded_val==saved['decoded_validation'],'Reload validation mismatch'
        status=dict(status='COMPLETE',id=config['id'],epochs_run=epoch,best_epochs=best_epochs,stale=stale,
            raw_best=str(dest/'raw_best.pt'),decoded_best=str(dest/'decoded_best.pt'),
            raw_best_checkpoint=str(dest/'raw_best.pt'),decoded_best_checkpoint=str(dest/'decoded_best.pt'),
            seconds=prior_seconds+time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),
            receptive_field_frames=2*model.radius+1,train_windows=len(windows),epoch_windows=epoch_windows,
            optimizer_steps_per_epoch=optimizer_steps_per_epoch,reload_verified=True,
            stopping_reason='both_selectors_patience' if all(v>=t['patience'] for v in stale.values()) else 'maximum_epochs')
        _json(status_path,status)
        return status
    except Exception as error:
        _json(status_path,dict(status='FAILED',error=type(error).__name__+': '+str(error),
                             last_epoch=epoch,seconds=prior_seconds+time.perf_counter()-started))
        raise


def synthetic_window_checks():
    """Small public-data-free test of masking, targets and intervention isolation."""
    from tiki_data.contracts import MatchSpec
    from tiki_data.p2.data import fit_normalization, make_window
    from raw_features import batch_from_raw
    n=160;spec=MatchSpec('synthetic_historical_factorial',n,25)
    t=np.arange(n,dtype=np.float32)
    raw=dict(home_positions=np.stack([np.column_stack((t*.01+j,np.full(n,j))) for j in range(3)],axis=1),
             away_positions=np.stack([np.column_stack((t*.01+8+j,np.full(n,5+j))) for j in range(3)],axis=1),
             home_speeds=np.ones((n,3),np.float32),away_speeds=np.ones((n,3),np.float32),
             ball_positions=np.column_stack((t*.02,np.ones(n))),ball_speeds=np.ones(n,np.float32))
    base,_=batch_from_raw(raw,spec,hide_speed=True,with_age=False)
    labels=np.where(t<80,0,1).astype(np.int64)
    norm=fit_normalization([base],allow_unobserved=('ball_speed_mps',));norm.update(mask_ball_speed=True,train_match_ids=[spec.match_id])
    config=dict(id='test',seed=42,boundary_weighting=False,augmentation=False,long_context=False)
    ordinary=HistoricalWindows([(base,labels)],{spec.match_id:raw},norm,64,62,config)
    x,y,w=ordinary[0]
    expected=make_window(base,norm,0,n,0,64,62)
    assert np.array_equal(x.numpy(),expected)
    assert np.array_equal(y.numpy(),labels[:64]) and np.all(w.numpy()==1)
    augmented=HistoricalWindows([(base,labels)],{spec.match_id:raw},norm,64,62,dict(config,augmentation=True))
    augmented.epoch=3;ax,ay,aw=augmented[1];bx,by,bw=augmented[1]
    assert torch.equal(ax,bx) and torch.equal(ay,by) and torch.equal(aw,bw)
    assert torch.equal(ay,ordinary[1][1]) and torch.equal(aw,ordinary[1][2])
    assert not bool(ax[0].any()) and not bool(ax[19].any())
    weighted=HistoricalWindows([(base,labels)],{spec.match_id:raw},norm,64,62,dict(config,boundary_weighting=True))
    wx,wy,ww=weighted[1]
    assert torch.equal(wx,ordinary[1][0]) and torch.equal(wy,ordinary[1][1])
    assert torch.equal(ww[6:26],torch.full((20,),2.))
    assert int((ww==2).sum())==20
    original=TCN(19,width=32,kernel=3,dilations=[1,2,4,8,16],dropout=.1)
    long=TCN(19,width=32,kernel=3,dilations=[1,2,4,8,48],dropout=.1)
    assert sum(p.numel() for p in original.parameters())==sum(p.numel() for p in long.parameters())
    assert 2*original.radius+1==125 and 2*long.radius+1==253
    sample_data=list(range(19));cap=7;generator=torch.Generator().manual_seed(43)
    sampler=RandomSampler(sample_data,replacement=False,num_samples=cap,generator=generator)
    first=list(sampler);state=generator.get_state();second=list(sampler)
    repeated_generator=torch.Generator();repeated_generator.set_state(state)
    repeated=list(RandomSampler(sample_data,replacement=False,num_samples=cap,generator=repeated_generator))
    assert len(first)==cap and len(set(first))==cap and second==repeated
    assert len(DataLoader(sample_data,batch_size=3,sampler=sampler,shuffle=False))==3
    return dict(unaugmented_matches_original_window=True,raw_dropout_deterministic=True,
                targets_unchanged=True,ball_speed_always_hidden=True,boundary_weighting_isolated=True,
                context_parameter_count_equal=True,receptive_fields_125_and_253=True,
                capped_sampling_without_replacement=True,capped_sampler_generator_resume_exact=True)
