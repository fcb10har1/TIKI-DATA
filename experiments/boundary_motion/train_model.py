"""Comparable external-only training and cached evaluation for one ablation."""
import argparse
import gc
from common import *
import torch
from torch.utils.data import DataLoader
from tiki_data.p2.model import seed_all
from tiki_data.p2.data import fit_normalization
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.train import validate
from tiki_data.artifacts import write_batch
from feature_engine import Factory,TrainingWindows,schema
from models import fresh_model,load_model,classification_loss,consistency_penalty
from boundary import boundary_probabilities,evaluate_boundary
from decoder import BASE,decode

def evaluate_cached(probability_root,candidate_id,settings,family='model'):
    started=time.perf_counter();rows=[];diags=[]
    for spec,truth in truth_records():
        for scenario in SCENARIOS:
            probabilities=read_batch(Path(probability_root)/spec.match_id/scenario/'probabilities.npz',spec,'probabilities').probabilities
            visible=np.load(cache_folder(spec.match_id,scenario)/'visible.npy')
            labels=decode(probabilities,visible,spec,settings)
            store_decoded(candidate_id,spec,scenario,labels)
            rows.append(dict(candidate=candidate_id,match_id=spec.match_id,scenario=scenario,**metrics(truth,labels)))
            diags.append(dict(candidate=candidate_id,match_id=spec.match_id,scenario=scenario,**diagnostic(truth,labels)))
    return dict(id=candidate_id,family=family,settings=settings,status='COMPLETE',runtime_seconds=time.perf_counter()-started,
        mean=summarize(rows),rows=rows,diagnostics=diags)

def train(config):
    started=time.perf_counter();folder=OUT/'models'/config['id'];folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'config.json',config,exclusive=True);write_json(folder/'status.json',dict(status='RUNNING'))
    prior=read_json(OUT/'baseline/training_config.json');t=prior['training'];seed=config['seed']
    try:
        data=external_data();factories={};clean={}
        for split in data:
            factories[split]=[Factory(base,raw_load(base.match.match_id),config['speed']) for base,_ in data[split]]
            clean[split]=[(f.clean(),y) for f,(_,y) in zip(factories[split],data[split])]
        norm=fit_normalization([b for b,_ in clean['train']])
        norm.update(mask_ball_speed=False,train_match_ids=[b.match.match_id for b,_ in clean['train']],speed_policy=config['speed'])
        if config['speed']=='recorded':
            saved_norm=read_json(OUT/'baseline/normalization.json')
            for key in ('mean','scale','observed_count','feature_names'):assert norm[key]==saved_norm[key],'Reference preprocessing mismatch'
        write_json(folder/'normalization.json',norm)
        seed_all(seed,t['threads'])
        model=fresh_model(len(norm['feature_names']),prior['model'],config['architecture'],config['speed'])
        windows=TrainingWindows(data['train'],factories['train'],norm,t['core_frames'],model.radius,
            prior['augmentation'],seed,config['boundary_radius'],config['boundary_weight'])
        loader=DataLoader(windows,batch_size=t['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(seed))
        optimizer=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay'])
        assert t['class_weights']=='none'
        best=None;stale=0
        for epoch in range(1,t['epochs']+1):
            windows.epoch=epoch;model.train();loss_sum=0.;steps=0
            for x,y,w in loader:
                optimizer.zero_grad(set_to_none=True)
                per_frame,logits=classification_loss(model,x,y,model.radius,t['core_frames'])
                weights=w*(y>=0);loss=(per_frame*weights).sum()/weights.sum()
                if config['consistency']:
                    loss=loss+config['consistency']*consistency_penalty(logits,y)
                assert torch.isfinite(loss),'Nonfinite objective'
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),t['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();loss_sum+=float(loss.detach());steps+=1
            val=validate(model,clean['validation'],norm,t['core_frames'],'cpu')
            key=(val['spell_f1_mean'],val['macro_f1']);improved=best is None or key>best
            row=dict(epoch=epoch,training_objective=loss_sum/steps,validation=val,selected=improved)
            with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(model=config['id'],epoch=epoch,spellF1=key[0],macroF1=key[1],selected=improved)),flush=True)
            if improved:
                best=key;stale=0
                checkpoint=dict(schema_version=1,class_order=list(CLASS_ORDER),state_dict={k:v.detach().clone() for k,v in model.state_dict().items()},
                    model_config=prior['model'],normalization=norm,feature_schema=schema(config['speed']),seed=seed,fps=25,
                    core_frames=t['core_frames'],data_kind='external',epoch=epoch,validation=val,architecture=config['architecture'],
                    ablation=config,augmentation=prior['augmentation'])
                torch.save(checkpoint,folder/'best.pt')
            else:stale+=1
            if stale>=t['patience'] or epoch==t['epochs']:break
            if time.perf_counter()-started>read_json(OUT/'manifest.json')['training_budget']['per_run_seconds']:
                raise TimeoutError('Predeclared per-run training budget exceeded before the unchanged stopping rule completed')
        reloaded,saved=load_model(folder/'best.pt')
        repeated=validate(reloaded,clean['validation'],norm,t['core_frames'],'cpu')
        assert repeated==saved['validation'],'Checkpoint reload metrics differ'
        status=dict(status='COMPLETE',seconds=time.perf_counter()-started,epochs_run=epoch,best_epoch=saved['epoch'],
            parameters=sum(p.numel() for p in model.parameters()),train_windows=len(windows),reload_verified=True,
            checkpoint=file_record(folder/'best.pt'),training_match_ids=norm['train_match_ids'])
        write_json(folder/'status.json',status)
    except Exception as error:
        write_json(folder/'status.json',dict(status='FAILED',seconds=time.perf_counter()-started,error=type(error).__name__+': '+str(error)))
        raise

def cache_and_evaluate(model_id):
    folder=OUT/'models'/model_id;config=read_json(folder/'config.json');seed_all(config['seed'],1)
    model,saved=load_model(folder/'best.pt');records=external_data()['validation']
    status=read_json(folder/'status.json')
    if status.get('status')!='COMPLETE':
        clean=[(Factory(base,raw_load(base.match.match_id),config['speed']).clean(),y) for base,y in records]
        assert validate(model,clean,saved['normalization'],saved['core_frames'],'cpu')==saved['validation'],'Partial checkpoint reload mismatch'
    probability_root=OUT/'model_predictions'/model_id
    started=time.perf_counter()
    for base,_ in records:
        factory=Factory(base,raw_load(base.match.match_id),config['speed'])
        for scenario in SCENARIOS:
            missing=np.load(cache_folder(base.match.match_id,scenario)/'missing.npy')
            features=factory.corrupted(missing,scenario in ('speed_missing','combined'))
            probabilities=predict_batch(features,model,saved['normalization'],saved['core_frames'])
            write_batch(probability_root/base.match.match_id/scenario/'probabilities.npz',probabilities)
            if config['architecture']=='boundary':
                bp=boundary_probabilities(features,model,saved['normalization'],saved['core_frames'])
                np.save(probability_root/base.match.match_id/scenario/'boundary.npy',bp)
    result=evaluate_cached(probability_root,model_id+'__baseline_decoder',BASE)
    result['training_config']=config;result['training_runtime_seconds']=status.get('seconds')
    result['inference_and_evaluation_seconds']=time.perf_counter()-started
    write_json(folder/'evaluation.json',result)
    if config['architecture']=='boundary':
        grid=[evaluate_boundary(probability_root,model_id+'_strength_'+str(v),BASE,v) for v in (.5,1.,2.)]
        write_json(folder/'boundary_grid.json',grid)
    print(json.dumps(dict(model=model_id,external_mean=result['mean'])),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['train','evaluate']);parser.add_argument('value')
    args=parser.parse_args()
    if args.stage=='train':train(read_json(args.value))
    else:cache_and_evaluate(args.value)

if __name__=='__main__':main()
