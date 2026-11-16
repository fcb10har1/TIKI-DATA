from common import *
import torch
from numba import njit
from tiki_data.p2.data import make_window,timeline_blocks
from decoder import chain_for,calibrated_log_probabilities

@torch.inference_mode()
def boundary_probabilities(features,model,norm,core):
    out=np.zeros((features.match.n_frames,3))
    model.eval()
    for a,b in timeline_blocks(features,include_outside=True):
        for start in range(a,b,core):
            stop=min(start+core,b);x=make_window(features,norm,a,b,start,core,model.radius)
            _,logits=model.components(torch.from_numpy(x[None]))
            out[start:stop]=torch.sigmoid(logits[0,:,model.radius:model.radius+stop-start].double()).T.numpy()
    return out

@njit(cache=True)
def boundary_viterbi(logpi,pred,weights,emissions,boundaries,kinds,strength):
    n,k=emissions.shape;delta=logpi+emissions[0];back=np.zeros((n,k),np.int16)
    for t in range(1,n):
        nxt=np.empty(k)
        for dest in range(k):
            best=-np.inf;source=0
            for j in range(pred.shape[1]):
                prev=pred[dest,j]
                if prev<0:break
                kind=kinds[dest,j];adjust=0.
                if kind>=0:adjust=strength*(2*boundaries[t,kind]-1)
                score=delta[prev]+weights[dest,j]+adjust
                if score>best:best=score;source=prev
            nxt[dest]=best+emissions[t,dest];back[t,dest]=source
        delta=nxt
    path=np.empty(n,np.int16);path[-1]=np.argmax(delta)
    for t in range(n-1,0,-1):path[t-1]=back[t,path[t]]
    return path

def decode_boundary(probabilities,visible,spec,settings,boundaries,strength):
    _,chain,pred,weights=chain_for(settings,spec.fps)
    kinds=np.full(pred.shape,-1,np.int8)
    for dest in range(chain.K):
        for j,prev in enumerate(pred[dest]):
            if prev<0:continue
            a,b=chain.base[prev],chain.base[dest]
            if a<2 and b<2 and a!=b:kinds[dest,j]=0
            elif a<2 and b>=2:kinds[dest,j]=1
            elif a>=2 and b<2:kinds[dest,j]=2
    labels=np.full(spec.n_frames,'X',dtype='<U1')
    for a,b in spec.periods:
        e=calibrated_log_probabilities(probabilities[a:b],settings['temperature'])
        e*=np.where(visible[a:b],1.,settings['missing_weight'])[:,None]
        path=boundary_viterbi(chain.logpi,pred,weights,e[:,chain.team],boundaries[a:b],kinds,strength)
        labels[a:b]=np.asarray(['H','A','D'])[chain.team[path]]
    return labels

def evaluate_boundary(root,name,settings,strength):
    started=time.perf_counter();rows=[];diags=[]
    for spec,truth in truth_records():
        for scenario in SCENARIOS:
            path=Path(root)/spec.match_id/scenario
            p=read_batch(path/'probabilities.npz',spec,'probabilities').probabilities
            boundary=np.load(path/'boundary.npy');visible=np.load(cache_folder(spec.match_id,scenario)/'visible.npy')
            labels=decode_boundary(p,visible,spec,settings,boundary,strength)
            store_decoded(name,spec,scenario,labels)
            rows.append(dict(candidate=name,match_id=spec.match_id,scenario=scenario,**metrics(truth,labels)))
            diags.append(dict(candidate=name,match_id=spec.match_id,scenario=scenario,**diagnostic(truth,labels)))
    return dict(id=name,status='COMPLETE',settings=settings,strength=strength,mean=summarize(rows),rows=rows,diagnostics=diags,runtime_seconds=time.perf_counter()-started)
