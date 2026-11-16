"""Frozen sparse Viterbi implementation for the selected possession ensemble."""
import numpy as np
from numba import njit
from p3.decoder import DecoderConfig, _Chain
from .selected_features import age_seconds

@njit(cache=True)
def sparse_viterbi(logpi, predecessors, weights, emissions):
    n,k = emissions.shape
    delta = logpi + emissions[0]
    back = np.zeros((n,k),dtype=np.int16)
    for t in range(1,n):
        next_delta = np.empty(k)
        for dest in range(k):
            best = -np.inf
            source = 0
            for j in range(predecessors.shape[1]):
                pred = predecessors[dest,j]
                if pred < 0: break
                score = delta[pred] + weights[dest,j]
                if score > best:
                    best,source = score,pred
            next_delta[dest] = best + emissions[t,dest]
            back[t,dest] = source
        delta = next_delta
    path = np.empty(n,dtype=np.int16)
    path[-1] = np.argmax(delta)
    for t in range(n-1,0,-1): path[t-1] = back[t,path[t]]
    return path


def chain_for(settings,fps):
    cfg=DecoderConfig(fps=fps,min_poss_s=settings['min_live_s'],min_dead_s=settings['min_dead_s'],
        temperature=settings['temperature'],missing_weight=settings['missing_weight'],seed=42)
    chain=_Chain(cfg);loga=chain.logA.copy()
    if settings['duration']=='soft':
        for state in range(4):
            entry=int(chain.offsets[state]);length=chain.lengths[state];free=entry+length-1
            dest=np.flatnonzero(np.isfinite(loga[free])&(chain.base!=state))
            for age in range(1,length):
                loga[entry+age-1,dest]=loga[free,dest]-settings['short_penalty']*(length-age)/length
    base=chain.base
    turn=((base[:,None]==0)&(base[None,:]==1))|((base[:,None]==1)&(base[None,:]==0))
    stoppage=(base[:,None]<2)!=(base[None,:]<2)
    loga[turn]-=settings['turnover_penalty'];loga[stoppage]-=settings['stoppage_penalty']
    width=int(np.isfinite(loga).sum(axis=0).max())
    predecessors=np.full((chain.K,width),-1,dtype=np.int16);weights=np.full((chain.K,width),-np.inf)
    for j in range(chain.K):
        i=np.flatnonzero(np.isfinite(loga[:,j]));predecessors[j,:len(i)]=i;weights[j,:len(i)]=loga[i,j]
    return cfg,chain,predecessors,weights


def calibrated_log_probabilities(probabilities,temperature):
    # log(P) recovers logits up to a frame-wise constant. Retain the baseline's
    # fixed 1e-4 floor BEFORE scaling, then explicitly softmax scaled logits.
    logits=np.log(np.clip(np.asarray(probabilities,dtype=np.float64),1e-4,1.))
    scaled=logits/temperature
    scaled-=scaled.max(axis=1,keepdims=True)
    return scaled-np.log(np.exp(scaled).sum(axis=1,keepdims=True))


def decode(probabilities,visible,spec,settings):
    cfg,chain,pred,weights=chain_for(settings,spec.fps)
    labels=np.full(spec.n_frames,'X',dtype='<U1')
    for a,b in spec.periods:
        lp=calibrated_log_probabilities(probabilities[a:b],settings['temperature'])
        w=np.where(visible[a:b],1.,settings['missing_weight'])
        if settings['evidence']=='age_decay':
            age=age_seconds(visible[a:b],spec.fps)
            w=np.where(visible[a:b],1.,.25+.5*np.exp(-age/2.))
        emission=(lp*w[:,None])[:,chain.team]
        path=sparse_viterbi(chain.logpi,pred,weights,emission)
        labels[a:b]=np.asarray(['H','A','D'])[chain.team[path]]
    return labels
