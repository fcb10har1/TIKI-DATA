"""The continued verifier's frozen sparse duration decoder."""
import numpy as np
from numba import njit
from p3.decoder import DecoderConfig, _Chain, _log_emission
from tiki_data.contracts import CLASS_ORDER

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


def transitions(base, duration='hard', transition_penalty=0.0, short_penalty=1.0):
    if duration not in ('hard','soft'): raise ValueError('Unknown duration mode')
    if short_penalty < 0 or not np.isfinite([transition_penalty,short_penalty]).all():
        raise ValueError('Invalid penalty')
    chain = _Chain(base)
    loga = chain.logA.copy()
    if duration == 'soft':
        for state in range(4):
            entry = int(chain.offsets[state])
            length = chain.lengths[state]
            free = entry + length - 1
            destinations = np.flatnonzero(np.isfinite(loga[free]) & (chain.base != state))
            for age in range(1,length):
                source = entry+age-1
                loga[source,destinations] = loga[free,destinations] - short_penalty*(length-age)/length
    switching = chain.base[:,None] != chain.base[None,:]
    loga[switching] -= transition_penalty
    width = int(np.isfinite(loga).sum(axis=0).max())
    predecessors = np.full((chain.K,width),-1,dtype=np.int16)
    weights = np.full((chain.K,width),-np.inf)
    for dest in range(chain.K):
        sources = np.flatnonzero(np.isfinite(loga[:,dest]))
        predecessors[dest,:len(sources)] = sources
        weights[dest,:len(sources)] = loga[sources,dest]
    return chain,predecessors,weights


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
