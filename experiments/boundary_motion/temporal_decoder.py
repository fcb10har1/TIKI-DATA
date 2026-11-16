"""Exact sparse Viterbi, with optional finite penalties for short spells.

Hard mode at transition_penalty=0 reproduces the existing decoder. Soft mode
adds early exits with the normal exit cost plus short_penalty*(L-age)/L.
All one-frame-or-longer durations remain possible; this is an energy model,
so the extra edge scores are deliberately not row-normalized probabilities.
"""
import numpy as np
from numba import njit
from p3.decoder import DecoderConfig, _Chain, _log_emission


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


def decode(probabilities, visible, spec, settings):
    cfg = DecoderConfig(fps=spec.fps, temperature=4.0, missing_weight=settings['missing_weight'],
        min_poss_s=1.0,min_dead_s=1.0,mean_poss_s=14.0,mean_dead_s=10.0,
        turnover_frac=.5,restart_switch=.8,start_alive=.5,prob_floor=1e-4,use_mbr=False,seed=42)
    chain,pred,weights = transitions(cfg,settings['duration'],settings['transition_penalty'],settings['short_penalty'])
    output = np.full(spec.n_frames,'X',dtype='<U1')
    for a,b in spec.periods:
        e = _log_emission(probabilities[a:b],visible[a:b],cfg,chain)
        path = sparse_viterbi(chain.logpi,pred,weights,e)
        output[a:b] = np.asarray(['H','A','D'])[chain.team[path]]
    return output
