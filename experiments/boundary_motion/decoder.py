"""Separate transition costs, duration choices, logit temperature and age weights."""
from dataclasses import replace
import numpy as np
from p3.decoder import DecoderConfig,_Chain
from temporal_decoder import sparse_viterbi
from raw_features import age_seconds

BASE=dict(min_live_s=1.,min_dead_s=1.,duration='hard',short_penalty=1.,
    turnover_penalty=-2.,stoppage_penalty=-1.,temperature=2.,missing_weight=1.,evidence='fixed')

def candidates():
    out=[dict(id='baseline',family='reference',settings=BASE.copy())]
    def add(name,family,**changes):out.append(dict(id=name,family=family,settings=dict(BASE,**changes)))
    for field,tag in (('min_live_s','live'),('min_dead_s','dead')):
        for value in (.5,.2,.04):add('duration_'+tag+'_'+str(value),'minimum_'+tag,**{field:value})
    add('duration_soft','soft_duration',duration='soft')
    for turn in (-2.,-1.,0.):
        for dead in (-2.,-1.,0.):
            if turn==dead==-1.:continue
            add('transition_'+str(turn)+'_'+str(dead),'transitions',turnover_penalty=turn,stoppage_penalty=dead)
    for temp in (2.,3.,5.,6.):add('temperature_'+str(temp),'temperature',temperature=temp)
    for weight in (0.,.25,.75,1.):add('evidence_'+str(weight),'evidence',missing_weight=weight)
    add('evidence_age_decay','evidence',evidence='age_decay')
    return out

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
