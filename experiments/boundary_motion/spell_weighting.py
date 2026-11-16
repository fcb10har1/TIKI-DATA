"""Capped live-spell weighting computed before windowing from training labels only."""
import numpy as np


def complete_live_spells(labels,periods):
    """Yield uncensored H/A spells; unknown runs and periods break continuity."""
    labels=np.asarray(labels)
    for left,right in periods:
        edges=np.r_[left,np.flatnonzero(labels[left+1:right]!=labels[left:right-1])+left+1,right]
        for start,stop in zip(edges[:-1],edges[1:]):
            if labels[start] not in (0,1):continue
            if start==left or stop==right:continue
            if labels[start-1]<0 or labels[stop]<0:continue
            yield int(start),int(stop)


def fit_reference_seconds(records):
    durations=[(b-a)/fps for labels,periods,fps in records for a,b in complete_live_spells(labels,periods)]
    if not durations:raise ValueError('No complete external training possessions')
    return float(np.median(durations))


def spell_weights(labels,periods,fps,reference_seconds,cap=4.):
    if fps<=0 or reference_seconds<=0 or cap<1:raise ValueError('Invalid spell-weighting configuration')
    labels=np.asarray(labels);weights=np.zeros(len(labels),np.float32)
    for a,b in periods:weights[a:b]=np.where(labels[a:b]>=0,1.,0.)
    for a,b in complete_live_spells(labels,periods):
        weights[a:b]=np.clip(np.sqrt(reference_seconds/((b-a)/fps)),1.,cap)
    return weights


def train_worker(config_path):
    from common import OUT,read_json,write_json
    import train_model
    from feature_engine import TrainingWindows
    config=read_json(config_path)

    class WeightedWindows(TrainingWindows):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            records=self.records
            fitted=fit_reference_seconds([(y,b.match.periods,b.match.fps) for b,y in records])
            self.loss_weights=[spell_weights(y,b.match.periods,b.match.fps,fitted,config['spell_weight_cap']) for b,y in records]
            folder=OUT/'models'/config['id']
            norm=read_json(folder/'normalization.json');old=read_json(OUT/'models/A42/normalization.json')
            keys=('feature_names','mean','scale','observed_count','train_match_ids')
            assert all(norm[k]==old[k] for k in keys),'Reference preprocessing changed'
            write_json(folder/'spell_weighting.json',dict(reference_seconds=fitted,cap=config['spell_weight_cap'],
                formula='clip(sqrt(training_median_complete_live_spell_seconds / spell_seconds),1,cap)',
                train_match_ids=[b.match.match_id for b,_ in records],
                complete_training_spells=sum(1 for b,y in records for _ in complete_live_spells(y,b.match.periods)),
                unknown_weight=0,censored_weight=1,dead_ball_weight=1,reference_normalization_identical=True),exclusive=True)
    train_model.TrainingWindows=WeightedWindows
    train_model.train(config)


if __name__=='__main__':
    import sys
    train_worker(sys.argv[1])
