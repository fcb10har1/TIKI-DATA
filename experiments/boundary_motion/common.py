"""Shared external-only experiment utilities. No competition data are opened."""
import csv
import json
from pathlib import Path
import sys
import time

SOURCE=Path(__file__).resolve().parent
ROOT=SOURCE.parents[1]
OUT=ROOT/'runs/boundary-motion-20261116'
PREVIOUS=ROOT/'runs/possession-controlled-study-20261116'
OLD=ROOT/'runs/missingness-decoder-20261116'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD/'deps'))

import numpy as np
from tiki_data.artifacts import read_batch,read_json,read_predictions,write_json
from tiki_data.benchmark import load_manifest,score_match
from tiki_data.contracts import MatchSpec,PredictionBatch,CLASS_ORDER
from tiki_data.provenance import file_record
from evaluation import possession_metrics as pm

SCENARIOS=('clean','point_10pct','gap_2s','gap_10s','speed_missing','combined')
METRICS=('accuracy_pct','macroF1','spellF1','dead_ball_recall')

def external_entries(splits=('train','validation')):
    # load_manifest validates paths/groups; unlike load_data it never opens or
    # hashes competition ground truth. Only permitted external entries are read.
    m=load_manifest(ROOT/'data/external/idsse_processed/manifest.json')
    assert m['kind']=='external'
    return [e for e in m['matches'] if e['split'] in splits]

def external_data():
    data={'train':[],'validation':[]}
    for e in external_entries():
        spec=MatchSpec.from_dict(e['match'])
        base=read_batch(ROOT/'data/external/idsse_processed/matches'/spec.match_id/'features.npz',spec,'features')
        truth=read_predictions(e['ground_truth'],spec)
        codes=np.full(spec.n_frames,-100,dtype=np.int64)
        for i,label in enumerate(CLASS_ORDER):codes[truth.labels==label]=i
        data[e['split']].append((base,codes))
    return data

def raw_load(mid):
    keys=('home_positions','away_positions','home_speeds','away_speeds','ball_positions','ball_speeds')
    return {k:np.load(OLD/'raw_arrays'/mid/(k+'.npy'),mmap_mode='r') for k in keys}

def cache_folder(mid,scenario):return PREVIOUS/'baseline/predictions'/mid/scenario

def truth_records():
    return [(MatchSpec.from_dict(e['match']),read_predictions(e['ground_truth'],MatchSpec.from_dict(e['match']))) for e in external_entries(('validation',))]

def metrics(truth,labels):
    result=score_match(truth,PredictionBatch(truth.match,truth.frames,labels),truth.match)
    cm=np.asarray(result['confusion'])
    return dict(accuracy_pct=float(100*np.trace(cm[:3,:3])/cm[:3].sum()),
        macroF1=float(result['macro_f1']),spellF1=float(result['spell_f1_mean']),
        dead_ball_recall=float(cm[2,2]/cm[2].sum()))

def csv_write(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def summarize(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in METRICS}

def diagnostic(truth,labels):
    """Official greedy same-team IoU>=0.5 matches plus overlap diagnostics.

    A split/merge requires positive same-team overlap with >=2 spells. Halves
    and unlabelled runs are independent. Timing uses matched live possessions.
    """
    gt=pm.label_codes(truth.labels);pred=pm.label_codes(labels)
    totals=dict(gt_possessions=0,pred_possessions=0,split_gt_possessions=0,merged_pred_possessions=0,
        matched_possessions=0,missed_possessions=0,extra_possessions=0,wrong_team_spells=0,
        wrong_team_frames=0,gt_live_frames=0,short_gt_possessions=0,matched_short_possessions=0)
    start_errors=[];end_errors=[]
    for a,b in truth.match.periods:
        runs=pm.valid_runs(gt[a:b])
        gs=pm.possession_spells(gt[a:b],runs);ps=pm.possession_spells(pred[a:b],runs)
        overlaps=pm.spell_overlaps(gs,ps)
        gc=np.zeros(len(gs),int);pc=np.zeros(len(ps),int)
        for _,i,j in overlaps:gc[i]+=1;pc[j]+=1
        usedg=set();usedp=set()
        for iou,i,j in overlaps:
            if iou<.5:break
            if i in usedg or j in usedp:continue
            usedg.add(i);usedp.add(j)
            start_errors.append((ps[j][0]-gs[i][0])/truth.match.fps)
            end_errors.append((ps[j][1]-gs[i][1])/truth.match.fps)
        wrong=[]
        for i,(s,t,c) in enumerate(gs):
            for j,(u,v,d) in enumerate(ps):
                overlap=min(t,v)-max(s,u)
                if overlap>0 and c!=d:
                    iou=overlap/(max(t,v)-min(s,u))
                    if iou>=.5:wrong.append((-iou,i,j))
        wg=set();wp=set()
        for _,i,j in sorted(wrong):
            if i not in wg and j not in wp:wg.add(i);wp.add(j)
        live=np.isin(gt[a:b],[0,1]);pred_live=np.isin(pred[a:b],[0,1])
        short={i for i,(s,t,_) in enumerate(gs) if t-s<truth.match.fps}
        add=dict(gt_possessions=len(gs),pred_possessions=len(ps),split_gt_possessions=int((gc>1).sum()),
            merged_pred_possessions=int((pc>1).sum()),matched_possessions=len(usedg),
            missed_possessions=len(gs)-len(usedg),extra_possessions=len(ps)-len(usedp),
            wrong_team_spells=len(wg),wrong_team_frames=int((live&pred_live&(gt[a:b]!=pred[a:b])).sum()),
            gt_live_frames=int(live.sum()),short_gt_possessions=len(short),matched_short_possessions=len(short&usedg))
        for key,value in add.items():totals[key]+=value
    totals.update(start_bias_s=float(np.mean(start_errors)) if start_errors else None,
        end_bias_s=float(np.mean(end_errors)) if end_errors else None,
        start_mae_s=float(np.mean(np.abs(start_errors))) if start_errors else None,
        end_mae_s=float(np.mean(np.abs(end_errors))) if end_errors else None)
    return totals

def store_decoded(candidate,spec,scenario,labels):
    p=OUT/'decoded'/candidate/spec.match_id/(scenario+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,labels=labels)

def saved_decoded(candidate,mid,scenario):
    with np.load(OUT/'decoded'/candidate/mid/(scenario+'.npz')) as d:return d['labels'].copy()

def check_preserved():
    for rec in read_json(PREVIOUS/'baseline/source_records.json'):
        assert file_record(rec['path'])['sha256']==rec['sha256'],'Baseline source changed'

def now():return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
