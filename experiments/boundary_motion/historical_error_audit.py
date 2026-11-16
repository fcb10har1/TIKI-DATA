"""Aggregate-only frozen-reference diagnosis. No training or parameter search."""
from common import *
import subprocess
from temporal_decoder import decode
from games import GAME,GAMES

DEST=ROOT/'runs/historical-two-reference-diagnosis-20261116'


def aggregate(truth,labels):
    result=score_match(truth,PredictionBatch(truth.match,truth.frames,labels),truth.match)
    return dict(**metrics(truth,labels),spell_counts=dict(ground_truth=result['n_gt_spells'],predicted=result['n_pred_spells']),
                per_iou=result['spell_f1'],error_summary=diagnostic(truth,labels))


def main():
    DEST.mkdir(exist_ok=False)
    source=ROOT/'runs/full-context-model-comparison-20261116'
    old=read_json(source/'freeze.json')
    names=['historical_best_spell_recipe','historical_90_12_accuracy_recipe']
    variants=[r for r in old['variants'] if r['id'] in names]
    freeze=dict(references=variants,checkpoint=old['models']['legacy_B']['checkpoint'],
        predictions=[file_record(source/'predictions'/(n+'.csv')) for n in names],
        source=file_record(__file__),scorer=file_record(ROOT/'evaluation/possession_metrics.py'),
        purpose='User-designated references for further work. Diagnostic aggregates only; no active submission change or public-based tuning.')
    write_json(DEST/'references.json',freeze,exclusive=True)
    external=[]
    for spec,truth in truth_records():
        for scenario in ('speed_available','gap_2s','gap_10s'):
            folder=ROOT/'runs/abcd-comparison-20261111/external'/spec.match_id/scenario
            probabilities=read_batch(folder/'B_probabilities.npz',spec,'probabilities').probabilities
            with np.load(folder/'ball_observed.npz') as saved:visible=saved['ball_observed']
            for v in variants:
                labels=decode(probabilities,visible,spec,v['settings'])
                external.append(dict(candidate=v['id'],match_id=spec.match_id,scenario=scenario,**aggregate(truth,labels)))
    write_json(DEST/'external_aggregates.json',external,exclusive=True)
    spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25)
    truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
    public=[]
    for name in names:
        labels=read_predictions(source/'predictions'/(name+'.csv'),spec).labels
        public.append(dict(candidate=name,**aggregate(truth,labels)))
    write_json(DEST/'public_aggregates.json',public,exclusive=True)
    means=[]
    for name in names:
        for scenario in ('speed_available','gap_2s','gap_10s'):
            rows=[r for r in external if r['candidate']==name and r['scenario']==scenario]
            means.append(dict(candidate=name,scenario=scenario,**summarize(rows),
                error_totals={k:sum(r['error_summary'][k] for r in rows) for k in ['gt_possessions','pred_possessions','matched_possessions','missed_possessions','extra_possessions','split_gt_possessions','merged_pred_possessions','short_gt_possessions','matched_short_possessions']},
                per_iou_mean={str(t):float(np.mean([r['per_iou'][t]['f1'] for r in rows])) for t in pm.SPELL_IOUS}))
    write_json(DEST/'external_means.json',means,exclusive=True)
    (DEST/'command.txt').write_text('python experiments/boundary_motion/historical_error_audit.py\n')
    for r in freeze['predictions']+[freeze['checkpoint']]:assert file_record(r['path'])['sha256']==r['sha256']
    audit=DEST/'verification.json';paths={p.relative_to(ROOT).as_posix() for p in DEST.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    result=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',capture_output=True,text=True,check=True)
    assert set(result.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(DEST.relative_to(ROOT))],cwd=ROOT)
    write_json(audit,dict(aggregate_only=True,generated_files_ignored=True,tracked_outputs=0,references_preserved=True,training_or_tuning=False))
    print(json.dumps(dict(public_aggregate_diagnosis=public,external_aggregate_diagnosis=means)),flush=True)


if __name__=='__main__':main()
