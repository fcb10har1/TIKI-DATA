"""User-requested diagnostic public scoring of preserved incomplete A/B models."""
from common import *
import subprocess
from public_full import raw_players
from games import GAME,GAMES
from tiki_data.p1.metrica import build_metrica_features
from tiki_data.p1.kalman import metrica_positions
from tiki_data.p2.selected_features import build_features
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.model import seed_all
from tiki_data.artifacts import write_predictions
from models import load_model
from feature_engine import Factory
from boundary import boundary_probabilities,decode_boundary
from decoder import decode,BASE


def main():
    dest=OUT/'public_ab_diagnostic';dest.mkdir(exist_ok=False)
    started=time.perf_counter()
    old=read_json(OUT/'freeze.json')
    preserved=old['source_hashes']+old['configuration_hashes']+old['selected_checkpoints']+old['reference_checkpoints']
    for r in preserved:assert file_record(r['path'])['sha256']==r['sha256']
    training={r['model']:r for r in csv.DictReader((OUT/'training_summary.csv').open())}
    checkpoints={name:file_record(OUT/'models'/name/'best.pt') for name in ('A42','B42')}
    for name,r in checkpoints.items():assert r['sha256']==training[name]['checkpoint_sha256']
    grid=read_json(OUT/'models/A42/boundary_grid.json')
    enabled=max(grid,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
    strength=enabled['strength']
    active=ROOT/'runs/active_possession_candidate.json'
    reference_csv=OUT/'public/selected_full_match.csv'
    protected=[file_record(reference_csv),file_record(OUT/'freeze.json')]
    if active.exists():protected.append(file_record(active))
    freeze=dict(created_utc=now(),purpose='Explicit user-requested additional public diagnostic batch; no selection or tuning',
        checkpoints=checkpoints,decoder=BASE,A_external_best_strength=0.,A_enabled_external_best_strength=strength,
        variants=['reference','A_boundary_disabled','A_boundary_enabled','B_relative_motion'],
        training_status='Both A/B are time-capped incomplete seed42 checkpoints',selected_unchanged=old['selected'],
        full_match_context=True,source=file_record(__file__),protected=protected)
    write_json(dest/'freeze.json',freeze,exclusive=True)
    (dest/'command.txt').write_text('python experiments/boundary_motion/public_ab.py\n')
    print('Frozen A/B diagnostic settings; preparing full-match features locally.',flush=True)
    seed_all(42,1)
    spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25);pitch=GAMES[GAME]['pitch']
    metadata=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.xml'
    tracking=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.txt'
    base=build_metrica_features(spec,metadata,tracking,pitch)
    ball=metrica_positions(spec,tracking,pitch);visible=np.isfinite(ball).all(1)
    features=build_features(base,ball)
    model,saved=load_model(checkpoints['A42']['path'])
    p=predict_batch(features,model,saved['normalization'],saved['core_frames']).probabilities
    boundary=boundary_probabilities(features,model,saved['normalization'],saved['core_frames'])
    predictions={
        'A_boundary_disabled':decode(p,visible,spec,BASE),
        'A_boundary_enabled':decode_boundary(p,visible,spec,BASE,boundary,strength)}
    print('A inference complete; computing identity-continuous relative-motion features for B.',flush=True)
    raw=raw_players(spec,metadata,tracking,pitch,ball)
    relative=Factory(base,raw,'relative').clean()
    model,saved=load_model(checkpoints['B42']['path'])
    p=predict_batch(relative,model,saved['normalization'],saved['core_frames']).probabilities
    predictions['B_relative_motion']=decode(p,visible,spec,BASE)
    for name,labels in predictions.items():
        write_predictions(dest/(name+'.csv'),PredictionBatch(spec,base.frames,labels).validate(spec))
    reference=read_predictions(reference_csv,spec)
    # All model outputs and configurations were fixed before this labels read.
    truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
    rows=[dict(candidate='reference',**metrics(truth,reference.labels))]
    rows += [dict(candidate=name,**metrics(truth,labels)) for name,labels in predictions.items()]
    previous=read_json(OUT/'public/comparison.json')['rows'][0]
    for key in METRICS:assert abs(rows[0][key]-previous[key])<1e-12
    for row in rows:
        for key in METRICS:row['change_'+key]=row[key]-rows[0][key]
    for r in protected+list(checkpoints.values())+preserved:
        assert file_record(r['path'])['sha256']==r['sha256']
    write_json(dest/'comparison.json',dict(rows=rows,diagnostic_only=True,selected_unchanged=old['selected'],
        A_enabled_strength=strength,seconds=time.perf_counter()-started),exclusive=True)
    csv_write(dest/'comparison.csv',rows)
    audit=dest/'verification.json'
    paths={p.relative_to(ROOT).as_posix() for p in dest.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    result=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',capture_output=True,text=True,check=True)
    assert set(result.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(dest.relative_to(ROOT))],cwd=ROOT)
    write_json(audit,dict(generated_files_ignored=True,tracked_generated_files=0,frozen_assets_unchanged=True,
        selected_unchanged=old['selected'],no_training_or_tuning=True,source=file_record(__file__)),exclusive=True)
    print(json.dumps(dict(public_aggregate_metrics=rows,selected_unchanged=old['selected'])),flush=True)


if __name__=='__main__':main()
