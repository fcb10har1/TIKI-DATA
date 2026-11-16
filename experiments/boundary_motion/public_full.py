"""One frozen local public evaluation; never print rows, tracking or labels."""
from common import *
import subprocess
import hashlib
from games import GAME,GAMES
from tiki_data.p1.metrica import build_metrica_features,team_slots,parse_observation
from tiki_data.p1.kalman import metrica_positions
from tiki_data.p2.selected import predict as predict_reference
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.model import seed_all
from tiki_data.artifacts import write_predictions
from models import load_model
from feature_engine import Factory
from boundary import boundary_probabilities,decode_boundary
from decoder import decode

def raw_players(spec,metadata,path,pitch,ball):
    home,slots=team_slots(metadata);indices={s:[i for i,x in enumerate(slots) if (x==home)==(s=='home')] for s in ('home','away')}
    raw=dict(ball_positions=ball,ball_speeds=np.full(spec.n_frames,np.nan,dtype=np.float32))
    for side in indices:
        raw[side+'_positions']=np.full((spec.n_frames,len(indices[side]),2),np.nan,dtype=np.float32)
        raw[side+'_speeds']=np.full((spec.n_frames,len(indices[side])),np.nan,dtype=np.float32)
    with Path(path).open() as handle:
        for n,line in enumerate(handle):
            frame,players,_=line.rstrip().split(':');assert int(frame)==n+1
            groups=players.split(';');assert len(groups)==len(slots)
            for side,ids in indices.items():
                for j,i in enumerate(ids):
                    item=parse_observation(groups[i])
                    if item is not None:
                        raw[side+'_positions'][n,j]=(item[0]*pitch[0],item[1]*pitch[1])
                        if item[2] is not None:raw[side+'_speeds'][n,j]=item[2]
    assert n+1==spec.n_frames
    return raw

def main():
    freeze=read_json(OUT/'freeze.json')
    for rec in freeze['source_hashes']+freeze['configuration_hashes']+freeze['selected_checkpoints']+freeze['reference_checkpoints']:
        assert file_record(rec['path'])['sha256']==rec['sha256'],'Frozen inference asset changed'
    public=OUT/'public';public.mkdir(exist_ok=False)
    write_json(public/'started.json',dict(utc=now(),freeze=file_record(OUT/'freeze.json')),exclusive=True)
    seed_all(42,1)
    metadata=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.xml'
    tracking=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.txt'
    spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25);pitch=GAMES[GAME]['pitch']
    base=build_metrica_features(spec,metadata,tracking,pitch);ball=metrica_positions(spec,tracking,pitch)
    _,reference=predict_reference(base,ball)
    selected_features=None
    if freeze['selected']!='reference':
        raw=raw_players(spec,metadata,tracking,pitch,ball)
        selected_features=Factory(base,raw,freeze['policy']).clean()
    def selected_predict():
        if freeze['selected']=='reference':return predict_reference(base,ball)[1]
        probs=[];boundaries=[]
        for rec in freeze['selected_checkpoints']:
            model,saved=load_model(rec['path']);probs.append(predict_batch(selected_features,model,saved['normalization'],saved['core_frames']).probabilities)
            if freeze['boundary_strength']:boundaries.append(boundary_probabilities(selected_features,model,saved['normalization'],saved['core_frames']))
        p=np.mean(probs,axis=0);visible=np.isfinite(ball).all(1)
        labels=decode_boundary(p,visible,spec,freeze['decoder'],np.mean(boundaries,axis=0),freeze['boundary_strength']) if boundaries else decode(p,visible,spec,freeze['decoder'])
        return PredictionBatch(spec,base.frames.copy(),labels).validate(spec)
    selected=selected_predict();first=public/'selected_full_match.csv';write_predictions(first,selected)
    # Re-load all weights and repeat inference, not merely duplicate the first CSV.
    repeated=selected_predict();second=public/'selected_full_match_repeat.csv';write_predictions(second,repeated)
    assert hashlib.sha256(first.read_bytes()).digest()==hashlib.sha256(second.read_bytes()).digest()
    command=[sys.executable,'evaluation/validate_submission.py',str(first)]
    validation=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (public/'validator.log').write_text(validation.stdout+validation.stderr)
    assert validation.returncode==0,'Official CSV validator failed; inspect local log'
    # Only after the external freeze, load public labels for one aggregate batch.
    truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
    rows=[dict(role='reference',candidate='reference',**metrics(truth,reference.labels)),dict(role='selected',candidate=freeze['selected'],**metrics(truth,selected.labels))]
    for row in rows:
        for key in METRICS:row['change_'+key]=row[key]-rows[0][key]
    write_json(public/'comparison.json',dict(rows=rows,diagnostic_only=True,selected_unchanged=freeze['selected'],full_match_context=True),exclusive=True)
    csv_write(public/'comparison.csv',rows)
    write_json(public/'verification.json',dict(validator_passed=True,rows=spec.n_frames,repeat_sha256_identical=True,csv=file_record(first),repeat=file_record(second),command=command),exclusive=True)
    print(json.dumps(dict(public_aggregate_metrics=rows,validator_passed=True,rows=spec.n_frames,repeat_identical=True)),flush=True)

if __name__=='__main__':main()
