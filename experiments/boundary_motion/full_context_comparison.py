"""Frozen saved-model audit: full tracking input, public-window scoring only.

No training or public-based tuning. Retain historical named recipes and choose
one previously evaluated decoder per other model using external results only.
Private arrays and predictions are written only below the ignored run directory.
"""
from common import *
import hashlib
import importlib.util
import subprocess
from games import GAME, GAMES
from public_full import raw_players
from tiki_data.p1.metrica import build_metrica_features
from tiki_data.p1.kalman import metrica_positions, augment, DEFAULT_SETTINGS
from tiki_data.p1.features import apply_feature_policy
from tiki_data.p2.selected_features import build_features
from tiki_data.p2.inference import predict_batch
from tiki_data.p2.model import seed_all
from tiki_data.contracts import ProbabilityBatch
from tiki_data.artifacts import write_predictions
from models import load_model
from feature_engine import Factory
from raw_features import batch_from_raw
from boundary import boundary_probabilities, decode_boundary
from decoder import decode, BASE
from temporal_decoder import decode as legacy_decode

DEST = ROOT/'runs/full-context-model-comparison-20261116'
BROAD = ROOT/'runs/broad-study-20261116'
OLD_BASE = dict(BASE, temperature=4., missing_weight=.5, turnover_penalty=-1.)


def registry():
    models = {}
    variants = []
    def model(name, path, policy, status='saved checkpoint'):
        models[name] = dict(checkpoint=file_record(path), policy=policy, status=status)
    def add(name, members, settings, kind='modern', strength=0., rationale='Previously frozen recipe'):
        variants.append(dict(id=name, members=members, settings=settings, kind=kind,
                             strength=strength, rationale=rationale))
    def legacy(weight, penalty=0.):
        return dict(duration='hard', missing_weight=weight, transition_penalty=penalty, short_penalty=1.)
    artifacts = dict(original='p2-idsse-20261111', legacy_A='p2-idsse-cd715a3',
                     legacy_B='p2-idsse-no-ball-speed-20261111',
                     legacy_C='p2-idsse-c-optional-speed-20261111', legacy_D='p2-idsse-d-kalman-20261111')
    for name, folder in artifacts.items():
        model(name, ROOT/'artifacts'/folder/'best.pt', 'legacy_kalman' if name=='legacy_D' else 'base19')
    add('original_three_state', ['original'], dict(switch_cost=6., dead_cost=3.), 'three_state')
    add('legacy_A_original', ['legacy_A'], legacy(0), 'legacy')
    add('historical_best_spell_recipe', ['legacy_B'], legacy(0), 'legacy')
    add('historical_90_12_accuracy_recipe', ['legacy_B'], legacy(.5), 'legacy')
    selected_weight=read_json(ROOT/'runs/abcd-comparison-20261111/results.json')['decoder_selection']['selected_weight']
    for name in ('legacy_A','legacy_B','legacy_C','legacy_D'):
        for w in sorted({selected_weight,.5}):
            # Original A/C/D comparison explicitly hid the provider speed.
            add(name+'_abcd_weight_'+str(w), [name], legacy(w), 'legacy_mask_speed')
    for name in ('moderate','strong'):
        model(name, OLD/'training'/name/'best.pt', 'recorded')
    add('moderate_augmentation', ['moderate'], legacy(0), 'legacy')
    for name, recipe in read_json(OLD/'selection.json')['variants'].items():
        member='legacy_A' if recipe['model']=='baseline' else recipe['model']
        add('missingness_'+name, [member], recipe['decoder'], 'legacy')

    previous=read_json(PREVIOUS/'all_results.json')
    groups={}
    for folder in sorted((PREVIOUS/'models').iterdir()):
        path=folder/'best.pt'
        if not path.exists(): continue
        name=folder.name
        policy={'speed_removed':'removed','speed_difference':'difference',
                'speed_kalman':'kalman','speed_kalman_seed43':'kalman'}.get(name,'recorded')
        model(name,path,policy)
    for r in previous:
        if r.get('status')!='COMPLETE':continue
        name=r['id']
        if name.startswith('seed_confirm_'):continue
        if name.startswith('model_plus_decoder__'): group=name.split('__',1)[1]
        elif '__' in name:group=name.split('__',1)[0]
        else:group='strong'
        groups.setdefault(group,[]).append(r)
    for group, rows in groups.items():
        best=max(rows,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
        members={'ensemble_reference':['strong','reference_seed43'],
                 'ensemble_new_speed_kalman':['speed_kalman','speed_kalman_seed43']}.get(group,[group])
        assert all(m in models for m in members)
        add(group+'_external_best',members,best['settings'],rationale='Best previously evaluated external mean spellF1: '+best['id'])
    add('current_saved_ensemble',['speed_kalman','speed_kalman_seed43'],BASE)
    for name, policy in [('A42','kalman'),('B42','relative'),('AB42','relative'),('SW42','kalman')]:
        model(name,OUT/'models'/name/'best.pt',policy,
              'time-capped incomplete' if name in ('A42','B42') else 'completed')
        add(name+'_boundary_disabled' if name in ('A42','AB42') else name,[name],BASE)
        if name in ('A42','AB42'):
            grid=read_json(OUT/'models'/name/'boundary_grid.json')
            best=max(grid,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
            add(name+'_boundary_enabled',[name],BASE,'boundary',best['strength'],
                'Best enabled strength in saved external boundary grid')
    model('dropout_02',BROAD/'models/dropout_02/best.pt','recorded','incomplete cancelled training; saved epoch 29')
    add('dropout_02_incomplete',['dropout_02'],OLD_BASE,rationale='Original experiment default; no completed external decoder selection')
    for r in read_json(BROAD/'cached_results.json'):
        member='strong' if r['id'].startswith('directional') else 'speed_kalman' if r['id'].endswith('42') else 'speed_kalman_seed43'
        add(r['id'],[member],r['settings'],'directional')
    r=read_json(BROAD/'symmetry_results.json')[0]
    add('kalman_team_swap_tta',['speed_kalman'],r['settings'],'tta')
    # Same checkpoint/feature/decoder recipes are aliases, not separate ranks.
    unique=[]; seen={}
    for v in variants:
        kind=v['kind']
        if kind=='legacy_mask_speed' and v['members'][0] in ('legacy_B','legacy_D'):kind='legacy'
        if kind=='directional' and 'live_to_dead_penalty' not in v['settings']:kind='modern'
        key=json.dumps([v['members'],v['settings'],kind,v['strength']],sort_keys=True)
        if key in seen:seen[key]['aliases'].append(v['id'])
        else:
            v['aliases']=[];seen[key]=v;unique.append(v)
    return models,unique


def main():
    started=time.perf_counter()
    DEST.mkdir(exist_ok=False)
    models,variants=registry()
    protected=[file_record(ROOT/'runs/active_possession_candidate.json')]
    protected += [file_record(p) for p in (ROOT/'models/possession-kalman-ensemble-20261116').glob('*') if p.is_file()]
    freeze=dict(created_utc=now(),models=models,variants=variants,
        protocol='Full supplied tracking timeline, half resets, identical official public-five-minute scoring. No training, tuning, checkpoint changes or public-based adoption.',
        coverage='Every available unique trained checkpoint in recorded studies, prior ensembles and TTA; externally best saved decoder per model plus named historical and recent recipes. Not a rerun of every tuning-grid cell.',
        excluded={'ballfree':'No saved checkpoint; preprocessing failed','geometryless':'No saved checkpoint; preprocessing failed',
                  'unrun_seeds_and_future_context':'No saved checkpoints'},
        protected=protected,source=file_record(__file__),
        sources=[file_record(p) for p in Path(__file__).parent.glob('*.py')]+[file_record(BROAD/'decoder.py')])
    write_json(DEST/'freeze.json',freeze,exclusive=True)
    (DEST/'command.txt').write_text('python experiments/boundary_motion/full_context_comparison.py\n',encoding='utf-8')
    print(f'Frozen {len(models)} checkpoints and {len(variants)} unique recipes before public scoring.',flush=True)
    seed_all(42,1)
    spec=MatchSpec(GAME,GAMES[GAME]['n_video_frames'],25);pitch=GAMES[GAME]['pitch']
    metadata=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.xml'
    tracking=Path.home()/'Downloads/Telegram Desktop/BAR-ATH_tactical_atd.txt'
    base=build_metrica_features(spec,metadata,tracking,pitch)
    ball=metrica_positions(spec,tracking,pitch);visible=np.isfinite(ball).all(1)
    raw=raw_players(spec,metadata,tracking,pitch,ball)
    features={'base19':base,'base19_masked':apply_feature_policy(base,True),
              'kalman':build_features(base,ball)}
    for policy in ('recorded','removed','difference','relative'):
        features[policy]=Factory(base,raw,policy).clean()
    features['legacy_kalman']=augment(base,ball,DEFAULT_SETTINGS)
    # Reproduce the previously tested team-swap transformation without targets.
    swapped=dict(raw,home_positions=raw['away_positions'],away_positions=raw['home_positions'],
                 home_speeds=raw['away_speeds'],away_speeds=raw['home_speeds'])
    swapped_base,_=batch_from_raw(swapped,spec)
    features['kalman_swapped']=Factory(swapped_base,swapped,'kalman').clean()
    print('Full-timeline features ready; running saved checkpoints.',flush=True)
    probabilities={};boundaries={}
    (DEST/'probabilities').mkdir()
    for i,(name,record) in enumerate(models.items(),1):
        model,saved=load_model(record['checkpoint']['path'])
        assert saved['data_kind']=='external'
        f=features[record['policy']]
        p=predict_batch(f,model,saved['normalization'],saved['core_frames']).probabilities
        probabilities[name]=p;np.save(DEST/'probabilities'/(name+'.npy'),p)
        if name in ('legacy_A','legacy_C'):
            probabilities[name+'_masked']=predict_batch(features['base19_masked'],model,saved['normalization'],saved['core_frames']).probabilities
        if name in ('A42','AB42'):
            boundaries[name]=boundary_probabilities(f,model,saved['normalization'],saved['core_frames'])
        if name=='speed_kalman':
            q=predict_batch(features['kalman_swapped'],model,saved['normalization'],saved['core_frames']).probabilities
            probabilities['kalman_team_swap_tta']=(p+q[:,[1,0,2]])/2
        print(f'Checkpoint {i}/{len(models)} complete: {name}',flush=True)
    loader=importlib.util.spec_from_file_location('frozen_directional_decoder',BROAD/'decoder.py')
    directional=importlib.util.module_from_spec(loader);loader.loader.exec_module(directional)
    (DEST/'predictions').mkdir()
    for i,v in enumerate(variants,1):
        p=np.mean([probabilities[m] for m in v['members']],axis=0)
        kind=v['kind'];settings=v['settings']
        if kind=='legacy_mask_speed' and v['members'][0] in ('legacy_A','legacy_C'):
            p=probabilities[v['members'][0]+'_masked']
        if kind.startswith('legacy'):labels=legacy_decode(p,visible,spec,settings)
        elif kind=='boundary':labels=decode_boundary(p,visible,spec,settings,boundaries[v['members'][0]],v['strength'])
        elif kind=='directional':labels=directional.decode(p,visible,spec,settings)
        elif kind=='three_state':
            from tiki_data.p3.decoder import decode as three_decode
            labels=three_decode(ProbabilityBatch(spec,base.frames,p,base.periods),{},settings).labels
        else:
            if kind=='tta':p=probabilities['kalman_team_swap_tta']
            labels=decode(p,visible,spec,settings)
        write_predictions(DEST/'predictions'/(v['id']+'.csv'),PredictionBatch(spec,base.frames,labels).validate(spec))
        print(f'Decoded {i}/{len(variants)}: {v["id"]}',flush=True)
    write_json(DEST/'inference_complete.json',dict(created_utc=now(),
        predictions=[file_record(p) for p in (DEST/'predictions').glob('*.csv')]),exclusive=True)
    # Public labels are opened only after every prediction and setting is fixed.
    truth=read_predictions(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv',spec)
    rows=[]
    for v in variants:
        path=DEST/'predictions'/(v['id']+'.csv')
        labels=read_predictions(path,spec).labels
        rows.append(dict(candidate=v['id'],members='+'.join(v['members']),aliases=';'.join(v['aliases']),
                         **metrics(truth,labels),prediction_sha256=file_record(path)['sha256']))
    rows.sort(key=lambda r:(-r['spellF1'],-r['macroF1'],r['candidate']))
    for i,r in enumerate(rows,1):r['rank']=i
    csv_write(DEST/'leaderboard.csv',rows)
    write_json(DEST/'results.json',dict(rows=rows,checkpoint_count=len(models),recipe_count=len(variants),
        diagnostic_only=True,selected_unchanged=True,seconds=time.perf_counter()-started),exclusive=True)
    # A matching current-reference result is a useful end-to-end regression check.
    current=next(r for r in rows if r['candidate']=='current_saved_ensemble' or 'current_saved_ensemble' in r['aliases'].split(';'))
    previous=read_json(OUT/'public/comparison.json')['rows'][0]
    for k in METRICS:assert abs(current[k]-previous[k])<1e-12
    for r in protected+[m['checkpoint'] for m in models.values()]+freeze['sources']:
        assert file_record(r['path'])['sha256']==r['sha256']
    validations=[]
    for r in rows[:3]:
        result=subprocess.run([sys.executable,'evaluation/validate_submission.py',str(DEST/'predictions'/(r['candidate']+'.csv'))],cwd=ROOT,capture_output=True,text=True)
        (DEST/(r['candidate']+'_validation.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        assert result.returncode==0
        validations.append(r['candidate'])
    report=['Full-context public diagnostic comparison',
        f'{len(models)} unique saved checkpoints; {len(variants)} frozen recipes. All inference rerun.',
        freeze['protocol'],freeze['coverage'],
        'No submission candidate changed. Incomplete saved A42, B42 and dropout_02 are labelled in freeze.json.',
        'Historical crop-only scores are not mixed into this leaderboard. All scores below use full tracking input.',
        'Rank | Candidate | SpellF1 | Accuracy % | MacroF1']
    report += [f'{r["rank"]} | {r["candidate"]} | {r["spellF1"]:.9f} | {r["accuracy_pct"]:.6f} | {r["macroF1"]:.9f}' for r in rows]
    report += ['Limitations: public diagnostic only; repeated use of the small public window cannot establish unseen-match generalization.',
               'Only saved runnable models covered; failed or unrun checkpoints excluded as documented in freeze.json.',
               'Model-specific feature definitions and previously frozen settings preserved; no new external evaluation in this inference-only audit.',
               'Command: python experiments/boundary_motion/full_context_comparison.py']
    (DEST/'REPORT.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
    audit=DEST/'verification.json'
    paths={p.relative_to(ROOT).as_posix() for p in DEST.rglob('*') if p.is_file()};paths.add(audit.relative_to(ROOT).as_posix())
    check=subprocess.run(['git','check-ignore','--no-index','-z','--stdin'],cwd=ROOT,input='\0'.join(sorted(paths))+'\0',capture_output=True,text=True,check=True)
    assert set(check.stdout.rstrip('\0').split('\0'))==paths
    assert not subprocess.check_output(['git','ls-files','--',str(DEST.relative_to(ROOT))],cwd=ROOT)
    write_json(audit,dict(all_outputs_ignored=True,tracked_outputs=0,protected_assets_unchanged=True,
        current_reference_reproduced=True,validated_top_three=validations,no_training_or_tuning=True),exclusive=True)
    print(json.dumps(dict(top_three=rows[:3],historical=[r for r in rows if r['candidate'].startswith('historical')],
                          reference=current,checkpoints=len(models),recipes=len(variants),verified=True)),flush=True)


if __name__=='__main__':main()
