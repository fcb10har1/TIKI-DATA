from common import *
from budget import ledger
from decoder import BASE

def main():
    reference=read_json(OUT/'reference.json');results=read_json(OUT/'finalist_results.json')
    ranked=sorted([reference]+[r for r in results if r['id'].endswith('_ensemble')],key=lambda r:(-r['mean']['spellF1'],-r['mean']['macroF1']))
    selected=reference;audit=[]
    for r in ranked:
        if r['id']=='reference':continue
        changes={m:summarize([x for x in r['rows'] if x['match_id']==m])['spellF1']-summarize([x for x in reference['rows'] if x['match_id']==m])['spellF1'] for m in ('J03WPY','J03WQQ')}
        accepted=r['mean']['spellF1']-reference['mean']['spellF1']>=.01 and min(changes.values())>=0
        audit.append(dict(candidate=r['id'],per_match_changes=changes,accepted=accepted))
        if accepted and selected['id']=='reference':selected=r
    cfg=read_json(ROOT/'models/possession-kalman-ensemble-20261116/configuration.json')
    refpaths=[ROOT/'models/possession-kalman-ensemble-20261116'/x['file'] for x in cfg['checkpoints']]
    paths=refpaths if selected['id']=='reference' else [OUT/'models'/(selected['family']+str(seed))/'best.pt' for seed in (42,43)]
    data=dict(selected=selected['id'],created_utc=now(),external_only=True,public_accessed=False,reference=reference['mean'],selected_metrics=selected['mean'],
        ranking=[dict(id=r['id'],**r['mean']) for r in ranked],audit=audit,
        screening_decisions=read_json(OUT/'screening_decisions.json'),
        training_attempts=[dict(id=r['id'],status=r['status'],reason=r.get('reason')) for r in ledger()],
        incomplete_diagnostics=file_record(OUT/'incomplete_results.json'),
        rationale='Highest mean external spellF1 with macroF1 secondary; require >=0.01 gain and nonnegative changes on both validation matches. Otherwise retain preserved reference.',
        selected_checkpoints=[file_record(p) for p in paths],reference_checkpoints=[file_record(p) for p in refpaths],decoder=BASE,
        boundary_strength=selected.get('boundary_strength',0.),policy='relative' if selected.get('family')=='B' else 'kalman',
        inference_command='python experiments/boundary_motion/public_full.py',
        source_hashes=[file_record(p) for p in sorted(list(SOURCE.glob('*.py'))+list((ROOT/'tiki_data').rglob('*.py'))+list((ROOT/'p3').rglob('*.py'))+[ROOT/'games.py',ROOT/'evaluation/possession_metrics.py',ROOT/'evaluation/validate_submission.py'])],
        configuration_hashes=[file_record(p) for p in [OUT/'manifest.json',OUT/'repeat_protocol.json',ROOT/'models/possession-kalman-ensemble-20261116/configuration.json']+list((OUT/'run_configs').glob('*.json'))+list((OUT/'models').glob('*/normalization.json'))],
        fresh_confirmation=dict(status='NOT RUN',reason='J03WR9 already has historical scoring; no unused IDSSE match exists.'),
        no_reselection_after_public=True)
    write_json(OUT/'freeze.json',data,exclusive=True)
    print(json.dumps(dict(selected=data['selected'],metrics=data['selected_metrics'])),flush=True)

if __name__=='__main__':main()
