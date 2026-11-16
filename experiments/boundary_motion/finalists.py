from common import *
from budget import run_one,ledger
from train_model import evaluate_cached
from boundary import evaluate_boundary
from decoder import BASE
from tiki_data.artifacts import write_batch
from tiki_data.contracts import ProbabilityBatch

def main():
    from tiki_data.provenance import environment
    write_json(OUT/'environment.json',environment(),exclusive=True)
    # Descriptive scores cannot qualify an unfinished training run for selection.
    from incomplete import main as score_incomplete
    score_incomplete()
    entries={x['id']:x for x in ledger()};results=[];decisions=[]
    single=evaluate_cached(PREVIOUS/'model_predictions/speed_kalman','reference_seed42',BASE)
    write_json(OUT/'single_seed_reference.json',single,exclusive=True)
    single43=evaluate_cached(PREVIOUS/'model_predictions/speed_kalman_seed43','reference_seed43',BASE)
    write_json(OUT/'single_seed_reference43.json',single43,exclusive=True)
    for label in ('A','B'):
        item=entries.get(label+'42')
        if not item or item['status']!='COMPLETE':
            decisions.append(dict(family=label,status='NOT RUN',reason='Seed42 did not complete'));continue
        folder=OUT/'models'/(label+'42');candidates=[read_json(folder/'evaluation.json')]
        if label=='A':candidates+=read_json(folder/'boundary_grid.json')
        best=max(candidates,key=lambda r:(r['mean']['spellF1'],r['mean']['macroF1']))
        results+=candidates
        per_match=[summarize([r for r in best['rows'] if r['match_id']==m])['spellF1']-summarize([r for r in single['rows'] if r['match_id']==m])['spellF1'] for m in ('J03WPY','J03WQQ')]
        if best['mean']['spellF1']<=single['mean']['spellF1'] or min(per_match)<-.005:
            decisions.append(dict(family=label,status='NOT RUN',reason='Seed42 screening failed positive mean / no match regression >0.005 rule',per_match_changes=per_match));continue
        decision=dict(family=label,chosen_decoder=best.get('strength',0.),seed42_metrics=best['mean'],per_match_changes=per_match)
        write_json(OUT/(label+'_repeat_decision.json'),decision,exclusive=True)
        repeat=run_one(dict(item['config'],id=label+'43',seed=43))
        if repeat['status']!='COMPLETE':decisions.append(dict(decision,status=repeat['status'],reason=repeat.get('reason')));continue
        root=OUT/'ensemble_predictions'/label
        for spec,_ in truth_records():
            for scenario in SCENARIOS:
                paths=[OUT/'model_predictions'/(label+str(seed))/spec.match_id/scenario for seed in (42,43)]
                p=[read_batch(path/'probabilities.npz',spec,'probabilities') for path in paths]
                mean=(p[0].probabilities+p[1].probabilities)/2
                write_batch(root/spec.match_id/scenario/'probabilities.npz',ProbabilityBatch(spec,p[0].frames,mean,p[0].periods))
                if label=='A':np.save(root/spec.match_id/scenario/'boundary.npy',(np.load(paths[0]/'boundary.npy')+np.load(paths[1]/'boundary.npy'))/2)
        r=evaluate_boundary(root,label+'_ensemble',BASE,decision['chosen_decoder']) if label=='A' and decision['chosen_decoder'] else evaluate_cached(root,label+'_ensemble',BASE)
        r['family']=label;r['boundary_strength']=decision['chosen_decoder'];results.append(r)
        decisions.append(dict(decision,status='COMPLETE',ensemble_metrics=r['mean']))
        write_json(OUT/'finalist_results.json',results)
    write_json(OUT/'finalist_results.json',results)
    write_json(OUT/'screening_decisions.json',decisions)
    print(json.dumps(decisions),flush=True)

if __name__=='__main__':main()
