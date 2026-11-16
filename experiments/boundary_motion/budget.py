"""One sequential training worker with a shared finite study budget."""
import subprocess
from common import *

def ledger():return read_json(OUT/'training_ledger.json') if (OUT/'training_ledger.json').exists() else []

def run_one(config):
    items=ledger();budget=read_json(OUT/'manifest.json')['training_budget']
    if any(r['id']==config['id'] for r in items):raise RuntimeError('Preserving existing run '+config['id'])
    used=sum(r.get('training_seconds',0) for r in items)
    attempted=sum(r['status']!='NOT RUN' for r in items)
    reason=None
    if attempted>=budget['max_new_runs']:reason='Predeclared maximum number of new training runs reached'
    elif budget['cumulative_seconds']-used<budget['per_run_seconds']:
        reason='Less than one complete per-run allocation remains in the cumulative training budget'
    if reason:
        item=dict(id=config['id'],config=config,status='NOT RUN',reason=reason,training_seconds=0,evaluation_seconds=0)
        items.append(item);write_json(OUT/'training_ledger.json',items);print(json.dumps(item),flush=True);return item
    path=OUT/'run_configs'/(config['id']+'.json');write_json(path,config,exclusive=True)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    started=time.perf_counter()
    item=dict(id=config['id'],config=config,status='FAILED',training_seconds=0,evaluation_seconds=0)
    try:
        with (logs/(config['id']+'_train.log')).open('x') as handle:
            subprocess.run([sys.executable,str(SOURCE/'train_model.py'),'train',str(path)],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=budget['per_run_seconds']+20)
        item['training_seconds']=time.perf_counter()-started
        item['training_status']=read_json(OUT/'models'/config['id']/'status.json')
        infer_started=time.perf_counter()
        with (logs/(config['id']+'_evaluate.log')).open('x') as handle:
            subprocess.run([sys.executable,str(SOURCE/'train_model.py'),'evaluate',config['id']],cwd=ROOT,
                stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=360)
        item['evaluation_seconds']=time.perf_counter()-infer_started
        item['status']='COMPLETE';item['mean']=read_json(OUT/'models'/config['id']/'evaluation.json')['mean']
    except Exception as error:
        if not item['training_seconds']:item['training_seconds']=time.perf_counter()-started
        item['reason']=type(error).__name__+': '+str(error)
        # Preserve partial checkpoints/logs; never pass an incomplete run to selection.
    items.append(item);write_json(OUT/'training_ledger.json',items)
    print(json.dumps({k:v for k,v in item.items() if k!='training_status'}),flush=True)
    return item
