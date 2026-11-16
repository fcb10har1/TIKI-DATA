"""Score preserved time-capped checkpoints descriptively, never for selection."""
from common import *
from budget import ledger
import subprocess


def main():
    results=[]
    for item in ledger():
        folder=OUT/'models'/item['id']
        child=read_json(folder/'status.json') if (folder/'status.json').exists() else {}
        time_capped='Timeout' in (item.get('reason','')+' '+child.get('error',''))
        if item['status']!='FAILED' or not time_capped or not (folder/'best.pt').exists():
            continue
        try:
            with (OUT/'logs'/(item['id']+'_partial_evaluate.log')).open('x') as handle:
                subprocess.run([sys.executable,str(SOURCE/'train_model.py'),'evaluate',item['id']],cwd=ROOT,
                    stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=360)
            rows=[read_json(folder/'evaluation.json')]
            if (folder/'boundary_grid.json').exists():rows+=read_json(folder/'boundary_grid.json')
            for r in rows:
                r.update(status='INCOMPLETE',selection_eligible=False,
                    reason='Time-capped training did not finish the original stopping rule. Best available checkpoint; descriptive results only.',
                    training_runtime_seconds=item['training_seconds'])
            results+=rows
        except Exception as error:
            results.append(dict(id=item['id']+'_partial_evaluation',status='FAILED',reason=type(error).__name__+': '+str(error)))
    write_json(OUT/'incomplete_results.json',results,exclusive=True)
    print(json.dumps([dict(id=r['id'],status=r['status'],mean=r.get('mean')) for r in results]),flush=True)


if __name__=='__main__':main()
