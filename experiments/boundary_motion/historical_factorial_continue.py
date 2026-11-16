"""Durable continuation after all factorial training/evaluation artifacts exist."""
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'runs/historical-factorial-20261116'
status_path=DEST/'continuation_status.json'

def status(**values):
    status_path.write_text(json.dumps(dict(driver_pid=os.getpid(),**values),indent=2)+'\n')

def main():
    with (DEST/'continuation.lock').open('x') as f:f.write(str(os.getpid()))
    status(stage='waiting_for_factorial')
    while True:
        matrix=json.loads((DEST/'training_matrix.json').read_text())
        complete=True
        for arm in matrix:
            p=DEST/'training_status'/(arm['id']+'.json')
            if not p.exists() or json.loads(p.read_text()).get('status')!='COMPLETE':complete=False;break
            if any(not (DEST/'evaluations'/(arm['id']+'_'+s+'.json')).exists() for s in ('raw_best','decoded_best')):complete=False;break
        if complete:break
        time.sleep(10)
    commands=[['experiments/boundary_motion/historical_factorial.py','finish'],
              ['experiments/boundary_motion/historical_factorial_report.py']]
    for i,args in enumerate(commands):
        with (DEST/('continuation_stage_'+str(i)+'.log')).open('x') as log:
            job=subprocess.Popen([sys.executable,*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status(stage='confirmation_and_public' if i==0 else 'report',child_pid=job.pid)
            code=job.wait()
            if code:
                status(stage='FAILED',exit_code=code,log='continuation_stage_'+str(i)+'.log');return
    status(stage='COMPLETE')

if __name__=='__main__':main()
