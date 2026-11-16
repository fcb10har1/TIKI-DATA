"""Continue once both seed42 training attempts have finished."""
from common import *
import subprocess
while True:
    path=OUT/'training_ledger.json'
    if path.exists() and {'A42','B42'}<={x['id'] for x in read_json(path)}:break
    time.sleep(20)
for script in ('finalists.py','freeze.py','public_full.py','report.py'):
    subprocess.run([sys.executable,str(SOURCE/script)],cwd=ROOT,check=True)
