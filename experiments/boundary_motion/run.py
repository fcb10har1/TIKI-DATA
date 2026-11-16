from common import *
from budget import run_one
from train_model import evaluate_cached
from decoder import BASE

check_preserved()
reference=evaluate_cached(PREVIOUS/'ensemble_predictions/ensemble_new_speed_kalman','reference',BASE)
expected=read_json(ROOT/'models/possession-kalman-ensemble-20261116/configuration.json')['evaluation']['metrics']
for k in METRICS:assert abs(reference['mean'][k]-expected[k])<1e-12
write_json(OUT/'reference.json',reference,exclusive=True)
print(json.dumps(dict(reference_reproduced=reference['mean'])),flush=True)
for name,arch,speed in [('A42','boundary','kalman'),('B42','tcn','relative')]:
    run_one(dict(id=name,architecture=arch,speed=speed,boundary_radius=0,boundary_weight=1.,consistency=0.,seed=42))
print('Seed42 screening completed',flush=True)
