"""PPO reference evaluation (not part of the comparison): the submitted implementation at the main budget
(three seeds) and at four times the budget (two seeds), all conditions, fixed 1.2x reserve."""
import os, numpy as np
from evallib import load, Policy, run, mission_table, point_metrics, make_cfg
from uavenv import FAMILIES
CONDS = {'nominal': ((), 0.0), 'wind': (('wind',), 1.0), 'sensor': (('sensor',), 1.0), 'traffic': (('traffic',), 1.0),
         'energy': (('energy',), 1.0), 'compound': (FAMILIES, 0.5)}
CIDX = {c: i for i, c in enumerate(CONDS)}
os.makedirs('res3/ppo', exist_ok=True)
for tag, path in [('ppo25k_s%d' % s, f'runs_legacy/ppo_s{s}.pkl') for s in range(3)] + [('ppo100k_s%d' % s, f'runs_legacy/ppo100k_s{s}.pkl') for s in range(2)]:
    import pickle
    ck = pickle.load(open(path, 'rb')); s = int(tag[-1])
    for c, (fam, sv) in CONDS.items():
        f = f'res3/ppo/{tag}_std_{c}.npz'
        if os.path.exists(f):
            continue
        env, eps, legs = run(Policy(ck), make_cfg(fam, sv), guard='std', k=5, E=32, seed=20000 + 1000 * CIDX[c] + s)
        M = mission_table(eps); np.savez_compressed(f, legs=legs, **M)
        pm = point_metrics(M, legs); print(tag, c, 'succ %.3f loss %.3f coll_b %.3f coll_a %.3f' % (pm['succ'], pm['loss'], pm['coll_b'], pm['coll_a']), flush=True)
open('res3/ppo/done.txt', 'w').write('ok\n')
