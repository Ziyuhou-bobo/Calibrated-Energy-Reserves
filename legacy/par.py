"""Reserve trade-off study for a given policy: fixed multiplicative margins versus online conformal reserves."""
import sys, os, json, numpy as np
from evallib import *
m, seed = sys.argv[1], int(sys.argv[2])
out = f'res/s{seed}'; meta = json.load(open(f'{out}/meta.json')); CAL = meta['calib']
ab = meta.get('ab', {}).get(m, 0.5)
pol = Policy(load(m, seed), alpha_base=ab)
sc = np.load(f'{out}/calscores_{m}.npy')
CONDS = {'nominal': ((), 0.0), 'wind': (('wind',), 1.0), 'energy': (('energy',), 1.0)}
CIDX = {'nominal': 0, 'wind': 1, 'energy': 4}
for c, (fam, s) in CONDS.items():
    for mg in [1.0, 1.2, 1.5, 2.0, 2.5]:
        name = f'par_{m}_fixed{mg}_{c}'
        if os.path.exists(f'{out}/{name}.npz'): continue
        cal = dict(CAL[m]); cal['margin'] = mg
        env, eps, legs = run(pol, make_cfg(fam, s), guard='fixed', calib=cal, k=5, E=32, seed=20000 + 1000 * CIDX[c] + seed)
        np.savez_compressed(f'{out}/{name}.npz', legs=legs, lam_final=np.array([env.lam.mean()]), **mission_table(eps))
        pm = point_metrics(mission_table(eps), legs, 'fixed'); print(name, 'succ %.3f dep %.3f rth %.3f cov %.3f' % (pm['succ'], pm['dep'], pm['rth'], pm['cov']), flush=True)
    for dl in [0.02, 0.05, 0.10, 0.20, 0.30]:
        name = f'par_{m}_online{dl}_{c}'
        if os.path.exists(f'{out}/{name}.npz'): continue
        cal = dict(CAL[m]); cal['lam'] = conformal_quantile(sc, dl)
        env, eps, legs = run(pol, make_cfg(fam, s), guard='online', calib=cal, k=5, E=32, seed=20000 + 1000 * CIDX[c] + seed, conf=dict(delta_l=dl))
        np.savez_compressed(f'{out}/{name}.npz', legs=legs, lam_final=np.array([env.lam.mean()]), **mission_table(eps))
        pm = point_metrics(mission_table(eps), legs, 'online'); print(name, 'succ %.3f dep %.3f rth %.3f cov %.3f' % (pm['succ'], pm['dep'], pm['rth'], pm['cov']), flush=True)
