"""Evaluation suite for one training seed (resumable: finished evaluations are skipped).
usage: python3 suite2.py SEED PART   with PART in {main, abl, sev, pareto, dial, all}"""
import sys, json, os, time
import numpy as np
from evallib import *

seed = int(sys.argv[1]); only = sys.argv[2] if len(sys.argv) > 2 else 'all'
out = f'res/s{seed}'; os.makedirs(out, exist_ok=True)
ALL = ['sac', 'td3', 'ppo', 'dsacn', 'dsaccvar', 'wcsac', 'crestret', 'crest']
SCHED = ('crest', 'crestret')
CONDS = {'nominal': ((), 0.0), 'wind': (('wind',), 1.0), 'sensor': (('sensor',), 1.0), 'traffic': (('traffic',), 1.0),
         'energy': (('energy',), 1.0), 'compound': (FAMILIES, 0.5)}
CIDX = {c: i for i, c in enumerate(list(CONDS) + ['sev'])}
t0 = time.time()
meta_f = f'{out}/meta.json'
meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}
meta.setdefault('calib', {}); meta.setdefault('ab', {}); meta.setdefault('ab_sel', {}); meta.setdefault('spread', {})
if 'alpha_base_crestret' in meta:
    meta['ab'].setdefault('crestret', meta['alpha_base_crestret'])
if 'spread_ref_crestret' in meta:
    meta['spread'].setdefault('crestret', meta['spread_ref_crestret'])


def dump():
    json.dump(meta, open(meta_f, 'w'))


def log(*a):
    print('[%6.0fs]' % (time.time() - t0), *a, flush=True)


def done(name):
    return os.path.exists(f'{out}/{name}.npz')


def ev(name, pol, fam, s, guard, calib, k, envseed, conf=None):
    if done(name):
        return
    env, eps, legs = run(pol, make_cfg(fam, s), guard=guard, calib=calib, k=k, E=32, seed=envseed, conf=conf)
    M = mission_table(eps)
    np.savez_compressed(f'{out}/{name}.npz', legs=legs, lam_final=np.array([env.lam.mean()]), **M)
    pm = point_metrics(M, legs, guard)
    log(name, ' '.join('%s=%.3f' % (kk, pm[kk]) for kk in ('succ', 'coll', 'dep', 'loss', 'rth', 'energy', 'tdel', 'ret_cvar', 'cov') if kk in pm))


METHODS = [m for m in ALL if os.path.exists(f'runs/{m}_s{seed}.pkl')]
cks = {m: load(m, seed) for m in METHODS}
log('methods', METHODS)
# ---- 1. base risk level of scheduled policies, selected on nominal validation districts (criterion: CVaR_0.1 of mission return)
for m in SCHED:
    if m in METHODS and m not in meta['ab']:
        best, sel = None, {}
        for ab in [0.25, 0.5, 0.75, 1.0]:
            env, eps, legs = run(Policy(cks[m], alpha_base=ab), make_cfg(), guard='std', k=3, E=32, seed=5000 + seed)
            pm = point_metrics(mission_table(eps)); sel[str(ab)] = pm
            log(m, 'alpha_base', ab, 'ret_cvar %.3f succ %.3f loss %.3f' % (pm['ret_cvar'], pm['succ'], pm['loss']))
            if best is None or pm['ret_cvar'] > best[1]:
                best = (ab, pm['ret_cvar'])
        meta['ab'][m] = best[0]; meta['ab_sel'][m] = sel; dump()


def pol_of(m, **kw):
    if m in SCHED:
        return Policy(cks[m], alpha_base=kw.get('alpha_base', meta['ab'][m]), alpha_mode=kw.get('alpha_mode', 'sched'),
                      sched=kw.get('sched', 'full'), spread_ref=kw.get('spread_ref'))
    return Policy(cks[m])


# ---- 2. split-conformal calibration of every policy on nominal districts
for m in METHODS:
    if m not in meta['calib']:
        cal, sc, S = calibrate(pol_of(m), seed=7000 + 10 * seed)
        np.save(f'{out}/calscores_{m}.npy', sc); meta['calib'][m] = cal; dump(); log('calib', m, cal)
CAL = meta['calib']
# ---- 3. main conditions: every policy under the fixed 1.2x reserve and the online conformal reserve; CREST under all reserves
if only in ('all', 'main'):
    for c, (fam, s) in CONDS.items():
        for m in METHODS:
            for g in (['online', 'std', 'static', 'std15'] if m == 'crest' else ['std', 'online']):
                ev(f'main_{m}_{g}_{c}', pol_of(m), fam, s, g, CAL[m], 5, 20000 + 1000 * CIDX[c] + seed)
# ---- 4. risk-regulation ablations of the scheduled policies (online reserve)
if only in ('all', 'abl'):
    for m in [x for x in SCHED if x in METHODS]:
        if m not in meta['spread']:
            pu = pol_of(m, sched='unc', spread_ref=(0.0, 1e9)); pu.spreads = []
            run(pu, make_cfg(), guard='online', calib=CAL[m], k=2, E=32, seed=6000 + seed)
            sp = np.concatenate([x.ravel() for x in pu.spreads])
            meta['spread'][m] = [float(np.quantile(sp, 0.1)), float(np.quantile(sp, 0.9))]; dump()
        tagp = 'abl' if m == 'crest' else 'ablret'
        for c, (fam, s) in CONDS.items():
            for tag, kw in [('fixedalpha', dict(alpha_mode='fixed')), ('noenergy', dict(sched='noenergy')), ('prox', dict(sched='prox')),
                            ('unc', dict(sched='unc', spread_ref=tuple(meta['spread'][m])))]:
                ev(f'{tagp}_{tag}_{c}', pol_of(m, **kw), fam, s, 'online', CAL[m], 5, 20000 + 1000 * CIDX[c] + seed)
# ---- 5. severity sweeps (fixed reserve for every policy; CREST additionally with its online reserve)
if only in ('all', 'sev'):
    for f in FAMILIES:
        for s in [0.25, 0.5, 0.75]:
            for m in METHODS:
                for g in (['std', 'online'] if m == 'crest' else ['std']):
                    ev(f'sev_{m}_{g}_{f}_{s}', pol_of(m), (f,), s, g, CAL[m], 3, 30000 + 100 * FAMILIES.index(f) + int(s * 10) + seed)
# ---- 6. reserve Pareto study with the CREST policy
if only in ('all', 'pareto') and 'crest' in METHODS:
    sc = np.load(f'{out}/calscores_crest.npy')
    for c in ['nominal', 'wind', 'energy']:
        fam, s = CONDS[c]
        for mg in [1.0, 1.2, 1.5, 2.0, 2.5]:
            cal = dict(CAL['crest']); cal['margin'] = mg
            ev(f'par_fixed{mg}_{c}', pol_of('crest'), fam, s, 'fixed', cal, 5, 20000 + 1000 * CIDX[c] + seed)
        for dl in [0.02, 0.05, 0.10, 0.20, 0.30]:
            cal = dict(CAL['crest']); cal['lam'] = conformal_quantile(sc, dl)
            ev(f'par_online{dl}_{c}', pol_of('crest'), fam, s, 'online', cal, 5, 20000 + 1000 * CIDX[c] + seed, conf=dict(delta_l=dl))
# ---- 7. risk dial of the CREST policy
if only in ('all', 'dial') and 'crest' in METHODS:
    for c in ['nominal', 'sensor', 'traffic', 'compound']:
        fam, s = CONDS[c]
        for ab in [0.1, 0.25, 0.5, 0.75, 1.0]:
            ev(f'dial_{ab}_{c}', pol_of('crest', alpha_base=ab), fam, s, 'std', CAL['crest'], 3, 20000 + 1000 * CIDX[c] + seed)
log('SUITE DONE', only)
