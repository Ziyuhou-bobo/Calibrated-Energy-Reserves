"""Write plain numeric matrices (no headers) for the Octave figure scripts."""
import numpy as np, csv, os, sys
D = 'pa/data'
M6 = ['sac', 'td3', 'ppo', 'dsacn', 'dsaccvar', 'crest']          # main comparison order
CONDS = ['nominal', 'wind', 'sensor', 'traffic', 'energy', 'compound']
FAMS = ['wind', 'sensor', 'traffic', 'energy']


def rd(name):
    with open(f'{D}/{name}.csv') as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def save(name, A):
    np.savetxt(f'{D}/{name}.txt', np.asarray(A, float), fmt='%.4f', delimiter=' ')


main = {(r['method'], r['guard'], r['cond']): r for r in rd('main')}
# 1. main comparison: rows = method x condition, cols = metrics
for g in ['std', 'online']:
    A = []
    for m in M6:
        for c in CONDS:
            r = main.get((m, g, c))
            A.append([num(r[k]) if r else np.nan for k in ('loss', 'coll', 'coll_a', 'coll_b', 'dep', 'succ', 'rth', 'energy', 'tdel', 'minsep', 'retcvar', 'cov', 'energy_cvar', 'loss_lo', 'loss_hi', 'coll_lo', 'coll_hi', 'succ_lo', 'succ_hi')])
    save(f'fig_main_{g}', A)
# 2. coverage and reserve comparison for the CREST policy (and SAC) across guards
A = []
for m in ['crest', 'sac']:
    for g in ['std', 'std15', 'static', 'online']:
        for c in CONDS:
            r = main.get((m, g, c))
            A.append([num(r[k]) if r else np.nan for k in ('cov', 'tight', 'dep', 'rth', 'succ', 'loss')])
save('fig_cov', A)
# 3. severity sweeps: rows = family x severity(0,.25,.5,.75,1) x method, cols = loss, coll, dep, succ
if os.path.exists(f'{D}/sev.csv'):
    sev = {(r['method'], r['guard'], r['family'], r['sev']): r for r in rd('sev')}
    A = []
    for f in FAMS:
        for s in ['0.0', '0.25', '0.5', '0.75', '1.0']:
            for m in M6:
                if s == '0.0':
                    r = main.get((m, 'std', 'nominal'))
                elif s == '1.0':
                    r = main.get((m, 'std', f))
                else:
                    r = sev.get((m, 'std', f, s))
                A.append([num(r[k]) if r else np.nan for k in ('loss', 'coll', 'dep', 'succ')])
    save('fig_sev', A)
# 4. reserve trade-off: rows = policy x rule x condition
if os.path.exists(f'{D}/pareto.csv'):
    par = {(r['pol'], r['rule'], r['cond']): r for r in rd('pareto')}
    RULES = ['fixed1.0', 'fixed1.2', 'fixed1.5', 'fixed2.0', 'fixed2.5', 'online0.02', 'online0.05', 'online0.1', 'online0.2', 'online0.3']
    A = []
    for pol in ['sac', 'crest']:
        for ru in RULES:
            for c in ['nominal', 'wind', 'energy']:
                r = par.get((pol, ru, c)) or par.get(('', ru, c))
                A.append([num(r[k]) if r else np.nan for k in ('rth', 'dep', 'succ', 'cov', 'loss', 'energy')])
    save('fig_pareto', A)
# 5. risk dial: rows = alpha x condition
if os.path.exists(f'{D}/dial.csv'):
    dl = {(r['alpha'], r['cond']): r for r in rd('dial')}
    A = []
    for a in ['0.1', '0.25', '0.5', '0.75', '1.0']:
        for c in ['nominal', 'sensor', 'traffic', 'compound']:
            r = dl.get((a, c))
            A.append([num(r[k]) if r else np.nan for k in ('coll', 'loss', 'succ', 'tdel', 'minsep', 'energy')])
    save('fig_dial', A)
# 6. risk-regulation ablations: rows = variant x condition
if os.path.exists(f'{D}/abl.csv'):
    ab = {(r['family'], r['variant'], r['cond']): r for r in rd('abl')}
    A = []
    for fam in ['abl', 'ablret']:
        for v in ['fixedalpha', 'noenergy', 'prox', 'unc']:
            for c in CONDS:
                r = ab.get((fam, v, c))
                A.append([num(r[k]) if r else np.nan for k in ('loss', 'coll', 'dep', 'succ', 'rth', 'tdel')])
    # reference rows: the full schedule (main_crest_online_* and main_crestret_online_*)
    for m in ['crest', 'crestret']:
        for c in CONDS:
            r = main.get((m, 'online', c))
            A.append([num(r[k]) if r else np.nan for k in ('loss', 'coll', 'dep', 'succ', 'rth', 'tdel')])
    save('fig_abl', A)
# 7. learning curves: mean +- sd over seeds on a common grid
lr = rd('learn')
grid = np.arange(250, 25001, 250)
A = []
for m in M6 + ['crestret']:
    seeds = sorted({r['seed'] for r in lr if r['algo'] == m})
    for k in ('succ', 'coll', 'ret'):
        Y = []
        for s in seeds:
            d = {int(r['step']): num(r[k]) for r in lr if r['algo'] == m and r['seed'] == s}
            Y.append([d.get(int(g), np.nan) for g in grid])
        Y = np.array(Y) if Y else np.full((1, len(grid)), np.nan)
        A.append(np.nanmean(Y, 0)); A.append(np.nanstd(Y, 0))
save('fig_learn', A)
print('figure data written')


# 8. separation distribution under the traffic shift (mission-level minimum air clearance)
from agg import load_table
import numpy as _np
for m in ['sac', 'dsacn', 'crest']:
    M, _, _ = load_table(f'main_{m}_std_traffic', (0, 1, 2))
    if M is None:
        continue
    v = _np.sort(_np.clip(M['minair'], -5, 80))
    q = _np.linspace(0, 1, len(v))
    k = _np.linspace(0, len(v) - 1, min(400, len(v))).astype(int)
    _np.savetxt(f'{D}/ecdf_{m}.txt', _np.stack([v[k], q[k]], -1), fmt='%.3f')
