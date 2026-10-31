"""Turn saved evaluation results into LaTeX macros, tables and figure CSVs."""
import numpy as np, glob, os, json, sys
from agg import metrics, load_table

SEEDS = tuple(int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else '0,1,2'))
os.makedirs('pa/data', exist_ok=True)
MT = dict(sac='Sac', td3='Tdthree', ppo='Ppo', dsacn='Dsacn', dsaccvar='Dsaccvar', crest='Crestc', crestret='Crestr')
GT = dict(std='Std', online='Online', static='Static', std15='Stdfifteen', fixed='Fixed')
CT = dict(nominal='Nominal', wind='Wind', sensor='Sensor', traffic='Traffic', energy='Energy', compound='Compound')
PCT = ('succ', 'coll', 'coll_a', 'coll_b', 'dep', 'oob', 'rth', 'loss', 'cov', 'safe_ret', 'tmo')
MET = dict(succ='Succ', coll='Coll', coll_a='Colla', coll_b='Collb', dep='Dep', rth='Rth', loss='Loss', cov='Cov',
           energy='Energy', energy_cvar='Energycvar', tdel='Tdel', minsep_q05='Minsep', ret_cvar='Retcvar', ret='Ret',
           minbld_q05='Minbld', tight='Tight', n='Nmiss', oob='Oob')
ROM = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', '.': 'p'}


def fmt(k, v):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '--'
    if k in PCT:
        return '%.1f' % (100 * v)
    if k == 'n':
        return '%d' % v
    return '%.2f' % v if k in ('minsep_q05', 'minbld_q05') else '%.1f' % v


def numname(x):
    return ''.join(ROM[int(c)] if c.isdigit() else ('p' if c == '.' else c) for c in str(x))


macros, rows = {}, {}
for f in sorted(glob.glob('res/s%d/*.npz' % SEEDS[0])):
    nm = os.path.basename(f)[:-4]
    if nm.startswith('calscores'):
        continue
    pm = metrics(nm, seeds=SEEDS)
    if pm is None:
        continue
    rows[nm] = pm
    parts = nm.split('_')
    if parts[0] == 'main' and len(parts) == 4:
        _, m, g, c = parts
        if m in MT:
            for k, tok in MET.items():
                if k in pm:
                    macros['R%s%s%s%s' % (MT[m], GT[g], tok, CT[c])] = fmt(k, pm[k])
    elif parts[0] in ('abl', 'ablret') and len(parts) == 3:
        tag = {'fixedalpha': 'Fixedalpha', 'noenergy': 'Noenergy', 'prox': 'Prox', 'unc': 'Unc'}[parts[1]]
        pre = 'A' if parts[0] == 'abl' else 'Aret'
        for k, tok in MET.items():
            if k in pm:
                macros['%s%s%s%s' % (pre, tag, tok, CT[parts[2]])] = fmt(k, pm[k])
    elif parts[0] == 'par' and parts[-1] in CT:
        for k, tok in MET.items():
            if k in pm:
                macros['P%s%s%s' % (numname('_'.join(parts[1:-1])).replace('.', 'p').replace('_', ''), tok, CT[parts[-1]])] = fmt(k, pm[k])
    elif parts[0] == 'dial' and parts[-1] in CT:
        for k, tok in MET.items():
            if k in pm:
                macros['D%s%s%s' % (numname('_'.join(parts[1:-1])).replace('.', 'p').replace('_', ''), tok, CT[parts[-1]])] = fmt(k, pm[k])

# ---------------- seed spread and bootstrap CI for the headline comparison
ci_rows = {}
for nm in [n for n in rows if n.startswith('main_') and ('_std_' in n or '_online_' in n)]:
    pm = metrics(nm, seeds=SEEDS, boot=400)
    ci_rows[nm] = pm
    parts = nm.split('_'); m, g, c = parts[1], parts[2], parts[3]
    if m in MT:
        for k in ('loss', 'coll', 'succ', 'dep'):
            lo, hi = pm['ci'][k]
            macros['C%s%s%s%sLo' % (MT[m], GT[g], MET[k], CT[c])] = fmt(k, lo)
            macros['C%s%s%s%sHi' % (MT[m], GT[g], MET[k], CT[c])] = fmt(k, hi)
        ps = [pm['per_seed'][s]['loss'] for s in pm['seeds']]
        macros['S%s%s%s%s' % (MT[m], GT[g], 'Loss', CT[c])] = fmt('loss', float(np.std(ps)))

with open('pa/data/macros.tex', 'w') as f:
    f.write('%% generated from evaluation results, seeds %s\n' % (SEEDS,))
    for k in sorted(macros):
        f.write('\\newcommand{\\%s}{%s}\n' % (k, macros[k]))
    f.write('\\newcommand{\\Nseeds}{%d}\n' % len(SEEDS))

# ---------------- CSVs for figures
def wcsv(path, header, lines):
    with open(path, 'w') as f:
        f.write(header + '\n')
        f.write('\n'.join(lines) + '\n')


L = []
for nm, pm in sorted(rows.items()):
    p = nm.split('_')
    if p[0] == 'main' and len(p) == 4:
        ci = ci_rows.get(nm, {}).get('ci', {})
        extra = [fmt('loss', ci.get('loss', (None, None))[0]), fmt('loss', ci.get('loss', (None, None))[1]),
                 fmt('coll', ci.get('coll', (None, None))[0]), fmt('coll', ci.get('coll', (None, None))[1]),
                 fmt('succ', ci.get('succ', (None, None))[0]), fmt('succ', ci.get('succ', (None, None))[1])]
        L.append('%s,%s,%s,%s,%s' % (p[1], p[2], p[3], ','.join(fmt(k, pm.get(k)) for k in
                 ('succ', 'coll', 'coll_a', 'coll_b', 'dep', 'oob', 'rth', 'loss', 'energy', 'energy_cvar', 'tdel', 'minsep_q05', 'ret_cvar', 'cov', 'tight')), ','.join(extra)))
wcsv('pa/data/main.csv', 'method,guard,cond,succ,coll,coll_a,coll_b,dep,oob,rth,loss,energy,energy_cvar,tdel,minsep,retcvar,cov,tight,loss_lo,loss_hi,coll_lo,coll_hi,succ_lo,succ_hi', L)

L = []
for nm, pm in sorted(rows.items()):
    p = nm.split('_')
    if p[0] == 'sev' and len(p) == 5:
        L.append('%s,%s,%s,%s,%s' % (p[1], p[2], p[3], p[4], ','.join(fmt(k, pm.get(k)) for k in ('succ', 'coll', 'dep', 'loss', 'rth', 'energy', 'minsep_q05', 'cov'))))
if L:
    wcsv('pa/data/sev.csv', 'method,guard,family,sev,succ,coll,dep,loss,rth,energy,minsep,cov', L)

L = []
for nm, pm in sorted(rows.items()):
    p = nm.split('_')
    if p[0] == 'par' and p[-1] in CT:
        L.append('%s,%s,%s,%s' % ('_'.join(p[1:-2]), p[-2], p[-1], ','.join(fmt(k, pm.get(k)) for k in ('succ', 'dep', 'rth', 'loss', 'cov', 'tight', 'energy'))))
if L:
    wcsv('pa/data/pareto.csv', 'pol,rule,cond,succ,dep,rth,loss,cov,tight,energy', L)

L = []
for nm, pm in sorted(rows.items()):
    p = nm.split('_')
    if p[0] == 'dial' and p[-1] in CT:
        L.append('%s,%s,%s' % (p[-2], p[-1], ','.join(fmt(k, pm.get(k)) for k in ('succ', 'coll', 'dep', 'loss', 'tdel', 'energy', 'minsep_q05', 'ret_cvar'))))
if L:
    wcsv('pa/data/dial.csv', 'alpha,cond,succ,coll,dep,loss,tdel,energy,minsep,retcvar', L)

L = []
for nm, pm in sorted(rows.items()):
    p = nm.split('_')
    if p[0] in ('abl', 'ablret'):
        L.append('%s,%s,%s,%s' % (p[0], p[1], p[2], ','.join(fmt(k, pm.get(k)) for k in ('succ', 'coll', 'dep', 'loss', 'rth', 'energy', 'tdel', 'minsep_q05', 'ret_cvar'))))
if L:
    wcsv('pa/data/abl.csv', 'family,variant,cond,succ,coll,dep,loss,rth,energy,tdel,minsep,retcvar', L)

# learning curves from the training logs
L = []
for f in sorted(glob.glob('logs/train_*_s*.log')):
    base = os.path.basename(f)[6:-4]; algo, sd = base.rsplit('_s', 1)
    if int(sd) not in SEEDS:
        continue
    for line in open(f):
        i = line.find('{')
        if i < 0:
            continue
        try:
            d = json.loads(line[i:])
        except Exception:
            continue
        if 'step' in d:
            L.append('%s,%s,%d,%.4f,%.4f,%.4f,%.4f' % (algo, sd, d['step'], d.get('succ', np.nan), d.get('coll', np.nan), d.get('dep', np.nan), d.get('ret', np.nan)))
wcsv('pa/data/learn.csv', 'algo,seed,step,succ,coll,dep,ret', L)

# leg-level calibrator trace and running miscoverage (energy shift, CREST online reserve)
for tag, nm in [('energy', 'main_crest_online_energy'), ('wind', 'main_crest_online_wind'), ('nominal', 'main_crest_online_nominal')]:
    M, LG, per = load_table(nm, SEEDS)
    if LG is None or not len(LG):
        continue
    s0 = per[0][2]
    err = s0[:, 10]; lam = s0[:, 12]
    run = np.cumsum(err) / np.arange(1, len(err) + 1)
    wcsv('pa/data/lam_%s.csv' % tag, 'j,err,lam,running_miscov',
         ['%d,%.0f,%.4f,%.4f' % (j + 1, err[j], lam[j], run[j]) for j in range(len(err))])
print('macros', len(macros), 'rows', len(rows))
