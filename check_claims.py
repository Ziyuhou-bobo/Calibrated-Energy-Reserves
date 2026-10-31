"""Print the quantities behind each qualitative statement of the manuscript, so that the wording can be checked
against the final analysis (numbers quoted in the text are macros and need no check; this covers the words).
usage: python3 check_claims.py            (reads analysis/summary.pkl, or $ANALYSIS/summary.pkl)
"""
import os, pickle
import numpy as np

ANA = os.environ.get('ANALYSIS', 'analysis')
S = pickle.load(open(f'{ANA}/summary.pkl', 'rb'))
C = S['cells']; P = S['paired']
CONDS = ['nominal', 'wind', 'sensor', 'traffic', 'energy', 'compound']
LEARN = ['sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest']
TAIL = ['dsaccvar', 'crestret', 'crest']


def pp(x):
    return '%+.2f' % (100 * x)


def cell(nm, arm='fixed'):
    return C.get((arm, nm))


print('seeds', S['seeds'])
print('\n== 1. paired comparisons of tail-risk agents (loss), significant after Holm or with an interval excluding zero')
for g in ('std', 'online'):
    for ref in ('sac', 'dsacn'):
        for m in TAIL:
            for c in CONDS:
                d = P.get((g, ref, m, c, 'loss'))
                if d and (d['p_holm'] < 0.05 or d['lo'] > 0 or d['hi'] < 0):
                    print('  %-6s %-5s %-8s %-8s mean %s [%s, %s] p_holm %.3f  equiv %s' % (g, ref, m, c, pp(d['mean']), pp(d['lo']), pp(d['hi']), d['p_holm'], d['equiv']))

print('\n== 2. risk dial (alpha_b 0.1 minus 1.0), collisions and loss; * = 95% interval excludes zero')
for m in ('crest', 'crestret'):
    for c in CONDS:
        for k in ('coll', 'loss', 'spd', 'prox50', 'minsep_q05', 'tdel'):
            d = S['dial'].get((m, c, 'diff', k))
            if d:
                f = 100 if k in ('coll', 'loss') else 1
                star = '*' if (d['lo'] > 0 or d['hi'] < 0) else ' '
                print('  %-8s %-8s %-10s %+.2f [%+.2f, %+.2f] %s' % (m, c, k, f * d['mean'], f * d['lo'], f * d['hi'], star))

print('\n== 3. ablations (variant minus full schedule, loss), * = 95% interval excludes zero, H = Holm significant')
for (m, v, c), d in sorted(S['abl'].items()):
    star = ('*' if (d['lo'] > 0 or d['hi'] < 0) else ' ') + ('H' if d['p_holm'] < 0.05 else ' ')
    print('  %-8s %-10s %-8s %s [%s, %s] %s' % (m, v, c, pp(d['mean']), pp(d['lo']), pp(d['hi']), star))
for m in ('crest', 'crestret'):
    for v in ('fixedalpha', 'noenergy', 'prox', 'unc'):
        dd = [S['abl'][(m, v, c)] for c in CONDS if (m, v, c) in S['abl']]
        if dd:
            print('  %-8s %-10s mean %s, positive in %d of %d, CI>0 in %d, Holm %d' % (m, v, pp(np.mean([d['mean'] for d in dd])), sum(d['mean'] > 0 for d in dd), len(dd),
                  sum(d['lo'] > 0 for d in dd), sum(d['p_holm'] < 0.05 for d in dd)))

print('\n== 4. differences between agents (range of loss over the six learned agents) against ablation effects')
for c in CONDS:
    v = [cell(f'main_{m}_std_{c}') for m in LEARN]
    v = [x['loss'] for x in v if x]
    r = cell(f'main_reactive_std_{c}')
    print('  %-8s learned loss %.1f-%.1f %%   reactive %.1f %%' % (c, 100 * min(v), 100 * max(v), 100 * r['loss'] if r else np.nan))

print('\n== 5. hardest condition (fixed reserve): loss and success per agent')
for m in LEARN + ['reactive']:
    L = {c: cell(f'main_{m}_std_{c}') for c in CONDS}
    lo = max(CONDS, key=lambda c: L[c]['loss'] if L[c] else -1); su = min(CONDS, key=lambda c: L[c]['succ'] if L[c] else 2)
    print('  %-9s highest loss: %-8s (%.1f%%)   lowest success: %-8s (%.1f%%)' % (m, lo, 100 * L[lo]['loss'], su, 100 * L[su]['succ']))

print('\n== 6. composition of losses')
for m in ('sac', 'crest', 'reactive'):
    for c in ('nominal', 'energy', 'traffic', 'sensor', 'compound'):
        x = cell(f'main_{m}_std_{c}')
        if x:
            print('  %-8s %-8s loss %.1f  coll %.1f (aircraft %.1f: fleet %.1f, det %.1f, undet %.1f; building %.1f)  dep %.1f  oob %.1f  rth %.1f  spd %.1f' % (
                m, c, 100 * x['loss'], 100 * x['coll'], 100 * x['coll_a'], 100 * x['coll_af'], 100 * x['coll_ai_det'], 100 * x['coll_ai_undet'],
                100 * x['coll_b'], 100 * x['dep'], 100 * x['oob'], 100 * x['rth'], x['spd']))

print('\n== 7. one-factor sub-shifts (SAC, aircraft and building collisions, %)')
for nm in ('sensor_nav', 'sensor_windmeas', 'sensor_miss', 'sensor_intnoise', 'traffic_fleet', 'traffic_intruders'):
    for m in ('sac', 'crest', 'reactive'):
        x = cell(f'sub_{m}_std_{nm}')
        if x:
            print('  %-18s %-8s coll_a %.1f [%.1f, %.1f]  coll_b %.1f' % (nm, m, 100 * x['coll_a'], 100 * x['ci_coll_a']['lo'], 100 * x['ci_coll_a']['hi'], 100 * x['coll_b']))
for m in ('sac', 'crest'):
    x = cell(f'main_{m}_std_nominal')
    print('  nominal            %-8s coll_a %.1f [%.1f, %.1f]  coll_b %.1f' % (m, 100 * x['coll_a'], 100 * x['ci_coll_a']['lo'], 100 * x['ci_coll_a']['hi'], 100 * x['coll_b']))

print('\n== 8. severity sweeps: learned-agent range and reactive rule (loss %)')
for fam in ('wind', 'sensor', 'traffic', 'energy'):
    for sv in (0.25, 0.5, 0.75):
        v = [cell(f'sev_{m}_std_{fam}_{sv}') for m in LEARN]
        v = [x for x in v if x]
        r = cell(f'sev_reactive_std_{fam}_{sv}')
        if v:
            print('  %-8s %.2f learned %.1f-%.1f  (CI lows %.1f-%.1f, highs %.1f-%.1f)  reactive %s' % (
                fam, sv, 100 * min(x['loss'] for x in v), 100 * max(x['loss'] for x in v), 100 * min(x['ci_loss']['lo'] for x in v),
                100 * max(x['ci_loss']['lo'] for x in v), 100 * min(x['ci_loss']['hi'] for x in v), 100 * max(x['ci_loss']['hi'] for x in v),
                '%.1f' % (100 * r['loss']) if r else '--'))

print('\n== 9. reserve frontier: conformal 5% point against fixed margins (abandoned %, depletion %)')
FIX = ['fixed1.0', 'fixed1.2', 'fixed1.35', 'fixed1.5', 'fixed1.75', 'fixed2.0', 'fixed2.5']
ONL = ['online0.02', 'online0.05', 'online0.1', 'online0.2', 'online0.3']
for m in ('sac', 'crest', 'reactive'):
    for c in ('nominal', 'wind', 'energy'):
        F = [(ru, S['frontier'].get((m, c, ru))) for ru in FIX]
        F = [(ru, p) for ru, p in F if p]
        if not F:
            continue
        xs = np.array([p['safe_ret'] for _, p in F]); ys = np.array([p['dep'] for _, p in F])
        o = np.argsort(xs)
        for ru in ONL:
            q = S['frontier'].get((m, c, ru))
            if not q:
                continue
            dom_by = [r for r, p in F if p['safe_ret'] <= q['safe_ret'] and p['dep'] <= q['dep']]
            doms = [r for r, p in F if p['safe_ret'] >= q['safe_ret'] and p['dep'] >= q['dep']]
            yi = np.interp(q['safe_ret'], xs[o], ys[o])
            print('  %-8s %-8s %-10s rth %.1f [%.1f,%.1f] dep %.2f [%.2f,%.2f]  frontier dep at same rth %.2f  dominated by %s  dominates %s' % (
                m, c, ru, 100 * q['safe_ret'], 100 * q['ci_safe_ret']['lo'], 100 * q['ci_safe_ret']['hi'], 100 * q['dep'], 100 * q['ci_dep']['lo'],
                100 * q['ci_dep']['hi'], 100 * yi, dom_by or '-', doms or '-'))
        print('    fixed:', ', '.join('%s %.1f/%.2f' % (ru[5:], 100 * p['safe_ret'], 100 * p['dep']) for ru, p in F))
    onl_nom = [S['frontier'].get((m, 'nominal', ru)) for ru in ONL]
    print('  %s nominal abandonment by target:' % m, ['%.1f' % (100 * p['safe_ret']) if p else '--' for p in onl_nom])

print('\n== 10. online reserve coverage and calibrator')
allr = [r for d in S['calib_diag'].values() for r in d['rows']]
print('  coverage min %.1f max %.1f; runs below 94%%: %d of %d' % (100 * min(r['cov'] for r in allr), 100 * max(r['cov'] for r in allr),
      sum(r['cov'] < 0.94 for r in allr), len(allr)))
print('  excursions %d, at bound %d, max disp %.3f' % (sum(r['excursion'] for r in allr), sum(r['atb'] > 0 for r in allr), max(r['disp'] for r in allr)))
for (m, c), d in sorted(S['calib_diag'].items()):
    ex = [r for r in d['rows'] if r['excursion']]
    if ex:
        print('   %-9s %-8s excursions %d of %d: rth %s vs others %s' % (m, c, len(ex), len(d['rows']), ['%.1f' % (100 * r['rth']) for r in ex],
              ['%.1f' % (100 * r['rth']) for r in d['rows'] if not r['excursion']]))
lt = S['lam_trace'].get(('crest', 'energy')); ln = S['lam_trace'].get(('crest', 'nominal'))
if lt is not None and ln is not None:
    print('  fig lam (a): CREST-C seed 0 mean lambda energy %.2f nominal %.2f (second half)' % (lt[len(lt) // 2:, 0].mean(), ln[len(ln) // 2:, 0].mean()))

print('\n== 11. training-reserve arm')
print('  seeds of the online-trained arm:', S.get('tr_seeds'))
for k, r in sorted(S['train_reserve'].items()):
    if not isinstance(r, dict):
        continue
    print('  ', k, {kk: (round(100 * vv, 2) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk in ('mean', 'lo', 'hi', 'p', 'n')})

print('\n== 12. airspeed (m/s) and delivery time at nominal')
for m in LEARN + ['reactive']:
    x = cell(f'main_{m}_std_nominal')
    print('  %-9s spd %.1f  tdel %.1f  succ %.1f' % (m, x['spd'], x['tdel'], 100 * x['succ']))
