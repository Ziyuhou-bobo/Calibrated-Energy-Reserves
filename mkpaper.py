"""Generate ms/data/macros.tex and the generated tables ms/tab_*.tex from analysis/summary.pkl.
Every number quoted in the manuscript is one of these macros; tables are written from the same summary.
usage: python3 mkpaper.py
"""
import json, os, pickle, re
import numpy as np

ANA = os.environ.get('ANALYSIS', 'analysis')
MS = os.environ.get('MSDIR', 'ms')
S = pickle.load(open(f'{ANA}/summary.pkl', 'rb'))
os.makedirs(f'{MS}/data', exist_ok=True)
MT = dict(sac='Sac', td3='Tdthree', dsacn='Dsacn', dsaccvar='Dsaccvar', crestret='Crestr', crest='Crestc', reactive='Reac', ppo='Ppo')
LAB = dict(sac='SAC', td3='TD3', dsacn='DSAC-N', dsaccvar='DSAC-CVaR', crestret='CREST-R', crest='CREST-C', reactive='reactive rule')
POLNAME = dict(sac='the soft actor-critic', td3='TD3', dsacn='DSAC-N', dsaccvar='DSAC-CVaR', crestret='CREST-R', crest='CREST-C',
               reactive='the reactive rule')
CONDNAME = dict(nominal='at nominal', wind='under the wind shift', sensor='under the sensor shift', traffic='under the traffic shift',
                energy='under the energy shift', compound='in the compound condition')
GT = dict(std='Std', online='Online', std15='Stdfifteen', static='Static')
CONDS = ['nominal', 'wind', 'sensor', 'traffic', 'energy', 'compound']
CT = {c: c.capitalize() for c in CONDS}
KT = dict(succ='Succ', coll='Coll', coll_a='Colla', coll_b='Collb', dep='Dep', safe_ret='Rth', loss='Loss', energy='Energy',
          tdel='Tdel', cov='Cov', tight='Tight', spd='Spd', prox50='Proxfifty', fprox50='Fproxfifty', prox25='Proxtwentyfive',
          minsep_q05='Minsep', coll_af='Collaf', coll_ai='Collai', coll_ai_det='Collaidet', coll_ai_undet='Collaiundet',
          coll_a_outslot='Collaout', cancel='Cancel', n='Nmiss', oob='Oob', tflight='Tflight', gspd='Gspd', act='Act')
PCT = {'succ', 'coll', 'coll_a', 'coll_b', 'dep', 'safe_ret', 'loss', 'cov', 'coll_af', 'coll_ai', 'coll_ai_det', 'coll_ai_undet',
       'coll_a_outslot', 'cancel', 'oob', 'fprox50'}
ALT = {0.1: 'Zpone', 0.25: 'Zptwofive', 0.5: 'Zpfive', 0.75: 'Zpsevenfive', 1.0: 'Onepzero'}
RULET = {'fixed1.0': 'Fone', 'fixed1.2': 'Fonetwo', 'fixed1.35': 'Fonethreefive', 'fixed1.5': 'Fonefive', 'fixed1.75': 'Fonesevenfive',
         'fixed2.0': 'Ftwo', 'fixed2.5': 'Ftwofive', 'online0.02': 'Otwo', 'online0.05': 'Ofive', 'online0.1': 'Oten',
         'online0.2': 'Otwenty', 'online0.3': 'Othirty'}
SUBT = dict(sensor_nav='Nav', sensor_windmeas='Windmeas', sensor_miss='Miss', sensor_intnoise='Intnoise', traffic_fleet='Fleet',
            traffic_intruders='Intruders')
MAC = {}
WORDS = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten', 11: 'eleven',
         12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen'}


def NWORD(n):
    return WORDS.get(n, str(n))


def OFN(k, n):
    """'all six', 'none of the six', 'four of the six' (counts written out for the prose)."""
    k, n = int(k), int(n)
    if k == n:
        return 'both' if n == 2 else 'all ' + NWORD(n)
    if k == 0:
        return 'none of the ' + NWORD(n)
    return NWORD(k) + ' of the ' + NWORD(n)


def CNT(k):
    return 'none' if int(k) == 0 else NWORD(int(k))


def num(v, d=1, pct=False, sign=False):
    if v is None or not np.isfinite(v):
        return '--'
    x = 100 * v if pct else v
    s = ('%+.' + str(d) + 'f') % x if sign else ('%.' + str(d) + 'f') % x
    if s.startswith('-'):
        s = r'\ensuremath{-}' + s[1:]
    elif s.startswith('+'):
        s = r'\ensuremath{+}' + s[1:]
    if s in (r'\ensuremath{-}0.0', r'\ensuremath{+}0.0'):
        s = '0.0'
    return s


def fmt(k, v):
    if k == 'n':
        return '%d' % v
    if k in PCT:
        return num(v, 1, pct=True)
    if k in ('tight', 'minsep_q05'):
        return num(v, 2)
    return num(v, 1)


def pval(p):
    if p is None or not np.isfinite(p):
        return '--'
    return r'\ensuremath{<}0.001' if p < 0.001 else '%.3f' % p


def put(name, val):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    MAC[name] = val


def cell(arm, name):
    return S['cells'].get((arm, name))


# ------------------------------------------------------------------ main cells (both arms) and reserve rules
for arm, pre in (('fixed', 'R'), ('online', 'Ro')):
    for (a, nm), ce in S['cells'].items():
        if a != arm or ce is None or '|' in nm:
            continue
        p = nm.split('_')
        if p[0] == 'main' and len(p) == 4 and p[1] in MT and p[2] in GT:
            for k, tok in KT.items():
                if k in ce:
                    put(f'{pre}{MT[p[1]]}{GT[p[2]]}{tok}{CT[p[3]]}', fmt(k, ce[k]))
            for k in ('loss', 'succ', 'coll', 'coll_a', 'dep', 'safe_ret', 'cov'):
                c = ce.get('ci_' + k)
                if c and c['n'] >= 2:
                    put(f'{pre}{MT[p[1]]}{GT[p[2]]}{KT[k]}{CT[p[3]]}Lo', fmt(k, c['lo']))
                    put(f'{pre}{MT[p[1]]}{GT[p[2]]}{KT[k]}{CT[p[3]]}Hi', fmt(k, c['hi']))
                    put(f'{pre}{MT[p[1]]}{GT[p[2]]}{KT[k]}{CT[p[3]]}Sd', fmt(k, c['sd']))
        if p[0] == 'disp' and len(p) == 4:
            for k, tok in KT.items():
                if k in ce:
                    put(f'Disp{MT[p[1]]}{GT[p[2]]}{tok}{CT[p[3]]}', fmt(k, ce[k]))
        if p[0] == 'sub' and len(p) >= 4:
            sub = '_'.join(p[3:])
            for k, tok in KT.items():
                if k in ce:
                    put(f'Sub{MT[p[1]]}{SUBT[sub]}{tok}', fmt(k, ce[k]))
        if p[0] == 'sev' and len(p) == 5:
            for k in ('loss', 'coll', 'succ'):
                put(f'Sev{MT[p[1]]}{CT[p[3]]}{ {"0.25": "Q", "0.5": "H", "0.75": "Tq"}[p[4]] }{KT[k]}', fmt(k, ce[k]))
# missions per cell
seeds_f = S['seeds']['fixed']; seeds_o = S['seeds']['online']
put('Nseeds', str(len(seeds_f))); put('Nseedsonline', str(len(seeds_o)))
ce = cell('fixed', 'main_sac_std_nominal'); put('Nmissnominal', '{:,}'.format(ce['n']).replace(',', '{,}') if ce else '--')
ce = cell('fixed', 'main_sac_std_traffic'); put('Nmisstraffic', '{:,}'.format(ce['n']).replace(',', '{,}') if ce else '--')

# ------------------------------------------------------------------ paired differences
PD = S['paired']
for (g, ref, m, c, key), r in PD.items():
    base = f'D{MT[ref]}{MT[m]}{GT[g]}{KT[key]}{CT[c]}'
    put(base + 'Mean', num(r['mean'], 1, pct=True, sign=True)); put(base + 'Lo', num(r['lo'], 1, pct=True)); put(base + 'Hi', num(r['hi'], 1, pct=True))
    put(base + 'Lon', num(r['lo90'], 1, pct=True)); put(base + 'Hin', num(r['hi90'], 1, pct=True)); put(base + 'Mde', num(r['mde'], 1, pct=True))
    put(base + 'P', pval(r['p'])); put(base + 'Ph', pval(r.get('p_holm', np.nan)))
    if 'hb' in r:
        put(base + 'Hblo', num(r['hb'][0], 1, pct=True)); put(base + 'Hbhi', num(r['hb'][1], 1, pct=True))
TAIL = ['dsaccvar', 'crestret', 'crest']
for g in ('std', 'online'):
    for ref in ('sac', 'dsacn'):
        rows = [PD[(g, ref, m, c, 'loss')] for m in TAIL for c in CONDS if (g, ref, m, c, 'loss') in PD]
        if not rows:
            continue
        tag = f'D{MT[ref]}Tail{GT[g]}'
        put(tag + 'Bound', num(max(max(abs(r['lo']), abs(r['hi'])) for r in rows), 1, pct=True))
        put(tag + 'Maxabs', num(max(abs(r['mean']) for r in rows), 1, pct=True))
        put(tag + 'Eqcount', str(sum(r['equiv'] for r in rows))); put(tag + 'Ncomp', str(len(rows)))
        put(tag + 'Sigcount', str(sum(r.get('p_holm', 1) < 0.05 for r in rows)))
        put(tag + 'Rawsigcount', str(sum(r['p'] < 0.05 for r in rows)))
        put(tag + 'Mdemin', num(min(r['mde'] for r in rows), 1, pct=True)); put(tag + 'Mdemax', num(max(r['mde'] for r in rows), 1, pct=True))
        neg = sum(r['mean'] < 0 for r in rows); put(tag + 'Negcount', str(neg))
        put(tag + 'Sigw', CNT(sum(r.get('p_holm', 1) < 0.05 for r in rows))); put(tag + 'Negw', CNT(neg))
        for m in TAIL:
            rr = [PD[(g, ref, m, c, 'loss')] for c in CONDS if (g, ref, m, c, 'loss') in PD]
            if rr:
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossMinmean', num(min(r['mean'] for r in rr), 1, pct=True, sign=True))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossMaxmean', num(max(r['mean'] for r in rr), 1, pct=True, sign=True))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossBound', num(max(max(abs(r['lo']), abs(r['hi'])) for r in rr), 1, pct=True))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossEqcount', str(sum(r['equiv'] for r in rr)))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossSigcount', str(sum(r.get('p_holm', 1) < 0.05 for r in rr)))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossPoscount', str(sum(r['mean'] > 0 for r in rr)))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossNegcount', str(sum(r['mean'] < 0 for r in rr)))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossCicount', str(sum((r['lo'] > 0) or (r['hi'] < 0) for r in rr)))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossAvg', num(np.mean([r['mean'] for r in rr]), 1, pct=True, sign=True))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossAvgu', num(abs(np.mean([r['mean'] for r in rr])), 1, pct=True))
                n_ = len(rr)
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossPosof', OFN(sum(r['mean'] > 0 for r in rr), n_))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossNegof', OFN(sum(r['mean'] < 0 for r in rr), n_))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossCiof', OFN(sum((r['lo'] > 0) or (r['hi'] < 0) for r in rr), n_))
                put(f'D{MT[ref]}{MT[m]}{GT[g]}LossSigw', CNT(sum(r.get('p_holm', 1) < 0.05 for r in rr)))
# which condition is hardest for whom, and the reactive rule's aircraft collisions
LEARNED = ['sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest']
_hl = {}
for m in LEARNED:
    L_ = {c: cell('fixed', f'main_{m}_std_{c}') for c in CONDS}
    if all(L_.values()):
        w_ = max(CONDS, key=lambda c: L_[c]['loss']); _hl[w_] = _hl.get(w_, 0) + 1
for c in CONDS:
    put(f'Hardest{CT[c]}w', CNT(_hl.get(c, 0)))
_ls = 0; _np_ = 0
for m in LEARNED + ['reactive']:
    L_ = {c: cell('fixed', f'main_{m}_std_{c}') for c in CONDS}
    if all(L_.values()):
        _np_ += 1; _ls += min(CONDS, key=lambda c: L_[c]['succ']) == 'energy'
put('LowestSuccEnergyof', OFN(_ls, _np_))
_rc = [cell('fixed', f'main_reactive_std_{c}') for c in CONDS]
if all(_rc):
    put('RReacCollaMax', num(max(x['coll_a'] for x in _rc), 1, pct=True))
# spread of the learned agents in the severity sweeps
_sp = []
for fam in ('wind', 'sensor', 'traffic', 'energy'):
    for sv in (0.25, 0.5, 0.75):
        v_ = [cell('fixed', f'sev_{m}_std_{fam}_{sv}') for m in LEARNED]
        v_ = [x['loss'] for x in v_ if x]
        if len(v_) > 1:
            _sp.append(max(v_) - min(v_))
if _sp:
    put('SevSpreadMax', num(max(_sp), 1, pct=True))
# every learned agent against SAC (context: how far the risk-neutral agents are from SAC)
for g in ('std', 'online'):
    for m in ('td3', 'dsacn', 'dsaccvar', 'crestret', 'crest'):
        rr = [PD[(g, 'sac', m, c, 'loss')] for c in CONDS if (g, 'sac', m, c, 'loss') in PD]
        if rr:
            put(f'DSac{MT[m]}{GT[g]}LossLow', num(min(r['mean'] for r in rr), 1, pct=True))
            put(f'DSac{MT[m]}{GT[g]}LossHigh', num(max(r['mean'] for r in rr), 1, pct=True))
    rr = [PD[(g, 'sac', m, c, 'loss')] for m in ('td3', 'dsacn') for c in CONDS if (g, 'sac', m, c, 'loss') in PD]
    if rr:
        put(f'DSacRiskneutral{GT[g]}LossLow', num(min(r['mean'] for r in rr), 1, pct=True))
        put(f'DSacRiskneutral{GT[g]}LossHigh', num(max(r['mean'] for r in rr), 1, pct=True))
        put(f'DSacRiskneutral{GT[g]}LossPosof', OFN(sum(r['mean'] > 0 for r in rr), len(rr)))
# learned agents versus the reactive rule
for g in ('std', 'online'):
    rows = [PD[(g, 'reactive', m, c, 'loss')] for m in MT if m not in ('reactive', 'ppo') for c in CONDS if (g, 'reactive', m, c, 'loss') in PD]
    if rows:
        put(f'DReacAll{GT[g]}LossMin', num(min(r['mean'] for r in rows), 1, pct=True))
        put(f'DReacAll{GT[g]}LossMax', num(max(r['mean'] for r in rows), 1, pct=True))
        put(f'DReacAll{GT[g]}LossLomin', num(min(r['lo'] for r in rows), 1, pct=True))

# ------------------------------------------------------------------ seed-level spread of the main loss rates
for g in ('std', 'online'):
    for m in MT:
        sds = [S['cells'][('fixed', f'main_{m}_{g}_{c}')]['ci_loss']['sd'] for c in CONDS if S['cells'].get(('fixed', f'main_{m}_{g}_{c}'))]
        if sds and np.all(np.isfinite(sds)):
            put(f'Sdrange{MT[m]}{GT[g]}Lo', num(min(sds), 1, pct=True)); put(f'Sdrange{MT[m]}{GT[g]}Hi', num(max(sds), 1, pct=True))
# coverage ranges across conditions for every policy and rule
for g in ('std', 'std15', 'static', 'online'):
    for m in MT:
        v = [S['cells'][('fixed', f'main_{m}_{g}_{c}')]['cov'] for c in CONDS if S['cells'].get(('fixed', f'main_{m}_{g}_{c}')) and 'cov' in S['cells'][('fixed', f'main_{m}_{g}_{c}')]]
        if v:
            put(f'Covrange{MT[m]}{GT[g]}Lo', fmt('cov', min(v))); put(f'Covrange{MT[m]}{GT[g]}Hi', fmt('cov', max(v)))
# overall range of catastrophic loss of the six learned agents over all conditions (fixed reserve)
allv = [S['cells'][('fixed', f'main_{m}_std_{c}')]['loss'] for m in ('sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest') for c in CONDS
        if S['cells'].get(('fixed', f'main_{m}_std_{c}'))]
if allv:
    put('RngAllStdLossLo', fmt('loss', min(allv))); put('RngAllStdLossHi', fmt('loss', max(allv)))
# ranges across the six off-policy agents
for g in ('std', 'online'):
    for c in CONDS:
        for k in ('loss', 'succ', 'coll_a', 'coll_b', 'energy', 'tdel', 'spd'):
            vals = [S['cells'][('fixed', f'main_{m}_{g}_{c}')][k] for m in ('sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest')
                    if S['cells'].get(('fixed', f'main_{m}_{g}_{c}')) and k in S['cells'][('fixed', f'main_{m}_{g}_{c}')]]
            if vals:
                put(f'Rng{GT[g]}{KT[k]}{CT[c]}Lo', fmt(k, min(vals))); put(f'Rng{GT[g]}{KT[k]}{CT[c]}Hi', fmt(k, max(vals)))

# ------------------------------------------------------------------ risk dial
for key, v in S['dial'].items():
    if len(key) == 3:
        m, c, ab = key
        for k in ('coll', 'loss', 'succ', 'tdel', 'spd', 'prox50', 'fprox50', 'minsep_q05', 'coll_a', 'gspd', 'act'):
            if k in v:
                put(f'Dial{MT[m]}{ALT[ab]}{KT[k]}{CT[c]}', fmt(k, v[k]))
    else:
        m, c, _, k = key
        tok = f'Dialdiff{MT[m]}{KT[k]}{CT[c]}'
        f = (lambda x: num(x, 1, pct=True, sign=True)) if k in PCT else (lambda x: num(x, 2 if k in ('minsep_q05', 'spd') else 1, sign=True))
        g = (lambda x: num(x, 1, pct=True)) if k in PCT else (lambda x: num(x, 2 if k in ('minsep_q05', 'spd') else 1))
        put(tok + 'Mean', f(v['mean'])); put(tok + 'Lo', g(v['lo'])); put(tok + 'Hi', g(v['hi'])); put(tok + 'P', pval(v['p']))
put('DialCheckN', str(S['dial_check']['n'])); put('DialCheckMax', num(100 * S['dial_check']['max_abs'], 2))
# summaries of the dial differences (alpha_b = 0.1 minus 1.0) over the conditions in which the dial was evaluated
for m in ('crest', 'crestret'):
    for k in ('coll', 'loss', 'prox50', 'minsep_q05', 'spd', 'tdel', 'succ'):
        rr = [v for (mm, c, *rest), v in S['dial'].items() if mm == m and len(rest) == 2 and rest[1] == k]
        if not rr:
            continue
        tok = f'Dialsum{MT[m]}{KT[k]}'
        f = (lambda x: num(x, 1, pct=True, sign=True)) if k in PCT else (lambda x: num(x, 2 if k == 'minsep_q05' else 1, sign=True))
        put(tok + 'Neg', str(sum(v['mean'] < 0 for v in rr))); put(tok + 'Pos', str(sum(v['mean'] > 0 for v in rr)))
        put(tok + 'Ci', str(sum((v['lo'] > 0) or (v['hi'] < 0) for v in rr))); put(tok + 'Nc', str(len(rr)))
        put(tok + 'Min', f(min(v['mean'] for v in rr))); put(tok + 'Max', f(max(v['mean'] for v in rr)))
        put(tok + 'Ciup', str(sum(v['lo'] > 0 for v in rr))); put(tok + 'Cidown', str(sum(v['hi'] < 0 for v in rr)))
        put(tok + 'Negof', OFN(sum(v['mean'] < 0 for v in rr), len(rr))); put(tok + 'Ciof', OFN(sum((v['lo'] > 0) or (v['hi'] < 0) for v in rr), len(rr)))
        put(tok + 'Ciw', CNT(sum((v['lo'] > 0) or (v['hi'] < 0) for v in rr)))
        a_ = max(abs(v['mean']) for v in rr)
        put(tok + 'Absmax', num(a_, 1, pct=True) if k in PCT else num(a_, 2 if k == 'minsep_q05' else 1))
# selected base levels
for m in ('crest', 'crestret'):
    vals = [S['ab_sel'][(m, s)] for s in seeds_f if (m, s) in S['ab_sel']]
    if vals:
        cnt = {a: vals.count(a) for a in sorted(set(vals))}
        parts = ['%g in %d' % (a, n) for a, n in cnt.items()]
        put(f'Absel{MT[m]}', parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1])
        put(f'Abmode{MT[m]}', '%g' % max(cnt, key=cnt.get)); put(f'Abmodecount{MT[m]}', str(max(cnt.values())))

# ------------------------------------------------------------------ ablations
AB = S['abl']
for (m, v, c), r in AB.items():
    tok = f'Abl{MT[m]}{v.capitalize()}{CT[c]}'
    put(tok + 'Mean', num(r['mean'], 1, pct=True, sign=True)); put(tok + 'Lo', num(r['lo'], 1, pct=True)); put(tok + 'Hi', num(r['hi'], 1, pct=True))
    put(tok + 'Ph', pval(r.get('p_holm', np.nan)))
for m in ('crest', 'crestret'):
    for v in ('fixedalpha', 'noenergy', 'prox', 'unc'):
        rr = [AB[(m, v, c)] for c in CONDS if (m, v, c) in AB]
        if rr:
            put(f'Abl{MT[m]}{v.capitalize()}Avg', num(np.mean([r['mean'] for r in rr]), 1, pct=True, sign=True))
            put(f'Abl{MT[m]}{v.capitalize()}Avgu', num(abs(np.mean([r['mean'] for r in rr])), 1, pct=True))
            put(f'Abl{MT[m]}{v.capitalize()}Minmean', num(min(r['mean'] for r in rr), 1, pct=True, sign=True))
            put(f'Abl{MT[m]}{v.capitalize()}Maxmean', num(max(r['mean'] for r in rr), 1, pct=True, sign=True))
            put(f'Abl{MT[m]}{v.capitalize()}Poscount', str(sum(r['mean'] > 0 for r in rr)))
            put(f'Abl{MT[m]}{v.capitalize()}Sigcount', str(sum(r.get('p_holm', 1) < 0.05 for r in rr)))
            put(f'Abl{MT[m]}{v.capitalize()}Rawsigcount', str(sum(r['p'] < 0.05 for r in rr)))
            put(f'Abl{MT[m]}{v.capitalize()}Cicount', str(sum((r['lo'] > 0) or (r['hi'] < 0) for r in rr)))
            put(f'Abl{MT[m]}{v.capitalize()}Posof', OFN(sum(r['mean'] > 0 for r in rr), len(rr)))
            put(f'Abl{MT[m]}{v.capitalize()}Ciw', CNT(sum((r['lo'] > 0) or (r['hi'] < 0) for r in rr)))
            put(f'Abl{MT[m]}{v.capitalize()}Sigw', CNT(sum(r.get('p_holm', 1) < 0.05 for r in rr)))
for m in ('crest', 'crestret'):
    cm = [sum((AB[(m, v, c)]['lo'] > 0) or (AB[(m, v, c)]['hi'] < 0) for c in CONDS if (m, v, c) in AB) for v in ('noenergy', 'prox', 'unc')]
    put(f'Abl{MT[m]}OtherCimaxw', CNT(max(cm)) if cm else '--')
# k=3 references for the ablation and dial tables
for m in ('crest', 'crestret'):
    for c in CONDS:
        ce = cell('fixed', f'main_{m}_std_{c}|k3')
        if ce:
            put(f'Ref{MT[m]}Loss{CT[c]}', fmt('loss', ce['loss']))

# ------------------------------------------------------------------ calibrator diagnostics and continual deployment
for (m, c), d in S['calib_diag'].items():
    tok = f'Cal{MT[m]}{CT[c]}'
    put(tok + 'Cov', num(d['cov']['mean'], 1, pct=True)); put(tok + 'Covhund', num(d['cov100']['mean'], 1, pct=True))
    put(tok + 'Covhundlo', num(min(r['cov100'] for r in d['rows']), 1, pct=True))
    put(tok + 'Covrest', num(d['cov_rest']['mean'], 1, pct=True))
    put(tok + 'Disp', num(d['disp']['mean'], 2)); put(tok + 'Absdisp', num(d['absdisp']['mean'], 2))
    put(tok + 'Atb', num(d['atb']['mean'], 1, pct=True)); put(tok + 'Lamend', num(d['lam_end']['mean'], 2)); put(tok + 'Lammax', num(max(r['lam_max'] for r in d['rows']), 2))
    put(tok + 'Legspermin', num(d['legs_per_min']['mean'], 0)); put(tok + 'Minperhund', num(100 / d['legs_per_min']['mean'], 1))
    put(tok + 'Simmin', num(d['sim_min']['mean'], 0)); put(tok + 'Nlegs', num(d['n']['mean'], 0))
    put(tok + 'Depepzero', num(d['dep_ep0']['mean'], 1, pct=True)); put(tok + 'Depeprest', num(d['dep_ep1p']['mean'], 1, pct=True))
allrows = [r for d in S['calib_diag'].values() for r in d['rows']]
if allrows:
    put('CalAllOvershootN', str(int(sum(r['overshoot'] for r in allrows))))
    put('CalAllNatb', str(int(sum(r['atb'] > 0 for r in allrows)))); put('CalAllNatbw', CNT(sum(r['atb'] > 0 for r in allrows)))
    put('CalAllExcursionN', str(int(sum((r['overshoot'] > 0) or (r['burst'] >= 0.25) for r in allrows))))
    put('CalTrafficExcursionN', str(int(sum(((r['overshoot'] > 0) or (r['burst'] >= 0.25)) for (m, c), d in S['calib_diag'].items() if c == 'traffic' for r in d['rows']))))
    put('CalTrafficNruns', str(int(sum(1 for (m, c), d in S['calib_diag'].items() if c == 'traffic' for r in d['rows']))))
    put('CalTrafficInflight', num(np.nanmean([r['inflight'] for (m, c), d in S['calib_diag'].items() if c == 'traffic' for r in d['rows']]), 0))
    ex = S.get('lam_example')
    if ex:
        put('CalExampleLammax', num(float(np.max(ex['lam'])), 1)); put('CalExampleRatio', num(ex['ratio'], 1))
        ia = int(np.argmax(ex['lam'])); p_ = ex['name'].split('_')
        put('CalExampleLamzero', num(float(ex['lam_before'][0]), 2)); put('CalExampleLammin', num(float(ex['lam'][:ia + 1].min()), 2))
        put('CalExampleLamend', num(float(ex['lam'][-1]), 1)); put('CalExampleNlegs', str(len(ex['lam'])))
        put('CalExamplePeakleg', str(ia + 1)); put('CalExamplePol', POLNAME[p_[1]]); put('CalExampleCond', CONDNAME[p_[3]])
        put('CalExampleSeed', str(int(ex['seed'])))
    put('CalAllBurstN', str(int(sum(r['burst'] >= 0.25 for r in allrows))))
    put('CalAllInflight', num(np.nanmean([r['inflight'] for r in allrows if np.isfinite(r['inflight'])]), 0))
    put('CalAllLegdur', num(np.nanmean([r['leg_dur'] for r in allrows if np.isfinite(r['leg_dur'])]), 0))
    nomr = [r for (m, c), d in S['calib_diag'].items() if c == 'nominal' for r in d['rows']]
    put('CalNomInflight', num(np.nanmean([r['inflight'] for r in nomr]), 0))
    ov = [r for r in allrows if r['excursion']]; nov = [r for r in allrows if not r['excursion']]
    put('CalAllDispCorrMax', num(100 * max(r['dispcorr'] for r in allrows), 2))
    if ov:
        put('CalBurstNdist', num(np.nanmean([r['burst_ndist'] for r in ov]), 0)); put('CalBurstEpshare', num(np.nanmean([r['burst_epshare'] for r in ov]), 0, pct=True))
        put('CalExcDipN', str(int(sum(r['lam_min_pre'] < r['lam0'] - 0.1 for r in ov))))
    # abandoned deliveries in excursion runs, compared within the same policy and condition
    dif = []
    for (m, c), d in S['calib_diag'].items():
        a = [r['rth'] for r in d['rows'] if r['excursion']]; b = [r['rth'] for r in d['rows'] if not r['excursion']]
        if a and b:
            dif.append(np.mean(a) - np.mean(b))
    if dif:
        put('CalExcRthWithin', num(100 * np.mean(dif), 1)); put('CalExcRthCells', str(len(dif)))
    difd = []
    for (m, c), d in S['calib_diag'].items():
        a = [r['dep'] for r in d['rows'] if r['excursion']]; b = [r['dep'] for r in d['rows'] if not r['excursion']]
        if a and b:
            difd.append(np.mean(a) - np.mean(b))
    if difd:
        put('CalExcDepWithin', num(100 * np.mean(difd), 2, sign=True))
    # the run whose abandoned deliveries exceed the median of the other seeds of its policy and condition the most
    cand = []
    for (m, c), d in S['calib_diag'].items():
        for r in d['rows']:
            oth = [x['rth'] for x in d['rows'] if x['seed'] != r['seed']]
            if oth:
                cand.append((r['rth'] - float(np.median(oth)), r['rth'], float(np.median(oth)), m, c, r['seed'], r['lam_max']))
    if cand:
        b = max(cand)
        put('CalRthMaxRun', num(b[1], 1, pct=True)); put('CalRthMaxRunMed', num(b[2], 1, pct=True))
        put('CalRthMaxRunPol', POLNAME[b[3]]); put('CalRthMaxRunCond', CONDNAME[b[4]]); put('CalRthMaxRunLammax', num(b[6], 1))
    nomlpm = [r['legs_per_min'] for (m, c), d in S['calib_diag'].items() if c == 'nominal' for r in d['rows']]
    put('CalNomMinPerUnit', num(400.0 / np.mean(nomlpm), 0))
    if ov and nov:
        put('CalOvershootRth', num(np.mean([r['rth'] for r in ov]), 1, pct=True)); put('CalNoovershootRth', num(np.mean([r['rth'] for r in nov]), 1, pct=True))
if allrows:
    put('CalAllAtbMax', num(max(r['atb'] for r in allrows), 1, pct=True)); put('CalAllDispMax', num(max(r['disp'] for r in allrows), 2))
    put('CalAllAbsdispMax', num(max(r['absdisp'] for r in allrows), 2)); put('CalAllLamMax', num(max(r['lam_max'] for r in allrows), 2))
    put('CalAllNruns', str(len(allrows)))
    covs = [r['cov'] for r in allrows]; put('CalAllCovMin', num(min(covs), 1, pct=True)); put('CalAllCovMax', num(max(covs), 1, pct=True))
    c100 = [r['cov100'] for r in allrows]; put('CalAllCovhundMin', num(min(c100), 1, pct=True)); put('CalAllCovhundMean', num(np.mean(c100), 1, pct=True))
    lpm = [r['legs_per_min'] for r in allrows]; put('CalAllLegsperminMin', num(min(lpm), 0)); put('CalAllLegsperminMax', num(max(lpm), 0))
CT2 = S['cont']
for m in ('sac', 'crest', 'reactive'):
    rows = [v for (mm, s), v in CT2.items() if mm == m]
    if not rows:
        continue
    nb = len(rows[0]['blocks'])
    for b in range(nb):
        blk = [r['blocks'][b] for r in rows]
        tok = f'Cont{MT[m]}B{"ABCDEFGH"[b]}'
        put(tok + 'Cov', num(np.nanmean([x['cov'] for x in blk]), 1, pct=True)); put(tok + 'Covhund', num(np.nanmean([x['cov100'] for x in blk]), 1, pct=True))
        put(tok + 'Covhundmin', num(np.nanmin([x['cov100'] for x in blk]), 1, pct=True))
        put(tok + 'Dep', num(np.nanmean([x['dep'] for x in blk]), 1, pct=True)); put(tok + 'Lamend', num(np.nanmean([x['lam_end'] for x in blk]), 2))
        put(tok + 'Lamstart', num(np.nanmean([x['lam_start'] for x in blk]), 2))
    put(f'Cont{MT[m]}Cov', num(np.mean([r['cov_all'] for r in rows]), 1, pct=True))
    put(f'Cont{MT[m]}Covmin', num(np.min([r['cov_all'] for r in rows]), 1, pct=True))
    put(f'Cont{MT[m]}Atb', num(np.mean([r['atb'] for r in rows]), 1, pct=True)); put(f'Cont{MT[m]}Disp', num(np.max([r['disp'] for r in rows]), 2))
    put(f'Cont{MT[m]}Nseeds', str(len(rows)))

# ------------------------------------------------------------------ training-reserve check
put('TrNseeds', str(len(S.get('tr_seeds', [])))); put('TrNseedsWord', NWORD(len(S.get('tr_seeds', []))))
for key, r in S['train_reserve'].items():
    g, m, c, what = key
    if what.startswith('rate_'):
        continue
    if m == 'crest-sac':
        tok = f'Trcs{GT[g]}{CT[c]}{what.capitalize()}'
    else:
        tok = f'Tr{MT[m]}{GT[g]}{CT[c]}{"Arm" if what == "arm" else "Armsucc"}'
    put(tok + 'Mean', num(r['mean'], 1, pct=True, sign=True)); put(tok + 'Lo', num(r['lo'], 1, pct=True)); put(tok + 'Hi', num(r['hi'], 1, pct=True))
    put(tok + 'P', pval(r['p']))
for g in ('std', 'online'):
    rr = [r for (gg, m, c, w), r in S['train_reserve'].items() if gg == g and w == 'arm']
    if rr:
        put(f'TrAll{GT[g]}Bound', num(max(max(abs(r['lo']), abs(r['hi'])) for r in rr), 1, pct=True))
        put(f'TrAll{GT[g]}Maxabs', num(max(abs(r['mean']) for r in rr), 1, pct=True))
        put(f'TrAll{GT[g]}Rawsig', str(sum(r['p'] < 0.05 for r in rr))); put(f'TrAll{GT[g]}N', str(len(rr)))
        put(f'TrAll{GT[g]}Rawsigw', CNT(sum(r['p'] < 0.05 for r in rr)))

# ------------------------------------------------------------------ frontier
# dominance of the 5% conformal target by fixed margins, over policies and conditions of the sweep
FIXR = ['fixed1.0', 'fixed1.2', 'fixed1.35', 'fixed1.5', 'fixed1.75', 'fixed2.0', 'fixed2.5']
_np, _dom, _dom175, _domsq, _cdom = 0, 0, 0, 0, 0
for m in ('sac', 'crest', 'reactive'):
    for c in ('nominal', 'wind', 'energy'):
        q = S['frontier'].get((m, c, 'online0.05'))
        F = [(r, S['frontier'].get((m, c, r))) for r in FIXR]
        F = [(r, p) for r, p in F if p]
        if not q or not F:
            continue
        _np += 1
        dm = [r for r, p in F if p['safe_ret'] <= q['safe_ret'] and p['dep'] <= q['dep'] and (p['safe_ret'] < q['safe_ret'] or p['dep'] < q['dep'])]
        _dom += bool(dm); _dom175 += 'fixed1.75' in dm
        _cdom += any(q['safe_ret'] <= p['safe_ret'] and q['dep'] <= p['dep'] and (q['safe_ret'] < p['safe_ret'] or q['dep'] < p['dep']) for r, p in F)
put('FrNpairs', str(_np)); put('FrDomByN', str(_dom)); put('FrDomBySevenfiveN', str(_dom175)); put('FrConfDomN', str(_cdom))
for (m, c, rule), ce in S['frontier'].items():
    for k in ('safe_ret', 'dep', 'succ', 'cov', 'loss'):
        if k not in ce:
            continue
        put(f'Fr{MT[m]}{RULET[rule]}{KT[k]}{CT[c]}', fmt(k, ce[k]))
        cc = ce.get('ci_' + k)
        if cc and cc['n'] >= 2:
            put(f'Fr{MT[m]}{RULET[rule]}{KT[k]}{CT[c]}Lo', fmt(k, cc['lo'])); put(f'Fr{MT[m]}{RULET[rule]}{KT[k]}{CT[c]}Hi', fmt(k, cc['hi']))

# ------------------------------------------------------------------ critic diagnostics
CR = S['critic']
if CR:
    alphas = list(CR[seeds_f[0]]['alphas']) if seeds_f and seeds_f[0] in CR else [0.1, 0.25, 0.5, 0.75, 1.0]
    for algo in ('crest', 'crestret'):
        for cn in ('nominal', 'traffic'):
            sp = [CR[s][f'sens_{algo}_{cn}_spd'].mean(1) for s in CR if f'sens_{algo}_{cn}_spd' in CR[s]]
            dh = [CR[s][f'sens_{algo}_{cn}_dhead'].mean(1) for s in CR if f'sens_{algo}_{cn}_dhead' in CR[s]]
            if sp:
                sp = np.array(sp); dh = np.array(dh)
                put(f'Sens{MT[algo]}{CT[cn]}Spdlow', num(sp[:, 0].mean(), 1)); put(f'Sens{MT[algo]}{CT[cn]}Spdhigh', num(sp[:, -1].mean(), 1))
                put(f'Sens{MT[algo]}{CT[cn]}Spddiff', num((sp[:, 0] - sp[:, -1]).mean(), 2, sign=True))
                put(f'Sens{MT[algo]}{CT[cn]}Dhead', num(dh[:, 0].mean(), 1)); put(f'Sens{MT[algo]}{CT[cn]}Dheadmin', num(dh[:, 0].min(), 1))
                put(f'Sens{MT[algo]}{CT[cn]}Dheadmax', num(dh[:, 0].max(), 1))
                put(f'Sens{MT[algo]}{CT[cn]}Dheadhalf', num(dh[:, 2].mean(), 1))
                # heading change in high-threat states
                hc = [(CR[s][f'sens_{algo}_{cn}_dhead'][0][CR[s][f'sens_{algo}_{cn}_hc'] > 0.3].mean()) for s in CR if f'sens_{algo}_{cn}_hc' in CR[s]]
                put(f'Sens{MT[algo]}{CT[cn]}Dheadthreat', num(np.nanmean(hc), 1))
    for cn in ('nominal', 'traffic'):
        R = [CR[s][f'cal_crest_{cn}'] for s in CR if f'cal_crest_{cn}' in CR[s] and len(CR[s][f'cal_crest_{cn}'])]
        if R:
            R = np.concatenate(R)
            put(f'Crit{CT[cn]}Predmean', num(R[:, 0].mean(), 3)); put(f'Crit{CT[cn]}Realmean', num(R[:, 4].mean(), 3))
            put(f'Crit{CT[cn]}Ratio', num(R[:, 0].mean() / max(R[:, 4].mean(), 1e-9), 1))
            q = np.quantile(R[:, 0], [0.5, 0.9]); lo = R[:, 0] <= q[0]; hi = R[:, 0] >= q[1]
            put(f'Crit{CT[cn]}Reallow', num(R[lo, 4].mean(), 3)); put(f'Crit{CT[cn]}Realhigh', num(R[hi, 4].mean(), 3))
            put(f'Crit{CT[cn]}Evlow', num(R[lo, 5].mean(), 1, pct=True)); put(f'Crit{CT[cn]}Evhigh', num(R[hi, 5].mean(), 1, pct=True))
            from scipy.stats import spearmanr
            rho = spearmanr(R[:, 0], R[:, 4]).correlation; put(f'Crit{CT[cn]}Rho', num(rho, 2))
            # area under the ROC curve of the predicted mean cost for a catastrophe within 10 s
            y = R[:, 5] > 0.5; sc = R[:, 0]
            if y.any() and (~y).any():
                order = np.argsort(sc); ranks = np.empty(len(sc)); ranks[order] = np.arange(1, len(sc) + 1)
                auc = (ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())
                put(f'Crit{CT[cn]}Auc', num(auc, 2))
            put(f'Crit{CT[cn]}Nsteps', '{:,}'.format(len(R)).replace(',', '{,}'))

# ------------------------------------------------------------------ PPO reference and misc
pp = S.get('ppo', {})
PPOT = {'ppo25k': 'Ppotwofivek', 'ppo100k': 'Ppoonezerozerok'}
for tag, d in pp.items():
    for c, v in d.items():
        for k in ('succ', 'loss', 'coll_b', 'coll_a'):
            put(f'Ppo{PPOT[tag]}{KT[k]}{CT[c]}', fmt(k, v[k]))
for k, v in S.get('misc', {}).items():
    put(k, v)
def _json(name):
    """Selection of the reactive rule: read from res3/ (outcome archive) or from the copy kept in analysis/."""
    for d in (os.environ.get('RES3', 'res3'), ANA):
        if os.path.exists(f'{d}/{name}'):
            return json.load(open(f'{d}/{name}'))
    return None


RP = _json('reactive_params.json') or {}
for k, tok in (('U', 'RpU'), ('cb', 'RpCb'), ('kb', 'RpKb'), ('ds', 'RpDs'), ('ka', 'RpKa'), ('tau', 'RpTau')):
    if k in RP:
        put(tok, '%g' % RP[k])
put('DeltaEq', '%g' % (100 * S['delta_eq']))
G = _json('reactive_grid.json')
if G:
    dflt = [r for r in G if r['U'] == 14.0 and r['kb'] == 1.6 and r['cb'] == 45.0 and r['ka'] == 1.6 and r['ds'] == 35.0]
    best = max(G, key=lambda r: r['ret'])
    if dflt:
        put('RpDefaultLoss', num(dflt[0]['loss'], 1, pct=True)); put('RpDefaultSucc', num(dflt[0]['succ'], 1, pct=True))
    put('RpSelectedLoss', num(best['loss'], 1, pct=True)); put('RpSelectedSucc', num(best['succ'], 1, pct=True))
    put('RpGridLossMin', num(min(r['loss'] for r in G), 1, pct=True)); put('RpGridLossMax', num(max(r['loss'] for r in G), 1, pct=True))

with open(f'{MS}/data/macros.tex', 'w') as f:
    f.write('%% generated by mkpaper.py from analysis/summary.pkl; seeds fixed=%s online=%s\n' % (seeds_f, seeds_o))
    for k in sorted(MAC):
        f.write('\\newcommand{\\%s}{%s}\n' % (k, MAC[k]))
print('macros', len(MAC))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mktables3.py')).read())
