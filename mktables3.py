# Generated tables of the revised manuscript (executed at the end of mkpaper.py, which defines S, fmt, num, pval, ...)
POLS7 = ['sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest', 'reactive']


def wtab(name, s):
    open(f'{MS}/{name}.tex', 'w').write(s + '\n')


def ci_txt(ce, k):
    c = ce.get('ci_' + k) if ce else None
    if not c or c['n'] < 2:
        return fmt(k, ce[k]) if ce else '--'
    return '%s [%s, %s]' % (fmt(k, ce[k]), fmt(k, c['lo']), fmt(k, c['hi']))


def tab_outcomes(g, label, cap):
    L = [r'\begin{table}[tb]', r'\centering', r'\caption{%s}' % cap, r'\label{%s}' % label, r'\footnotesize', r'\setlength{\tabcolsep}{3.6pt}',
         r'\begin{tabular}{llrrrrrrlr}', r'\toprule',
         r'condition & policy & success & collision & aircraft & building & depletion & abandoned & loss [95\% CI] & energy \\',
         r' &  & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (kJ) \\', r'\midrule']
    for c in CONDS:
        for i, m in enumerate(POLS7):
            ce = S['cells'].get(('fixed', f'main_{m}_{g}_{c}'))
            if not ce:
                continue
            L.append('%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\' % (
                c if i == 0 else '', LAB[m], fmt('succ', ce['succ']), fmt('coll', ce['coll']), fmt('coll_a', ce['coll_a']),
                fmt('coll_b', ce['coll_b']), fmt('dep', ce['dep']), fmt('safe_ret', ce['safe_ret']), ci_txt(ce, 'loss'), fmt('energy', ce['energy'])))
        L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


N_S = len(S['seeds']['fixed'])
cap_main = (r'Mission outcomes with the conventional fixed $1.2\times$ return-to-home reserve. Rates are percentages of missions, '
            r'pooled over \rev{ten training seeds (9600 missions per policy and condition, 19{,}200 under the traffic shift). The interval '
            r'of the catastrophic loss rate is a 95\% $t$ interval over the ten seed-level rates, so it includes training variability; for '
            r'the reactive rule, which involves no training, it reflects only the different scenario sets of the ten seed indices.} '
            r'Loss is the catastrophic loss rate: collision, energy depletion or airspace exit. Energy is the mean consumption per mission.')
wtab('tab_main', tab_outcomes('std', 'tab:main', cap_main))
wtab('tab_online', tab_outcomes('online', 'tab:online', r'Mission outcomes with the online conformal return-to-home reserve, otherwise as in Table~\ref{tab:main}.'))

# ---- paired differences against SAC (fixed reserve)
L = [r'\begin{table}[tb]', r'\centering',
     r'\caption{\rev{Paired differences in catastrophic loss between each tail-risk agent and the soft actor-critic with the fixed reserve, in percentage points '
     r'(positive: the tail-risk agent loses more missions). Mean of the ten seed-level differences with its 95\% $t$ interval and the '
     r'hierarchical bootstrap interval (seeds, then scenarios); equivalence is declared when the 90\% interval lies within $\pm$2 points; '
     r'MDE is the difference detectable with 80\% power; $p_{\mathrm{Holm}}$ is adjusted over the 18 comparisons of the table.}}',
     r'\label{tab:paired}', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llrlllcrr}', r'\toprule',
     r'agent & condition & difference & 95\% $t$ interval & bootstrap interval & 90\% interval & equivalent & MDE & $p_{\mathrm{Holm}}$ \\', r'\midrule']
for m in ('dsaccvar', 'crestret', 'crest'):
    for i, c in enumerate(CONDS):
        r = S['paired'].get(('std', 'sac', m, c, 'loss'))
        if not r:
            continue
        hb = r.get('hb', (np.nan, np.nan))
        L.append('%s & %s & %s & [%s, %s] & [%s, %s] & [%s, %s] & %s & %s & %s \\\\' % (
            LAB[m] if i == 0 else '', c, num(r['mean'], 1, pct=True, sign=True), num(r['lo'], 1, pct=True), num(r['hi'], 1, pct=True),
            num(hb[0], 1, pct=True), num(hb[1], 1, pct=True), num(r['lo90'], 1, pct=True), num(r['hi90'], 1, pct=True),
            'yes' if r['equiv'] else 'no', num(r['mde'], 1, pct=True), pval(r.get('p_holm', np.nan))))
    L.append(r'\addlinespace[2pt]')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_paired', '\n'.join(L))

# ---- risk dial (CREST-C, fixed reserve)
L = [r'\begin{table}[tb]', r'\centering',
     r'\caption{Deployment risk dial of the risk-conditioned CREST-C policy with the fixed reserve: collision rate, catastrophic loss, success, '
     r'delivery time \rev{and exposure statistics} as the base risk level is varied, \rev{pooled over ten seeds (three episodes per district). '
     r'Airspeed is the mean over all flight time; proximity is the mean time per mission spent within 50~m of another aircraft; separation is '
     r'the 5\% quantile of the per-mission minimum clearance to other aircraft.} A lower base level is more cautious.}',
     r'\label{tab:dial}', r'\footnotesize', r'\setlength{\tabcolsep}{4.5pt}', r'\begin{tabular}{llrrrrrrr}', r'\toprule',
     r'condition & $\alpha_b$ & collision & loss & success & delivery & \rev{airspeed} & \rev{proximity} & \rev{separation} \\',
     r' &  & (\%) & (\%) & (\%) & time (s) & \rev{(m s$^{-1}$)} & \rev{(s)} & \rev{(m)} \\', r'\midrule']
for c in ('nominal', 'sensor', 'traffic', 'compound'):
    for i, ab in enumerate((0.1, 0.25, 0.5, 0.75, 1.0)):
        v = S['dial'].get(('crest', c, ab))
        if not v:
            continue
        L.append('%s & %g & %s & %s & %s & %s & %s & %s & %s \\\\' % (c if i == 0 else '', ab, fmt('coll', v['coll']), fmt('loss', v['loss']),
                 fmt('succ', v['succ']), fmt('tdel', v['tdel']), fmt('spd', v['spd']), fmt('prox50', v['prox50']), fmt('minsep_q05', v['minsep_q05'])))
    L.append(r'\addlinespace[2pt]')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_dial', '\n'.join(L))

# ---- ablations (fixed reserve, three episodes per district)
V = [('fixedalpha', r'constant risk level $\alpha_t=\alpha_b$'), ('noenergy', 'no energy cue'), ('prox', 'proximity cue only'), ('unc', 'return-spread driven')]
L = [r'\begin{table}[tb]', r'\centering',
     r'\caption{Risk-regulation ablations \rev{with the fixed reserve}. Catastrophic loss rate (\%), with the full hazard-adaptive schedule as reference, '
     r'\rev{pooled over ten seeds on the same scenarios (three episodes per district). The last column is the mean over the six conditions of the '
     r'paired difference to the full schedule, in percentage points, with the number of conditions in which the variant loses more missions; '
     r'per-condition intervals are shown in Fig.~\ref{fig:abl}.}}',
     r'\label{tab:abl}', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{ll' + 'r' * len(CONDS) + 'r}', r'\toprule',
     'agent & risk regulation & ' + ' & '.join(CONDS) + r' & \rev{mean difference} \\', r'\midrule']
for m in ('crest', 'crestret'):
    row = [cell('fixed', f'main_{m}_std_{c}|k3') for c in CONDS]
    L.append('%s & full schedule & %s & -- \\\\' % (LAB[m], ' & '.join(fmt('loss', r['loss']) if r else '--' for r in row)))
    for v, vn in V:
        rr = [cell('fixed', f'abl_{m}_{v}_{c}') for c in CONDS]
        dd = [S['abl'].get((m, v, c)) for c in CONDS]
        dd = [d for d in dd if d]
        md = '%s (%d of %d)' % (num(np.mean([d['mean'] for d in dd]), 1, pct=True, sign=True), sum(d['mean'] > 0 for d in dd), len(dd)) if dd else '--'
        L.append(' & %s & %s & %s \\\\' % (vn, ' & '.join(fmt('loss', r['loss']) if r else '--' for r in rr), md))
    L.append(r'\addlinespace[2pt]')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_abl', '\n'.join(L))

# ---- composition of aircraft collisions (fixed reserve)
DPOL = ('sac', 'dsacn', 'crest', 'reactive')
_dc = [cell('fixed', f'main_{m}_std_{c}') for c in CONDS for m in DPOL]
# the 'outside view' column is shown only if it carries information beyond the undetected-intruder column
OUTCOL = any(abs(ce['coll_a_outslot'] - ce['coll_ai_undet']) > 5e-4 for ce in _dc if ce)
cap = (r'\caption{\rev{Composition of aircraft collisions with the fixed reserve, as percentages of missions pooled over ten seeds. Each collision is attributed '
       r"to the aircraft with the smallest clearance at the collision step. ``Detected'' means that the intruder was among the detections of the "
       r'observation on which the policy acted at that step')
cap += (r"; ``outside view'' means that the other aircraft was not among the four aircraft the policy observed.}}" if OUTCOL else
        r'. In every cell, the other aircraft was among the four aircraft the policy observed unless it was an undetected intruder.}}')
L = [r'\begin{table}[tb]', r'\centering', cap, r'\label{tab:decomp}', r'\footnotesize', r'\setlength{\tabcolsep}{4.5pt}',
     r'\begin{tabular}{llrrrrr}' if OUTCOL else r'\begin{tabular}{llrrrr}', r'\toprule',
     r'condition & policy & aircraft & with fleet & with detected & with undetected' + (r' & partner outside \\' if OUTCOL else r' \\'),
     r' &  & collisions & aircraft & intruders & intruders' + (r' & view \\' if OUTCOL else r' \\'), r'\midrule']
for c in CONDS:
    for i, m in enumerate(DPOL):
        ce = cell('fixed', f'main_{m}_std_{c}')
        if not ce:
            continue
        row = [c if i == 0 else '', LAB[m], fmt('coll_a', ce['coll_a']), fmt('coll_af', ce['coll_af']), fmt('coll_ai_det', ce['coll_ai_det']),
               fmt('coll_ai_undet', ce['coll_ai_undet'])] + ([fmt('coll_a_outslot', ce['coll_a_outslot'])] if OUTCOL else [])
        L.append(' & '.join(row) + ' \\\\')
    L.append(r'\addlinespace[2pt]')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_decomp', '\n'.join(L))

# ---- reserve rules (SAC policy), with the pre-dispatch check
G = [('std', r'fixed $1.2\times$', 'main'), ('std15', r'fixed $1.5\times$', 'main'), ('static', 'split conformal', 'main'), ('online', 'online conformal', 'main'),
     ('std', r'\rev{dispatch check + fixed $1.2\times$}', 'disp'), ('online', r'\rev{dispatch check + online}', 'disp')]
L = [r'\begin{table}[tb]', r'\centering',
     r'\caption{Return-to-home reserve rules \rev{with the soft actor-critic policy, pooled over ten seeds.} Coverage is the fraction of return legs whose energy '
     r'stayed within the reserve predicted at the start of the leg \rev{under the rule that was active in the run}; the target of the conformal rules is 95\%. '
     r'\rev{Reserve/used is the median ratio of the reserve predicted at the start of a return leg to the energy the leg consumed. The last two rows of each block '
     r'add the pre-dispatch feasibility check; cancelled missions never take off. The corresponding coverage for CREST-C and the reactive rule is shown in Fig.~\ref{fig:cov}.}}',
     r'\label{tab:reserve}', r'\footnotesize', r'\setlength{\tabcolsep}{3.6pt}', r'\begin{tabular}{llrrrrrrr}', r'\toprule',
     r'condition & reserve & coverage & \rev{reserve/used} & depletion & abandoned & \rev{cancelled} & success & loss \\',
     r' &  & (\%) &  & (\%) & (\%) & (\%) & (\%) & (\%) \\', r'\midrule']
for c in CONDS:
    first = True
    for g, gn, kind in G:
        nm = f'main_sac_{g}_{c}' if kind == 'main' else f'disp_sac_{g}_{c}'
        ce = cell('fixed', nm)
        if not ce:
            continue
        L.append('%s & %s & %s & %s & %s & %s & %s & %s & %s \\\\' % (c if first else '', gn, fmt('cov', ce.get('cov', np.nan)), fmt('tight', ce.get('tight', np.nan)),
                 fmt('dep', ce['dep']), fmt('safe_ret', ce['safe_ret']), fmt('cancel', ce['cancel']), fmt('succ', ce['succ']), fmt('loss', ce['loss'])))
        first = False
    L.append(r'\addlinespace[2pt]')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_reserve', '\n'.join(L))

# ---- calibrator diagnostics (online reserve), SAC and CREST-C
L = [r'\begin{table}[tb]', r'\centering',
     r'\caption{\rev{Diagnostics of the online conformal reserve, mean over ten seeds (SAC / CREST-C). Coverage over all return legs of a run and over its '
     r'first 100 legs; depletion in the first episode of each district (the transient after deployment, when the calibrator starts from its '
     r'nominal value) and in the later episodes; cumulative projection displacement $\sum_j D_j$ of Proposition~\ref{prop:coverage}; fraction of legs '
     r'with the calibrator on a bound; calibrator at the end of the run; number of the ten runs with an excursion, in which $\lambda$ more than doubled '
     r'from its starting value or more than a quarter of 80 consecutive legs were uncovered; and simulated minutes needed for 100 return legs with '
     r'the 192 aircraft (32 districts of six) flying at once.}}',
     r'\label{tab:calib}', r'\footnotesize', r'\setlength{\tabcolsep}{2.9pt}', r'\begin{tabular}{lrrrrrrrrr}', r'\toprule',
     r'condition & coverage & first 100 & depletion, & depletion, & $\sum_j D_j$ & on bound & final $\lambda$ & excursions & minutes per \\',
     r' & (\%) & legs (\%) & 1st ep. (\%) & later (\%) &  & (\%) &  & (runs) & 100 legs \\', r'\midrule']
for c in CONDS:
    a = S['calib_diag'].get(('sac', c)); b = S['calib_diag'].get(('crest', c))
    if not a or not b:
        continue
    def pr(k, d=1, pct=True):
        return '%s / %s' % (num(a[k]['mean'], d, pct=pct), num(b[k]['mean'], d, pct=pct))
    nex = '%d / %d' % (sum(r['excursion'] for r in a['rows']), sum(r['excursion'] for r in b['rows']))
    L.append('%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\' % (c, pr('cov'), pr('cov100'), pr('dep_ep0'), pr('dep_ep1p'), pr('disp', 2, False),
             pr('atb'), pr('lam_end', 2, False), nex, '%s / %s' % (num(100 / a['legs_per_min']['mean'], 1), num(100 / b['legs_per_min']['mean'], 1))))
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_calib', '\n'.join(L))

# ---- reserve frontier (SAC)
RU = [('fixed1.0', r'fixed $1.0\times$'), ('fixed1.2', r'fixed $1.2\times$'), ('fixed1.35', r'\rev{fixed $1.35\times$}'), ('fixed1.5', r'fixed $1.5\times$'),
      ('fixed1.75', r'\rev{fixed $1.75\times$}'), ('fixed2.0', r'fixed $2.0\times$'), ('fixed2.5', r'fixed $2.5\times$'),
      ('online0.02', r'conformal $\delta_l=0.02$'), ('online0.05', r'conformal $\delta_l=0.05$'), ('online0.1', r'conformal $\delta_l=0.10$'),
      ('online0.2', r'conformal $\delta_l=0.20$'), ('online0.3', r'conformal $\delta_l=0.30$')]
L = [r'\begin{table}[tb]', r'\centering',
     r'\caption{Reserve sweep with the soft actor-critic policy: deliveries abandoned, energy depletion, success and reserve coverage for fixed margins and '
     r'for online conformal reserves at five miscoverage targets, \rev{pooled over ten seeds on the same scenarios (three episodes per district), with 95\% '
     r'$t$ intervals over seeds for the abandonment and depletion rates. Coverage is that of the rule active in the run. The frontier for CREST-C and the '
     r'reactive rule is shown in Fig.~\ref{fig:pareto}.}}',
     r'\label{tab:par}', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{llllrr}', r'\toprule',
     r'condition & reserve rule & abandoned (\%) & depletion (\%) & success (\%) & coverage (\%) \\', r'\midrule']
for c in ('nominal', 'wind', 'energy'):
    for i, (ru, rn) in enumerate(RU):
        ce = S['frontier'].get(('sac', c, ru))
        if not ce:
            continue
        L.append('%s & %s & %s & %s & %s & %s \\\\' % (c if i == 0 else '', rn, ci_txt(ce, 'safe_ret'), ci_txt(ce, 'dep'), fmt('succ', ce['succ']), fmt('cov', ce.get('cov', np.nan))))
    L.append(r'\addlinespace[2pt]')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_par', '\n'.join(L))

# ---- training-reserve check
L = [r'\begin{table}[tb]', r'\centering',
     r'\caption{\rev{Effect of the reserve active during training. Catastrophic loss rate (\%) with the fixed reserve at test time for agents trained with the '
     r'fixed $1.2\times$ reserve (main comparison) and with the online conformal reserve, mean over the same ' + NWORD(len(S.get('tr_seeds', []))) + r' seeds '
     r'and scenarios for both. The last column is the mean over conditions of the paired difference (online-trained minus fixed-trained) in percentage points.}}',
     r'\label{tab:trainres}', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{ll' + 'r' * len(CONDS) + 'r}', r'\toprule',
     'agent & trained with & ' + ' & '.join(CONDS) + r' & difference \\', r'\midrule']
for m in ('sac', 'dsacn', 'crest'):
    for arm, an in (('fixed', r'fixed $1.2\times$'), ('online', 'online conformal')):
        rr = [S['train_reserve'].get(('std', m, c, 'rate_' + arm)) for c in CONDS]
        if not any(r is not None for r in rr):
            continue
        if arm == 'online':
            dd = [S['train_reserve'].get(('std', m, c, 'arm')) for c in CONDS]
            dd = [d for d in dd if d]
            md = num(np.mean([d['mean'] for d in dd]), 1, pct=True, sign=True) if dd else '--'
        else:
            md = ''
        L.append('%s & %s & %s & %s \\\\' % (LAB[m] if arm == 'fixed' else '', an, ' & '.join(fmt('loss', r) if r is not None else '--' for r in rr), md))
    L.append(r'\addlinespace[2pt]')
L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
wtab('tab_trainres', '\n'.join(L))
print('tables written')
