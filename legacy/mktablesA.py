"""LaTeX tables built from the aggregated evaluation CSVs."""
import csv
D = 'pa/data'
M6 = [('sac', 'SAC'), ('td3', 'TD3'), ('ppo', 'PPO'), ('dsacn', 'DSAC-N'), ('dsaccvar', 'DSAC-CVaR'), ('crest', 'CREST-C')]
CONDS = [('nominal', 'nominal'), ('wind', 'wind'), ('sensor', 'sensor'), ('traffic', 'traffic'), ('energy', 'energy'), ('compound', 'compound')]
main = {(r['method'], r['guard'], r['cond']): r for r in csv.DictReader(open(f'{D}/main.csv'))}


def tab_main(guard, label, cap):
    L = [r'\begin{table}[tb]', r'\centering', r'\caption{%s}' % cap, r'\label{%s}' % label,
         r'\small', r'\begin{tabular}{llrrrrrrrr}', r'\toprule',
         r'condition & policy & success & collision & aircraft & building & depletion & abandoned & loss & energy \\',
         r' &  & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (\%) & (kJ) \\', r'\midrule']
    for c, cn in CONDS:
        for i, (m, mn) in enumerate(M6):
            r = main.get((m, guard, c))
            if not r:
                continue
            L.append('%s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\' % (
                cn if i == 0 else '', mn, r['succ'], r['coll'], r['coll_a'], r['coll_b'], r['dep'], r['rth'], r['loss'], r['energy']))
        L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def tab_reserve():
    G = [('std', r'fixed $1.2\times$'), ('std15', r'fixed $1.5\times$'), ('static', 'split conformal'), ('online', 'online conformal')]
    L = [r'\begin{table}[tb]', r'\centering',
         r'\caption{Return-to-home reserve rules, reported as CREST-C / SAC. Coverage is the fraction of return legs whose energy stayed within the reserve predicted at the start of the leg; the target of the conformal rules is 95\%. The fixed and split-conformal rules lose coverage as conditions drift, while the online rule holds its target and converts the residual risk into abandoned deliveries.}',
         r'\label{tab:reserve}', r'\small', r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'condition & reserve & coverage (\%) & depletion (\%) & abandoned (\%) & success (\%) & loss (\%) \\', r'\midrule']
    for c, cn in CONDS:
        for i, (g, gn) in enumerate(G):
            r = main.get(('crest', g, c)); r2 = main.get(('sac', g, c))
            if not r:
                continue
            L.append('%s & %s & %s / %s & %s / %s & %s / %s & %s / %s & %s / %s \\\\' % (
                cn if i == 0 else '', gn, r['cov'], r2['cov'], r['dep'], r2['dep'], r['rth'], r2['rth'], r['succ'], r2['succ'], r['loss'], r2['loss']))
        L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def tab_abl():
    ab = {(r['family'], r['variant'], r['cond']): r for r in csv.DictReader(open(f'{D}/abl.csv'))}
    V = [('fixedalpha', 'constant risk level'), ('noenergy', 'no energy cue'), ('prox', 'proximity cue only'), ('unc', 'return-spread driven')]
    L = [r'\begin{table}[tb]', r'\centering',
         r'\caption{Risk-regulation ablations under the online conformal reserve. Catastrophic loss rate (\%), with the full hazard-adaptive schedule as reference.}',
         r'\label{tab:abl}', r'\small', r'\begin{tabular}{ll' + 'r' * len(CONDS) + '}', r'\toprule',
         'agent & risk regulation & ' + ' & '.join(cn for _, cn in CONDS) + r' \\', r'\midrule']
    for fam, an in [('abl', 'CREST-C'), ('ablret', 'CREST-R')]:
        m = 'crest' if fam == 'abl' else 'crestret'
        row = [main.get((m, 'online', c)) for c, _ in CONDS]
        L.append('%s & full schedule & %s \\\\' % (an, ' & '.join(r['loss'] if r else '--' for r in row)))
        for v, vn in V:
            rr = [ab.get((fam, v, c)) for c, _ in CONDS]
            L.append(' & %s & %s \\\\' % (vn, ' & '.join(r['loss'] if r else '--' for r in rr)))
        L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


open(f'pa/tab_main.tex', 'w').write(tab_main('std', 'tab:main',
    r'Mission outcomes with the conventional fixed $1.2\times$ return-to-home reserve. Rates are percentages of missions, pooled over three training seeds and about 2900 missions per policy and condition (5800 under the traffic shift). Loss is the catastrophic loss rate: collision, energy depletion or airspace exit. Energy is the mean consumption per mission.'))
open(f'pa/tab_online.tex', 'w').write(tab_main('online', 'tab:online',
    r'Mission outcomes with the online conformal return-to-home reserve, otherwise as in Table~\ref{tab:main}.'))
open(f'pa/tab_reserve.tex', 'w').write(tab_reserve())
open(f'pa/tab_abl.tex', 'w').write(tab_abl())
print('tables written')

# ---- severity sweep, risk dial and reserve sweep numbers
def tab_sev():
    sev = {(r['method'], r['guard'], r['family'], r['sev']): r for r in csv.DictReader(open(f'{D}/sev.csv'))}
    SS = ['0.25', '0.5', '0.75']
    L = [r'\begin{table}[tb]', r'\centering',
         r'\caption{Catastrophic loss rate (\%) against shift severity for each family, with the fixed $1.2\times$ reserve. Severity 0 is the nominal condition and severity 1 is the fully shifted condition reported in Table~\ref{tab:main}.}',
         r'\label{tab:sev}', r'\small', r'\begin{tabular}{ll' + 'r' * 5 + '}', r'\toprule',
         r'family & policy & $s=0$ & $s=0.25$ & $s=0.5$ & $s=0.75$ & $s=1$ \\', r'\midrule']
    for f in ['wind', 'sensor', 'traffic', 'energy']:
        for i, (m, mn) in enumerate(M6):
            cells = [main.get((m, 'std', 'nominal'))['loss']]
            cells += [sev[(m, 'std', f, s)]['loss'] if (m, 'std', f, s) in sev else '--' for s in SS]
            cells += [main.get((m, 'std', f))['loss']]
            L.append('%s & %s & %s \\\\' % (f if i == 0 else '', mn, ' & '.join(cells)))
        L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def tab_dial():
    dl = {(r['alpha'], r['cond']): r for r in csv.DictReader(open(f'{D}/dial.csv'))}
    A = ['0.1', '0.25', '0.5', '0.75', '1.0']
    CC = [('nominal', 'nominal'), ('sensor', 'sensor'), ('traffic', 'traffic'), ('compound', 'compound')]
    L = [r'\begin{table}[tb]', r'\centering',
         r'\caption{Deployment risk dial of the risk-conditioned CREST-C policy with the fixed reserve: collision rate, catastrophic loss, success and delivery time as the base risk level is varied. A lower base level is more cautious.}',
         r'\label{tab:dial}', r'\small', r'\begin{tabular}{llrrrr}', r'\toprule',
         r'condition & $\alpha_b$ & collision (\%) & loss (\%) & success (\%) & delivery time (s) \\', r'\midrule']
    for c, cn in CC:
        for i, a in enumerate(A):
            r = dl.get((a, c))
            if not r:
                continue
            L.append('%s & %s & %s & %s & %s & %s \\\\' % (cn if i == 0 else '', a, r['coll'], r['loss'], r['succ'], r['tdel']))
        L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


def tab_par():
    par = {(r['pol'], r['rule'], r['cond']): r for r in csv.DictReader(open(f'{D}/pareto.csv'))}
    RU = [('fixed1.0', r'fixed $1.0\times$'), ('fixed1.2', r'fixed $1.2\times$'), ('fixed1.5', r'fixed $1.5\times$'),
          ('fixed2.0', r'fixed $2.0\times$'), ('fixed2.5', r'fixed $2.5\times$'),
          ('online0.02', r'conformal $\delta_l=0.02$'), ('online0.05', r'conformal $\delta_l=0.05$'),
          ('online0.1', r'conformal $\delta_l=0.10$'), ('online0.2', r'conformal $\delta_l=0.20$'), ('online0.3', r'conformal $\delta_l=0.30$')]
    L = [r'\begin{table}[tb]', r'\centering',
         r'\caption{Reserve sweep with the soft actor-critic policy: deliveries abandoned, energy depletion, success and reserve coverage for fixed margins and for online conformal reserves at five miscoverage targets.}',
         r'\label{tab:par}', r'\small', r'\begin{tabular}{llrrrr}', r'\toprule',
         r'condition & reserve rule & abandoned (\%) & depletion (\%) & success (\%) & coverage (\%) \\', r'\midrule']
    for c, cn in [('nominal', 'nominal'), ('wind', 'wind'), ('energy', 'energy')]:
        for i, (ru, rn) in enumerate(RU):
            r = par.get(('sac', ru, c))
            if not r:
                continue
            L.append('%s & %s & %s & %s & %s & %s \\\\' % (cn if i == 0 else '', rn, r['rth'], r['dep'], r['succ'], r['cov']))
        L.append(r'\addlinespace[2pt]')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(L)


open('pa/tab_sev.tex', 'w').write(tab_sev())
open('pa/tab_dial.tex', 'w').write(tab_dial())
open('pa/tab_par.tex', 'w').write(tab_par())
print('extra tables written')
