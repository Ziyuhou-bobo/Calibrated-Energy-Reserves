"""All figures of the revised manuscript (matplotlib, MATLAB-like style; vector PDF and EPS).
usage: python3 figs3.py [name ...]      reads analysis/summary.pkl and analysis/qual.npz
"""
import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, Circle
from figstyle import *
from uavenv import rotor_power, VCAND, PHI

ANA = os.environ.get('ANALYSIS', 'analysis')
S = pickle.load(open(f'{ANA}/summary.pkl', 'rb')) if os.path.exists(f'{ANA}/summary.pkl') else None
MAIN = ['sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest', 'reactive']


def C(arm, name):
    return S['cells'].get((arm, name)) if S else None


def ci_of(ce, key):
    c = ce.get('ci_' + key) if ce else None
    if not c:
        return np.nan, np.nan, np.nan
    return c['mean'], c['lo'], c['hi']


# ----------------------------------------------------------------------------------------- energy model
def fig_power():
    fig, axs = plt.subplots(1, 3, figsize=(18 * CM, 8.2 * CM))
    c4 = [(0.16, 0.52, 0.55), (0.35, 0.44, 0.66), (0.82, 0.60, 0.20), (0.78, 0.16, 0.14)]
    ls4 = ['-', (0, (6, 2)), (0, (4, 1.5, 1, 1.5)), (0, (1.5, 1.2))]
    ax = axs[0]; v = np.linspace(0, 22, 300)
    for k, pl in enumerate([0, 0.6, 1.2, 1.8]):
        ax.plot(v, rotor_power(v, 2 + pl), color=c4[k], ls=ls4[k], lw=1.3, label='payload %.1f kg' % pl)
    ax.set_xlabel(r'airspeed $v$ (m s$^{-1}$)'); ax.set_ylabel(r'electrical power $P(v,m)$ (W)'); ax.set_xlim(0, 22)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6.3, handlelength=2.4, frameon=False); style(ax); panel_title(ax, '(a) power against airspeed')
    ax = axs[1]; v = np.linspace(6, 22, 300); m = 3.0
    wv = [-8, -4, 0, 4, 8]
    cw = [(0.16, 0.52, 0.55), (0.35, 0.44, 0.66), (0.2, 0.2, 0.2), (0.82, 0.60, 0.20), (0.78, 0.16, 0.14)]
    lsw = [(0, (1.5, 1.2)), (0, (4, 1.5, 1, 1.5)), '-', (0, (6, 2)), (0, (8, 2, 2, 2))]
    for k, w in enumerate(wv):
        g = np.maximum(v - w, 0.5); e = rotor_power(v, m) * 1000 / g / 1e3
        ax.plot(v, e, color=cw[k], ls=lsw[k], lw=1.3, label=r'headwind %+d m s$^{-1}$' % w)
        gc = np.maximum(VCAND - w, 0.5); ec = rotor_power(VCAND, m) * 1000 / gc / 1e3; j = np.argmin(ec)
        ax.plot(VCAND[j], ec[j], 'o', ms=4.5, mfc=cw[k], mec='k', mew=0.4)
    ax.set_ylim(5, 45); ax.set_xlim(6, 22)
    ax.set_xlabel(r'airspeed $v$ (m s$^{-1}$)'); ax.set_ylabel('energy per ground km (kJ)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6.0, handlelength=2.4, frameon=False); style(ax); panel_title(ax, '(b) energy per kilometre')
    ax = axs[2]; w = np.linspace(-10, 10, 201); m = 2.0; d = 1000.0
    efix = rotor_power(14.0, m) * PHI * d / np.maximum(14 - w, 1) / 1e3
    eopt, vopt = [], []
    for wi in w:
        gc = np.maximum(VCAND - wi, 1); ec = rotor_power(VCAND, m) * PHI * d / gc; j = np.argmin(ec); eopt.append(ec[j] / 1e3); vopt.append(VCAND[j])
    h1, = ax.plot(w, eopt, '-', color=POL['crest']['col'], lw=1.5, label='energy-optimal airspeed')
    h2, = ax.plot(w, efix, '--', color=(0.35, 0.35, 0.35), lw=1.3, label=r'fixed 14 m s$^{-1}$ cruise')
    ax.set_xlim(-10, 10); ax.set_ylim(0, 35); ax.set_ylabel('return energy for 1 km (kJ)')
    ax.set_xlabel(r'headwind along return track (m s$^{-1}$)'); style(ax)
    ax2 = ax.twinx(); h3, = ax2.step(w, vopt, '-', where='mid', color=POL['dsaccvar']['col'], lw=0.9, label='selected airspeed (right axis)')
    ax2.set_ylim(6, 22); ax2.set_yticks(range(6, 23, 4)); ax2.set_ylabel(r'selected airspeed (m s$^{-1}$)', color=POL['dsaccvar']['col'])
    ax2.tick_params(axis='y', direction='in', colors=POL['dsaccvar']['col']); ax2.grid(False)
    ax.tick_params(right=False)
    ax.legend([h1, h2, h3], [h1.get_label(), h2.get_label(), h3.get_label()], loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=1, fontsize=6.0, handlelength=2.2, frameon=False)
    panel_title(ax, '(c) nominal return energy')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_power')


# ----------------------------------------------------------------------------------------- schematic
def _box(ax, x, y, w, h, lines, fc, ec, fs=7.0, bold_first=True, align='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.18', fc=fc, ec=ec, lw=0.9))
    n = len(lines)
    if align == 'center':
        for i, t in enumerate(lines):
            yy = y + h / 2 + (n / 2 - i - 0.5) * 0.36
            ax.text(x + w / 2, yy, t, ha='center', va='center', fontsize=fs, fontweight='bold' if (i == 0 and bold_first) else 'normal', color='#1a1a1a')
    else:
        for i, t in enumerate(lines):
            ax.text(x + 0.18, y + h - 0.32 - i * 0.36, t, ha='left', va='center', fontsize=fs,
                    fontweight='bold' if (i == 0 and bold_first) else 'normal', color='#1a1a1a')


def _arrow(ax, pts, col, lw=1.0):
    pts = np.asarray(pts, float)
    ax.plot(pts[:-1, 0], pts[:-1, 1], '-', color=col, lw=lw, solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle='-|>', color=col, lw=lw, mutation_scale=8, shrinkA=0, shrinkB=0))


def fig_schematic():
    fig = plt.figure(figsize=(18 * CM, 8.8 * CM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 18); ax.set_ylim(0, 8.8); ax.axis('off')
    cEnv, eEnv = (0.93, 0.95, 0.97), (0.35, 0.44, 0.66)
    cRisk, eRisk = (0.99, 0.93, 0.90), POL['crest']['col']
    cLearn, eLearn = (0.92, 0.96, 0.95), POL['dsaccvar']['col']
    cGuard, eGuard = (0.97, 0.95, 0.90), (0.66, 0.52, 0.36)
    _box(ax, 0.2, 1.25, 4.25, 6.25, ['Stochastic urban district', '- mean wind, OU turbulence, gusts', '- GNSS drift and multipath',
                                     '- fleet traffic, non-cooperative', '   intruders (missed at random)', '- battery, payload, in-flight faults',
                                     '- test shifts, one family at a time:', '   wind, sensor, traffic, energy', '- paired test scenarios,',
                                     '   10 training seeds per agent'], cEnv, eEnv, fs=6.8, align='left')
    _box(ax, 5.25, 6.05, 3.55, 1.3, ['Hazard cues', r'$h_t=(h_c,h_e,h_d,h_w)$'], cRisk, eRisk)
    _box(ax, 9.6, 6.05, 4.55, 1.3, ['Hazard-adaptive risk level (CREST)', r'$\alpha_t=\alpha_{min}+(\alpha_b-\alpha_{min})(1-H(h_t))$'], cRisk, eRisk)
    _box(ax, 9.6, 3.35, 4.55, 1.85, ['Policy under test', r'SAC, TD3, DSAC-N, DSAC-CVaR,', 'CREST-R, CREST-C', 'reference: reactive rule (no learning)'], cLearn, eLearn)
    _box(ax, 15.0, 3.35, 2.8, 4.0, ['Quantile critics', r'return $Z(o,a;\alpha)$', 'CREST-C adds', 'catastrophe cost', r'$Z^c(o,a;\alpha)$'], cLearn, eLearn)
    _box(ax, 5.25, 0.35, 5.45, 2.1, ['Energy reserve under test', r'fixed 1.0$\times$ to 2.5$\times$, split conformal,', r'online conformal ($\kappa_t$, $q_t$, fleet $\lambda_j$),',
                                     'optional pre-dispatch check'], cGuard, eGuard)
    _box(ax, 11.5, 0.35, 6.3, 2.1, ['Return-to-home trigger', r'$E_t<B_t$ (+ drop allowance while loaded)', 'goal switched to nearest pad;',
                                    'return leg flown by the same policy'], cGuard, eGuard)
    _arrow(ax, [(4.45, 6.7), (5.25, 6.7)], eRisk); _arrow(ax, [(8.8, 6.7), (9.6, 6.7)], eRisk)
    _arrow(ax, [(11.875, 6.05), (11.875, 5.2)], eRisk); ax.text(12.0, 5.6, r'$\alpha_t$', fontsize=7.5, color=eRisk)
    _arrow(ax, [(4.45, 4.55), (9.6, 4.55)], eLearn); ax.text(6.0, 4.72, r'observation $o_t$ (59-d)', fontsize=6.8, color=eLearn)
    _arrow(ax, [(15.0, 4.25), (14.15, 4.25)], eLearn); ax.text(14.58, 4.42, 'tail', ha='center', fontsize=6.4, color=eLearn)
    _arrow(ax, [(11.875, 3.35), (11.875, 2.95), (4.45, 2.95)], eLearn); ax.text(6.0, 3.1, r'action $a_t$ (goal-frame velocity)', fontsize=6.8, color=eLearn)
    _arrow(ax, [(4.45, 1.4), (5.25, 1.4)], eGuard); ax.text(4.5, 1.1, 'telemetry', fontsize=6.2, color=eGuard)
    _arrow(ax, [(10.7, 1.4), (11.5, 1.4)], eGuard)
    _arrow(ax, [(14.65, 0.35), (14.65, 0.12), (2.3, 0.12), (2.3, 1.25)], eGuard)
    _arrow(ax, [(2.3, 7.5), (2.3, 8.25), (16.4, 8.25), (16.4, 7.35)], (0.45, 0.45, 0.45), 0.8)
    ax.text(9.35, 8.42, r'training: every agent with the fixed 1.2$\times$ reserve (second arm: online conformal);'
            ' hazard cues replayed, risk level relabelled', fontsize=6.3, color=(0.3, 0.3, 0.3), ha='center')
    save(fig, 'fig_schematic')


# ----------------------------------------------------------------------------------------- main loss bars
def _groupbars(ax, conds, pols, getter, width=0.84, ci=True, hatch_reactive=True):
    n = len(pols); w = width / n; hs = []
    for i, m in enumerate(pols):
        x = np.arange(len(conds)) + (i - (n - 1) / 2) * w
        y, lo, hi = zip(*[getter(m, c) for c in conds])
        y = np.array(y, float); lo = np.array(lo, float); hi = np.array(hi, float)
        h = ax.bar(x, y, w * 0.9, color=light(POL[m]['col'], 0.0) if m != 'reactive' else (0.85, 0.85, 0.85),
                   edgecolor='#333333', lw=0.4, hatch='xxxx' if (m == 'reactive' and hatch_reactive) else None, label=POL[m]['lab'])
        if ci:
            ok = np.isfinite(lo) & np.isfinite(hi)
            if ok.any():
                ax.errorbar(x[ok], y[ok], yerr=np.vstack([y[ok] - lo[ok], hi[ok] - y[ok]]), fmt='none', ecolor='#1a1a1a', elinewidth=0.6, capsize=1.4)
        hs.append(h)
    ax.set_xticks(np.arange(len(conds))); ax.set_xticklabels([CONDLAB[c] for c in conds])
    ax.set_xlim(-0.55, len(conds) - 0.45)
    return hs


def fig_main():
    fig, axs = plt.subplots(2, 1, figsize=(18 * CM, 11.5 * CM), sharex=True)
    for ax, g, t in zip(axs, ('std', 'online'), (r'(a) fixed 1.2$\times$ return-to-home reserve', '(b) online conformal reserve')):
        def get(m, c):
            ce = C('fixed', f'main_{m}_{g}_{c}'); v, lo, hi = ci_of(ce, 'loss')
            return 100 * v, 100 * lo, 100 * hi
        hs = _groupbars(ax, COND, MAIN, get)
        ax.set_ylabel('catastrophic loss rate (%)'); style(ax, grid_x=False); panel_title(ax, t)
        ax.set_ylim(0, max(5, np.nanmax([get(m, c)[2] for m in MAIN for c in COND]) * 1.12))
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=1.0)
    legend_below(fig, hs, [POL[m]['lab'] for m in MAIN], ncol=7, y=0.0)
    save(fig, 'fig_main')


# ----------------------------------------------------------------------------------------- learning curves
def _curves(paths):
    import pickle as pk
    out = []
    for p in paths:
        if os.path.exists(p):
            L = pk.load(open(p, 'rb'))['log']
            out.append({r['step']: r for r in L})
    return out


def fig_learn():
    fig, axs = plt.subplots(1, 4, figsize=(18 * CM, 6.6 * CM))
    keys = [('succ', 'training success rate'), ('coll', 'training collision rate'), ('ret', 'mean mission return')]
    hs = []
    for m in ('sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest'):
        cur = _curves([f'runs/{m}_s{s}.pkl' for s in range(10)])
        if not cur:
            continue
        grid = np.arange(250, 25001, 250)
        for ax, (k, yl) in zip(axs[:3], keys):
            Y = np.array([[c.get(int(g), {}).get(k, np.nan) for g in grid] for c in cur], float)
            mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
            ax.fill_between(grid / 1000, mu - sd, mu + sd, color=light(POL[m]['col'], 0.82), lw=0)
            h, = ax.plot(grid / 1000, mu, color=POL[m]['col'], ls=POL[m]['ls'], lw=1.1, label=POL[m]['lab'])
        hs.append(h)
    cur = _curves([f'runs_legacy/ppo_s{s}.pkl' for s in range(3)])
    if cur:
        grid = np.arange(250, 25001, 250)
        for ax, (k, yl) in zip(axs[:3], keys):
            Y = np.array([[c.get(int(g), {}).get(k, np.nan) for g in grid] for c in cur], float)
            h, = ax.plot(grid / 1000, np.nanmean(Y, 0), color=POL['ppo']['col'], ls=POL['ppo']['ls'], lw=1.0, label='PPO (3 seeds)')
        hs.append(h)
    for ax, (k, yl), t in zip(axs[:3], keys, ('(a) delivery success', '(b) collisions', '(c) mission return')):
        ax.set_xlabel(r'environment steps ($10^3$)'); ax.set_ylabel(yl); ax.set_xlim(0, 25); style(ax); panel_title(ax, t)
    ax = axs[3]
    cur = _curves([f'runs_legacy/ppo100k_s{s}.pkl' for s in range(2)])
    if cur:
        grid = np.arange(250, 100001, 250)
        win = 8                         # moving average over 2000 environment steps, curves labelled directly
        for k, lsx, lab, dy in (('succ', '-', 'PPO success', 0.09), ('coll', (0, (4, 1.5)), 'PPO collisions', -0.13)):
            Y = np.nanmean(np.array([[c.get(int(g), {}).get(k, np.nan) for g in grid] for c in cur], float), 0)
            Ys = np.convolve(Y, np.ones(win) / win, mode='valid'); gx = grid[win - 1:] / 1000
            ax.plot(gx, Ys, color=POL['ppo']['col'], ls=lsx, lw=1.1)
            ax.text(gx[-1] - 2, Ys[-1] + dy, lab, ha='right', va='center', fontsize=6.2, color=POL['ppo']['col'])
        sac = _curves([f'runs/sac_s{s}.pkl' for s in range(10)])
        if sac:
            g2 = np.arange(20250, 25001, 250)
            for k, lsx, lab, dy in (('succ', '-', 'SAC, end of training', -0.05), ('coll', (0, (4, 1.5)), 'SAC, end of training', 0.05)):
                v = np.nanmean([[c.get(int(g), {}).get(k, np.nan) for g in g2] for c in sac])
                ax.axhline(v, color=POL['sac']['col'], ls=lsx, lw=0.8)
                ax.text(3, v + dy, lab, ha='left', va='center', fontsize=6.0, color=POL['sac']['col'])
    ax.set_xlabel(r'environment steps ($10^3$)'); ax.set_ylabel('rate'); ax.set_xlim(0, 100); ax.set_ylim(0, 1)
    style(ax); panel_title(ax, '(d) PPO at 4$\\times$ budget')
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
    legend_below(fig, hs, [h.get_label() for h in hs], ncol=7, y=0.0)
    save(fig, 'fig_learn')


# ----------------------------------------------------------------------------------------- paired differences (forest plot)
def fig_forest():
    PD = S['paired']; d = 100 * S['delta_eq']
    fig, axs = plt.subplots(1, 3, figsize=(18 * CM, 10.0 * CM), sharey=True)
    combos = [('std', 'sac', '(a) vs SAC, fixed reserve'), ('std', 'dsacn', '(b) vs DSAC-N, fixed reserve'),
              ('online', 'sac', '(c) vs SAC, online reserve')]
    tail = ['dsaccvar', 'crestret', 'crest']
    hs = {}
    for ax, (g, ref, t) in zip(axs, combos):
        neutral = [m for m in ('sac', 'td3', 'dsacn') if m != ref]
        agents = neutral + tail; n = len(agents)
        ax.axvspan(-d, d, color=(0.93, 0.93, 0.93), lw=0, zorder=0)
        ax.axvline(0, color='#555555', lw=0.7)
        lim = 4.0
        for ci, c in enumerate(COND):
            for ai, m in enumerate(agents):
                r = PD.get((g, ref, m, c, 'loss'))
                if not r:
                    continue
                y = ci + (ai - (n - 1) / 2) * 0.15
                col = POL[m]['col'] if m in tail else (0.55, 0.55, 0.55)
                mfc = (col if r['equiv'] else 'white')
                h = ax.errorbar(100 * r['mean'], y, xerr=[[100 * (r['mean'] - r['lo'])], [100 * (r['hi'] - r['mean'])]], fmt=POL[m]['mk'],
                                color=col, mfc=mfc, mec=col, ms=3.8, elinewidth=0.9, capsize=1.4)
                hs[m] = h
                lim = max(lim, 1.1 * abs(100 * r['lo']), 1.1 * abs(100 * r['hi']))
        ax.set_yticks(range(len(COND))); ax.set_yticklabels([CONDLAB[c] for c in COND]); ax.invert_yaxis()
        ax.set_xlabel('difference in catastrophic loss (points)'); style(ax); panel_title(ax, t)
        ax.set_xlim(-lim, lim)
    fig.tight_layout(rect=(0, 0.15, 1, 1), w_pad=0.8)
    order = [m for m in ('sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest') if m in hs]
    hh = [hs[m] for m in order] + [Patch(fc=(0.93, 0.93, 0.93), ec='none')]
    labs = [POL[m]['lab'] + ('' if m in tail else ' (risk-neutral)') for m in order] + [r'equivalence margin $\pm$%g points' % d]
    legend_below(fig, hh, labs, ncol=4, y=0.035)
    fig.text(0.5, 0.008, 'filled markers: 90% interval inside the margin (equivalent); open markers: not shown equivalent', ha='center', fontsize=6.6, color='#333333')
    save(fig, 'fig_forest')


# ----------------------------------------------------------------------------------------- risk dial and exposure
def fig_dial():
    al = [0.1, 0.25, 0.5, 0.75, 1.0]
    cc = {'nominal': ((0.35, 0.44, 0.66), 'o', '-'), 'sensor': ((0.16, 0.52, 0.55), 's', (0, (6, 2))),
          'traffic': ((0.82, 0.60, 0.20), '^', (0, (4, 1.5, 1, 1.5))), 'compound': ((0.78, 0.16, 0.14), 'D', (0, (1.5, 1.2)))}
    panels = [('coll', 'collision rate (%)', 100, '(a) collisions'), ('loss', 'catastrophic loss rate (%)', 100, '(b) catastrophic loss'),
              ('spd', r'mean airspeed (m s$^{-1}$)', 1, '(c) airspeed'), ('prox50', 'time within 50 m (s per mission)', 1, '(d) exposure to other aircraft'),
              ('minsep_q05', r'5% quantile of min. separation (m)', 1, '(e) separation'), ('tdel', 'delivery time (s)', 1, '(f) delivery time')]
    fig, axs = plt.subplots(2, 3, figsize=(18 * CM, 11.6 * CM)); hs = []
    for ax, (k, yl, sc, t) in zip(axs.ravel(), panels):
        for c, (col, mk, ls) in cc.items():
            y, lo, hi = [], [], []
            for a in al:
                ce = S['dial'].get(('crest', c, a))
                v, l, h = ci_of(ce, k) if ce else (np.nan,) * 3
                y.append(sc * v); lo.append(sc * l); hi.append(sc * h)
            y, lo, hi = map(np.array, (y, lo, hi))
            h = ax.errorbar(al, y, yerr=np.vstack([np.nan_to_num(y - lo), np.nan_to_num(hi - y)]), color=col, marker=mk, ls=ls, ms=3.8, lw=1.1, elinewidth=0.7, capsize=1.5, label=c)
            if k == 'coll':
                hs.append(h)
        ax.set_xlabel(r'base risk level $\alpha_b$'); ax.set_ylabel(yl); ax.set_xlim(0.03, 1.07); style(ax); panel_title(ax, t)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=0.8)
    legend_below(fig, hs, [CONDLAB[c] + (' shift' if c in ('sensor', 'traffic') else '') for c in cc], ncol=4, y=0.0)
    save(fig, 'fig_dial')


# ----------------------------------------------------------------------------------------- critic and actor diagnostics
def fig_critic():
    CR = S['critic']
    if not CR:
        return
    fig, axs = plt.subplots(1, 3, figsize=(18 * CM, 7.8 * CM))
    ax = axs[0]
    for cn, col, mk in (('nominal', (0.35, 0.44, 0.66), 'o'), ('traffic', (0.82, 0.60, 0.20), '^')):
        R = np.concatenate([CR[s][f'cal_crest_{cn}'] for s in CR if f'cal_crest_{cn}' in CR[s] and len(CR[s][f'cal_crest_{cn}'])])
        q = np.quantile(R[:, 0], np.linspace(0, 1, 11)); xs, ys = [], []
        for lo, hi in zip(q[:-1], q[1:]):
            mm = (R[:, 0] >= lo) & (R[:, 0] <= hi); xs.append(R[mm, 0].mean()); ys.append(R[mm, 4].mean())
        ax.plot(xs, ys, marker=mk, color=col, ls='-', ms=3.8, lw=1.1, label=cn if cn == 'nominal' else 'traffic shift')
    lim = ax.get_xlim()[1]; ax.plot([0, lim], [0, lim], color='#555555', lw=0.7, ls='--', label='perfect calibration')
    ax.set_xlabel('predicted discounted catastrophe cost'); ax.set_ylabel('realised discounted cost')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.2, frameon=False)
    style(ax); panel_title(ax, '(a) cost critic of CREST-C (deciles)')
    al = list(CR[next(iter(CR))]['alphas'])
    for ax, key, yl, t in ((axs[1], 'spd', r'commanded speed (m s$^{-1}$)', '(b) speed command'), (axs[2], 'dhead', r'heading change vs $\alpha=1$ (deg)', '(c) heading command')):
        for algo in ('crest', 'crestret'):
            for cn, ls in (('nominal', '-'), ('traffic', '--')):
                arr = [CR[s][f'sens_{algo}_{cn}_{key}'].mean(1) for s in CR if f'sens_{algo}_{cn}_{key}' in CR[s]]
                if not arr:
                    continue
                A = np.array(arr); mu = A.mean(0); sd = A.std(0, ddof=1) if len(A) > 1 else np.zeros_like(mu)
                ax.errorbar(al, mu, yerr=sd, color=POL[algo]['col'], marker=POL[algo]['mk'], ls=ls, ms=3.6, lw=1.1, elinewidth=0.6, capsize=1.4,
                            label='%s, %s' % (POL[algo]['lab'], 'nominal' if cn == 'nominal' else 'traffic shift'))
        ax.set_xlabel(r'risk input $\alpha$'); ax.set_ylabel(yl); style(ax); panel_title(ax, t)
        if key == 'spd':
            ax.set_ylim(0, 21)
    axs[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=6.0, handlelength=2.4, frameon=False)
    fig.tight_layout(w_pad=0.9)
    save(fig, 'fig_critic')


# ----------------------------------------------------------------------------------------- example scenario and risk trace
def fig_map():
    Q = np.load(f'{ANA}/qual.npz') if os.path.exists(f'{ANA}/qual.npz') else None
    if Q is None:
        return
    fig, axs = plt.subplots(1, 3, figsize=(18 * CM, 6.9 * CM))
    uc = [(0.35, 0.44, 0.66), (0.16, 0.52, 0.55), (0.82, 0.60, 0.20), (0.45, 0.30, 0.62), (0.20, 0.50, 0.25), (0.78, 0.16, 0.14)]
    th = np.linspace(0, 2 * np.pi, 40)
    for ax, (m, t) in zip(axs, (('sac', '(a) SAC'), ('crest', '(b) CREST-C'), ('reactive', '(c) reactive rule'))):
        for o in Q['obs']:
            ax.fill(o[0] + o[2] * np.cos(th), o[1] + o[2] * np.sin(th), color=(0.84, 0.84, 0.86), ec=(0.55, 0.55, 0.58), lw=0.3)
        I = Q[f'{m}_ipos']
        for j in range(I.shape[1]):
            ax.plot(I[:, j, 0], I[:, j, 1], ':', color=(0.5, 0.5, 0.5), lw=0.7)
        X = Q[f'{m}_pos']; A = Q[f'{m}_act']; OU = Q[f'{m}_out']
        for j in range(X.shape[1]):
            k = np.where(A[:, j])[0]; k = np.r_[k, min(k[-1] + 1, len(X) - 1)] if len(k) else k
            ax.plot(X[k, j, 0], X[k, j, 1], '-', color=uc[j % 6], lw=1.0)
            ax.plot(X[0, j, 0], X[0, j, 1], 'o', mfc='white', mec=uc[j % 6], ms=3)
            if OU[j] in (3, 4, 5, 6):
                ax.plot(X[k[-1], j, 0], X[k[-1], j, 1], 'x', color=(0.85, 0.1, 0.1), ms=7, mew=1.5)
        ax.plot(Q['pads'][:, 0], Q['pads'][:, 1], 's', mfc=(0.35, 0.35, 0.35), mec='k', ms=5)
        ax.plot(Q['cust'][:, 0], Q['cust'][:, 1], 'D', mfc=(0.95, 0.85, 0.35), mec='k', ms=4.2)
        ax.set_aspect('equal'); ax.set_xlim(0, 1600); ax.set_ylim(0, 1600); ax.set_xticks(range(0, 1601, 400)); ax.set_yticks(range(0, 1601, 400))
        ax.set_xlabel('east (m)'); style(ax); panel_title(ax, t)
        if m == 'sac':
            ax.set_ylabel('north (m)')
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=0.6)
    hh = [Line2D([], [], color='k', lw=1.0), Line2D([], [], color=(0.5, 0.5, 0.5), ls=':', lw=0.8), Line2D([], [], ls='none', marker='s', mfc=(0.35, 0.35, 0.35), mec='k'),
          Line2D([], [], ls='none', marker='D', mfc=(0.95, 0.85, 0.35), mec='k'), Line2D([], [], ls='none', marker='x', color=(0.85, 0.1, 0.1), mew=1.5, ms=7),
          Patch(fc=(0.84, 0.84, 0.86), ec=(0.55, 0.55, 0.58))]
    legend_below(fig, hh, ['fleet trajectory', 'intruder track', 'pad', 'customer', 'catastrophic loss', 'building'], ncol=6, y=0.0)
    save(fig, 'fig_map')


def fig_alpha():
    Q = np.load(f'{ANA}/qual.npz') if os.path.exists(f'{ANA}/qual.npz') else None
    if Q is None:
        return
    A = Q['trace']; t = np.arange(1, len(A) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(12 * CM, 8.6 * CM), sharex=True)
    cc = [((0.78, 0.16, 0.14), '-', r'collision threat $h_c$'), ((0.82, 0.60, 0.20), (0, (6, 2)), r'energy pressure $h_e$'),
          ((0.16, 0.52, 0.55), (0, (4, 1.5, 1, 1.5)), r'traffic density $h_d$'), ((0.35, 0.44, 0.66), (0, (1.5, 1.2)), r'disturbance $h_w$')]
    for i, (col, ls, lab) in enumerate(cc):
        axs[0].plot(t, A[:, i], color=col, ls=ls, lw=1.1, label=lab)
    axs[0].set_ylabel('hazard cue'); axs[0].set_ylim(-0.03, 1.03); style(axs[0]); panel_title(axs[0], '(a) hazard cues along one delivery')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=6.6, handlelength=2.2)
    axs[1].plot(t, A[:, 4], color='#1a1a1a', lw=1.2, label=r'scheduled risk level $\alpha_t$')
    axs[1].axhline(0.1, color='#777777', ls='--', lw=0.7, label=r'floor $\alpha_{min}$')
    axs[1].set_ylabel(r'risk level $\alpha_t$'); axs[1].set_xlabel('mission time (s)'); axs[1].set_ylim(0, 1.05); style(axs[1]); panel_title(axs[1], '(b) scheduled risk level')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6.6, frameon=False)
    fig.tight_layout(h_pad=2.2)
    save(fig, 'fig_alphaA')


# ----------------------------------------------------------------------------------------- ablations (paired differences)
def fig_abl():
    V = [('fixedalpha', 'constant risk level', (0.45, 0.45, 0.45), 'o'), ('noenergy', 'no energy cue', (0.82, 0.60, 0.20), 's'),
         ('prox', 'proximity cue only', (0.16, 0.52, 0.55), '^'), ('unc', 'return-spread driven', (0.45, 0.30, 0.62), 'D')]
    fig, axs = plt.subplots(1, 2, figsize=(18 * CM, 7.6 * CM), sharey=True); hs = []
    for ax, m, t in ((axs[0], 'crest', '(a) CREST-C: tail risk on the catastrophe cost'), (axs[1], 'crestret', '(b) CREST-R: tail risk on the return')):
        ax.axhline(0, color='#555555', lw=0.7)
        for vi, (v, lab, col, mk) in enumerate(V):
            x = np.arange(len(COND)) + (vi - 1.5) * 0.18
            r = [S['abl'].get((m, v, c)) for c in COND]
            y = np.array([100 * q['mean'] if q else np.nan for q in r]); lo = np.array([100 * q['lo'] if q else np.nan for q in r]); hi = np.array([100 * q['hi'] if q else np.nan for q in r])
            h = ax.errorbar(x, y, yerr=np.vstack([np.nan_to_num(y - lo), np.nan_to_num(hi - y)]), fmt=mk, color=col, ms=4, elinewidth=0.9, capsize=1.6, label=lab)
            if m == 'crest':
                hs.append(h)
        ax.set_xticks(range(len(COND))); ax.set_xticklabels([CONDLAB[c] for c in COND]); ax.set_xlim(-0.5, len(COND) - 0.5)
        ax.set_ylabel('loss difference to full schedule (points)') if m == 'crest' else None
        style(ax, grid_x=False); panel_title(ax, t)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.8)
    legend_below(fig, hs, [h.get_label() for h in hs], ncol=4, y=0.0)
    save(fig, 'fig_abl')


# ----------------------------------------------------------------------------------------- severity sweeps
def fig_sev():
    pols = ['sac', 'dsacn', 'dsaccvar', 'crestret', 'crest', 'reactive']
    sv = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axs = plt.subplots(2, 2, figsize=(18 * CM, 11.0 * CM)); hs = []
    for ax, f in zip(axs.ravel(), ('wind', 'sensor', 'traffic', 'energy')):
        for m in pols:
            y, lo, hi = [], [], []
            for s_ in sv:
                nm = f'main_{m}_std_nominal' if s_ == 0 else (f'main_{m}_std_{f}' if s_ == 1 else f'sev_{m}_std_{f}_{s_}')
                v, l, h = ci_of(C('fixed', nm), 'loss'); y.append(100 * v); lo.append(100 * l); hi.append(100 * h)
            y, lo, hi = map(np.array, (y, lo, hi))
            hh = ax.errorbar(np.array(sv) + (pols.index(m) - 2.5) * 0.008, y, yerr=np.vstack([np.nan_to_num(y - lo), np.nan_to_num(hi - y)]), color=POL[m]['col'],
                             marker=POL[m]['mk'], ls=POL[m]['ls'], ms=3.6, lw=1.1, elinewidth=0.6, capsize=1.3, label=POL[m]['lab'])
            if f == 'wind':
                hs.append(hh)
        ax.set_xlabel('shift severity $s$'); ax.set_ylabel('catastrophic loss rate (%)'); ax.set_xlim(-0.04, 1.04); style(ax); panel_title(ax, f'({"abcd"[["wind", "sensor", "traffic", "energy"].index(f)]}) {f} shift')
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.0, w_pad=1.0)
    legend_below(fig, hs, [h.get_label() for h in hs], ncol=6, y=0.0)
    save(fig, 'fig_sev')


# ----------------------------------------------------------------------------------------- anatomy of residual risk
def fig_extra():
    fig = plt.figure(figsize=(18 * CM, 15.0 * CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], height_ratios=[1, 1.05], hspace=0.78, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0]); hs = []
    for m in ('sac', 'dsacn', 'crest', 'reactive'):
        v = S['minair'].get(m)
        if v is None:
            continue
        q = np.linspace(0, 1, len(v)); k = np.linspace(0, len(v) - 1, min(600, len(v))).astype(int)
        h, = ax.plot(v[k], 100 * q[k], color=POL[m]['col'], ls=POL[m]['ls'], lw=1.2, label=POL[m]['lab']); hs.append(h)
    ax.axvline(0, color='#333333', ls='--', lw=0.7)
    ax.set_xlim(-5, 60); ax.set_ylim(0, 40); ax.set_xlabel('minimum air clearance (m)'); ax.set_ylabel('missions below (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27), ncol=2, fontsize=6.4, frameon=False, handlelength=2.4)
    style(ax); panel_title(ax, '(a) separation, traffic shift')
    # (b) partner of aircraft collisions, horizontal stacked bars
    ax = fig.add_subplot(gs[0, 1])
    comp = [('coll_af', 'with fleet aircraft', (0.35, 0.44, 0.66), ''), ('coll_ai_det', 'with detected intruders', (0.82, 0.60, 0.20), '////'),
            ('coll_ai_undet', 'with undetected intruders', (0.78, 0.16, 0.14), 'xxxx')]
    rows = [(c, m) for c in COND for m in ('sac', 'crest')]
    y = np.arange(len(rows))[::-1].astype(float)
    for i_, (c, m) in enumerate(rows):
        y[i_] -= 0.35 * (COND.index(c))
    left = np.zeros(len(rows))
    for k, lab, col, hat in comp:
        w = np.array([100 * (C('fixed', f'main_{m}_std_{c}') or {}).get(k, np.nan) for c, m in rows])
        ax.barh(y, np.nan_to_num(w), 0.8, left=left, color=col, edgecolor='#333333', lw=0.35, hatch=hat or None, label=lab)
        left += np.nan_to_num(w)
    ax.set_yticks(y); ax.set_yticklabels(['%s, %s' % (c, POL[m]['lab']) for c, m in rows], fontsize=6.5)
    ax.set_xlabel('aircraft collisions (% of missions)'); ax.grid(True, axis='x'); ax.grid(False, axis='y'); style(ax); ax.grid(False, axis='y')
    ax.set_ylim(y.min() - 0.7, y.max() + 0.7); ax.set_xlim(0, max(1.0, np.nanmax(left)) * 1.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.45, -0.27), ncol=2, fontsize=6.3, frameon=False, columnspacing=1.2)
    panel_title(ax, '(b) partner of aircraft collisions (fixed reserve)')
    # (c) one-factor components, horizontal grouped bars
    ax = fig.add_subplot(gs[1, :])
    subs = [('nominal', 'nominal'), ('sensor_nav', 'sensor: navigation error only'), ('sensor_windmeas', 'sensor: wind measurement only'),
            ('sensor_miss', 'sensor: intruder misses only'), ('sensor_intnoise', 'sensor: intruder noise only'), ('sensor', 'sensor: full family'),
            ('traffic_fleet', 'traffic: fleet size only'), ('traffic_intruders', 'traffic: intruders only'), ('traffic', 'traffic: full family')]
    pols = ('sac', 'dsacn', 'crest', 'reactive'); hh = []
    for pi, m in enumerate(pols):
        yy = np.arange(len(subs))[::-1] + (1.5 - pi) * 0.2
        vals = []
        for sb, _ in subs:
            nm = f'main_{m}_std_{sb}' if sb in ('nominal', 'sensor', 'traffic') else f'sub_{m}_std_{sb}'
            vals.append([100 * x for x in ci_of(C('fixed', nm), 'coll_a')])
        V = np.array(vals); lo = np.clip(V[:, 1], 0, None)
        h = ax.barh(yy, V[:, 0], 0.19, color=POL[m]['col'] if m != 'reactive' else (0.85, 0.85, 0.85), edgecolor='#333333', lw=0.35,
                    hatch='xxxx' if m == 'reactive' else None, label=POL[m]['lab'])
        ok = np.isfinite(V[:, 1])
        ax.errorbar(V[ok, 0], yy[ok], xerr=np.vstack([V[ok, 0] - lo[ok], V[ok, 2] - V[ok, 0]]), fmt='none', ecolor='#1a1a1a', elinewidth=0.55, capsize=1.0)
        hh.append(h)
    ax.set_yticks(np.arange(len(subs))[::-1]); ax.set_yticklabels([l for _, l in subs], fontsize=7)
    ax.set_xlabel('aircraft collisions (% of missions), fixed reserve'); style(ax); ax.grid(False, axis='y')
    ax.legend(hh, [h.get_label() for h in hh], loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=6.8, ncol=4, frameon=False)
    panel_title(ax, '(c) one-factor components of the sensor and traffic shifts')
    fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.12)
    save(fig, 'fig_extra')


# ----------------------------------------------------------------------------------------- reserve coverage
def fig_cov():
    fig, axs = plt.subplots(2, 3, figsize=(18 * CM, 11.4 * CM)); hs = []
    rules = ['std', 'std15', 'static', 'online']
    for row, m in enumerate(('sac', 'crest')):
        for ax, (k, yl, t) in zip(axs[row], (('cov', 'return legs within reserve (%)', 'coverage'), ('dep', 'energy depletion (%)', 'depletion'),
                                             ('safe_ret', 'deliveries abandoned (%)', 'abandoned deliveries'))):
            n = len(rules); w = 0.84 / n
            for i, g in enumerate(rules):
                x = np.arange(len(COND)) + (i - (n - 1) / 2) * w
                vals = [ci_of(C('fixed', f'main_{m}_{g}_{c}'), k) for c in COND]
                y = np.array([100 * v[0] for v in vals]); lo = np.clip(np.array([100 * v[1] for v in vals]), 0, None); hi = np.array([100 * v[2] for v in vals])
                h = ax.bar(x, y, w * 0.9, color=GUARD[g]['col'], edgecolor='#333333', lw=0.35, hatch=GUARD[g]['hatch'] or None, label=GUARD[g]['lab'])
                ok = np.isfinite(lo)
                ax.errorbar(x[ok], y[ok], yerr=np.vstack([y[ok] - lo[ok], hi[ok] - y[ok]]), fmt='none', ecolor='#1a1a1a', elinewidth=0.55, capsize=1.1)
                if row == 0 and k == 'cov':
                    hs.append(h)
            if k == 'cov':
                ax.axhline(95, color='#1a1a1a', ls='--', lw=0.8); ax.set_ylim(0, 105)
            ax.set_xticks(range(len(COND))); ax.set_xticklabels([CONDLAB[c] for c in COND], rotation=30, ha='right'); ax.set_xlim(-0.5, len(COND) - 0.5)
            ax.set_ylabel(yl); style(ax, grid_x=False); panel_title(ax, '(%s) %s, %s' % ('abcdef'[row * 3 + ['cov', 'dep', 'safe_ret'].index(k)], POL[m]['lab'], t))
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.2, w_pad=0.8)
    legend_below(fig, hs + [Line2D([], [], color='#1a1a1a', ls='--', lw=0.8)], [h.get_label() for h in hs] + ['95% target'], ncol=5, y=0.0)
    save(fig, 'fig_cov')


# ----------------------------------------------------------------------------------------- online calibration and continual deployment
def fig_lam():
    fig, axs = plt.subplots(2, 2, figsize=(18 * CM, 12.4 * CM)); axs = axs.ravel()
    cc = {'nominal': ((0.35, 0.44, 0.66), '-'), 'wind': ((0.82, 0.60, 0.20), (0, (6, 2))), 'energy': ((0.78, 0.16, 0.14), (0, (4, 1.5, 1, 1.5)))}
    for c, (col, ls) in cc.items():
        T = S['lam_trace'].get(('crest', c))
        if T is None:
            continue
        j_ = np.arange(1, len(T) + 1)
        axs[0].plot(j_, T[:, 0], color=col, ls=ls, lw=1.1, label=c)
        axs[1].plot(j_, 100 * np.cumsum(T[:, 1]) / j_, color=col, ls=ls, lw=1.1, label=c)
    axs[0].set_xlabel('completed return legs'); axs[0].set_ylabel(r'fleet calibrator $\lambda_j$'); style(axs[0])
    panel_title(axs[0], '(a) calibrator, CREST-C, seed 0')
    axs[1].axhline(5, color='#1a1a1a', ls='--', lw=0.8); axs[1].set_ylim(0, 30); axs[1].set_xlabel('completed return legs')
    axs[1].set_ylabel('running miscoverage (%)'); style(axs[1]); panel_title(axs[1], '(b) realised against target miscoverage')
    ax = axs[2]; ex = S.get('lam_example')
    if ex:
        lam, err = ex['lam'], ex['err']; j_ = np.arange(1, len(lam) + 1); win = 80
        p = ex['name'].split('_'); pc = POL[p[1]]['col']
        ax.plot(j_, lam, color=pc, lw=1.0)
        ax2 = ax.twinx(); rm = np.convolve(err, np.ones(win) / win, mode='valid')
        ax2.plot(np.arange(win, len(lam) + 1), 100 * rm, color='#333333', lw=0.8, ls=(0, (3, 1.2)))
        ax2.axhline(5, color='#777777', ls='--', lw=0.6); ax2.set_ylim(0, 100); ax2.set_ylabel('80-leg miscoverage (%)'); ax2.grid(False)
        ax2.tick_params(axis='y', direction='in'); ax.tick_params(right=False)
        ax.set_xlabel('completed return legs'); ax.set_ylabel(r'fleet calibrator $\lambda_j$')
        style(ax); ax.tick_params(right=False)
        panel_title(ax, '(c) largest excursion: %s, %s, seed %d' % (POL[p[1]]['lab'], p[3], ex['seed']))
    ax = axs[3]
    rows = [v for (m, s_), v in S['cont'].items() if m == 'crest']
    if rows:
        v = [r for (m, s_), r in S['cont'].items() if m == 'crest' and s_ == min(s2 for (m2, s2) in S['cont'] if m2 == 'crest')][0]
        lam = v['lam']; blk = v['blk']; j_ = np.arange(1, len(lam) + 1)
        names = [b['cond'] for b in v['blocks']]
        shade = {'nominal': (1, 1, 1), 'energy': (0.98, 0.92, 0.90), 'wind': (0.99, 0.96, 0.88), 'compound': (0.92, 0.95, 0.98)}
        top = max(2.5, lam.max() * 1.15)
        for b in range(len(names)):
            idx = np.where(blk == b)[0]
            if len(idx):
                ax.axvspan(idx[0] + 1, idx[-1] + 1, color=shade.get(names[b], (1, 1, 1)), lw=0)
                ax.text((idx[0] + idx[-1]) / 2 + 1, top * 0.97, {'compound': 'comp.'}.get(names[b], names[b][:4] + '.' if names[b] != 'wind' else 'wind'),
                        ha='center', va='top', fontsize=6.0, color='#333333')
        ax.plot(j_, lam, color=POL['crest']['col'], lw=1.0)
        win = 100; rm = np.convolve(v['err'], np.ones(win) / win, mode='valid')
        ax2 = ax.twinx(); ax2.plot(np.arange(win, len(lam) + 1), 100 * rm, color='#333333', lw=0.8, ls=(0, (3, 1.2)))
        ax2.axhline(5, color='#777777', ls='--', lw=0.6); ax2.set_ylim(0, 40); ax2.set_ylabel('100-leg miscoverage (%)'); ax2.grid(False)
        ax2.tick_params(axis='y', direction='in'); ax.tick_params(right=False)
        ax.set_ylim(0.5, top); ax.set_xlabel('completed return legs'); ax.set_ylabel(r'fleet calibrator $\lambda_j$')
    style(ax); ax.tick_params(right=False); panel_title(ax, '(d) continual deployment, CREST-C, no reset')
    fig.tight_layout(rect=(0, 0.1, 1, 1), h_pad=1.4, w_pad=1.2)
    hh = [Line2D([], [], color=cc[c][0], ls=cc[c][1], lw=1.1) for c in cc] + [Line2D([], [], color='#1a1a1a', ls='--', lw=0.8),
          Line2D([], [], color='#777777', lw=1.0), Line2D([], [], color='#333333', lw=0.8, ls=(0, (3, 1.2)))]
    legend_below(fig, hh, ['nominal (a, b)', 'wind shift (a, b)', 'energy shift (a, b)', '5% target',
                           r'calibrator $\lambda_j$ (c, d; colour of the policy)', 'moving miscoverage (c, d; right axis)'], ncol=3, y=0.0)
    save(fig, 'fig_lam')


# ----------------------------------------------------------------------------------------- reserve frontier
def fig_pareto():
    FR = S['frontier']
    fixed = ['fixed1.0', 'fixed1.2', 'fixed1.35', 'fixed1.5', 'fixed1.75', 'fixed2.0', 'fixed2.5']
    onl = ['online0.02', 'online0.05', 'online0.1', 'online0.2', 'online0.3']
    pols = [m for m in ('sac', 'crest', 'reactive') if any(k[0] == m for k in FR)]
    fig, axs = plt.subplots(len(pols), 3, figsize=(18 * CM, (4.9 * len(pols) + 1.4) * CM), squeeze=False)
    for r_, m in enumerate(pols):
        for c_, c in enumerate(('nominal', 'wind', 'energy')):
            ax = axs[r_, c_]
            for rules, col, mk, ls, lab in ((fixed, (0.45, 0.45, 0.45), 'o', '-', 'fixed margin'), (onl, POL['crest']['col'], 's', '--', 'online conformal')):
                pts = [FR.get((m, c, ru)) for ru in rules]
                X = np.array([[100 * p['safe_ret'], 100 * p['ci_safe_ret']['lo'], 100 * p['ci_safe_ret']['hi'], 100 * p['dep'], 100 * p['ci_dep']['lo'], 100 * p['ci_dep']['hi']] if p else [np.nan] * 6 for p in pts])
                X[:, 1] = np.clip(X[:, 1], 0, None); X[:, 4] = np.clip(X[:, 4], 0, None)
                ax.errorbar(X[:, 0], X[:, 3], xerr=np.vstack([np.nan_to_num(X[:, 0] - X[:, 1]), np.nan_to_num(X[:, 2] - X[:, 0])]),
                            yerr=np.vstack([np.nan_to_num(X[:, 3] - X[:, 4]), np.nan_to_num(X[:, 5] - X[:, 3])]), color=col, marker=mk, ls=ls, ms=3.6, lw=1.0,
                            elinewidth=0.55, capsize=1.0, label=lab, mfc='white' if ls == '--' else col)
                for ru, x in zip(rules, X):
                    if ru in ('fixed1.2', 'fixed1.5', 'fixed2.0', 'online0.05') and np.isfinite(x[0]):
                        lab_ = {'fixed1.2': r'1.2$\times$', 'fixed1.5': r'1.5$\times$', 'fixed2.0': r'2.0$\times$', 'online0.05': r'$\delta_l$=0.05'}[ru]
                        off = (4, 6) if ru.startswith('fixed') else (4, -10)
                        ax.annotate(lab_, (x[0], x[3]), xytext=off, textcoords='offset points', fontsize=5.8, color=col)
            ax.set_xlabel('deliveries abandoned (%)') if r_ == len(pols) - 1 else None
            ax.set_ylabel('energy depletion (%)') if c_ == 0 else None
            style(ax); panel_title(ax, '(%s) %s, %s' % ('abcdefghi'[r_ * 3 + c_], POL[m]['lab'], c if c == 'nominal' else c + ' shift'))
            ax.set_ylim(-0.04 * ax.get_ylim()[1], ax.get_ylim()[1])
    fig.tight_layout(rect=(0, 0.05, 1, 1), h_pad=0.8, w_pad=0.8)
    hh = [Line2D([], [], color=(0.45, 0.45, 0.45), marker='o', ls='-', ms=3.6), Line2D([], [], color=POL['crest']['col'], marker='s', mfc='white', ls='--', ms=3.6)]
    legend_below(fig, hh, [r'fixed margin 1.0$\times$ to 2.5$\times$', r'online conformal, target $\delta_l$ from 0.02 to 0.30'], ncol=2, y=0.0)
    save(fig, 'fig_pareto')


if __name__ == '__main__':
    names = sys.argv[1:] or [n[4:] for n in dir() if n.startswith('fig_')]
    for n in names:
        globals()['fig_' + n]()
