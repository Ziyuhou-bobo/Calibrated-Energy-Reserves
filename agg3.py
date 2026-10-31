"""Aggregate the revised evaluation (res3/ARM/sS/*.npz) into one analysis file.

Everything reported in the manuscript is computed here from the saved per-mission outcomes:
pooled point estimates, seed-level means and t intervals (seeds are the experimental units), paired
differences between policies on identical scenarios, hierarchical (seed x scenario) bootstrap intervals,
equivalence tests (TOST) against a margin fixed before the ten-seed evaluation was run, minimum detectable
effects and Holm-adjusted p-values.  Output: analysis/summary.pkl (read by mkpaper.py and figs3.py).
usage: python3 agg3.py
"""
import glob, os, pickle, re, sys, json, time
import numpy as np
from scipy import stats as st
from evallib import point_metrics

RES = os.environ.get('RES3', 'res3')
OUT = os.environ.get('ANALYSIS', 'analysis')
os.makedirs(OUT, exist_ok=True)
DELTA_EQ = 0.02          # equivalence margin on catastrophic loss (2 percentage points), fixed before the 10-seed runs
B_BOOT = int(os.environ.get('BOOT', 2000))
RNG = np.random.default_rng(20261027)
AGENTS = ['sac', 'td3', 'dsacn', 'dsaccvar', 'crestret', 'crest']
POLS = AGENTS + ['reactive']
TAIL = ['dsaccvar', 'crestret', 'crest']
CONDS = ['nominal', 'wind', 'sensor', 'traffic', 'energy', 'compound']
RATE_KEYS = ('succ', 'coll', 'coll_a', 'coll_b', 'dep', 'oob', 'loss', 'safe_ret', 'cancel', 'tmo')
CLUSTER_KEYS = ('loss', 'succ', 'coll', 'coll_a', 'dep', 'safe_ret')


def seeds_of(arm):
    return sorted(int(os.path.basename(d)[1:]) for d in glob.glob(f'{RES}/{arm}/s*') if os.path.isdir(d))


def cluster_summary(M, kmax=None):
    """Per-scenario (district, episode) counts used by the paired hierarchical bootstrap."""
    keep = np.ones(len(M['out']), bool) if kmax is None else M['ep'] < kmax
    key = M['env'][keep] * 1000 + M['ep'][keep]
    out = M['out'][keep]
    ind = dict(loss=np.isin(out, (3, 4, 5, 6)), succ=out == 1, coll=np.isin(out, (3, 4)), coll_a=out == 4, dep=out == 5,
               safe_ret=out == 2)
    uk, inv = np.unique(key, return_inverse=True)
    res = dict(keys=uk, n=np.bincount(inv, minlength=len(uk)).astype(float))
    for k, v in ind.items():
        res[k] = np.bincount(inv, weights=v.astype(float), minlength=len(uk))
    return res


def restrict(M, kmax):
    m = M['ep'] < kmax
    return {k: (v[m] if hasattr(v, '__len__') and len(v) == len(m) else v) for k, v in M.items()}


# ---------------------------------------------------------------------------------------------- loading
def load_all():
    data = dict(pm={}, cl={}, legs={}, minair={}, cont={}, extra={})
    t0 = time.time()
    for arm in ('fixed', 'online'):
        for s in seeds_of(arm):
            d = f'{RES}/{arm}/s{s}'
            if os.path.exists(f'{d}/meta.json'):
                data['extra'][(arm, s, 'meta')] = json.load(open(f'{d}/meta.json'))
            for f in sorted(glob.glob(f'{d}/*.npz')):
                nm = os.path.basename(f)[:-4]
                if nm.endswith('.tmp') or nm.startswith('calscores'):
                    continue
                z = np.load(f, allow_pickle=False)
                if nm == 'critic_check':
                    data['extra'][(arm, s, 'critic')] = {k: z[k] for k in z.files}
                    continue
                M = {k: z[k] for k in z.files if k not in ('legs', 'lam_final', 'gstep', 'blocks')}
                legs = z['legs']
                if nm.startswith('cont_'):
                    data['cont'][(arm, s, nm)] = dict(M=M, legs=legs, blocks=list(z['blocks']))
                    continue
                data['pm'][(arm, nm, s)] = point_metrics(M, legs)
                data['pm'][(arm, nm, s)]['gstep'] = int(z['gstep'][0]) if 'gstep' in z.files else 0
                if nm.startswith('main_'):
                    data['cl'][(arm, nm, s)] = cluster_summary(M)
                    data['cl'][(arm, nm + '|k3', s)] = cluster_summary(M, 3)
                    data['pm'][(arm, nm + '|k3', s)] = point_metrics(restrict(M, 3), legs[legs[:, 13] < 3] if len(legs) else legs)
                    if '_online_' in nm or '_std_' in nm:
                        data['legs'][(arm, nm, s)] = legs
                    if nm.endswith('_traffic') and '_std_' in nm:
                        data['minair'][(arm, nm, s)] = M['minair'].astype(np.float32)
                    if '_online_' in nm:
                        ep = M['ep']; out = M['out']
                        data['extra'][(arm, s, nm, 'dep_ep0')] = float((out[ep == 0] == 5).mean())
                        data['extra'][(arm, s, nm, 'dep_ep1p')] = float((out[ep >= 1] == 5).mean())
                        ok = (out == 1) & np.isfinite(M['tdel']) & np.isfinite(M['tend'])
                        data['extra'][(arm, s, nm, 'leg_dur')] = float(np.mean(M['tend'][ok] - M['tdel'][ok])) if ok.any() else np.nan
                if nm.startswith(('abl_', 'dial_', 'disp_', 'par_')):
                    data['cl'][(arm, nm, s)] = cluster_summary(M)
    print('loaded in %.0fs: %d evaluations' % (time.time() - t0, len(data['pm'])))
    return data


# ---------------------------------------------------------------------------------------------- statistics
def tci(v, level=0.95):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    n = len(v)
    if n == 0:
        return dict(mean=np.nan, sd=np.nan, lo=np.nan, hi=np.nan, n=0)
    m = float(v.mean())
    if n < 2:
        return dict(mean=m, sd=np.nan, lo=np.nan, hi=np.nan, n=n)
    sd = float(v.std(ddof=1)); h = st.t.ppf(0.5 + level / 2, n - 1) * sd / np.sqrt(n)
    return dict(mean=m, sd=sd, lo=m - h, hi=m + h, n=n)


def paired(va, vb):
    """Seed-level paired analysis of d_s = a_s - b_s (seed index s shares the scenario set)."""
    d = np.asarray(va, float) - np.asarray(vb, float)
    d = d[np.isfinite(d)]
    n = len(d)
    r = dict(n=n, d=d)
    if n < 2:
        r.update(mean=float(d.mean()) if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan, lo90=np.nan, hi90=np.nan,
                 p=np.nan, mde=np.nan, p_tost=np.nan, equiv=False)
        return r
    m, sd = float(d.mean()), float(d.std(ddof=1)); se = sd / np.sqrt(n)
    t975, t95, t80 = st.t.ppf(0.975, n - 1), st.t.ppf(0.95, n - 1), st.t.ppf(0.80, n - 1)
    p = float(2 * st.t.sf(abs(m) / se, n - 1)) if se > 0 else (0.0 if m != 0 else 1.0)
    p_lo = float(st.t.sf((m + DELTA_EQ) / se, n - 1)) if se > 0 else 0.0     # H0: diff <= -Delta
    p_hi = float(st.t.cdf((m - DELTA_EQ) / se, n - 1)) if se > 0 else 0.0    # H0: diff >= +Delta
    r.update(mean=m, sd=sd, lo=m - t975 * se, hi=m + t975 * se, lo90=m - t95 * se, hi90=m + t95 * se, p=p,
             mde=float((t975 + t80) * se), p_tost=max(p_lo, p_hi), equiv=bool(max(p_lo, p_hi) < 0.05))
    return r


def hboot_diff(cla, clb, key='loss', B=B_BOOT):
    """Hierarchical bootstrap of the pooled rate difference a - b: seed indices are resampled with replacement,
    and within every resampled seed the scenario clusters are resampled jointly for both policies."""
    seeds = sorted(set(cla) & set(clb))
    if len(seeds) < 2:
        return (np.nan, np.nan)
    A, Bb = [], []
    for s in seeds:
        a, b = cla[s], clb[s]
        assert np.array_equal(a['keys'], b['keys']), 'scenario sets differ'
        A.append(a); Bb.append(b)
    S = len(seeds); diffs = np.empty(B)
    for i in range(B):
        ss = RNG.integers(0, S, S)
        na = la = nb = lb = 0.0
        for j in ss:
            C = len(A[j]['keys'])
            w = RNG.multinomial(C, np.full(C, 1.0 / C))
            na += w @ A[j]['n']; la += w @ A[j][key]; nb += w @ Bb[j]['n']; lb += w @ Bb[j][key]
        diffs[i] = la / na - lb / nb
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def hboot_rate(cla, key='loss', B=B_BOOT):
    seeds = sorted(cla)
    if len(seeds) < 2:
        return (np.nan, np.nan)
    A = [cla[s] for s in seeds]; S = len(seeds); v = np.empty(B)
    for i in range(B):
        ss = RNG.integers(0, S, S); n = l = 0.0
        for j in ss:
            C = len(A[j]['keys']); w = RNG.multinomial(C, np.full(C, 1.0 / C))
            n += w @ A[j]['n']; l += w @ A[j][key]
        v[i] = l / n
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def holm(pvals):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (n - r) * p[i])); adj[i] = run
    return adj


# ---------------------------------------------------------------------------------------------- summaries
def cell(data, arm, name, seeds=None):
    """Pooled and seed-level summary of one evaluation cell."""
    if 'index' not in data:
        idx = {}
        for (a, n, s) in data['pm']:
            idx.setdefault((a, n), []).append(s)
        data['index'] = idx
    per = {s: data['pm'][(arm, name, s)] for s in data['index'].get((arm, name), []) if seeds is None or s in seeds}
    if not per:
        return None
    keys = set.intersection(*[set(v) for v in per.values()])
    res = dict(seeds=sorted(per), per=per)
    for k in keys:
        vals = [per[s][k] for s in sorted(per)]
        if all(isinstance(x, (int, float, np.floating)) for x in vals):
            res[k] = float(np.nanmean(vals)) if k != 'n' else int(np.sum(vals))
            res['ci_' + k] = tci(vals)
    return res


def main():
    data = load_all()
    S = {}
    # ---- every evaluation cell, both arms
    names = sorted({(a, n) for (a, n, s) in data['pm']})
    S['cells'] = {(a, n): cell(data, a, n) for a, n in names}
    # ---- paired differences against reference policies (main conditions, both reserves, fixed arm)
    seeds_f = seeds_of('fixed')
    PD = {}
    for g in ('std', 'online'):
        for ref in ('sac', 'dsacn', 'reactive'):
            for m in POLS:
                if m == ref:
                    continue
                for c in CONDS:
                    na, nb = f'main_{m}_{g}_{c}', f'main_{ref}_{g}_{c}'
                    ca = {s: data['pm'][('fixed', na, s)] for s in seeds_f if ('fixed', na, s) in data['pm']}
                    cb = {s: data['pm'][('fixed', nb, s)] for s in seeds_f if ('fixed', nb, s) in data['pm']}
                    common = sorted(set(ca) & set(cb))
                    if len(common) < 2:
                        continue
                    for key in ('loss', 'succ', 'coll', 'coll_a', 'dep'):
                        r = paired([ca[s][key] for s in common], [cb[s][key] for s in common])
                        if key in ('loss', 'succ'):
                            cla = {s: data['cl'][('fixed', na, s)] for s in common}
                            clb = {s: data['cl'][('fixed', nb, s)] for s in common}
                            r['hb'] = hboot_diff(cla, clb, key)
                        PD[(g, ref, m, c, key)] = r
    # Holm adjustment within each (reserve, comparator, metric) family of tail-risk comparisons
    for g in ('std', 'online'):
        for ref in ('sac', 'dsacn'):
            for key in ('loss', 'succ'):
                fam = [(g, ref, m, c, key) for m in TAIL for c in CONDS if (g, ref, m, c, key) in PD]
                if fam:
                    adj = holm([PD[f]['p'] for f in fam])
                    for f, a in zip(fam, adj):
                        PD[f]['p_holm'] = float(a)
    S['paired'] = PD
    # ---- seed-level and hierarchical intervals for the main loss rates
    HB = {}
    for g in ('std', 'online'):
        for m in POLS:
            for c in CONDS:
                nm = f'main_{m}_{g}_{c}'
                cl = {s: data['cl'][('fixed', nm, s)] for s in seeds_f if ('fixed', nm, s) in data['cl']}
                if len(cl) >= 2:
                    HB[(g, m, c)] = dict(loss=hboot_rate(cl, 'loss'), succ=hboot_rate(cl, 'succ'))
    S['hboot'] = HB
    # ---- ablations: paired differences of each variant against the full schedule on the same scenarios (k=3)
    AB = {}
    for m in ('crest', 'crestret'):
        for v in ('fixedalpha', 'noenergy', 'prox', 'unc'):
            for c in CONDS:
                na, nb = f'abl_{m}_{v}_{c}', f'main_{m}_std_{c}|k3'
                ca = {s: data['pm'][('fixed', na, s)] for s in seeds_f if ('fixed', na, s) in data['pm']}
                cb = {s: data['pm'][('fixed', nb, s)] for s in seeds_f if ('fixed', nb, s) in data['pm']}
                common = sorted(set(ca) & set(cb))
                if len(common) >= 2:
                    AB[(m, v, c)] = paired([ca[s]['loss'] for s in common], [cb[s]['loss'] for s in common])
    fam = list(AB)
    if fam:
        adj = holm([AB[f]['p'] for f in fam])
        for f, a in zip(fam, adj):
            AB[f]['p_holm'] = float(a)
    S['abl'] = AB
    # ---- risk dial: seed-level summaries per alpha_b, and the consistency check against the main runs
    DI, chk = {}, []
    for m in ('crest', 'crestret'):
        for c in ('nominal', 'sensor', 'traffic', 'compound'):
            for ab in (0.1, 0.25, 0.5, 0.75, 1.0):
                nm = f'dial_{m}_{ab}_{c}'
                ce = S['cells'].get(('fixed', nm))
                if ce:
                    DI[(m, c, ab)] = ce
                for s in seeds_f:
                    meta = data['extra'].get(('fixed', s, 'meta'))
                    if meta and meta['ab'].get(m) == ab and ('fixed', nm, s) in data['pm'] and ('fixed', f'main_{m}_std_{c}|k3', s) in data['pm']:
                        a, b = data['pm'][('fixed', nm, s)], data['pm'][('fixed', f'main_{m}_std_{c}|k3', s)]
                        chk.append(abs(a['loss'] - b['loss']) + abs(a['succ'] - b['succ']))
            # paired difference between the most cautious and the risk-neutral end of the dial
            for key in ('coll', 'loss', 'spd', 'prox50', 'fprox50', 'minsep_q05', 'tdel', 'succ'):
                ca = {s: data['pm'][('fixed', f'dial_{m}_0.1_{c}', s)] for s in seeds_f if ('fixed', f'dial_{m}_0.1_{c}', s) in data['pm']}
                cb = {s: data['pm'][('fixed', f'dial_{m}_1.0_{c}', s)] for s in seeds_f if ('fixed', f'dial_{m}_1.0_{c}', s) in data['pm']}
                common = sorted(set(ca) & set(cb))
                if len(common) >= 2:
                    DI[(m, c, 'diff', key)] = paired([ca[s][key] for s in common], [cb[s][key] for s in common])
    S['dial'] = DI
    S['dial_check'] = dict(n=len(chk), max_abs=float(max(chk)) if chk else np.nan)
    # ---- selected base risk levels
    S['ab_sel'] = {(m, s): data['extra'][('fixed', s, 'meta')]['ab'].get(m) for s in seeds_f for m in ('crest', 'crestret')
                   if ('fixed', s, 'meta') in data['extra']}
    S['calib'] = {(m, s): data['extra'][('fixed', s, 'meta')]['calib'].get(m) for s in seeds_f for m in POLS
                  if ('fixed', s, 'meta') in data['extra']}
    # ---- calibrator diagnostics for the online reserve runs
    CD = {}
    for m in POLS:
        for c in CONDS:
            nm = f'main_{m}_online_{c}'
            rows = []
            for s in seeds_f:
                L = data['legs'].get(('fixed', nm, s))
                if L is None or not len(L):
                    continue
                err = L[:, 10]; n = len(err)
                gst = data['pm'][('fixed', nm, s)]['gstep']
                rows.append(dict(cov=1 - err.mean(), cov100=1 - err[:100].mean(), cov_rest=1 - err[100:].mean() if n > 100 else np.nan,
                                 disp=float(L[:, 18].sum()) if L.shape[1] > 18 else np.nan,
                                 absdisp=float(np.abs(L[:, 18]).sum()) if L.shape[1] > 18 else np.nan,
                                 atb=float(np.mean((L[:, 12] <= 0.5 + 1e-12) | (L[:, 12] >= 4.0 - 1e-12))),
                                 lam_end=float(L[-1, 12]), lam_max=float(L[:, 12].max()), n=n,
                                 legs_per_min=(n - 1) / max((L[-1, 16] - L[0, 16]) / 60.0, 1e-9), sim_min=(L[-1, 16] - L[0, 16]) / 60.0,
                                 dep_ep0=data['extra'].get(('fixed', s, nm, 'dep_ep0'), np.nan),
                                 dep_ep1p=data['extra'].get(('fixed', s, nm, 'dep_ep1p'), np.nan),
                                 leg_dur=data['extra'].get(('fixed', s, nm, 'leg_dur'), np.nan),
                                 lam0=float(L[0, 11]), overshoot=float(L[:, 12].max() >= 2.0 * L[0, 11]),
                                 burst=float(max(err[i:i + 80].mean() for i in range(0, max(n - 79, 1), 20))),
                                 rth=data['pm'][('fixed', nm, s)]['safe_ret'], dep=data['pm'][('fixed', nm, s)]['dep'], seed=s))
                rows[-1]['inflight'] = rows[-1]['legs_per_min'] / 60.0 * rows[-1]['leg_dur']
                # where the worst 80-leg window of misses came from: districts and episode index of its misses
                w = 80 if n >= 80 else n
                i0 = int(np.argmax(np.convolve(err, np.ones(w), mode='valid'))); seg = L[i0:i0 + w]; mm = seg[:, 10] == 1
                if mm.any():
                    _, cnt = np.unique(seg[mm, 13].astype(int), return_counts=True)
                    rows[-1]['burst_ndist'] = float(len(np.unique(seg[mm, 0]))); rows[-1]['burst_epshare'] = float(cnt.max() / mm.sum())
                else:
                    rows[-1]['burst_ndist'] = np.nan; rows[-1]['burst_epshare'] = np.nan
                ia = int(np.argmax(L[:, 12]))
                rows[-1]['lam_min_pre'] = float(L[:ia + 1, 12].min()); rows[-1]['dispcorr'] = float(rows[-1]['disp'] / (0.05 * n))
                rows[-1]['excursion'] = float(rows[-1]['overshoot'] > 0 or rows[-1]['burst'] >= 0.25)
            if rows:
                CD[(m, c)] = {k: tci([r[k] for r in rows]) for k in rows[0]}
                CD[(m, c)]['rows'] = rows
    S['calib_diag'] = CD
    # the run with the largest calibrator excursion (figure: delayed-feedback overshoot)
    best = None
    for (arm, nm, s_), L in data['legs'].items():
        if arm == 'fixed' and '_online_' in nm and len(L):
            v = float(L[:, 12].max() / max(L[0, 11], 1e-9))
            if best is None or v > best[0]:
                best = (v, nm, s_, L[:, 12].copy(), L[:, 10].copy())
    S['lam_example'] = dict(name=best[1], seed=best[2], lam=best[3], err=best[4], ratio=best[0]) if best else None
    if best:
        Lb = data['legs'][('fixed', best[1], best[2])]
        S['lam_example'].update(ep=Lb[:, 13].copy(), district=Lb[:, 0].copy(), gstep=Lb[:, 16].copy(), lam_before=Lb[:, 11].copy())
    # traces of the calibrator for seed 0 (figure)
    S['lam_trace'] = {}
    for m in ('sac', 'crest', 'reactive'):
        for c in ('nominal', 'wind', 'energy'):
            L = data['legs'].get(('fixed', f'main_{m}_online_{c}', seeds_f[0] if seeds_f else 0))
            if L is not None and len(L):
                S['lam_trace'][(m, c)] = np.stack([L[:, 12], L[:, 10], L[:, 16] if L.shape[1] > 16 else np.arange(len(L))], 1)
    # ---- continual deployment
    CT = {}
    for (arm, s, nm), v in data['cont'].items():
        L = v['legs']; M = v['M']
        if not len(L):
            continue
        blk = L[:, 19].astype(int); err = L[:, 10]
        per_block = []
        for b in range(len(v['blocks'])):
            e = err[blk == b]
            mb = M['out'][M['block'] == b]
            per_block.append(dict(cond=v['blocks'][b], n=len(e), cov=1 - e.mean() if len(e) else np.nan,
                                  cov100=1 - e[:100].mean() if len(e) else np.nan, dep=float((mb == 5).mean()) if len(mb) else np.nan,
                                  lam_start=float(L[blk == b][0, 11]) if len(e) else np.nan,
                                  lam_end=float(L[blk == b][-1, 12]) if len(e) else np.nan))
        CT[(nm[5:], s)] = dict(blocks=per_block, lam=L[:, 12], err=err, blk=blk, cov_all=1 - err.mean(),
                               disp=float(L[:, 18].sum()), atb=float(np.mean((L[:, 12] <= 0.5 + 1e-12) | (L[:, 12] >= 4.0 - 1e-12))))
    S['cont'] = CT
    # ---- training-reserve check: online-trained arm against the fixed-trained arm (same seeds, same scenarios)
    # only seeds for which the online-trained arm is complete (all three agents, both test reserves, every condition)
    # are used, and the fixed-trained arm is summarised over the same seeds, so that both arms meet the same scenarios
    seeds_o = [s for s in seeds_of('online') if s in seeds_f and all(('online', f'main_{m}_{g}_{c}', s) in data['pm'] and
               ('fixed', f'main_{m}_{g}_{c}', s) in data['pm'] for m in ('sac', 'dsacn', 'crest') for g in ('std', 'online') for c in CONDS)]
    TR = {}
    for g in ('std', 'online'):
        for m in ('sac', 'dsacn', 'crest'):
            for c in CONDS:
                nm = f'main_{m}_{g}_{c}'
                if len(seeds_o) >= 2:
                    lo_ = [data['pm'][('online', nm, s)]['loss'] for s in seeds_o]; lf_ = [data['pm'][('fixed', nm, s)]['loss'] for s in seeds_o]
                    TR[(g, m, c, 'arm')] = paired(lo_, lf_)
                    TR[(g, m, c, 'arm_succ')] = paired([data['pm'][('online', nm, s)]['succ'] for s in seeds_o],
                                                       [data['pm'][('fixed', nm, s)]['succ'] for s in seeds_o])
                    TR[(g, m, c, 'rate_online')] = float(np.mean(lo_)); TR[(g, m, c, 'rate_fixed')] = float(np.mean(lf_))
            for c in CONDS:
                for arm in ('online', 'fixed'):
                    if len(seeds_o) >= 2:
                        TR[(g, 'crest-sac', c, arm)] = paired([data['pm'][(arm, f'main_crest_{g}_{c}', s)]['loss'] for s in seeds_o],
                                                              [data['pm'][(arm, f'main_sac_{g}_{c}', s)]['loss'] for s in seeds_o])
    S['train_reserve'] = TR
    S['tr_seeds'] = seeds_o
    # ---- reserve frontier: seed-level summaries for every rule
    FR = {}
    for m in ('sac', 'crest', 'reactive', 'dsacn'):
        for c in ('nominal', 'wind', 'energy'):
            rules = [('fixed1.0', f'par_{m}_fixed1.0_{c}'), ('fixed1.2', f'main_{m}_std_{c}|k3'), ('fixed1.35', f'par_{m}_fixed1.35_{c}'),
                     ('fixed1.5', f'main_{m}_std15_{c}|k3'), ('fixed1.75', f'par_{m}_fixed1.75_{c}'), ('fixed2.0', f'par_{m}_fixed2.0_{c}'),
                     ('fixed2.5', f'par_{m}_fixed2.5_{c}'), ('online0.02', f'par_{m}_online0.02_{c}'), ('online0.05', f'main_{m}_online_{c}|k3'),
                     ('online0.1', f'par_{m}_online0.1_{c}'), ('online0.2', f'par_{m}_online0.2_{c}'), ('online0.3', f'par_{m}_online0.3_{c}')]
            for rule, nm in rules:
                ce = S['cells'].get(('fixed', nm))
                if ce is None and nm.endswith('|k3'):
                    continue
                if ce:
                    FR[(m, c, rule)] = ce
    S['frontier'] = FR
    # ---- traffic-shift minimum clearance distribution
    S['minair'] = {}
    for m in POLS:
        v = [data['minair'][k] for k in data['minair'] if k[0] == 'fixed' and k[1] == f'main_{m}_std_traffic']
        if v:
            S['minair'][m] = np.sort(np.clip(np.concatenate(v), -5, 300))
    # ---- critic diagnostics
    S['critic'] = {s: data['extra'][('fixed', s, 'critic')] for s in seeds_f if ('fixed', s, 'critic') in data['extra']}
    # ---- PPO reference (not part of the comparison)
    PP = {}
    for tag, seeds in (('ppo25k', (0, 1, 2)), ('ppo100k', (0, 1))):
        for c in CONDS:
            tabs = [f'{RES}/ppo/{tag}_s{s}_std_{c}.npz' for s in seeds]
            tabs = [t for t in tabs if os.path.exists(t)]
            if not tabs:
                continue
            Ms = []
            for t in tabs:
                z = np.load(t); Ms.append({k: z[k] for k in z.files if k != 'legs'})
            M = {k: np.concatenate([m[k] for m in Ms]) for k in Ms[0]}
            PP.setdefault(tag, {})[c] = point_metrics(M)
    S['ppo'] = PP
    # ---- PPO implementation check (reward normalisation and 10 epochs x 16 minibatches), from the test log
    misc = {}
    lg = 'logs/ppo_v2_test.log'
    if os.path.exists(lg):
        recs = [json.loads(l[l.index('{'):]) for l in open(lg) if '{' in l]
        if recs:
            last = recs[-1]; misc['PpoVtwoSteps'] = '{:,}'.format(last['step']).replace(',', '{,}')
            tail = [r for r in recs if r['step'] > last['step'] - 2000]
            misc['PpoVtwoSucc'] = '%.0f' % (100 * np.mean([r['succ'] for r in tail]))
            misc['PpoVtwoColl'] = '%.0f' % (100 * np.mean([r['coll'] for r in tail]))
    import pickle as _p
    for s_ in (0, 1, 2):
        f = f'runs_legacy/ppo_s{s_}.pkl'
        if os.path.exists(f) and 'PpoVtwoSteps' in misc:
            L_ = _p.load(open(f, 'rb'))['log']; st_ = int(misc['PpoVtwoSteps'].replace('{,}', ''))
            v = [r['succ'] for r in L_ if st_ - 2000 < r['step'] <= st_]
            misc.setdefault('_leg', []).extend(v)
    if '_leg' in misc:
        misc['PpoLegacySuccAtSteps'] = '%.0f' % (100 * np.mean(misc.pop('_leg')))
    # compute budget of the study from the job log (wall time of every finished job, one core per job); checkpoints
    # trained before the job log existed (v1.0) are counted with the mean logged time of the same algorithm
    try:
        import datetime as _dt, re as _re
        starts, dur = {}, {}
        day = '10-27'
        for line in open('jobs/runner.log'):
            m_ = _re.match(r'^(?:(\d\d-\d\d) )?(\d\d:\d\d:\d\d) (start|done)(?: \(adopted\))? (\S+)', line)
            if not m_:
                continue
            day = m_.group(1) or day
            t = _dt.datetime.strptime('2026-%s %s' % (day, m_.group(2)), '%Y-%m-%d %H:%M:%S')
            if m_.group(3) == 'start':
                starts[m_.group(4)] = t
            elif m_.group(4) in starts:
                dur[m_.group(4)] = (t - starts.pop(m_.group(4))).total_seconds() / 3600
        ev = sum(v for k, v in dur.items() if not k.startswith('runs'))
        tr = 0.0; per_algo = {}
        for k, v in dur.items():
            if k.startswith('runs'):
                per_algo.setdefault(os.path.basename(k).split('_s')[0], []).append(v)
        for f in glob.glob('runs/*.pkl') + glob.glob('runs_online/*.pkl') + glob.glob('runs_legacy/*.pkl'):
            if f in dur:
                tr += dur[f]
            else:
                a = os.path.basename(f).split('_s')[0]
                if a == 'ppo' and 'ppo100k' in per_algo:
                    tr += np.mean(per_algo['ppo100k']) / 4
                elif a in per_algo:
                    tr += np.mean(per_algo[a])
        misc['RepEvalHours'] = '%d' % round(ev + 0.5)
        misc['RepTotalHours'] = '%d' % round(ev + tr + 0.5)
    except Exception as e:
        print('compute budget not available:', e)
        misc['RepEvalHours'] = '--'; misc['RepTotalHours'] = '--'
    S['misc'] = misc
    S['seeds'] = dict(fixed=seeds_f, online=seeds_o)
    S['delta_eq'] = DELTA_EQ
    pickle.dump(S, open(f'{OUT}/summary.pkl', 'wb'))
    print('summary written: %d cells, %d paired comparisons, seeds fixed=%s online=%s' % (len(S['cells']), len(PD), seeds_f, seeds_o))


if __name__ == '__main__':
    main()
