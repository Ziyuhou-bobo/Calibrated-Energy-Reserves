"""Evaluation harness for the multi-UAV delivery study.

Policies are evaluated deterministically.  Each evaluation runs E parallel districts until every
district has completed k episodes; only the first k episodes of every district are kept, so long
episodes are not under-sampled.  Return legs are filtered the same way.
"""
import numpy as np, pickle, math
import jax, jax.numpy as jnp
from uavenv import MultiUAVEnv, make_cfg, FAMILIES
from rl import act_tanh_det, td3_act, ppo_det, dac_quantiles
from train import schedule, ALPHA_MIN as ALPHA_MIN_EVAL

DAC = ('dsacn', 'dsaccvar', 'crest', 'crestret', 'wcsac')
ALPHA_FIXED = {'dsacn': 1.0, 'dsaccvar': 0.25, 'wcsac': 0.25}
GUARD_COL = {'std': 3, 'static': 4, 'online': 5, 'std15': 7, 'fixed': 15}   # leg-log column of each bound (15 = active guard)


def load(algo, seed, d='runs'):
    with open(f'{d}/{algo}_s{seed}.pkl', 'rb') as f:
        return pickle.load(f)


class Policy:
    """Deterministic deployment policy.  For CREST, alpha_mode in {'sched', 'fixed'} and sched in
    {'full', 'noenergy', 'prox', 'unc'} select the risk-regulation variant.  'unc' adapts the risk level
    to the spread of the learned return distribution (uncertainty-driven adaptation in the spirit of
    adaptive risk-tendency methods); spread_ref = (s_lo, s_hi) normalises the 10-90 inter-quantile range."""

    def __init__(self, ck, alpha_base=0.5, alpha_mode='sched', sched='full', spread_ref=None):
        self.algo = ck['algo']
        self.p = jax.tree_util.tree_map(jnp.asarray, ck['actor'])
        self.z = jax.tree_util.tree_map(jnp.asarray, ck['critic']) if ck.get('critic') is not None else None
        self.alpha_base, self.alpha_mode, self.sched = alpha_base, alpha_mode, sched
        self.spread_ref = spread_ref
        self.trace = None
        self.spreads = None

    def _spread(self, obs, al):
        E, N, D = obs.shape
        x = np.concatenate([obs, al[..., None]], -1).reshape(E * N, D + 1).astype(np.float32)
        a = act_tanh_det(self.p, x)
        q = np.asarray(dac_quantiles(self.z, x, a))
        n = q.shape[1]
        return (q[:, int(0.9 * n)] - q[:, int(0.1 * n)]).reshape(E, N)

    def alpha(self, cues, obs=None):
        E, N = cues.shape[:2]
        if self.algo in ALPHA_FIXED:
            return np.full((E, N), ALPHA_FIXED[self.algo])
        if self.alpha_mode == 'fixed':
            return np.full((E, N), self.alpha_base)
        if self.sched == 'unc':
            sp = self._spread(obs, np.full((E, N), self.alpha_base, np.float32))
            if self.spreads is not None:
                self.spreads.append(sp.copy())
            lo, hi = self.spread_ref if self.spread_ref is not None else (0.0, 1.0)
            u = np.clip((sp - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            return ALPHA_MIN_EVAL + (self.alpha_base - ALPHA_MIN_EVAL) * (1.0 - u)
        return schedule(cues, self.alpha_base, self.sched)

    def __call__(self, obs, cues):
        E, N, D = obs.shape
        if self.algo in DAC:
            al = self.alpha(cues, obs)
            if self.trace is not None:
                self.trace.append(al.copy())
            x = np.concatenate([obs, al[..., None]], -1).reshape(E * N, D + 1).astype(np.float32)
            a = act_tanh_det(self.p, x)
        elif self.algo == 'td3':
            a = td3_act(self.p, obs.reshape(E * N, D))
        elif self.algo == 'ppo':
            a = ppo_det(self.p, obs.reshape(E * N, D))
        else:
            a = act_tanh_det(self.p, obs.reshape(E * N, D))
        return np.asarray(a).reshape(E, N, 2)


def run(pol, cfg, guard='std', calib=None, k=5, E=32, seed=0, collect=False, record=False, conf=None,
        max_steps=100000):
    env = MultiUAVEnv(cfg, E, seed, guard=guard, calib=calib, conf=conf, record_traj=record)
    env.collect = collect
    obs, cues = env.last, env.cues
    steps = 0
    while env.ep_count.min() < k and steps < max_steps:
        out = env.step(pol(obs, cues))
        obs, cues = out['obs'], out['cues']
        steps += 1
    eps = [r for r in env.completed if r['ep'] < k]
    legs = np.array([l for l in env.leg_log if l[13] < k]) if env.leg_log else np.zeros((0, 16))
    return env, eps, legs


# ---------------------------------------------------------------------- calibration
def conformal_quantile(scores, delta):
    s = np.sort(np.asarray(scores))
    n = len(s)
    r = int(math.ceil((n + 1) * (1 - delta)))
    return float(s[min(r, n) - 1]) if r <= n else float('inf')


def calibrate(pol, seed=7000, k=5, E=32, delta_k=0.10, delta_q=0.10, delta_l=0.05):
    """Split-conformal calibration on nominal (training-distribution) episodes.
    kappa_hat, q_hat: (1-delta) quantiles of the per-step power-residual ratio and headwind innovation;
    lam_hat: conformal (1-delta_l) quantile of return-leg scores actual / B_tilde."""
    env, _, _ = run(pol, make_cfg(), guard='std', k=2, E=E, seed=seed, collect=True)
    S = np.concatenate(env.score_buf)
    kap = float(np.quantile(S[:, 0], 1 - delta_k)); q = float(max(np.quantile(S[:, 1], 1 - delta_q), 0.0))
    calib0 = dict(kappa=kap, q=q, lam=1.0)
    _, _, legs = run(pol, make_cfg(), guard='static', calib=calib0, k=k, E=E, seed=seed + 1)
    act, bsta, dep = legs[:, 6], legs[:, 4], legs[:, 9]
    sc = np.where(dep > 0, np.inf, act / bsta)
    lam = conformal_quantile(sc, delta_l)
    return dict(kappa=kap, q=q, lam=lam, n_scores=int(len(S)), n_legs=int(len(sc))), sc, S


# ---------------------------------------------------------------------- metrics
def _cat(eps, key):
    return np.concatenate([r[key] for r in eps])


def mission_table(eps):
    """Per-mission arrays plus the episode index of each mission (bootstrap clusters)."""
    out = _cat(eps, 'out'); cl = np.concatenate([np.full(len(r['out']), j) for j, r in enumerate(eps)])
    return dict(out=out, cl=cl, energy=_cat(eps, 'energy') / 1e3, tdel=_cat(eps, 'tdel'), tend=_cat(eps, 'tend'),
                minair=_cat(eps, 'minair'), minbld=_cat(eps, 'minbld'), rth=_cat(eps, 'rth'), ret=_cat(eps, 'retsum'),
                E0=_cat(eps, 'E0') / 1e3, fault=_cat(eps, 'fault'))


def cvar_low(x, a=0.1):
    x = np.sort(x[np.isfinite(x)]); m = max(1, int(np.ceil(a * len(x))))
    return float(x[:m].mean())


def cvar_high(x, a=0.1):
    x = np.sort(x[np.isfinite(x)]); m = max(1, int(np.ceil(a * len(x))))
    return float(x[-m:].mean())


def point_metrics(M, legs=None, guard=None):
    out = M['out']
    succ = out == 1
    res = dict(n=int(len(out)), succ=succ.mean(), coll=((out == 3) | (out == 4)).mean(), coll_b=(out == 3).mean(),
               coll_a=(out == 4).mean(), dep=(out == 5).mean(), oob=(out == 6).mean(), tmo=(out == 7).mean(),
               safe_ret=(out == 2).mean(), rth=M['rth'].mean())
    res['loss'] = res['coll'] + res['dep'] + res['oob']
    trig = M['rth'].astype(bool)
    res['rth_ok'] = float((out[trig] == 2).mean()) if trig.any() else float('nan')
    res['energy'] = float(M['energy'].mean())
    res['energy_del'] = float(M['energy'][succ].mean()) if succ.any() else float('nan')
    res['energy_cvar'] = cvar_high(M['energy'], 0.1)
    td = M['tdel'][succ]
    res['tdel'] = float(td.mean()) if succ.any() else float('nan')
    res['tdel_cvar'] = cvar_high(td, 0.1) if succ.any() else float('nan')
    ma = np.minimum(M['minair'], 250.0)
    res['minsep'] = float(ma.mean()); res['minsep_q05'] = float(np.quantile(ma, 0.05))
    mb = np.minimum(M['minbld'], 250.0)
    res['minbld'] = float(mb.mean()); res['minbld_q05'] = float(np.quantile(mb, 0.05))
    res['ret'] = float(M['ret'].mean()); res['ret_cvar'] = cvar_low(M['ret'], 0.1)
    if legs is not None and len(legs) and guard is not None:
        col = GUARD_COL[guard]
        viol = (legs[:, 9] > 0) | (legs[:, 6] > legs[:, col])
        res['cov'] = float(1 - viol.mean()); res['n_legs'] = int(len(legs))
        ok = legs[:, 9] == 0
        res['tight'] = float(np.nanmean(legs[ok, col] / legs[ok, 6]))
    return {k: (float(v) if isinstance(v, (np.floating, np.bool_)) else v) for k, v in res.items()}


def bootstrap(M, keys=('succ', 'coll', 'dep', 'loss', 'energy', 'tdel', 'minsep_q05', 'ret_cvar'), B=400, seed=0):
    rng = np.random.default_rng(seed)
    cl = M['cl']; ucl = np.unique(cl); idx_by = [np.where(cl == c)[0] for c in ucl]
    vals = {k: [] for k in keys}
    for b in range(B):
        pick = rng.integers(0, len(ucl), len(ucl))
        ii = np.concatenate([idx_by[j] for j in pick])
        Mb = {k: v[ii] for k, v in M.items()}
        pm = point_metrics(Mb)
        for k in keys:
            vals[k].append(pm[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def merge_tables(tabs):
    out = {}
    off = 0
    for T in tabs:
        for k, v in T.items():
            vv = v + off if k == 'cl' else v
            out.setdefault(k, []).append(vv)
        off += int(T['cl'].max()) + 1
    return {k: np.concatenate(v) for k, v in out.items()}
