"""Diagnostics of the risk-conditioned agents (revision, Reviewer 2 points 4d and 5a).

1. Sensitivity of the learned actor to its risk input: on states visited by the deployed policy, the commanded
   velocity is recomputed with the risk input set to each level in ALPHAS, everything else fixed.
2. Calibration of the catastrophe-cost critic of CREST-C: the stochastic training-time policy is rolled out with
   the selected base risk level; for every aircraft-step the mean and the upper-tail CVaR of the predicted
   discounted catastrophe cost are compared with the realised discounted cost-to-go of that aircraft.
usage: python3 critic_check.py --seed S [--arm fixed]   ->  res3/ARM/sS/critic_ALGO.npz
"""
import argparse, json, os
import numpy as np
import jax, jax.numpy as jnp
from evallib import load, Policy
from uavenv import MultiUAVEnv, make_cfg, VMAX
from rl import act_tanh_det, sample_tanh, mlp, NQ, cvar_upper_weights
from train import schedule, ALPHA_MIN
from rl import GAMMA_C

ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0]
ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--arm', default='fixed')
ap.add_argument('--resdir', default='res3')
A = ap.parse_args()
out = f'{A.resdir}/{A.arm}/s{A.seed}'
meta = json.load(open(f'{out}/meta.json'))
RUNS = 'runs' if A.arm == 'fixed' else 'runs_online'


@jax.jit
def cost_atoms(cps, x, a):
    oa = jnp.concatenate([x, a], -1)
    return jnp.sort(jnp.concatenate([mlp(cps[0], oa), mlp(cps[1], oa)], -1), -1)


@jax.jit
def act_stoch(ps, x, key):
    return sample_tanh(ps, x, key)


def sensitivity(algo, cond):
    ck = load(algo, A.seed, RUNS)
    ab = meta['ab'][algo]
    pol = Policy(ck, alpha_base=ab)
    env = MultiUAVEnv(make_cfg(*cond), 32, 60000 + A.seed, guard='std')
    obs, cues = env.last, env.cues
    X, H = [], []
    while env.ep_count.min() < 1:
        act = env.active.copy()
        X.append(obs[act]); H.append(cues[act])
        r = env.step(pol(obs, cues)); obs, cues = r['obs'], r['cues']
    X = np.concatenate(X); H = np.concatenate(H)
    rng = np.random.default_rng(A.seed)
    idx = rng.choice(len(X), min(6000, len(X)), replace=False); X, H = X[idx], H[idx]
    p = jax.tree_util.tree_map(jnp.asarray, ck['actor'])
    acts = []
    for al in ALPHAS:
        x = np.concatenate([X, np.full((len(X), 1), al, np.float32)], -1)
        acts.append(np.asarray(act_tanh_det(p, x)))
    acts = np.stack(acts)                                    # n_alpha x n x 2 (goal frame)
    spd = VMAX * np.minimum(np.linalg.norm(acts, axis=-1), 1.0)
    head = np.degrees(np.arctan2(acts[..., 1], acts[..., 0]))
    dhead = np.abs((head - head[-1] + 180.0) % 360.0 - 180.0)
    return dict(spd=spd, dhead=dhead, hc=H[:, 0], he=H[:, 1])


def calibration(algo, cond, k=2):
    ck = load(algo, A.seed, RUNS)
    ab = meta['ab'][algo]
    p = jax.tree_util.tree_map(jnp.asarray, ck['actor']); cps = jax.tree_util.tree_map(jnp.asarray, ck['cost_critic'])
    env = MultiUAVEnv(make_cfg(*cond), 32, 61000 + A.seed, guard='std')
    obs, cues = env.last, env.cues
    key = jax.random.PRNGKey(1000 + A.seed)
    E, N = env.E, env.N
    traj = {}            # (env, uav, ep) -> list of (step index, mean, cvar10, cvar_alpha, cost)
    recs = []
    open_ = {}
    while env.ep_count.min() < k:
        al = schedule(cues, ab).astype(np.float32)
        x = np.concatenate([obs, al[..., None]], -1).reshape(E * N, -1).astype(np.float32)
        key, k1 = jax.random.split(key)
        a, _ = act_stoch(p, x, k1)
        atoms = np.asarray(cost_atoms(cps, x, a))
        n = atoms.shape[1]
        w10 = np.asarray(cvar_upper_weights(jnp.full((E * N,), 0.1), n))
        wal = np.asarray(cvar_upper_weights(jnp.asarray(al.reshape(-1)), n))
        mean_c = atoms.mean(-1); cv10 = (atoms * w10).sum(-1); cval = (atoms * wal).sum(-1)
        act = env.active.copy().reshape(-1); epc = env.ep_count.copy()
        r = env.step(np.asarray(a).reshape(E, N, 2))
        cost = r['cost'].reshape(-1); term = r['term'].reshape(-1); trunc = r['trunc'].reshape(-1)
        for j in np.where(act)[0]:
            e, i = divmod(j, N)
            kk = (e, i, int(epc[e]))
            open_.setdefault(kk, []).append((mean_c[j], cv10[j], cval[j], al.reshape(-1)[j], cost[j]))
            if term[j] or trunc[j] or r['reset'][e]:
                seq = open_.pop(kk)
                if trunc[j] and not term[j]:
                    continue                      # truncated missions have no realised terminal outcome
                c = np.array([s[4] for s in seq])
                g = np.zeros(len(c)); acc = 0.0
                for t in range(len(c) - 1, -1, -1):
                    acc = c[t] + GAMMA_C * acc; g[t] = acc
                for t, s in enumerate(seq):
                    recs.append((s[0], s[1], s[2], s[3], g[t], float(c[t:min(len(c), t + 10)].sum() > 0)))
        obs, cues = r['obs'], r['cues']
    return np.array(recs)


res = {}
for algo in ('crest', 'crestret'):
    if not os.path.exists(f'{RUNS}/{algo}_s{A.seed}.pkl') or algo not in meta['ab']:
        continue
    for cn, cond in [('nominal', ((), 0.0)), ('traffic', (('traffic',), 1.0))]:
        S = sensitivity(algo, cond)
        for k2, v in S.items():
            res[f'sens_{algo}_{cn}_{k2}'] = v
        if algo == 'crest':
            R = calibration(algo, cond)
            res[f'cal_{algo}_{cn}'] = R
        print(algo, cn, 'speed by alpha', np.round(S['spd'].mean(1), 2), 'mean |dheading| vs alpha=1', np.round(S['dhead'].mean(1), 2), flush=True)
np.savez_compressed(f'{out}/critic_check.npz', alphas=np.array(ALPHAS), **res)
open(f'{out}/done_critic.txt', 'w').write('ok\n')
