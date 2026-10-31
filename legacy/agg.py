"""Aggregate saved evaluation results (res/s*/NAME.npz) into pooled metrics with cluster-bootstrap CIs."""
import numpy as np, glob, os, json, sys
from evallib import point_metrics, GUARD_COL

KEYS = ('succ', 'coll', 'coll_b', 'coll_a', 'dep', 'oob', 'tmo', 'safe_ret', 'rth', 'loss', 'energy', 'energy_del', 'energy_cvar',
        'tdel', 'tdel_cvar', 'minsep', 'minsep_q05', 'minbld', 'minbld_q05', 'ret', 'ret_cvar', 'cov', 'tight', 'n', 'n_legs')


def load_table(name, seeds=(0, 1, 2), base='res'):
    tabs, legs, per = [], [], []
    off = 0
    for s in seeds:
        f = f'{base}/s{s}/{name}.npz'
        if not os.path.exists(f):
            continue
        z = np.load(f)
        M = {k: z[k] for k in ('out', 'cl', 'energy', 'tdel', 'tend', 'minair', 'minbld', 'rth', 'ret', 'E0', 'fault')}
        M['cl'] = M['cl'] + off; off = int(M['cl'].max()) + 1
        tabs.append(M); legs.append(z['legs']); per.append((s, M, z['legs']))
    if not tabs:
        return None, None, []
    M = {k: np.concatenate([t[k] for t in tabs]) for k in tabs[0]}
    L = np.concatenate([l for l in legs if len(l)]) if any(len(l) for l in legs) else np.zeros((0, 16))
    return M, L, per


def metrics(name, guard=None, seeds=(0, 1, 2), boot=0, base='res'):
    M, L, per = load_table(name, seeds, base)
    if M is None:
        return None
    g = guard or ('online' if '_online_' in name else 'std' if '_std_' in name else 'static' if '_static_' in name
                  else 'std15' if '_std15_' in name else 'online')
    pm = point_metrics(M, L, g)
    pm['seeds'] = [s for s, _, _ in per]
    pm['per_seed'] = {int(s): point_metrics(m, l, g) for s, m, l in per}
    if boot:
        rng = np.random.default_rng(1)
        cl = M['cl']; u = np.unique(cl); idx = [np.where(cl == c)[0] for c in u]
        bs = {k: [] for k in ('succ', 'coll', 'dep', 'loss', 'energy', 'tdel', 'minsep_q05', 'ret_cvar', 'rth', 'energy_cvar')}
        for b in range(boot):
            ii = np.concatenate([idx[j] for j in rng.integers(0, len(u), len(u))])
            p = point_metrics({k: v[ii] for k, v in M.items()})
            for k in bs:
                bs[k].append(p[k])
        pm['ci'] = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in bs.items()}
    return pm


if __name__ == '__main__':
    pat = sys.argv[1] if len(sys.argv) > 1 else 'main_*'
    seeds = tuple(int(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (0, 1, 2)
    names = sorted({os.path.basename(f)[:-4] for f in glob.glob(f'res/s*/{pat}.npz')})
    cols = ('succ', 'coll', 'coll_a', 'coll_b', 'dep', 'rth', 'loss', 'energy', 'energy_cvar', 'tdel', 'minsep_q05', 'minbld_q05', 'ret', 'ret_cvar', 'cov')
    print('%-34s' % 'name', ' '.join('%8s' % c[:8] for c in cols))
    for nm in names:
        pm = metrics(nm, seeds=seeds)
        print('%-34s' % nm, ' '.join('%8.3f' % pm[c] if c in pm else '%8s' % '-' for c in cols))
