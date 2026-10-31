"""Qualitative data for the district map and the risk-schedule trace."""
import numpy as np, json
from evallib import *

seed = 0
meta = json.load(open(f'res/s{seed}/meta.json')); CAL = meta['calib']; AB = meta['ab']['crest']
out = {}
for m, guard in [('sac', 'std'), ('crest', 'online')]:
    ck = load(m, seed)
    pol = Policy(ck, alpha_base=AB) if m == 'crest' else Policy(ck)
    pol.trace = [] if m == 'crest' else None
    env = MultiUAVEnv(make_cfg(('sensor',), 1.0), n_env=1, seed=4242, guard=guard, calib=CAL[m], record_traj=True)
    obs, cues, cue_log = env.last, env.cues, []
    while env.ep_count[0] < 1:
        cue_log.append(cues[0].copy())
        r = env.step(pol(obs, cues)); obs, cues = r['obs'], r['cues']
    T = len(env.traj)
    pos = np.stack([t['pos'][0] for t in env.traj])          # T x N x 2
    ipos = np.stack([t['ipos'][0] for t in env.traj])
    act = np.stack([t['active'][0] for t in env.traj])
    rec = env.completed[0]
    out[m] = dict(pos=pos, ipos=ipos, act=act, out=rec['out'], cues=np.stack(cue_log),
                  alpha=(np.stack(pol.trace)[:, 0, :] if pol.trace else None))
    if m == 'sac':
        np.savetxt('pa/data/map_obs.txt', np.concatenate([env.oc[0], env.orad[0][:, None]], -1), fmt='%.2f')
        np.savetxt('pa/data/map_pads.txt', env.pads[0], fmt='%.2f')
        np.savetxt('pa/data/map_goal.txt', env.cust[0], fmt='%.2f')
    np.savetxt(f'pa/data/map_pos_{m}.txt', pos.reshape(T, -1), fmt='%.2f')
    np.savetxt(f'pa/data/map_int_{m}.txt', ipos.reshape(T, -1), fmt='%.2f')
    np.savetxt(f'pa/data/map_act_{m}.txt', act.astype(float), fmt='%d')
    np.savetxt(f'pa/data/map_out_{m}.txt', rec['out'][None, :], fmt='%d')
    print(m, 'steps', T, 'outcomes', rec['out'])
c = out['crest']
if c['alpha'] is not None:
    n = min(len(c['alpha']), len(c['cues']))
    k = int(np.argmax((c['alpha'][:n].max(0) - c['alpha'][:n].min(0))))   # most modulated aircraft
    M = np.concatenate([c['cues'][:n, k, :], c['alpha'][:n, k:k+1]], -1)
    np.savetxt('pa/data/alpha_trace.txt', M, fmt='%.4f')
    print('alpha trace uav', k, 'steps', n, 'range', c['alpha'][:n, k].min(), c['alpha'][:n, k].max())
