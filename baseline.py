"""Non-learning reference controller (revision).

A reactive goal-seeking controller with wind compensation, repulsion from nearby buildings and a
closest-point-of-approach (velocity-obstacle style) deflection from observed aircraft.  It acts on exactly
the same 59-dimensional goal-frame observation as the learned policies, so it sees the same estimated
positions, the same four nearest buildings and the same four nearest detected aircraft.  Its six
parameters are selected on nominal validation districts (never on test conditions) by maximising the
mean mission return, the objective the learning agents are trained on.
"""
import numpy as np
from uavenv import KN, MN, VMAX

S0 = 15                     # size of the own-state block of the observation
DEFAULT = dict(U=14.0, cb=45.0, kb=1.6, ds=35.0, ka=1.6, tau=8.0)


class ReactiveController:
    algo = 'reactive'

    def __init__(self, **kw):
        self.p = dict(DEFAULT); self.p.update(kw)

    def __call__(self, obs, cues=None):
        p = self.p
        E, N, _ = obs.shape
        dg = obs[..., 0] * 1000.0                                   # distance to goal (m), goal along +x
        west = obs[..., 6:8] * 10.0                                  # wind estimate in the goal frame
        vg = obs[..., 4:6] * 20.0                                    # (noisy) ground velocity in the goal frame
        U = np.minimum(p['U'], np.maximum(3.0, 0.6 * dg))
        v = np.zeros((E, N, 2)); v[..., 0] = U                        # desired ground velocity towards the goal
        # buildings: radial repulsion plus a tangential slide towards the goal side
        ob = obs[..., S0:S0 + 4 * KN].reshape(E, N, KN, 4)
        rel = ob[..., :2] * 200.0; clr = ob[..., 3] * 200.0
        d = np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-6); u = rel / d
        w = np.clip(1.0 - clr / p['cb'], 0.0, 1.0)[..., None] ** 2
        tang = np.stack([-u[..., 1], u[..., 0]], -1)
        side = np.sign(tang[..., 0:1] + 1e-9)                         # choose the tangent that keeps progress
        v += p['kb'] * p['U'] * (w * (-u + 0.8 * side * tang)).sum(-2)
        # aircraft: deflect away from the predicted closest-approach point within the horizon
        air = obs[..., S0 + 4 * KN:].reshape(E, N, MN, 7)
        ra = air[..., :2] * 200.0; rv = air[..., 2:4] * 20.0; cl = air[..., 4] * 200.0; pres = air[..., 6] > 0.5
        vv = (rv * rv).sum(-1) + 1e-6
        tc = np.clip(-(ra * rv).sum(-1) / vv, 0.0, p['tau'])
        mpt = ra + rv * tc[..., None]
        dm = np.linalg.norm(mpt, axis=-1)
        thr = np.clip(1.0 - dm / p['ds'], 0.0, 1.0) * (1.0 - tc / (p['tau'] + 1e-6)) * pres
        near = np.clip(1.0 - cl / (0.5 * p['ds']), 0.0, 1.0) * pres
        dirm = -mpt / np.maximum(dm, 1e-6)[..., None]
        dirn = -ra / np.maximum(np.linalg.norm(ra, axis=-1), 1e-6)[..., None]
        v += p['ka'] * p['U'] * ((thr[..., None] * dirm) + (near[..., None] * dirn)).sum(-2)
        # convert the desired ground velocity into a commanded air velocity and scale to the action disc
        va = v - west
        a = va / VMAX
        n = np.linalg.norm(a, axis=-1, keepdims=True)
        return np.where(n > 1.0, a / np.maximum(n, 1e-9), a)
