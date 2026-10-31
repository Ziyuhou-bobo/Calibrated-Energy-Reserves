"""Vectorised stochastic simulator for multi-UAV urban parcel delivery.

Each environment instance is a 1.6 km x 1.6 km urban district with landing pads, building
towers that penetrate the flight layer, cooperative delivery UAVs and non-cooperative
intruders.  Uncertainty sources: mean wind, Dryden-type turbulence, discrete 1-cosine gusts,
GNSS error with multipath outliers, noisy wind/intruder sensing, residual battery energy,
payload mass and in-flight battery/propulsion faults.

Rotor power follows the rotary-wing model of Zeng, Xu & Zhang (IEEE TWC 2019) with
momentum-theory weight scaling of the induced term.
"""
import numpy as np

# ---------------------------------------------------------------- physics constants
P0, PI0, UTIP, V0H, D0, RHO, SOL, AREA, M0 = 79.86, 88.63, 120.0, 4.03, 0.6, 1.225, 0.05, 0.503, 2.0
PARA = 0.5 * D0 * RHO * SOL * AREA
R_UAV, R_INT = 1.5, 2.0          # collision radii (m)
VMAX, AMAX, VC = 20.0, 6.0, 14.0  # max airspeed, max accel, nominal return cruise airspeed
PHI, T_LAND, T_DROP = 1.15, 8.0, 10.0
VCAND = np.array([8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
K_TURB, ETA_ACC = 0.10, 0.75
R_DELIVER, R_LAND = 12.0, 15.0
DT = 1.0
KN, MN = 4, 4                     # nearest static obstacles / aircraft in observation
D_SAFE = 25.0
OBS_DIM = 15 + 4 * KN + 7 * MN
SENSE_RANGE = 250.0


def rotor_power(v, m):
    """Electrical power (W) of a rotary-wing UAV at airspeed v (m/s) and mass m (kg)."""
    w = m / M0
    pi = PI0 * w ** 1.5
    x = v * v / (2.0 * (V0H ** 2) * w)
    induced = pi * np.sqrt(1.0 / (np.sqrt(1.0 + x * x) + x))
    return P0 * (1.0 + 3.0 * v * v / UTIP ** 2) + induced + PARA * v ** 3


BASE = dict(L=1600.0, n_pads=4, n_obs=30, r_obs=(15.0, 40.0), n_uav=6, n_int=4,
            int_speed=(4.0, 10.0), int_turn_p=0.01,
            wind_mean=(0.0, 6.0), turb_scale=0.17, turb_base=0.3, turb_tau=10.0,
            gust_rate=1.0 / 150.0, gust_amp=(3.0, 8.0), gust_dur=(4.0, 10.0),
            gnss_sigma=2.0, gnss_tau=20.0, mp_prob=0.005, mp_mag=(8.0, 20.0), mp_dur=(3.0, 10.0),
            wind_meas_sigma=1.0, int_miss=0.05, int_noise=0.02,
            batt=(24e3, 50e3), payload=(0.2, 1.2), fault_rate=1.0 / 3000.0,
            fault_mult=(1.25, 1.6), fault_drop=(0.05, 0.15),
            mission=(600.0, 1100.0), T_max=300)

SHIFTS = {
    'wind': dict(wind_mean=(5.0, 10.0), turb_scale=0.25, gust_rate=1.0 / 60.0, gust_amp=(5.0, 10.0)),
    'sensor': dict(gnss_sigma=5.0, mp_prob=0.03, mp_mag=(15.0, 40.0), wind_meas_sigma=2.0,
                   int_miss=0.2, int_noise=0.05),
    'traffic': dict(n_uav=12, n_int=10, int_speed=(6.0, 14.0), int_turn_p=0.03),
    'energy': dict(batt=(18e3, 40e3), payload=(1.0, 1.8), fault_rate=1.0 / 750.0),
}
FAMILIES = ('wind', 'sensor', 'traffic', 'energy')


def make_cfg(families=(), s=0.0):
    cfg = dict(BASE)
    for f in families:
        for k, v1 in SHIFTS[f].items():
            v0 = BASE[k]
            if isinstance(v0, tuple):
                cfg[k] = tuple(a + s * (b - a) for a, b in zip(v0, v1))
            elif isinstance(v0, int):
                cfg[k] = int(round(v0 + s * (v1 - v0)))
            else:
                cfg[k] = v0 + s * (v1 - v0)
    return cfg


def _unit(th):
    return np.stack([np.cos(th), np.sin(th)], -1)


class MultiUAVEnv:
    """guard in {'std', 'std15', 'static', 'online'} selects the return-to-home trigger."""

    def __init__(self, cfg, n_env, seed, guard='std', calib=None, conf=None, record_traj=False):
        self.cfg, self.E = cfg, n_env
        self.N, self.NI, self.K, self.P, self.L = cfg['n_uav'], cfg['n_int'], cfg['n_obs'], cfg['n_pads'], cfg['L']
        self.rng = np.random.default_rng(seed)
        self.seed_base = int(seed)
        self.guard = guard
        self.calib = dict(kappa=1.20, q=1.6, lam=1.0) if calib is None else dict(calib)
        self.calib.setdefault('lam', 1.0)
        c = dict(eta_k=0.02, delta_k=0.10, eta_q=0.15, delta_q=0.10, eta_l=0.05, delta_l=0.05)
        if conf is not None:
            c.update(conf)
        self.conf = c
        self.lam = np.full(n_env, float(self.calib['lam']))   # fleet-level leg calibrator, persists across episodes
        self.leg_log = []
        self.collect = False
        self.score_buf = []
        self.ep_count = np.zeros(n_env, np.int64)
        E, N, NI, K, P = self.E, self.N, self.NI, self.K, self.P
        self.pads = np.zeros((E, P, 2)); self.oc = np.zeros((E, K, 2)); self.orad = np.zeros((E, K))
        self.pos = np.zeros((E, N, 2)); self.vair = np.zeros((E, N, 2)); self.goal = np.zeros((E, N, 2))
        self.cust = np.zeros((E, N, 2)); self.phase = np.zeros((E, N), np.int8); self.active = np.zeros((E, N), bool)
        self.Erem = np.zeros((E, N)); self.E0 = np.zeros((E, N)); self.payload = np.zeros((E, N)); self.fmult = np.ones((E, N))
        self.wbar = np.zeros((E, 2)); self.turb = np.zeros((E, N, 2)); self.gust_t = np.full((E, N), -1.0)
        self.gust_T = np.ones((E, N)); self.gust_A = np.zeros((E, N, 2)); self.wind = np.zeros((E, N, 2))
        self.gerr = np.zeros((E, N, 2)); self.mp_off = np.zeros((E, N, 2)); self.mp_left = np.zeros((E, N))
        self.west = np.zeros((E, N, 2)); self.wvar = np.zeros((E, N))
        self.kappa = np.zeros((E, N)); self.qw = np.zeros((E, N))
        self.ipos = np.zeros((E, NI, 2)); self.ivel = np.zeros((E, NI, 2)); self.ispd = np.zeros((E, NI))
        self.t = np.zeros(E, np.int32)
        self.m_out = np.zeros((E, N), np.int8); self.m_tdel = np.full((E, N), np.nan); self.m_tend = np.full((E, N), np.nan)
        self.m_energy = np.zeros((E, N)); self.m_minair = np.full((E, N), 1e4); self.m_minbld = np.full((E, N), 1e4)
        self.m_rth = np.zeros((E, N), bool); self.m_ret = np.full((E, N, 9), np.nan); self.m_retsum = np.zeros((E, N))
        self.m_fault = np.zeros((E, N), bool); self.m_rthtime = np.full((E, N), np.nan)
        self.payload0 = np.zeros((E, N)); self.m_clipk = np.zeros((E, N)); self.m_nstep = np.zeros((E, N))
        self.completed = []
        self.record_traj = record_traj
        self.traj = []
        self.episode_seed_counter = 0
        self.reset_all()

    # ------------------------------------------------------------------ layout
    def _layout(self, e, rng):
        L, cfg, N, NI, K, P = self.L, self.cfg, self.N, self.NI, self.K, self.P
        pads = []
        while len(pads) < P:
            p = rng.uniform(150, L - 150, 2)
            if all(np.linalg.norm(p - q) > 450 for q in pads):
                pads.append(p)
        pads = np.array(pads)
        obs, rads, tries = [], [], 0
        while len(obs) < K:
            tries += 1
            c = rng.uniform(0, L, 2); r = rng.uniform(*cfg['r_obs'])
            if tries > 20000:  # fallback: park an obstacle outside the district
                obs.append(np.array([-5000.0 - 100 * len(obs), -5000.0])); rads.append(10.0); continue
            if np.min(np.linalg.norm(pads - c, axis=1)) < r + 70:
                continue
            if obs and np.min(np.linalg.norm(np.array(obs) - c, axis=1) - np.array(rads)) < r + 25:
                continue
            obs.append(c); rads.append(r)
        obs, rads = np.array(obs), np.array(rads)
        sp = rng.integers(0, P, N)
        starts = np.zeros((N, 2))
        base_ang = rng.uniform(0, 2 * np.pi, P)
        for pidx in range(P):
            ids = np.where(sp == pidx)[0]
            for k, i in enumerate(ids):
                ang = base_ang[pidx] + 2 * np.pi * k / max(len(ids), 1)
                starts[i] = pads[pidx] + 30.0 * np.array([np.cos(ang), np.sin(ang)])
        cust = np.zeros((N, 2))
        for i in range(N):
            for tr in range(400):
                c = rng.uniform(150, L - 150, 2)
                d = np.linalg.norm(c - starts[i])
                lo, hi = cfg['mission'] if tr < 300 else (400.0, cfg['mission'][1])
                if d < lo or d > hi:
                    continue
                if np.min(np.linalg.norm(obs - c, axis=1) - rads) < 30:
                    continue
                if np.min(np.linalg.norm(pads - c, axis=1)) < 200:
                    continue
                break
            cust[i] = c
        ipos = np.zeros((NI, 2))
        for j in range(NI):
            for tr in range(200):
                q = rng.uniform(50, L - 50, 2)
                if np.min(np.linalg.norm(starts - q, axis=1)) > 200:
                    break
            ipos[j] = q
        self.pads[e], self.oc[e], self.orad[e] = pads, obs, rads
        self.pos[e], self.cust[e], self.goal[e] = starts, cust, cust
        self.ipos[e] = ipos
        spd = rng.uniform(*cfg['int_speed'], NI)
        self.ispd[e] = spd
        self.ivel[e] = spd[:, None] * _unit(rng.uniform(0, 2 * np.pi, NI))

    def _reset_envs(self, idx):
        # every episode-level quantity (layout, wind mean, battery, payload, initial turbulence) is drawn
        # from a generator seeded by (seed, env, episode index) so that different policies face the
        # same k-th scenario in each environment slot (common random numbers)
        cfg, N = self.cfg, self.N
        for e in idx:
            r = np.random.default_rng((self.seed_base, int(e), int(self.ep_count[e])))
            self._layout(e, r)
            self.wbar[e] = r.uniform(*cfg['wind_mean']) * _unit(r.uniform(0, 2 * np.pi))
            sig = cfg['turb_base'] + cfg['turb_scale'] * np.linalg.norm(self.wbar[e])
            self.turb[e] = sig * r.standard_normal((N, 2))
            self.E0[e] = r.uniform(*cfg['batt'], N)
            self.payload[e] = r.uniform(*cfg['payload'], N)
            self.gerr[e] = cfg['gnss_sigma'] * r.standard_normal((N, 2))
        sig_t = cfg['turb_base'] + cfg['turb_scale'] * np.linalg.norm(self.wbar[idx], axis=-1)
        self.gust_t[idx] = -1.0; self.gust_A[idx] = 0.0
        self.wind[idx] = self.wbar[idx][:, None, :] + self.turb[idx]
        self.vair[idx] = -self.wind[idx]
        self.phase[idx] = 0; self.active[idx] = True
        self.Erem[idx] = self.E0[idx]
        self.fmult[idx] = 1.0
        rng = self.rng
        n = len(idx)
        self.mp_off[idx] = 0.0; self.mp_left[idx] = 0.0
        self.west[idx] = self.wind[idx] + cfg['wind_meas_sigma'] * rng.standard_normal((n, N, 2))
        self.wvar[idx] = (sig_t ** 2)[:, None] + cfg['wind_meas_sigma'] ** 2
        self.kappa[idx] = self.calib['kappa']; self.qw[idx] = self.calib['q']
        self.t[idx] = 0
        self.m_out[idx] = 0; self.m_tdel[idx] = np.nan; self.m_tend[idx] = np.nan; self.m_energy[idx] = 0.0
        self.m_minair[idx] = 1e4; self.m_minbld[idx] = 1e4; self.m_rth[idx] = False; self.m_ret[idx] = np.nan
        self.m_retsum[idx] = 0.0; self.m_fault[idx] = False; self.m_rthtime[idx] = np.nan
        self.payload0[idx] = self.payload[idx]; self.m_clipk[idx] = 0.0; self.m_nstep[idx] = 0.0

    def reset_all(self):
        self._reset_envs(np.arange(self.E))
        self.last = self._observe()
        return self.last

    # ------------------------------------------------------------------ guards
    def _bounds(self, phat):
        rel = self.pads[:, None, :, :] - phat[:, :, None, :]
        d = np.linalg.norm(rel, axis=-1)
        ip = np.argmin(d, -1)
        dmin = np.take_along_axis(d, ip[..., None], -1)[..., 0]
        rp = np.take_along_axis(rel, ip[..., None, None], -2)[..., 0, :]
        e = rp / np.maximum(dmin, 1e-6)[..., None]
        eperp = np.stack([-e[..., 1], e[..., 0]], -1)
        mass = M0 + self.payload
        p_h = rotor_power(np.zeros_like(mass), mass)

        def enom(weff):
            # nominal return energy with the energy-optimal airspeed for the estimated wind
            along = (weff * e).sum(-1)[..., None]; cross = (weff * eperp).sum(-1)[..., None]
            v = VCAND[None, None, :]
            g = np.maximum(along + np.sqrt(np.maximum(v * v - cross * cross, 0.0)), 1.0)
            en = rotor_power(v, mass[..., None]) * PHI * dmin[..., None] / g
            return en.min(-1)

        land = p_h * T_LAND
        e_w = enom(self.west)
        lam_s, kap_s = self.calib['lam'], self.calib['kappa']
        bt_sta = kap_s * enom(self.west - self.calib['q'] * e) + land
        bt_onl = self.kappa * enom(self.west - self.qw[..., None] * e) + land
        lam_o = self.lam[:, None]
        # committed drop reserve: hover with payload at the customer in the estimated wind
        p_drop = rotor_power(np.linalg.norm(self.west, axis=-1), mass) * T_DROP
        bnd = dict(std=1.2 * e_w + land, std15=1.5 * e_w + land, static=lam_s * bt_sta, online=lam_o * bt_onl)
        drop = dict(std=1.2 * p_drop, std15=1.5 * p_drop, static=lam_s * kap_s * p_drop, online=lam_o * self.kappa * p_drop)
        if 'margin' in self.calib:   # generic fixed-margin guard used for the Pareto study
            bnd['fixed'] = self.calib['margin'] * e_w + land; drop['fixed'] = self.calib['margin'] * p_drop
        return dict(bnd, drop=drop, bt_online=bt_onl, e=e, d=dmin, ip=ip)

    def _trigger_bound(self, B):
        g = self.guard
        return B[g] + np.where(self.phase == 0, B['drop'][g], 0.0)

    def _leg_record(self, B):
        return np.stack([self.Erem, B['std'], B['static'], B['online'], np.full_like(self.Erem, np.nan), B['std15'],
                         B['bt_online'], np.zeros_like(self.Erem), B[self.guard]], -1)

    def _close_legs(self, done_mask, depleted_mask):
        """Leg-level online conformal update of the fleet calibrator lambda (one update per completed leg)."""
        if not done_mask.any():
            return
        c = self.conf
        for e, i in zip(*np.where(done_mask)):
            rec = self.m_ret[e, i]
            if np.isnan(rec[0]):
                continue
            dep = bool(depleted_mask[e, i])
            err = 1.0 if dep else float(rec[4] > rec[3])
            lam_before = self.lam[e]
            # fleet-level calibrator shared by all UAVs and all parallel districts operating in the same regime
            self.lam[:] = float(np.clip(self.lam[e] + c['eta_l'] * (err - c['delta_l']), 0.5, 4.0))
            self.leg_log.append((int(e), int(self.t[e]), rec[0], rec[1], rec[2], rec[3], rec[4], rec[5], rec[6],
                                 float(dep), err, lam_before, self.lam[e], int(self.ep_count[e]), int(i), rec[8]))

    # ------------------------------------------------------------------ observation
    def _observe(self):
        E, N, NI, cfg, rng = self.E, self.N, self.NI, self.cfg, self.rng
        phat = self.pos + self.gerr + self.mp_off
        B = self._bounds(phat)
        self.B = B
        bact = self._trigger_bound(B)
        margin = (self.Erem - bact) / np.maximum(bact, 1.0)
        relg = self.goal - phat
        dg = np.linalg.norm(relg, axis=-1) + 1e-6
        ex = relg / dg[..., None]; ey = np.stack([-ex[..., 1], ex[..., 0]], -1)
        self.ex, self.ey = ex, ey

        def rot(v, exb, eyb):
            return np.stack([(v * exb).sum(-1), (v * eyb).sum(-1)], -1)
        vg = self.vair + self.wind
        vgh = vg + 0.5 * rng.standard_normal((E, N, 2))
        relp = B['e'] * B['d'][..., None]
        # static obstacles (map-based: inherits GNSS error)
        relo = self.oc[:, None, :, :] - phat[:, :, None, :]
        co = np.linalg.norm(relo, axis=-1) - self.orad[:, None, :] - R_UAV
        io = np.argpartition(co, KN, axis=-1)[..., :KN]
        order = np.argsort(np.take_along_axis(co, io, -1), -1)
        io = np.take_along_axis(io, order, -1)
        ro = np.take_along_axis(relo, io[..., None], -2)
        rr = np.take_along_axis(np.broadcast_to(self.orad[:, None, :], co.shape), io, -1)
        cs = np.take_along_axis(co, io, -1)
        # aircraft: cooperative UAVs via broadcast estimates, intruders via onboard sensing
        relu = phat[:, None, :, :] - phat[:, :, None, :]
        relvu = vgh[:, None, :, :] - vgh[:, :, None, :]
        validu = self.active[:, None, :] & ~np.eye(N, dtype=bool)[None]
        reli_t = self.ipos[:, None, :, :] - self.pos[:, :, None, :]
        rng_i = np.linalg.norm(reli_t, axis=-1)
        reli = reli_t + (1.0 + cfg['int_noise'] * rng_i)[..., None] * rng.standard_normal((E, N, NI, 2))
        relvi = self.ivel[:, None, :, :] - vg[:, :, None, :] + 0.5 * rng.standard_normal((E, N, NI, 2))
        validi = (rng_i < SENSE_RANGE) & (rng.random((E, N, NI)) > cfg['int_miss'])
        rela = np.concatenate([relu, reli], 2); relva = np.concatenate([relvu, relvi], 2)
        valida = np.concatenate([validu, validi], 2)
        rada = np.concatenate([np.full(N, R_UAV), np.full(NI, R_INT)])
        isu = np.concatenate([np.ones(N), np.zeros(NI)])
        ca = np.linalg.norm(rela, axis=-1) - R_UAV - rada[None, None, :]
        ca = np.where(valida, ca, 1e6)
        M = MN
        ia = np.argpartition(ca, M, axis=-1)[..., :M]
        order = np.argsort(np.take_along_axis(ca, ia, -1), -1)
        ia = np.take_along_axis(ia, order, -1)
        ra = np.take_along_axis(rela, ia[..., None], -2)
        rva = np.take_along_axis(relva, ia[..., None], -2)
        cla = np.take_along_axis(ca, ia, -1)
        pres = cla < 1e5
        radsel = rada[ia]
        # ------------------------------------------------ hazard cues
        exb, eyb = ex[:, :, None, :], ey[:, :, None, :]
        tau_h = 10.0

        def threat(r, v, rad, clr, present):
            vv = (v * v).sum(-1) + 1e-6
            tc = np.clip(-(r * v).sum(-1) / vv, 0.0, tau_h)
            dc = np.linalg.norm(r + v * tc[..., None], axis=-1) - rad - R_UAV
            th = (1.0 - tc / tau_h) * np.clip(1.0 - dc / 30.0, 0.0, 1.0)
            px = np.clip(1.0 - clr / 40.0, 0.0, 1.0)
            return np.where(present, np.maximum(th, px), 0.0).max(-1)
        vst = -np.broadcast_to(vgh[:, :, None, :], ro.shape)
        h_c = np.maximum(threat(ro, vst, rr, cs, np.ones_like(cs, bool)), threat(ra, rva, radsel, cla, pres))
        h_d = np.clip((valida & (ca < 150.0)).sum(-1) / 4.0, 0.0, 1.0)
        sig_hat = np.sqrt(self.wvar)
        h_w = np.clip((sig_hat - 1.0) / 3.0, 0.0, 1.0)
        h_e = np.clip((0.5 - margin) / 0.5, 0.0, 1.0)
        cues = np.stack([h_c, h_e, h_d, h_w], -1)
        # ------------------------------------------------ features
        tfrac = (self.t / self.cfg['T_max'])[:, None] * np.ones((1, N))
        selff = np.concatenate([
            np.clip(dg / 1000.0, 0, 2)[..., None], (self.phase >= 2).astype(float)[..., None],
            rot(self.vair, ex, ey) / 20.0, rot(vgh, ex, ey) / 20.0, rot(self.west, ex, ey) / 10.0,
            (sig_hat / 5.0)[..., None], (self.Erem / 50e3)[..., None], np.clip(margin, -1, 3)[..., None],
            (self.payload / 2.0)[..., None], np.clip(rot(relp, ex, ey) / 1000.0, -2, 2), tfrac[..., None]], -1)
        obsf = np.concatenate([np.clip(rot(ro, exb, eyb) / 200.0, -2, 2), (rr / 50.0)[..., None],
                               np.clip(cs / 200.0, -1, 2)[..., None]], -1).reshape(E, N, 4 * KN)
        airf = np.concatenate([np.clip(rot(ra, exb, eyb) / 200.0, -2, 2), np.clip(rot(rva, exb, eyb) / 20.0, -2, 2),
                               np.clip(cla / 200.0, -1, 2)[..., None], isu[ia][..., None], pres[..., None].astype(float)], -1)
        airf = np.where(pres[..., None], airf, np.array([0, 0, 0, 0, 2.0, 0, 0]))
        airf = airf.reshape(E, N, 7 * MN)
        obs = np.concatenate([selff, obsf, airf], -1).astype(np.float32)
        self.cues = cues.astype(np.float32)
        self.margin = margin
        return obs

    # ------------------------------------------------------------------ step
    def step(self, act):
        E, N, NI, cfg, rng = self.E, self.N, self.NI, self.cfg, self.rng
        act = np.clip(np.asarray(act, dtype=np.float64), -1.0, 1.0)
        nrm = np.linalg.norm(act, axis=-1, keepdims=True)
        act = np.where(nrm > 1.0, act / np.maximum(nrm, 1e-9), act)
        act = act[..., 0:1] * self.ex + act[..., 1:2] * self.ey   # goal frame -> world frame
        was = self.active.copy()
        self.t += 1
        d_prev = np.linalg.norm(self.goal - self.pos, axis=-1)
        # ---- wind
        a = np.exp(-DT / cfg['turb_tau'])
        sig_t = cfg['turb_base'] + cfg['turb_scale'] * np.linalg.norm(self.wbar, axis=-1)
        self.turb = a * self.turb + (sig_t * np.sqrt(1 - a * a))[:, None, None] * rng.standard_normal((E, N, 2))
        start = (self.gust_t < 0) & (rng.random((E, N)) < cfg['gust_rate'] * DT)
        if start.any():
            amp = rng.uniform(*cfg['gust_amp'], (E, N))
            self.gust_A = np.where(start[..., None], amp[..., None] * _unit(rng.uniform(0, 2 * np.pi, (E, N))), self.gust_A)
            self.gust_T = np.where(start, rng.uniform(*cfg['gust_dur'], (E, N)), self.gust_T)
            self.gust_t = np.where(start, 0.0, self.gust_t)
        on = self.gust_t >= 0
        gmag = np.where(on, 0.5 * (1.0 - np.cos(2 * np.pi * np.clip(self.gust_t / self.gust_T, 0, 1))), 0.0)
        wdev = self.turb + gmag[..., None] * self.gust_A
        self.gust_t = np.where(on, self.gust_t + DT, self.gust_t)
        self.gust_t = np.where(self.gust_t > self.gust_T, -1.0, self.gust_t)
        self.wind = self.wbar[:, None, :] + wdev
        # ---- UAV kinematics
        vold = self.vair.copy()
        dv = VMAX * act - self.vair
        dvn = np.linalg.norm(dv, axis=-1, keepdims=True)
        dv = dv * np.minimum(1.0, AMAX * DT / np.maximum(dvn, 1e-9))
        wm = was[..., None]
        self.vair = np.where(wm, self.vair + dv, self.vair)
        p_old = self.pos.copy()
        self.pos = np.where(wm, self.pos + (self.vair + self.wind) * DT, self.pos)
        # ---- intruders (non-cooperative, random-walk heading, rare sharp turns)
        i_old = self.ipos.copy()
        th = np.arctan2(self.ivel[..., 1], self.ivel[..., 0]) + 0.05 * rng.standard_normal((E, NI))
        sharp = rng.random((E, NI)) < cfg['int_turn_p']
        th = th + np.where(sharp, rng.choice([-1.0, 1.0], (E, NI)) * rng.uniform(np.pi / 4, np.pi / 2, (E, NI)), 0.0)
        self.ivel = self.ispd[..., None] * _unit(th)
        self.ipos = self.ipos + self.ivel * DT
        for k in range(2):
            lo = self.ipos[..., k] < 0; hi = self.ipos[..., k] > self.L
            self.ivel[..., k] = np.where(lo | hi, -self.ivel[..., k], self.ivel[..., k])
            self.ipos[..., k] = np.clip(self.ipos[..., k], 0, self.L)
        # ---- energy
        mass = M0 + self.payload
        spd = np.linalg.norm(self.vair, axis=-1)
        pnom = rotor_power(spd, mass)
        sur = 1.0 + K_TURB * np.linalg.norm(wdev, axis=-1) / 5.0
        kin = 0.5 * mass * (spd ** 2 - np.linalg.norm(vold, axis=-1) ** 2)
        pacc = np.maximum(kin, 0.0) / ETA_ACC / DT
        fault = was & (self.fmult == 1.0) & (rng.random((E, N)) < cfg['fault_rate'] * DT)
        if fault.any():
            self.fmult = np.where(fault, rng.uniform(*cfg['fault_mult'], (E, N)), self.fmult)
            self.Erem = np.where(fault, self.Erem - rng.uniform(*cfg['fault_drop'], (E, N)) * self.E0, self.Erem)
            self.m_fault |= fault
        estep = np.where(was, (self.fmult * pnom * sur + pacc) * DT, 0.0)
        self.Erem = self.Erem - estep
        self.m_energy += estep
        # measured-to-modelled steady-flight power (battery telemetry, net of kinetic-energy rate)
        ratio = self.fmult * sur * (1.0 + 0.03 * rng.standard_normal((E, N)))
        # ---- collisions (continuous check within the step)
        seg = self.pos - p_old
        rel0 = self.oc[:, None, :, :] - p_old[:, :, None, :]
        sl2 = np.maximum((seg ** 2).sum(-1), 1e-9)[..., None]
        tp = np.clip((rel0 * seg[:, :, None, :]).sum(-1) / sl2, 0, 1)
        clos = p_old[:, :, None, :] + tp[..., None] * seg[:, :, None, :]
        clr_b = np.linalg.norm(self.oc[:, None] - clos, axis=-1) - self.orad[:, None, :] - R_UAV
        clr_b_step = clr_b.min(-1)
        d0 = p_old[:, :, None, :] - p_old[:, None, :, :]; d1 = self.pos[:, :, None, :] - self.pos[:, None, :, :]
        dd = d1 - d0
        tt = np.clip(-(d0 * dd).sum(-1) / np.maximum((dd ** 2).sum(-1), 1e-9), 0, 1)
        dmu = np.linalg.norm(d0 + tt[..., None] * dd, axis=-1) - 2 * R_UAV
        pv = was[:, :, None] & was[:, None, :] & ~np.eye(N, dtype=bool)[None]
        dmu = np.where(pv, dmu, 1e4)
        e0 = p_old[:, :, None, :] - i_old[:, None, :, :]; e1 = self.pos[:, :, None, :] - self.ipos[:, None, :, :]
        de = e1 - e0
        ti = np.clip(-(e0 * de).sum(-1) / np.maximum((de ** 2).sum(-1), 1e-9), 0, 1)
        dmi = np.linalg.norm(e0 + ti[..., None] * de, axis=-1) - R_UAV - R_INT
        clr_a_step = np.minimum(dmu.min(-1), dmi.min(-1) if NI > 0 else 1e4)
        self.m_minair = np.where(was, np.minimum(self.m_minair, clr_a_step), self.m_minair)
        self.m_minbld = np.where(was, np.minimum(self.m_minbld, clr_b_step), self.m_minbld)
        hit_b = was & (clr_b_step < 0); hit_a = was & (clr_a_step < 0)
        out = was & ((self.pos < 0) | (self.pos > self.L)).any(-1)
        dep = was & (self.Erem <= 0)
        crash = hit_b | hit_a | dep | out
        alive = was & ~crash
        d_now = np.linalg.norm(self.goal - self.pos, axis=-1)
        # ---- delivery
        arrive = alive & (self.phase == 0) & (d_now < R_DELIVER)
        land_evt = np.zeros_like(was)
        if arrive.any():
            hov = self.fmult * rotor_power(np.linalg.norm(self.wind, axis=-1), mass) * T_DROP
            self.Erem = np.where(arrive, self.Erem - hov, self.Erem)
            self.m_energy += np.where(arrive, hov, 0.0)
            dep2 = arrive & (self.Erem <= 0)
            dep |= dep2; crash |= dep2; alive &= ~dep2; arrive &= ~dep2
            self.payload = np.where(arrive, 0.0, self.payload)
            self.phase = np.where(arrive, 2, self.phase).astype(np.int8)
            self.m_tdel = np.where(arrive, self.t[:, None].astype(float), self.m_tdel)
            relp = self.pads[:, None, :, :] - self.pos[:, :, None, :]
            ip = np.argmin(np.linalg.norm(relp, axis=-1), -1)
            newg = self.pads[np.arange(E)[:, None], ip]
            self.goal = np.where(arrive[..., None], newg, self.goal)
            phat = self.pos + self.gerr + self.mp_off
            B = self._bounds(phat)
            self.m_ret = np.where(arrive[..., None], self._leg_record(B), self.m_ret)
        # ---- landing
        dl = np.linalg.norm(self.goal - self.pos, axis=-1)
        land = alive & (self.phase >= 2) & (dl < R_LAND) & ~arrive
        if land.any():
            eland = self.fmult * rotor_power(np.zeros_like(mass), M0 + self.payload) * T_LAND
            self.Erem = np.where(land, self.Erem - eland, self.Erem)
            self.m_energy += np.where(land, eland, 0.0)
            dep3 = land & (self.Erem <= 0)
            dep |= dep3; crash |= dep3; alive &= ~dep3; land &= ~dep3
            used = self.m_ret[..., 0] - self.Erem
            self.m_ret[..., 4] = np.where(land, used, self.m_ret[..., 4])
            land_evt = land
            self._close_legs(land, np.zeros_like(land))
        ret_dep = dep & (self.phase >= 2)
        if ret_dep.any():
            self.m_ret[..., 7] = np.where(ret_dep, 1.0, self.m_ret[..., 7])
            self._close_legs(ret_dep, ret_dep)
        # ---- sensing and online conformal trackers
        ag = np.exp(-DT / cfg['gnss_tau'])
        self.gerr = ag * self.gerr + cfg['gnss_sigma'] * np.sqrt(1 - ag * ag) * rng.standard_normal((E, N, 2))
        self.mp_left = np.maximum(self.mp_left - DT, 0.0)
        self.mp_off = np.where((self.mp_left > 0)[..., None], self.mp_off, 0.0)
        newmp = (self.mp_left <= 0) & (rng.random((E, N)) < cfg['mp_prob'])
        if newmp.any():
            mag = rng.uniform(*cfg['mp_mag'], (E, N))
            self.mp_off = np.where(newmp[..., None], mag[..., None] * _unit(rng.uniform(0, 2 * np.pi, (E, N))), self.mp_off)
            self.mp_left = np.where(newmp, rng.uniform(*cfg['mp_dur'], (E, N)), self.mp_left)
        wmeas = self.wind + cfg['wind_meas_sigma'] * rng.standard_normal((E, N, 2))
        innov = wmeas - self.west
        e_pad = self.B['e']
        xi = -(innov * e_pad).sum(-1)
        c = self.conf
        live = alive
        if self.collect:
            self.score_buf.append(np.stack([ratio[live], xi[live]], -1))
        kraw = self.kappa + c['eta_k'] * ((ratio > self.kappa) - c['delta_k'])
        self.m_clipk += live & ((kraw > 3.0) | (kraw < 0.9))
        self.m_nstep += was
        self.kappa = np.where(live, np.clip(kraw, 0.9, 3.0), self.kappa)
        self.qw = np.where(live, np.clip(self.qw + c['eta_q'] * ((xi > self.qw) - c['delta_q']), 0.0, 15.0), self.qw)
        self.wvar = np.where(live, 0.9 * self.wvar + 0.1 * (innov ** 2).mean(-1), self.wvar)
        self.west = np.where(live[..., None], self.west + 0.15 * innov, self.west)
        # ---- return-to-home guard
        phat = self.pos + self.gerr + self.mp_off
        B = self._bounds(phat)
        trig = alive & (self.phase == 0) & (self.Erem < self._trigger_bound(B))
        if trig.any():
            self.phase = np.where(trig, 3, self.phase).astype(np.int8)
            self.m_rth |= trig
            self.m_rthtime = np.where(trig, self.t[:, None].astype(float), self.m_rthtime)
            newg = self.pads[np.arange(E)[:, None], B['ip']]
            self.goal = np.where(trig[..., None], newg, self.goal)
            self.m_ret = np.where(trig[..., None], self._leg_record(B), self.m_ret)
        # ---- outcomes
        self.m_out = np.where(hit_b, 3, self.m_out); self.m_out = np.where(hit_a & ~hit_b, 4, self.m_out)
        self.m_out = np.where(dep & ~(hit_a | hit_b), 5, self.m_out)
        self.m_out = np.where(out & ~(hit_a | hit_b | dep), 6, self.m_out)
        delivered = ~np.isnan(self.m_tdel)
        self.m_out = np.where(land_evt & delivered, 1, self.m_out); self.m_out = np.where(land_evt & ~delivered, 2, self.m_out)
        fin = crash | land_evt
        self.m_tend = np.where(fin, self.t[:, None].astype(float), self.m_tend)
        self.active = was & ~fin
        # ---- reward
        # proximity penalty from true end-of-step clearances
        cb = np.linalg.norm(self.oc[:, None, :, :] - self.pos[:, :, None, :], axis=-1) - self.orad[:, None, :] - R_UAV
        pen = (np.clip(1.0 - cb / D_SAFE, 0, 1) ** 2).sum(-1)
        du = np.linalg.norm(self.pos[:, :, None, :] - self.pos[:, None, :, :], axis=-1) - 2 * R_UAV
        du = np.where(pv, du, 1e4)
        di = np.linalg.norm(self.pos[:, :, None, :] - self.ipos[:, None, :, :], axis=-1) - R_UAV - R_INT
        pen += (np.clip(1.0 - du / D_SAFE, 0, 1) ** 2).sum(-1) + (np.clip(1.0 - di / D_SAFE, 0, 1) ** 2).sum(-1)
        r = 0.01 * (d_prev - d_now) - 0.08 * estep / 1000.0 - 0.01 - 0.3 * np.minimum(pen, 5.0)
        r += 8.0 * arrive + np.where(land_evt, np.where(delivered, 8.0, 2.0), 0.0)
        r -= 30.0 * (hit_a | hit_b | dep) + 20.0 * (out & ~(hit_a | hit_b | dep))
        # timeout
        tmo = (self.t >= cfg['T_max'])[:, None] & self.active
        self.m_out = np.where(tmo, 7, self.m_out)
        self.m_tend = np.where(tmo, self.t[:, None].astype(float), self.m_tend)
        self.active = self.active & ~tmo
        term = crash | land_evt
        # observation of the post-transition state (before any reset)
        nobs = self._observe()
        r = r - 0.1 * np.clip(-self.margin, 0.0, 1.0) * self.active
        r = np.where(was, r, 0.0)
        self.m_retsum += r
        if self.record_traj:
            self.traj.append(dict(pos=self.pos.copy(), ipos=self.ipos.copy(), active=self.active.copy(), phase=self.phase.copy(),
                                  wind=self.wind.copy(), Erem=self.Erem.copy(), cues=self.cues.copy(), t=self.t.copy()))
        ncues = self.cues
        done_env = (~self.active).all(1) | (self.t >= cfg['T_max'])
        cost = ((hit_a | hit_b | dep | out) & was).astype(np.float32)   # catastrophic event indicator
        out_d = dict(next_obs=nobs, next_cues=ncues, rew=r.astype(np.float32), cost=cost, term=term, trunc=tmo, mask=was, reset=done_env)
        if done_env.any():
            idx = np.where(done_env)[0]
            for e in idx:
                self.completed.append(self._episode_record(e))
                self.ep_count[e] += 1
            self._reset_envs(idx)
            obs = self._observe()
            out_d['obs'] = obs; out_d['cues'] = self.cues
        else:
            out_d['obs'] = nobs; out_d['cues'] = ncues
        self.last = out_d['obs']
        return out_d

    def _episode_record(self, e):
        return dict(out=self.m_out[e].copy(), tdel=self.m_tdel[e].copy(), tend=self.m_tend[e].copy(),
                    energy=self.m_energy[e].copy(), minair=self.m_minair[e].copy(), minbld=self.m_minbld[e].copy(),
                    rth=self.m_rth[e].copy(), ret=self.m_ret[e].copy(), retsum=self.m_retsum[e].copy(),
                    E0=self.E0[e].copy(), fault=self.m_fault[e].copy(), rthtime=self.m_rthtime[e].copy(),
                    wbar=np.linalg.norm(self.wbar[e]), env=int(e), ep=int(self.ep_count[e]),
                    payload0=self.payload0[e].copy(), clipk=self.m_clipk[e].copy(), nstep=self.m_nstep[e].copy())
