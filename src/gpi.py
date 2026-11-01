from dataclasses import dataclass
import os
import numpy as np
import utils


class GridValueFunction:
    """Dense grid value function V[t, ix, iy, ith] over the discretized
    time/error grid. Stored as a single (T, nx, ny, ntheta) array."""

    def __init__(self, T, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = np.asarray(ex_space)
        self.ey_space = np.asarray(ey_space)
        self.etheta_space = np.asarray(etheta_space)
        self.values = np.zeros(
            (T, len(self.ex_space), len(self.ey_space), len(self.etheta_space)),
            dtype=np.float64,
        )

DT = utils.time_step
SIGMA = utils.sigma  # [0.04, 0.04, 0.004]
R_SAFE = 0.8
WORLD = 3.0
M_COLLISION = 1e4


@dataclass
class GpiConfig:
    traj: callable
    obstacles: np.ndarray
    ex_space: np.ndarray
    ey_space: np.ndarray
    eth_space: np.ndarray
    v_space: np.ndarray
    w_space: np.ndarray
    Q: np.ndarray
    q: float
    R: np.ndarray
    gamma: float
    num_evals: int
    collision_margin: float
    V: GridValueFunction
    output_dir: str
    m_collision: float = 1e4


def make_default_config(output_dir=None, nx=31, ny=31, nth=24, nv=9, nw=15,
                        q_scale=3.0, m_collision=100.0, collision_margin=0.0,
                        qth=10.0):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpi_out")
    T = 100
    ex = np.linspace(-3, 3, nx)
    ey = np.linspace(-3, 3, ny)
    eth = np.linspace(-np.pi, np.pi, nth, endpoint=False)
    v = np.linspace(0.0, 1.0, nv)
    w = np.linspace(-1.0, 1.0, nw)
    obstacles = np.array([[2.35, 0.95, 0.5], [-2.35, -0.95, 0.5],
                          [1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]])
    V = GridValueFunction(T, ex, ey, eth)
    q_theta = (2.0 * q_scale) if qth is None else float(qth)
    return GpiConfig(
        traj=utils.lemniscate, obstacles=obstacles,
        ex_space=ex, ey_space=ey, eth_space=eth, v_space=v, w_space=w,
        Q=np.diag([10.0 * q_scale, 10.0 * q_scale]), q=q_theta,
        R=np.diag([0.1, 0.05]),
        gamma=0.95, num_evals=20, collision_margin=collision_margin, V=V,
        output_dir=output_dir, m_collision=m_collision,
    )


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = config
        c = config
        self.T = c.V.T
        self.ex = np.asarray(c.ex_space)
        self.ey = np.asarray(c.ey_space)
        self.eth = np.asarray(c.eth_space)
        self.vsp = np.asarray(c.v_space)
        self.wsp = np.asarray(c.w_space)
        self.nx = len(self.ex)
        self.ny = len(self.ey)
        self.nth = len(self.eth)
        self.nv = len(self.vsp)
        self.nw = len(self.wsp)
        self.na = self.nv * self.nw
        self.gamma = c.gamma
        self.m_collision = getattr(c, "m_collision", M_COLLISION)
        self.a_v = np.repeat(np.arange(self.nv), self.nw)   # (na,)
        self.a_w = np.tile(np.arange(self.nw), self.nv)     # (na,)
        self.theta_period = 2 * np.pi
        self.V_coll = self.m_collision / (1.0 - self.gamma)
        self.r = np.array([c.traj(t)[:2] for t in range(self.T)])      # (T,2)
        self.alpha = np.array([c.traj(t)[2] for t in range(self.T)])   # (T,)
        self.policy = np.zeros((self.T, self.nx, self.ny, self.nth), dtype=np.int32)
        self._make_noise_points()
        self._precomputed = False

    def _make_noise_points(self):
        offsets_1d = [
            np.array([-np.sqrt(3.0) * s, 0.0, np.sqrt(3.0) * s], dtype=float)
            for s in SIGMA
        ]
        weights_1d = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float)

        pts = []
        wts = []
        for ix, dx in enumerate(offsets_1d[0]):
            for iy, dy in enumerate(offsets_1d[1]):
                for ith, dth in enumerate(offsets_1d[2]):
                    pts.append([dx, dy, dth])
                    wts.append(weights_1d[ix] * weights_1d[iy] * weights_1d[ith])

        self.noise_pts = np.asarray(pts, dtype=float)
        self.noise_wts = np.asarray(wts, dtype=float)

    def state_metric_to_index(self, metric_state):
        ex, ey, eth = metric_state[0], metric_state[1], metric_state[2]
        ix = int(np.argmin(np.abs(self.ex - ex)))
        iy = int(np.argmin(np.abs(self.ey - ey)))
        ith = int(np.argmin(np.abs(_wrap(self.eth - eth))))
        return (ix, iy, ith)

    def state_index_to_metric(self, state_index):
        ix, iy, ith = state_index
        return np.array([self.ex[ix], self.ey[iy], self.eth[ith]])

    def control_metric_to_index(self, control_metric):
        v = np.digitize(control_metric[0], self.config.v_space, right=True)
        w = np.digitize(control_metric[1], self.config.w_space, right=True)
        return v, w

    def control_index_to_metric(self, v, w):
        return self.config.v_space[v], self.config.w_space[w]

    def _interp_neighbors_1d(self, space, targets):

        n = len(space)
        step = space[1] - space[0]
        rel = (targets - space[0]) / step
        lo = np.clip(np.floor(rel).astype(int), 0, n - 1)
        hi = np.clip(lo + 1, 0, n - 1)
        frac = np.clip(rel - lo, 0.0, 1.0)          # 0 at/below lo node, 1 at hi node
        whi = np.where(lo == hi, 0.0, frac)
        wlo = 1.0 - whi
        return lo, hi, wlo, whi

    def _theta_neighbors(self, targets):

        n = self.nth
        step = self.theta_period / n
        rel = (targets - self.eth[0]) / step
        lo = np.floor(rel).astype(int) % n
        hi = (lo + 1) % n
        frac = rel - np.floor(rel)
        return lo, hi, 1.0 - frac, frac
    def compute_transition_matrix(self):

        T, nth, nv, nw = self.T, self.nth, self.nv, self.nw
        eth = self.eth.reshape(1, nth, 1, 1)
        vv = self.vsp.reshape(1, 1, nv, 1)
        ww = self.wsp.reshape(1, 1, 1, nw)
        alpha_t = self.alpha.reshape(T, 1, 1, 1)
        tp1 = (np.arange(T) + 1) % T
        alpha_t1 = self.alpha[tp1].reshape(T, 1, 1, 1)
        half = ww * DT / 2.0
        half_safe = np.where(np.abs(half) < 1e-4, 1.0, half)
        s = np.where(np.abs(half) < 1e-4, 1.0 - half * half / 6.0, np.sin(half) / half_safe)
        eta = eth + alpha_t + half                          # (T,nth,1,nw)
        eta = np.broadcast_to(eta, (T, nth, nv, nw))
        sb = np.broadcast_to(s, (T, nth, nv, nw))
        vb = np.broadcast_to(vv, (T, nth, nv, nw))
        drx = (self.r[:, 0] - self.r[tp1, 0]).reshape(T, 1, 1, 1)
        dry = (self.r[:, 1] - self.r[tp1, 1]).reshape(T, 1, 1, 1)
        self.dpx = DT * sb * np.cos(eta) * vb + drx          # (T,nth,nv,nw)
        self.dpy = DT * sb * np.sin(eta) * vb + dry
        dalpha = _wrap(alpha_t - alpha_t1)                   # (T,1,1,1)
        eth_next = _wrap(eth + DT * ww + dalpha)             # (T,nth,1,nw)
        self.eth_next = np.broadcast_to(eth_next, (T, nth, nv, nw)).copy()
        self._build_unsafe_mask()
        self._precomputed = True

    def _build_unsafe_mask(self):
        T, nx, ny = self.T, self.nx, self.ny
        WX = self.ex.reshape(1, nx, 1) + self.r[:, 0].reshape(T, 1, 1)    # (T,nx,1)
        WY = self.ey.reshape(1, 1, ny) + self.r[:, 1].reshape(T, 1, 1)    # (T,1,ny)
        WX = np.broadcast_to(WX, (T, nx, ny))
        WY = np.broadcast_to(WY, (T, nx, ny))
        unsafe = (np.abs(WX) > WORLD) | (np.abs(WY) > WORLD)
        margin = self.config.collision_margin
        for (cx, cy, _r) in self.config.obstacles:
            d2 = (WX - cx) ** 2 + (WY - cy) ** 2
            unsafe = unsafe | (d2 < (R_SAFE + margin) ** 2)
        self.unsafe = unsafe                                  # (T,nx,ny) bool

    def compute_stage_costs(self):
        Q = self.config.Q
        R = self.config.R
        q = self.config.q
        ex = self.ex.reshape(self.nx, 1, 1, 1, 1)
        ey = self.ey.reshape(1, self.ny, 1, 1, 1)
        eth = self.eth.reshape(1, 1, self.nth, 1, 1)
        v = self.vsp.reshape(1, 1, 1, self.nv, 1)
        w = self.wsp.reshape(1, 1, 1, 1, self.nw)
        pos = Q[0, 0] * ex ** 2 + Q[1, 1] * ey ** 2
        head = q * (1 - np.cos(eth)) ** 2
        ctrl = R[0, 0] * v ** 2 + R[1, 1] * w ** 2
        stage = pos + head + ctrl
        self.stage = np.broadcast_to(
            stage, (self.nx, self.ny, self.nth, self.nv, self.nw)).copy()

    def _augmented_value(self, t):
        V = self.config.V.values[t]                          # (nx,ny,nth)
        return np.where(self.unsafe[t][:, :, None], self.V_coll, V)

    def _expected_next_value_for_action(self, t, iv, iw):


        nth = self.nth
        t1 = (t + 1) % self.T
        Vc = self._augmented_value(t1)
        dpx = self.dpx[t, :, iv, iw]
        dpy = self.dpy[t, :, iv, iw]                         # (nth,)
        eth_n = self.eth_next[t, :, iv, iw]
        K = self.noise_pts.shape[0]

        # theta interpolation for all quadrature points and source headings.
        eth_targets = _wrap(eth_n[None, :] + self.noise_pts[:, 2, None])
        tlo, thi, wtlo, wthi = self._theta_neighbors(eth_targets)      # (K,nth)
        Vth = (
            wtlo[None, None, :, :] * Vc[:, :, tlo]
            + wthi[None, None, :, :] * Vc[:, :, thi]
        )
        Vperm = np.transpose(Vth, (2, 3, 0, 1))

        karr = np.arange(K)[:, None, None]
        tharr = np.arange(nth)[None, :, None]

        # x interpolation: target = ex_grid[ix] + dpx[ith] + noise_x
        x_targets = (
            self.ex[None, None, :] + dpx[None, :, None]
            + self.noise_pts[:, 0, None, None]
        )
        xlo, xhi, wxlo, wxhi = self._interp_neighbors_1d(self.ex, x_targets)
        gx_lo = Vperm[karr, tharr, xlo, :]
        gx_hi = Vperm[karr, tharr, xhi, :]
        A = wxlo[:, :, :, None] * gx_lo + wxhi[:, :, :, None] * gx_hi

        # y interpolation: target = ey_grid[iy] + dpy[ith] + noise_y
        y_targets = (
            self.ey[None, None, :] + dpy[None, :, None]
            + self.noise_pts[:, 1, None, None]
        )
        ylo, yhi, wylo, wyhi = self._interp_neighbors_1d(self.ey, y_targets)
        ylo = np.broadcast_to(ylo[:, :, None, :], A.shape)
        yhi = np.broadcast_to(yhi[:, :, None, :], A.shape)
        gy_lo = np.take_along_axis(A, ylo, axis=3)
        gy_hi = np.take_along_axis(A, yhi, axis=3)
        B = wylo[:, :, None, :] * gy_lo + wyhi[:, :, None, :] * gy_hi

        return np.einsum("k,ktxy->xyt", self.noise_wts, B)

    def _q_for_action(self, t, a):
        iv = self.a_v[a]
        iw = self.a_w[a]
        EV = self._expected_next_value_for_action(t, iv, iw)            # (nx,ny,nth)
        stage = self.stage[:, :, :, iv, iw]                            #
        return stage + self.gamma * EV

    def _vi_sweep(self):

        newV = np.empty_like(self.config.V.values)
        for t in range(self.T):
            Qbest = None
            abest = None
            for a in range(self.na):
                Qa = self._q_for_action(t, a)
                if Qbest is None:
                    Qbest = Qa.copy()
                    abest = np.full((self.nx, self.ny, self.nth), a, dtype=np.int32)
                else:
                    better = Qa < Qbest
                    Qbest = np.where(better, Qa, Qbest)
                    abest = np.where(better, a, abest)
            newV[t] = Qbest
            self.policy[t] = abest
        delta = np.max(np.abs(newV - self.config.V.values))
        self.config.V.values[...] = newV
        return delta

    @utils.timer
    def policy_evaluation(self):

        return self._vi_sweep()

    @utils.timer
    def policy_improvement(self):
        return 0

    def init_value_function(self):
        self.config.V.values[...] = 0.0
        self.policy[...] = 0

    def compute_policy(self, num_iters, tol=1e-3, verbose=True):
        if not self._precomputed:
            self.compute_stage_costs()
            self.compute_transition_matrix()
        self.init_value_function()
        max_sweeps = max(int(num_iters), 1)
        for it in range(max_sweeps):
            delta = self._vi_sweep()
            if verbose and (it % 10 == 0 or delta < tol):
                utils.tqdm.write(f"[GPI] sweep {it}: max value change {delta:.4g}")
            if delta < tol:
                break
        self.save()

    def evaluate_value_function(self):
        raise NotImplementedError  # feature-based only

    def _cache_fingerprint(self):

        import hashlib
        import json
        c = self.config
        payload = {
            "T": int(self.T),
            "ex": [float(self.ex[0]), float(self.ex[-1]), int(self.nx)],
            "ey": [float(self.ey[0]), float(self.ey[-1]), int(self.ny)],
            "eth": [float(self.eth[0]), float(self.eth[-1]), int(self.nth)],
            "v": np.asarray(self.vsp, float).round(8).tolist(),
            "w": np.asarray(self.wsp, float).round(8).tolist(),
            "gamma": float(self.gamma),
            "Q": np.asarray(c.Q, float).round(8).tolist(),
            "q": float(c.q),
            "R": np.asarray(c.R, float).round(8).tolist(),
            "m_collision": float(self.m_collision),
            "collision_margin": float(c.collision_margin),
        }
        return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def save(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        np.save(os.path.join(self.config.output_dir, "policy.npy"), self.policy)
        np.save(os.path.join(self.config.output_dir, "values.npy"), self.config.V.values)
        with open(os.path.join(self.config.output_dir, "fingerprint.txt"), "w") as f:
            f.write(self._cache_fingerprint())

    def load(self):
        p = os.path.join(self.config.output_dir, "policy.npy")
        if not os.path.exists(p):
            return False
        fp = os.path.join(self.config.output_dir, "fingerprint.txt")
        saved = None
        if os.path.exists(fp):
            with open(fp) as f:
                saved = f.read().strip()
        if saved != self._cache_fingerprint():
            print("[gpi] cached policy in {} does not match the current config "
                  "(stale cache); ignoring and rebuilding.".format(self.config.output_dir))
            return False
        self.policy = np.load(p)
        self.config.V.values[...] = np.load(
            os.path.join(self.config.output_dir, "values.npy"))
        return True

    def __call__(self, t, cur_state, cur_ref_state):
        e = np.asarray(cur_state, float) - np.asarray(cur_ref_state, float)
        e[2] = _wrap(e[2])
        ix, iy, ith = self.state_metric_to_index(e)
        a = int(self.policy[int(t) % self.T, ix, iy, ith])
        v = self.config.v_space[self.a_v[a]]
        w = self.config.w_space[self.a_w[a]]
        return [float(v), float(w)]
