import os
import numpy as np
try:
    from scipy.linalg import cho_factor, cho_solve
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False
from gpi import GPI, make_default_config
from value_function import FeatureValueFunction
import utils


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class FeatureGPI:
    """Feature-based value-function-approximation controller (Part 3).

    Fitted value iteration: each iteration materializes V_hat = theta^T phi onto
    the tabular GPI grid, runs one GPI backup sweep to produce Bellman targets,
    then solves a closed-form ridge regression for theta on a fixed sampled
    training set. A greedy policy table is extracted from the final theta and
    used at runtime via nearest-neighbour lookup (same as tabular GPI)."""

    def __init__(self, ntc=20, nxc=8, nyc=8, nthc=6, n_train=8000,
                 lam=1e-4, n_fit=20, seed=0, gpi_config=None, output_dir=None):
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "feature_out")
        self.output_dir = output_dir
        self.lam = float(lam)
        self.n_fit = int(n_fit)
        self.n_train = int(n_train)
        self.seed = int(seed)
        self.gpi = GPI(gpi_config if gpi_config is not None else make_default_config())
        g = self.gpi
        self.T, self.nx, self.ny, self.nth = g.T, g.nx, g.ny, g.nth
        ct = np.linspace(0, self.T, ntc, endpoint=False)
        cx = np.linspace(g.ex[0], g.ex[-1], nxc)
        cy = np.linspace(g.ey[0], g.ey[-1], nyc)
        cth = np.linspace(-np.pi, np.pi, nthc, endpoint=False)
        self.vf = FeatureValueFunction(self.T, g.ex, g.ey, g.eth, ct, cx, cy, cth,
                                       poly=True)
        self.Nc = self.vf.Nc
        self.policy = np.zeros((self.T, self.nx, self.ny, self.nth), dtype=np.int32)
        self._build_grid_coords()

    def _build_grid_coords(self):
        g = self.gpi
        TT, XX, YY, HH = np.meshgrid(np.arange(self.T), g.ex, g.ey, g.eth,
                                     indexing="ij")
        self.grid_shape = TT.shape
        self.t_flat = TT.ravel().astype(float)
        self.ex_flat = XX.ravel()
        self.ey_flat = YY.ravel()
        self.eth_flat = HH.ravel()

    def _materialize(self):
        vals = self.vf.value_batch(self.t_flat, self.ex_flat, self.ey_flat, self.eth_flat)
        # The true value function is bounded: 0 <= V*(t,e) <= V_coll (stage costs
        # are non-negative and the absorbing collision value is the maximum).
        # Projecting V_hat into this interval before bootstrapping keeps fitted
        # value iteration stable (bounded approximate DP); without it the linear
        # RBF projection composed with the Bellman max operator diverges.
        np.clip(vals, 0.0, self.gpi.V_coll, out=vals)
        self.gpi.config.V.values[...] = vals.reshape(self.grid_shape)

    def _solve_ridge(self, A_or_factor, rhs):
        if _HAVE_SCIPY:
            return cho_solve(A_or_factor, rhs)
        return np.linalg.solve(A_or_factor, rhs)

    @utils.timer
    def fit(self, verbose=True):
        g = self.gpi
        g.compute_stage_costs()
        g.compute_transition_matrix()
        rng = np.random.default_rng(self.seed)
        ntot = self.t_flat.size
        D = rng.choice(ntot, size=min(self.n_train, ntot), replace=False)
        Phi = self.vf.features(self.t_flat[D], self.ex_flat[D],
                               self.ey_flat[D], self.eth_flat[D])      # (|D|, Nc)
        A = Phi.T @ Phi + self.lam * np.eye(self.Nc)
        fac = cho_factor(A) if _HAVE_SCIPY else A
        PhiT = Phi.T
        self.vf.theta[...] = 0.0
        for k in range(self.n_fit):
            self._materialize()                 # V.values <- V_hat_k
            delta = g._vi_sweep()               # V.values <- targets y_k; g.policy greedy
            y = g.config.V.values.ravel()[D]
            theta_new = self._solve_ridge(fac, PhiT @ y)
            dtheta = float(np.max(np.abs(theta_new - self.vf.theta)))
            self.vf.theta[...] = theta_new
            if verbose:
                utils.tqdm.write(f"[FGPI] iter {k}: backup delta {delta:.4g} "
                                 f"dtheta {dtheta:.4g}")
        # extract greedy policy from the final theta
        self._materialize()
        g._vi_sweep()
        self.policy = g.policy.copy()
        self.save()

    def save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(os.path.join(self.output_dir, "theta.npy"), self.vf.theta)
        np.save(os.path.join(self.output_dir, "policy.npy"), self.policy)

    def load(self):
        p = os.path.join(self.output_dir, "policy.npy")
        if os.path.exists(p):
            self.policy = np.load(p)
            self.vf.theta[...] = np.load(os.path.join(self.output_dir, "theta.npy"))
            return True
        return False

    def __call__(self, t, cur_state, cur_ref_state):
        g = self.gpi
        e = np.asarray(cur_state, float) - np.asarray(cur_ref_state, float)
        e[2] = _wrap(e[2])
        ix, iy, ith = g.state_metric_to_index(e)
        a = int(self.policy[int(t) % self.T, ix, iy, ith])
        v = g.config.v_space[g.a_v[a]]
        w = g.config.w_space[g.a_w[a]]
        return [float(np.clip(v, utils.v_min, utils.v_max)),
                float(np.clip(w, utils.w_min, utils.w_max))]
