import numpy as np

# Optional GPU backend (CuPy). The feature materialization is the dominant cost
# of the feature-based controller (Part 3); on a CUDA device it runs in float32
# for full throughput on consumer GPUs (which throttle float64). The public API
# (features/value_batch) always returns NumPy float64, so the CPU path and all
# callers/tests are unchanged when CuPy is absent.
try:
    import cupy as _cp
    # A device must exist AND CuPy must be able to JIT-compile an elementwise
    # kernel (needs CUDA headers). Probe both so we fall back to CPU cleanly
    # rather than crashing partway through a long run.
    if _cp.cuda.runtime.getDeviceCount() > 0:
        float((_cp.asarray([1.0], dtype=_cp.float32) * 2.0).sum())
        _GPU_AVAILABLE = True
    else:
        _GPU_AVAILABLE = False
except Exception:
    _cp = None
    _GPU_AVAILABLE = False


def _asnumpy(a):
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)


class ValueFunction:
    def __init__(self, T, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = np.asarray(ex_space)
        self.ey_space = np.asarray(ey_space)
        self.etheta_space = np.asarray(etheta_space)

    def copy_from(self, other):
        raise NotImplementedError

    def update(self, t, ex, ey, etheta, target_value):
        raise NotImplementedError

    def __call__(self, t, ex, ey, etheta):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


def _nearest_index(space, value):
    return int(np.argmin(np.abs(space - value)))


def _nearest_index_periodic(space, value, period):
    d = np.abs((space - value + period / 2) % period - period / 2)
    return int(np.argmin(d))


class GridValueFunction(ValueFunction):
    """Grid-based value function stored as a dense array (T, nx, ny, ntheta)."""

    def __init__(self, T, ex_space, ey_space, etheta_space):
        super().__init__(T, ex_space, ey_space, etheta_space)
        self.values = np.zeros(
            (T, len(self.ex_space), len(self.ey_space), len(self.etheta_space)),
            dtype=np.float64,
        )

    def copy_from(self, other):
        self.values[...] = other.values

    def update(self, t, ex, ey, etheta, target_value):
        ix = _nearest_index(self.ex_space, ex)
        iy = _nearest_index(self.ey_space, ey)
        ith = _nearest_index_periodic(self.etheta_space, etheta, 2 * np.pi)
        self.values[int(t) % self.T, ix, iy, ith] = target_value

    def __call__(self, t, ex, ey, etheta):
        ix = _nearest_index(self.ex_space, ex)
        iy = _nearest_index(self.ey_space, ey)
        ith = _nearest_index_periodic(self.etheta_space, etheta, 2 * np.pi)
        return float(self.values[int(t) % self.T, ix, iy, ith])

    def copy(self):
        new = GridValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space)
        new.values[...] = self.values
        return new


class FeatureValueFunction(ValueFunction):
    """Feature-based value function: V(t,e) ~= theta^T phi(t,e), where phi is a
    vector of radial basis functions centered on a downsampled time/error grid
    (Part 3). Time and heading distances are circular."""

    # number of appended polynomial error features (see _features_batch)
    N_POLY = 8

    def __init__(self, T, ex_space, ey_space, etheta_space,
                 ct_space, cx_space, cy_space, cth_space,
                 alpha=1.0, batch=4096, use_gpu=None, poly=False):
        super().__init__(T, ex_space, ey_space, etheta_space)
        if use_gpu is None:
            use_gpu = _GPU_AVAILABLE
        self.use_gpu = bool(use_gpu and _cp is not None)
        self._xp = _cp if self.use_gpu else np
        # GPU centers/features use float32 (consumer GPUs throttle float64);
        # the CPU path keeps float64 for bit-for-bit parity with the old code.
        self._dtype = self._xp.float32 if self.use_gpu else np.float64
        xp = self._xp
        CT, CX, CY, CTH = np.meshgrid(ct_space, cx_space, cy_space, cth_space,
                                      indexing="ij")
        self.ct = xp.asarray(CT.ravel(), dtype=self._dtype)
        self.cx = xp.asarray(CX.ravel(), dtype=self._dtype)
        self.cy = xp.asarray(CY.ravel(), dtype=self._dtype)
        self.cth = xp.asarray(CTH.ravel(), dtype=self._dtype)
        self.n_rbf = int(self.ct.size)
        # Optional polynomial error features give the value a sharp quadratic bowl
        # at e=0 (the smoothed RBFs alone cannot resolve small tracking errors, so
        # the greedy policy under-actuates and drifts off the reference).
        self.poly = bool(poly)
        self.Nc = self.n_rbf + (self.N_POLY if self.poly else 0)
        self.alpha = float(alpha)
        self.batch = int(batch)
        # kernel widths derived from center spacing (so they track the grid)
        sigma_t = T / len(ct_space)                                   # period / n_t^c
        sigma_e = (ex_space[-1] - ex_space[0]) / (len(cx_space) - 1)  # x/y spacing
        self.beta_t = 1.0 / (2.0 * sigma_t ** 2)
        self.beta_e = 1.0 / (2.0 * sigma_e ** 2)
        # theta is always stored as a NumPy float64 vector (public API, save/load).
        self.theta = np.zeros(self.Nc, dtype=np.float64)

    def _features_batch(self, t, ex, ey, eth):
        """Feature matrix (B, Nc) for a batch on the active backend (NumPy on CPU,
        CuPy on GPU): RBF kernels, optionally followed by stage-cost-shaped
        polynomial error features."""
        xp = self._xp
        t = xp.asarray(t, dtype=self._dtype)
        ex = xp.asarray(ex, dtype=self._dtype)
        ey = xp.asarray(ey, dtype=self._dtype)
        eth = xp.asarray(eth, dtype=self._dtype)
        dt = xp.abs(t[:, None] - self.ct[None, :])
        dt = xp.minimum(dt, self.T - dt)                              # circular time
        two_pi = self._dtype(2 * np.pi)
        dth = (eth[:, None] - self.cth[None, :] + self._dtype(np.pi)) % two_pi - self._dtype(np.pi)
        de2 = ((ex[:, None] - self.cx[None, :]) ** 2
               + (ey[:, None] - self.cy[None, :]) ** 2
               + dth ** 2)
        beta_t = self._dtype(self.beta_t)
        beta_e = self._dtype(self.beta_e)
        rbf = self._dtype(self.alpha) * xp.exp(-beta_t * dt ** 2 - beta_e * de2)
        if not self.poly:
            return rbf
        # Polynomial error features (positions scaled to ~O(1) for conditioning).
        sx = ex / self._dtype(3.0)
        sy = ey / self._dtype(3.0)
        hcos = self._dtype(1.0) - xp.cos(eth)
        one = xp.ones_like(ex)
        poly = xp.stack([sx, sy, sx * sx, sy * sy, sx * sy, hcos, hcos * hcos, one],
                        axis=1)
        return xp.concatenate([rbf, poly], axis=1)

    def features(self, t, ex, ey, eth):
        """RBF feature matrix (B, Nc) as NumPy float64."""
        phi = self._features_batch(np.asarray(t, float).ravel(),
                                   np.asarray(ex, float).ravel(),
                                   np.asarray(ey, float).ravel(),
                                   np.asarray(eth, float).ravel())
        return _asnumpy(phi).astype(np.float64)

    def value_batch(self, t, ex, ey, eth):
        """theta^T phi for a batch, chunked to bound peak memory. Returns NumPy
        float64. On GPU, features and the dot product run in float32 on-device
        and only the final values are copied back to the host."""
        t = np.asarray(t, float).ravel(); ex = np.asarray(ex, float).ravel()
        ey = np.asarray(ey, float).ravel(); eth = np.asarray(eth, float).ravel()
        out = np.empty(t.size, dtype=np.float64)
        theta_dev = self._xp.asarray(self.theta, dtype=self._dtype)
        for s in range(0, t.size, self.batch):
            e = s + self.batch
            phi = self._features_batch(t[s:e], ex[s:e], ey[s:e], eth[s:e])
            out[s:e] = _asnumpy(phi @ theta_dev).astype(np.float64)
        return out

    def __call__(self, t, ex, ey, etheta):
        return float(self.value_batch([int(t) % self.T], [ex], [ey], [etheta])[0])

    def copy_from(self, other):
        self.theta[...] = other.theta

    def copy(self):
        new = FeatureValueFunction.__new__(FeatureValueFunction)
        new.__dict__.update(self.__dict__)
        new.theta = self.theta.copy()
        return new
