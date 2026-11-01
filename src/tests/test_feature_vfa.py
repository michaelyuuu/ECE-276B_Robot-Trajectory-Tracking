import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from value_function import FeatureValueFunction


def _tiny_vf():
    T = 100
    ex = np.linspace(-3, 3, 7)
    ey = np.linspace(-3, 3, 7)
    eth = np.linspace(-np.pi, np.pi, 6, endpoint=False)
    ct = np.linspace(0, T, 4, endpoint=False)
    cx = np.linspace(-3, 3, 3)
    cy = np.linspace(-3, 3, 3)
    cth = np.linspace(-np.pi, np.pi, 2, endpoint=False)
    return FeatureValueFunction(T, ex, ey, eth, ct, cx, cy, cth), (4, 3, 3, 2)


def test_feature_shape():
    vf, (nt, nx, ny, nth) = _tiny_vf()
    assert vf.Nc == nt * nx * ny * nth
    Phi = vf.features([0, 1, 2], [0.0, 1.0, -1.0], [0.0, 0.5, -0.5], [0.0, 1.0, -1.0])
    assert Phi.shape == (3, vf.Nc)
    assert np.all(np.isfinite(Phi))


def test_time_periodicity():
    vf, _ = _tiny_vf()
    a = vf.features([3], [0.7], [-0.4], [0.2])
    b = vf.features([3 + vf.T], [0.7], [-0.4], [0.2])
    assert np.allclose(a, b)


def test_heading_periodicity():
    vf, _ = _tiny_vf()
    a = vf.features([5], [0.7], [-0.4], [0.2])
    b = vf.features([5], [0.7], [-0.4], [0.2 + 2 * np.pi])
    assert np.allclose(a, b)


def test_value_batch_matches_call():
    vf, _ = _tiny_vf()
    vf.theta[...] = np.random.default_rng(1).standard_normal(vf.Nc)
    ts = [0, 10, 50]; xs = [0.3, -1.1, 2.0]; ys = [-0.2, 0.9, -2.0]; hs = [0.1, -1.0, 2.0]
    batched = vf.value_batch(ts, xs, ys, hs)
    singles = np.array([vf(t, x, y, h) for t, x, y, h in zip(ts, xs, ys, hs)])
    assert np.allclose(batched, singles)


def test_ridge_recovers_linear_target():
    # y = Phi @ theta_true lies exactly in feature span -> ridge reproduces the
    # target VALUES. (Exact parameter recovery is ill-conditioned because the
    # overlapping RBF columns are highly collinear; function-space recovery is
    # the meaningful, well-conditioned property the fitted-VI loop relies on.)
    vf, _ = _tiny_vf()
    rng = np.random.default_rng(2)
    n = 5 * vf.Nc
    t = rng.integers(0, vf.T, n); x = rng.uniform(-3, 3, n)
    y_ = rng.uniform(-3, 3, n); h = rng.uniform(-np.pi, np.pi, n)
    Phi = vf.features(t, x, y_, h)
    theta_true = rng.standard_normal(vf.Nc)
    target = Phi @ theta_true
    lam = 1e-8
    A = Phi.T @ Phi + lam * np.eye(vf.Nc)
    theta_hat = np.linalg.solve(A, Phi.T @ target)
    assert np.all(np.isfinite(theta_hat))
    assert np.allclose(Phi @ theta_hat, target, atol=1e-4)


def test_feature_gpi_tiny_fit():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from gpi import make_default_config
    from feature_gpi import FeatureGPI
    cfg = make_default_config(nx=5, ny=5, nth=4)
    fgpi = FeatureGPI(ntc=4, nxc=3, nyc=3, nthc=2, n_train=200, n_fit=2,
                      gpi_config=cfg, output_dir=os.path.join(
                          os.path.dirname(__file__), "_tmp_feature_out"))
    fgpi.fit(verbose=False)
    assert np.all(np.isfinite(fgpi.vf.theta))
    assert fgpi.policy.shape == (fgpi.T, fgpi.nx, fgpi.ny, fgpi.nth)
    u = fgpi(0, np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert 0.1 <= u[0] <= 1.0 and -1.0 <= u[1] <= 1.0


def test_feature_gpi_save_load_roundtrip(tmp_path):
    from gpi import make_default_config
    from feature_gpi import FeatureGPI
    cfg = make_default_config(nx=5, ny=5, nth=4)
    out = str(tmp_path / "fout")
    a = FeatureGPI(ntc=4, nxc=3, nyc=3, nthc=2, gpi_config=cfg, output_dir=out)
    a.vf.theta[...] = np.arange(a.Nc, dtype=float)
    a.policy[...] = 7
    a.save()
    b = FeatureGPI(ntc=4, nxc=3, nyc=3, nthc=2,
                   gpi_config=make_default_config(nx=5, ny=5, nth=4), output_dir=out)
    assert b.load()
    assert np.allclose(b.vf.theta, a.vf.theta)
    assert np.array_equal(b.policy, a.policy)
    u = b(3, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))
    assert 0.1 <= u[0] <= 1.0 and -1.0 <= u[1] <= 1.0
