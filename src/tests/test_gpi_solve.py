import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
from gpi import GPI, GpiConfig
from value_function import GridValueFunction


def small_gpi(tmpdir):
    T = 100
    ex = np.linspace(-3, 3, 11)
    ey = np.linspace(-3, 3, 11)
    eth = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    v = np.linspace(0.1, 1.0, 4)
    w = np.linspace(-1.0, 1.0, 5)
    obstacles = np.array([[2.35, 0.95, 0.5], [-2.35, -0.95, 0.5],
                          [1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]])
    V = GridValueFunction(T, ex, ey, eth)
    cfg = GpiConfig(traj=utils.lemniscate, obstacles=obstacles, ex_space=ex, ey_space=ey,
                    eth_space=eth, v_space=v, w_space=w, Q=np.diag([10., 10.]), q=2.0,
                    R=np.diag([0.1, 0.05]), gamma=0.95, num_evals=20, collision_margin=0.0,
                    V=V, output_dir=str(tmpdir))
    return GPI(cfg)


def test_policy_iteration_runs_and_values_finite(tmp_path):
    g = small_gpi(tmp_path)
    g.compute_policy(num_iters=40, tol=1e-2)
    V = g.config.V.values
    assert np.all(np.isfinite(V))      # no NaN/Inf
    assert np.all(V >= -1e-6)          # costs are non-negative
    # on-reference (zero error) is not the catastrophic absorbing value
    ix0 = np.argmin(np.abs(g.ex)); iy0 = np.argmin(np.abs(g.ey))
    assert V[2, ix0, iy0, 0] < g.V_coll


def test_policy_saved_and_loadable(tmp_path):
    g = small_gpi(tmp_path)
    g.compute_policy(num_iters=3)
    g2 = small_gpi(tmp_path)
    assert g2.load()
    assert np.array_equal(g.policy, g2.policy)


def test_call_returns_in_bounds_control(tmp_path):
    g = small_gpi(tmp_path)
    g.compute_policy(num_iters=3)
    state = np.array(utils.lemniscate(2)) + np.array([0.3, -0.2, 0.1])
    u = g(2, state, np.array(utils.lemniscate(2)))
    assert 0.1 <= u[0] <= 1.0 and -1.0 <= u[1] <= 1.0
