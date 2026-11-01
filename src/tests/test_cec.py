import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
from cec import CEC, error_dynamics, sinc

OBST = np.array([[2.35, 0.95, 0.5], [-2.35, -0.95, 0.5],
                 [1.0, 0.0, 0.5], [-1.0, 0.0, 0.5]])


def test_sinc_limit():
    assert abs(float(sinc(0.0)) - 1.0) < 1e-9
    assert abs(float(sinc(1e-6)) - 1.0) < 1e-9
    assert abs(float(sinc(0.5)) - np.sin(0.5) / 0.5) < 1e-9


def test_error_dynamics_matches_sim_at_w_zero():
    # At w=0, exact-integration g reduces to the simulator's Euler step.
    t = 3
    e = np.array([0.1, -0.2, 0.3])
    u = np.array([0.5, 0.0])  # w = 0
    enext = np.array(error_dynamics(t, e, u)).flatten()
    ref_t = np.array(utils.lemniscate(t))
    ref_t1 = np.array(utils.lemniscate(t + 1))
    x_t = e + ref_t
    x_t[2] = np.arctan2(np.sin(x_t[2]), np.cos(x_t[2]))
    x_t1 = utils.car_next_state(utils.time_step, x_t, u, noise=False)
    e_expected = x_t1 - ref_t1
    e_expected[2] = np.arctan2(np.sin(e_expected[2]), np.cos(e_expected[2]))
    assert np.allclose(enext[:2], e_expected[:2], atol=1e-9)


def test_cec_returns_valid_control_in_bounds():
    cec = CEC(utils.lemniscate, OBST)
    x0 = np.array(utils.lemniscate(2))
    u = np.array(cec(2, x0, np.array(utils.lemniscate(2)))).flatten()
    assert u.shape == (2,)
    assert 0.1 - 1e-6 <= u[0] <= 1.0 + 1e-6
    assert -1.0 - 1e-6 <= u[1] <= 1.0 + 1e-6


def test_cec_reduces_tracking_error_over_rollout():
    # Track only in the obstacle-free initial regime (the reference passes within
    # 0.41 of obstacles c1/c2 around k=21/79, where deviation is mandatory).
    cec = CEC(utils.lemniscate, OBST)
    np.random.seed(0)
    state = np.array(utils.lemniscate(2)) + np.array([0.5, 0.5, 0.0])
    errs = []
    for k in range(2, 15):
        ref = np.array(utils.lemniscate(k))
        u = np.array(cec(k, state, ref)).flatten()
        state = utils.car_next_state(utils.time_step, state, u, noise=False)
        errs.append(np.linalg.norm(state[:2] - np.array(utils.lemniscate(k + 1))[:2]))
    # converges from ~0.7 down to well under 0.1
    assert errs[0] > 0.3
    assert min(errs) < 0.1
    assert errs[-1] < 0.15
