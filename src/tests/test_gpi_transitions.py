import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
from gpi import GPI, make_default_config


def make_gpi():
    return GPI(make_default_config())


def test_neighbor_weights_sum_to_one():
    g = make_gpi()
    targets = np.array([0.05, -0.27, 2.9, -3.5])
    lo, hi, wlo, whi = g._interp_neighbors_1d(g.ex, targets)
    assert np.allclose(wlo + whi, 1.0)
    # interior target between nodes splits proportionally (mass conserved)
    lo2, hi2, wlo2, whi2 = g._interp_neighbors_1d(g.ex, np.array([0.15]))
    assert wlo2[0] > 0 and whi2[0] > 0


def test_theta_neighbors_periodic_sum_to_one():
    g = make_gpi()
    targets = np.array([np.pi - 0.01, -np.pi + 0.01, 0.0])
    lo, hi, wlo, whi = g._theta_neighbors(targets)
    assert np.allclose(wlo + whi, 1.0)
    lo0, hi0, _, _ = g._theta_neighbors(np.array([np.pi - 1e-6]))
    assert hi0[0] == 0 or lo0[0] == g.nth - 1


def test_displacement_matches_error_dynamics():
    g = make_gpi()
    g.compute_transition_matrix()
    t, ith, iv, iw = 3, 5, 2, 4
    eth = g.eth[ith]; v = g.vsp[iv]; w = g.wsp[iw]
    from cec import error_dynamics
    enext = np.array(error_dynamics(t, np.array([0.0, 0.0, eth]),
                                    np.array([v, w]))).flatten()
    assert abs(g.dpx[t, ith, iv, iw] - enext[0]) < 1e-9
    assert abs(g.dpy[t, ith, iv, iw] - enext[1]) < 1e-9


def test_unsafe_mask_flags_obstacle_center():
    g = make_gpi()
    g.compute_transition_matrix()
    found = False
    for t in range(g.T):
        for ix in range(g.nx):
            for iy in range(g.ny):
                wx = g.ex[ix] + g.r[t, 0]; wy = g.ey[iy] + g.r[t, 1]
                if (wx - 1.0) ** 2 + (wy - 0.0) ** 2 < 0.1:
                    assert g.unsafe[t, ix, iy]
                    found = True
    assert found


def test_stage_cost_zero_at_origin_and_positive_for_motion():
    g = make_gpi()
    g.compute_stage_costs()
    ix0 = np.argmin(np.abs(g.ex)); iy0 = np.argmin(np.abs(g.ey))
    ith0 = np.argmin(np.abs(g.eth))
    assert np.isclose(g.stage[ix0, iy0, ith0, 0, g.nw // 2], 0.0)
    assert g.stage[ix0, iy0, ith0, -1, g.nw // 2] > 0.0
