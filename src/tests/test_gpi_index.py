import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils
from gpi import GPI, make_default_config


def make_gpi():
    return GPI(make_default_config())


def test_index_roundtrip_on_grid_points():
    g = make_gpi()
    metric = np.array([g.config.ex_space[3], g.config.ey_space[7],
                       g.config.eth_space[5]])
    idx = g.state_metric_to_index(metric)
    assert idx == (3, 7, 5)
    back = g.state_index_to_metric(idx)
    assert np.allclose(back, metric)


def test_theta_periodic_nearest():
    g = make_gpi()
    metric = np.array([0.0, 0.0, np.pi - 1e-6])
    ix, iy, ith = g.state_metric_to_index(metric)
    assert 0 <= ith < len(g.config.eth_space)


def test_clip_outside_range():
    g = make_gpi()
    metric = np.array([100.0, -100.0, 0.0])
    ix, iy, ith = g.state_metric_to_index(metric)
    assert ix == len(g.config.ex_space) - 1
    assert iy == 0
