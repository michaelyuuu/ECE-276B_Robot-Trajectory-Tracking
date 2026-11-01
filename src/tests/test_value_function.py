import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from value_function import GridValueFunction


def make_vf():
    ex = np.linspace(-3, 3, 21)
    ey = np.linspace(-3, 3, 21)
    eth = np.linspace(-np.pi, np.pi, 16, endpoint=False)
    return GridValueFunction(100, ex, ey, eth)


def test_shape_and_zero_init():
    vf = make_vf()
    assert vf.values.shape == (100, 21, 21, 16)
    assert np.all(vf.values == 0)


def test_update_and_call_nearest():
    vf = make_vf()
    vf.values[5, 10, 10, 0] = 7.0
    v = vf(5, 0.0, 0.0, -np.pi)
    assert abs(v - 7.0) < 1e-9


def test_copy_independent():
    vf = make_vf()
    c = vf.copy()
    c.values[0, 0, 0, 0] = 123.0
    assert vf.values[0, 0, 0, 0] == 0.0


def test_copy_from():
    vf = make_vf()
    other = make_vf()
    other.values[:] = 3.0
    vf.copy_from(other)
    assert np.all(vf.values == 3.0)
