import pytest
import numpy as np
import pandas as pd

from eml.utils.helpers import normalize_array, _are_equal, are_equal


def test_normalize():
    array = np.array([1, 2, 3, 4])
    normalized_array = normalize_array(array, axis=None)
    expected_array = np.array([.1, .2, .3, .4])

    np.testing.assert_allclose(normalized_array, expected_array)


def test_normalize_zero_sum():
    array = np.array([0, 0, 0, 0])
    normalized_array = normalize_array(array, axis=None)
    expected_array = np.array([.0, .0, .0, .0])

    np.testing.assert_allclose(normalized_array, expected_array)


def test_normalize_error():
    array = np.array(['a', 'b', 'c', 'd'])
    with pytest.raises(TypeError):
        normalize_array(array, axis=None)


def test_are_equal():
    class A:
        def __init__(self):
            self.a = 1
            self.b = np.array([1, 2, 3])
            self.c = pd.Series([2])

    class B:
        def __init__(self):
            self.a = 1
            self.b = np.array([1, 2, 3])
            self.c = pd.Series([2])

    a1, a2, a3, a4, b1 = A(), A(), A(), A(), B()
    setattr(a3, 'a', 2)
    setattr(a4, 'd', 1)

    assert are_equal(a1, a2)
    assert not (a1 is a2)
    assert not are_equal(a1, a3)
    assert not are_equal(a1, a4)
    assert not are_equal(a4, a1)
    assert not are_equal(a1, b1)


def test__are_equal():
    a1, a2 = 1000000, 1000000
    b1, b2 = np.array([1, 2, 3]), np.array([1, 2, 3])
    c1, c2 = pd.Series([1, 2, 3], index=[1, 2, 3], name='c'), pd.Series([1, 2, 3], index=[1, 2, 3], name='c')
    assert _are_equal(a1, a2)
    assert not (a1 is a2)
    assert _are_equal(b1, b2)
    assert not (b1 is b2)
    assert _are_equal(c1, c2)
    assert not (c1 is c2)
