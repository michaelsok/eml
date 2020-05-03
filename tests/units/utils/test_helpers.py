import pytest

import numpy as np

from eml.utils.helpers import _normalize


def test__normalize():
    array = np.array([1, 2, 3, 4])
    normalized_array = _normalize(array, axis=None)
    expected_array = np.array([.1, .2, .3, .4])

    np.testing.assert_allclose(normalized_array, expected_array)


def test__normalize_zero_sum():
    array = np.array([0, 0, 0, 0])
    normalized_array = _normalize(array, axis=None)
    expected_array = np.array([.0, .0, .0, .0])

    np.testing.assert_allclose(normalized_array, expected_array)
