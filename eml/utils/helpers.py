import numpy as np
import pandas as pd


def normalize(array, axis=None):
    '''Normalize a positive array (creates a copy of the array) along a given axis when feasible

    Parameters
    ----------
    array : np.ndarray
        positive array to normalize (ie. array / array.sum(axis=axis))
    axis : int or None
        axis to normalize onto

    Returns
    -------
    np.ndarray
        normalized array if normalizer is greater than 0 otherwise itself

    '''
    array = array.copy()
    normalizer = array.sum(axis=axis)
    if normalizer > .0:
        array = array / normalizer
    return array


def _are_equal(element1, element2):
    if isinstance(element1, (pd.DataFrame, pd.Series, pd.Index)):
        return element1.equals(element2)
    elif isinstance(element1, np.ndarray):
        return np.array_equal(element1, element2)
    return element1 == element2


def are_equal(obj1, obj2):
    if type(obj1) != type(obj2):
        return False

    for attribute in obj1.__dict__.keys():
        if attribute not in obj2.__dict__.keys():
            return False

    for attribute in obj2.__dict__.keys():
        if attribute not in obj1.__dict__.keys():
            return False

    for attribute in obj1.__dict__.keys():
        if not _are_equal(getattr(obj1, attribute), getattr(obj2, attribute)):
            return False

    return True


def as_array(element):
    if isinstance(element, (pd.DataFrame, pd.Series, pd.Index)):
        return element.values
    elif isinstance(element, np.ndarray):
        return element
    return np.array(element)
