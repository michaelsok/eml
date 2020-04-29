def _normalize(array, axis=None):
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
        array /= normalizer
    return array
