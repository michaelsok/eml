import copy
from itertools import product

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from eml.utils.helpers import as_array


class ScoreMethodMissing(Exception):
    """Score method missing"""


class ShapeError(Exception):
    """Non-expected shape"""


class PermutationImportances(BaseEstimator, TransformerMixin):
    def __init__(self, scoring=None):
        self.scoring = scoring

    @property
    def use_score_method(self):
        return self.scoring is None

    def add_scoring(self, scoring):
        self.scoring = scoring

    def fit(self, estimator, predict_method=None):
        self.estimator_sanity_check(estimator)
        self.estimator = estimator
        self.predict_method = predict_method
        if (predict_method is None) and (not self.use_score_method):
            if hasattr(predict_method, 'predict_proba'):
                self._estimator_predict = self.estimator.predict_proba
            else:
                self._estimator_predict = self.estimator.predict

    def _compute_score_from(self, X, y):
        if self.use_score_method:
            score = self.estimator.score(X, y)
        else:
            score = self.scoring(self._estimator_predict(X), y)
        return score

    def estimator_sanity_check(self, estimator):
        check_is_fitted(estimator)
        if self.use_score_method and (not hasattr(estimator, 'score')):
            message = ('scoring attribute is None and estimator does not have a score method. ' +
                       'Either add a scoring function with the `add_scoring` method or during initialization ' +
                       'or use an estimator with a score method.')
            raise ScoreMethodMissing(message)

    def interpret(self, X, y, n_iter=20, as_mean=True, how='permutation', random_state=None):
        self.data_sanity_check(X, y)
        np.random.seed(random_state)
        return self._compute_importances(X=X, y=y, n_iter=n_iter, as_mean=as_mean, how=how)

    def data_sanity_check(self, X, y):
        if hasattr(X, 'shape') and hasattr(y, 'shape'):
            if X.shape[0] != y.shape[0]:
                raise ShapeError('X and y do not have the same number of rows')

    def _compute_importances(self, X, y, n_iter, as_mean=True, how='permutation'):
        X, y = as_array(X), as_array(y)
        score = self._compute_score_from(X, y)

        importances = np.zeros((n_iter,) + X.shape[1:])
        for idx in range(n_iter):
            importances[idx, :] = self._compute_importances_one_iter(X, y, score, how)

        return importances.mean(axis=0) if as_mean else importances

    def _compute_importances_one_iter(self, X, y, score, how='permutation'):
        features_shape = X.shape[1:]
        importances = np.zeros(features_shape)
        indices_space = list(product(*[range(s) for s in features_shape]))

        for indices in indices_space:
            X_ = X.copy()
            X_[(slice(None),) + indices] = np.random.permutation(X[(slice(None),) + indices])
            permuted_score = self._compute_score_from(X_, y)
            importances[indices] = score - permuted_score

        return importances
