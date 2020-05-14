import copy
from itertools import product

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin

from eml.utils.helpers import as_array


<<<<<<< HEAD
class IndividualConditionalExpectation(BaseEstimator, TransformerMixin):
    def __init__(self, task='infer'):
        self.task = task

=======
class ICE(BaseEstimator, TransformerMixin):
>>>>>>> dev_partial
    def fit(self, estimator):
        check_is_fitted(estimator)
        if hasattr(estimator, 'predict_proba'):
            self._estimator_predict = estimator.predict_proba
        else:
            self._estimator_predict = estimator.predict
        self.estimator = estimator

    def interpret(self, X, feature, n_splits=20, how='quantile'):
        X, feature = self._get_X_array_and_feature_index_from(X, feature)
        abscissa = self._get_abscissa_from(X[:, feature], n_splits, how)
        X_ = X.copy()

        ordinates = []
        for a in abscissa:
            X_[:, feature] = a
            ordinates.append(self._estimator_predict(X_))
        ordinates = np.array(ordinates)

        return abscissa, ordinates

    def predict(self, X, feature, n_splits=20, how='quantile'):
        return self.interpret(X=X, feature=feature, n_splits=n_splits, how=how)

    @staticmethod
    def _get_X_array_and_feature_index_from(X, feature):
        X_ = as_array(X)
        n_features = np.prod(X_.shape[1:])
        indices_space = list(product(*[range(s) for s in X_.shape[1:]]))

        if not isinstance(feature, (str, int)):
            raise TypeError('feature should either be a str or an int')
        elif (not isinstance(X, pd.DataFrame)) and isinstance(feature, str):
            raise ValueError('feature is str type and X is not a DataFrame')
        elif isinstance(X, pd.DataFrame) and isinstance(feature, str) and (feature not in X.columns):
            raise ValueError('feature is not in X columns')
        elif isinstance(feature, int) and (feature >= n_features):
            raise ValueError(f'feature index {feature} greather than number of features {n_features}')

        if isinstance(feature, str):
            return X_, X.columns.tolist().index(feature)
        return X_, indices_space[feature]

    @staticmethod
    def _get_abscissa_from(array, n_splits, how):
        if how == 'quantile':
            return np.quantile(array, np.linspace(0, 1, n_splits + 1))
        elif how == 'equal':
            return np.linspace(np.min(array), np.max(array), n_splits + 1)
        raise ValueError('how should be either `equal` or `quantile`')
