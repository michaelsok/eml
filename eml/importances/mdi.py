import numpy as np
import pandas as pd
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble._forest import BaseForest
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.base import BaseEstimator, TransformerMixin

from eml.utils.helpers import _normalize
from eml.importances._nodes import get_sklearn_nodes_from, Node


class MeanDecreaseImpurity(BaseEstimator, TransformerMixin):
    def __init__(self, features, normalize=True, use_precompute=False):
        self.features = features
        self.normalize = normalize
        self.use_precompute = use_precompute
        self._base = None
        self._name = None
        self._is_forest = None

    def predict(self, estimator, weighted=True):
        self.sanity_check(estimator)
        if (self._name is not None) & self.use_precompute:
            if hasattr(estimator, self._name):
                return getattr(estimator, self._name)
        return self._compute_importances(estimator, weighted=weighted)

    def sanity_check(self, estimator):
        if isinstance(estimator, (BaseForest, BaseDecisionTree, BaseWeightBoosting, BaseGradientBoosting)):
            self._base = 'scikit-learn'
            self._name = 'feature_importances_'
            self._is_forest = isinstance(estimator, (BaseForest, BaseWeightBoosting, BaseGradientBoosting))
        else:
            raise TypeError('estimator is not an expected tree-like model')

    def _compute_importances(self, estimator, weighted=True):
        if self._base == 'scikit-learn':
            return self._compute_sklearn_importances(estimator, weighted=weighted)
        raise ValueError('only scikit-learn estimators are accepted')

    def _compute_sklearn_importances(self, estimator, weighted=True):
        if not self._is_forest:
            return self._compute_sklearn_tree_importances(estimator, weighted=weighted)
        importances = self._compute_sklearn_forest_importances(estimator.estimators_)
        return importances

    def _compute_sklearn_forest_importances(self, trees, weighted=True):
        importances = self._initialize_importances(self.features)
        n_estimators = 0

        for e in trees:
            if isinstance(e, BaseDecisionTree):
                n_estimators += 1
                importances += self._compute_sklearn_tree_importances(e, weighted=weighted, normalize=True)
            else:
                for e_ in e:
                    if e_.tree_.node_count > 1:
                        n_estimators += 1
                        importances += self._compute_sklearn_tree_importances(e_, weighted=weighted, normalize=False)

        importances /= n_estimators
        if self.normalize:
            importances = _normalize(importances, axis=None)

        return importances

    def _compute_sklearn_tree_importances(self, tree, weighted=True, normalize=True):
        importances = self._initialize_importances(self.features)
        nodes = get_sklearn_nodes_from(tree, weighted=weighted)
        for n in nodes:
            if isinstance(n, Node):
                left, right = nodes[n.left], nodes[n.right]
                importances[n.feature] += self._compute_impurity_importance_from(n, left, right)
        importances /= nodes[0].n_node_samples
        if normalize:
            importances = _normalize(importances, axis=None)
        return importances

    @staticmethod
    def _initialize_importances(features):
        return np.zeros(getattr(features, 'shape', len(features)))

    @staticmethod    
    def _compute_impurity_importance_from(node, left, right):
        return node.weighted_impurity - left.weighted_impurity - right.weighted_impurity
