import numpy as np
import pandas as pd
from sklearn.tree import BaseDecisionTree
from sklearn.ensemble._forest import BaseForest
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from eml.utils.helpers import normalize_array
from eml._tree.nodes import Node, get_sklearn_nodes_from


SKLEARN_BASE_MODELS = (BaseForest, BaseDecisionTree, BaseWeightBoosting, BaseGradientBoosting)
SKLEARN_FOREST_MODELS = (BaseForest, BaseWeightBoosting, BaseGradientBoosting)


class TreeAdditiveContributions(BaseEstimator, TransformerMixin):
    """Contributions using tree-interpreter from a decision tree-like model"""

    def __init__(self):
        """"""
        self.estimator = None
        self.n_features_ = None
        self._base = None
        self._name = None
        self._is_forest = None

    def fit(self, estimator):
        """Fit the interpreter on the estimator

        Parameters
        ----------
        estimator : any tree model from sklearn API
            estimator used to compute MDI importances

        """
        self.sanity_check(estimator)
        self.estimator = estimator

    def sanity_check(self, estimator):
        """Sanity checks on estimator
        - check if estimator is fitted
        - check if estimator comes from expected API
        - check if estimator is a tree model

        Parameters
        ----------
        estimator : any tree model from sklearn API
            subclass of BaseForest, BaseDecisionTree, BaseWeightBoosting, BaseGradientBoosting

        Raises
        ------
        TypeError
            model is not a sklearn API

        """
        check_is_fitted(estimator)
        if isinstance(estimator, SKLEARN_BASE_MODELS):
            self._base = 'scikit-learn'
            self._name = 'feature_importances_'
            self._is_forest = isinstance(estimator, SKLEARN_FOREST_MODELS)

            if hasattr(estimator, 'n_features_'):
                self.n_features_ = estimator.n_features_
            else:
                self.n_features_ = estimator.estimators_[0].n_features_
        else:
            raise TypeError('estimator is not an expected tree-like model')

    def interpret(self, X):
        """Interpret the data given as inputs

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute the contributions

        Returns
        -------
        np.ndarray
            contributions in same order than the features

        """
        if self._base == 'scikit-learn':
            return self._compute_sklearn_contributions(X=X)
        raise ValueError('only scikit-learn estimators are accepted')

    def predict(self, X):
        """Interpret the data given as inputs (same results than interpret method)

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute the contributions

        Returns
        -------
        np.ndarray
            contributions in same order than the features

        """
        return self.interpret(X=X)

    def _compute_sklearn_contributions(self, X):
        """Compute contributions following scikit-learn API

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute the contributions

        Returns
        -------
        np.ndarray
            contributions in same order than the features

        """
        if not self._is_forest:
            return self._compute_sklearn_tree_contributions(self.estimator, X=X)
        return self._compute_sklearn_forest_contributions(X=X)

    def _compute_sklearn_forest_contributions(self, X):
        """Compute contributions following scikit-learn API on a forest-like model

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute the contributions

        Returns
        -------
        np.ndarray
            contributions in same order than the features

        """
        trees = self.estimator.estimators_
        bias = np.zeros(self.estimator.classes_.shape)
        contributions = np.zeros(X.shape + self.estimator.classes_.shape)
        n_estimators = 0

        for e in trees:
            if isinstance(e, BaseDecisionTree):
                n_estimators += 1
                b, c = self._compute_sklearn_tree_contributions(e, X=X)
            else: # specific case of sklearn gradient boosting models
                for e_ in e:
                    if e_.tree_.node_count > 1:
                        n_estimators += 1
                        b, c = self._compute_sklearn_tree_contributions(e_, X=X)
            bias, contributions = bias + b, contributions + c

        bias, contributions = bias / n_estimators, contributions / n_estimators
        return bias, contributions

    def _compute_sklearn_tree_contributions(self, estimator, X):
        """Compute contributions following scikit-learn API on a tree-like model

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute the contributions

        Returns
        -------
        np.ndarray
            contributions in same order than the features

        """
        contributions = np.zeros(X.shape + estimator.classes_.shape)
        decision_paths = estimator.decision_path(X).toarray()
        nodes = get_sklearn_nodes_from(estimator, X=X, weighted=False)
        bias = nodes[0].share_value

        for n in nodes:
            if isinstance(n, Node):
                left, right = nodes[n.left], nodes[n.right]
                went_left = np.array(decision_paths[:, n.left] == 1).reshape(-1)
                contributions[went_left, n.feature, :] += self._compute_contribution_from(n, left)

                went_right = np.array(decision_paths[:, n.right] == 1).reshape(-1)
                contributions[went_right, n.feature, :] += self._compute_contribution_from(n, right)

        return bias, contributions

    @staticmethod
    def _compute_contribution_from(node, child):
        """Impurity performance between a node and its child

        Parameters
        ----------
        node : eml.importances._nodes.Node
            parent node used to compute impurity gain
        child : eml._tree.nodes.Node or eml._tree.nodes.Leaf
            child of parent node

        Returns
        -------
        float
            contributions with split from parent node

        """
        return child.share_value - node.share_value
