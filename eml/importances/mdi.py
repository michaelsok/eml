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


class MeanDecreaseImpurity(BaseEstimator, TransformerMixin):
    """Importances using MDI from a decision tree-like models"""

    def __init__(self, use_precompute=True):
        """
        Parameters
        ----------
        use_precompute : bool, optional
            use precomputed feature importances when available.
            precomputed MDI is weighted and normalized, by default True

        """
        self.use_precompute = use_precompute
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

    def interpret(self, X=None, weighted=True, normalize=True):
        """Interpret the data given as inputs

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute MDI; when None use original dataset, by default None
        weighted : bool, optional
            if MDI should be computed based on weighted node samples, by default True
        normalize : bool, optional
            if MDI should be normalized after computing, by default True

        Returns
        -------
        np.ndarray
            MDI importances in same order than the features

        """
        if (X is None) & (self._name is not None) & self.use_precompute:
            if hasattr(self.estimator, self._name):
                return getattr(self.estimator, self._name)
        return self._compute_importances(X=X, weighted=weighted, normalize=normalize)

    def predict(self, X=None, weighted=True, normalize=True):
        """Interpret the data given as inputs (same results than interpret method)

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute MDI; when None use original dataset, by default None
        weighted : bool, optional
            if MDI should be computed based on weighted node samples, by default True
        normalize : bool, optional
            if MDI should be normalized after computing, by default True

        Returns
        -------
        np.ndarray
            MDI importances in same order than the features

        """
        return self.interpret(X=X, weighted=weighted, normalize=normalize)

    def _compute_importances(self, X=None, weighted=True, normalize=True):
        """Compute MDI importances according to model API

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute MDI; when None use original dataset, by default None
        weighted : bool, optional
            if MDI should be computed based on weighted node samples, by default True
        normalize : bool, optional
            if MDI should be normalized after computing, by default True

        Returns
        -------
        np.ndarray
            MDI importances in same order than the features

        Raises
        ------
        ValueError
            Safecheck error when model does not come from expected API

        """
        if self._base == 'scikit-learn':
            return self._compute_sklearn_importances(X=X, weighted=weighted, normalize=normalize)
        raise ValueError('only scikit-learn estimators are accepted')

    def _compute_sklearn_importances(self, X=None, normalize=True, weighted=True):
        """Compute MDI importances following scikit-learn API

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute MDI; when None use original dataset, by default None
        weighted : bool, optional
            if MDI should be computed based on weighted node samples, by default True
        normalize : bool, optional
            if MDI should be normalized after computing, by default True

        Returns
        -------
        np.ndarray
            MDI importances in same order than the features

        """
        if not self._is_forest:
            return self._compute_sklearn_tree_importances(self.estimator, X=X, weighted=weighted, normalize=normalize)
        return self._compute_sklearn_forest_importances(X=X, weighted=weighted, normalize=normalize)

    def _compute_sklearn_forest_importances(self, X=None, weighted=True, normalize=True):
        """Compute MDI importances following scikit-learn API on a forest-like model

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute MDI; when None use original dataset, by default None
        weighted : bool, optional
            if MDI should be computed based on weighted node samples, by default True
        normalize : bool, optional
            if MDI should be normalized after computing, by default True

        Returns
        -------
        np.ndarray
            MDI importances in same order than the features

        """
        trees = self.estimator.estimators_
        importances = np.zeros(self.n_features_)
        n_estimators = 0

        for e in trees:
            if isinstance(e, BaseDecisionTree):
                n_estimators += 1
                importances += self._compute_sklearn_tree_importances(e, X=X, weighted=weighted, normalize=normalize)
            else: # specific case of sklearn gradient boosting models
                for e_ in e:
                    if e_.tree_.node_count > 1:
                        n_estimators += 1
                        importances += self._compute_sklearn_tree_importances(e_, X=X, weighted=weighted, normalize=False)

        importances /= n_estimators
        if normalize:
            importances = normalize_array(importances, axis=None)

        return importances

    def _compute_sklearn_tree_importances(self, estimator, X=None, weighted=True, normalize=True):
        """Compute MDI importances following scikit-learn API on a tree-like model

        Parameters
        ----------
        X : np.ndarray, pd.DataFrame or None, optional
            data used to compute MDI; when None use original dataset, by default None
        weighted : bool, optional
            if MDI should be computed based on weighted node samples, by default True
        normalize : bool, optional
            if MDI should be normalized after computing, by default True

        Returns
        -------
        np.ndarray
            MDI importances in same order than the features

        """
        importances = np.zeros(self.n_features_)
        nodes = get_sklearn_nodes_from(estimator, X=X, weighted=weighted)
        for n in nodes:
            if isinstance(n, Node):
                left, right = nodes[n.left], nodes[n.right]
                importances[n.feature] += self._compute_impurity_importance_from(n, left, right)
        importances /= nodes[0].n_node_samples
        if normalize:
            importances = normalize_array(importances, axis=None)
        return importances

    @staticmethod
    def _compute_impurity_importance_from(node, left, right):
        """Impurity performance between a node and its children

        Parameters
        ----------
        node : eml._tree.nodes.Node
            parent node used to compute impurity gain
        left : eml._tree.nodes.Node or eml._tree.nodes.Leaf
            left child of parent node
        right : eml._tree.nodes.Node or eml._tree.nodes.Leaf
            right child of parent node

        Returns
        -------
        float
            impurity gain with split from parent node

        """
        return node.weighted_impurity - left.weighted_impurity - right.weighted_impurity
