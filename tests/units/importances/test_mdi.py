import math

import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, BaseDecisionTree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble._forest import BaseForest
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.svm import SVC
from sklearn.exceptions import NotFittedError
from scipy.sparse import csr_matrix

from eml._tree.nodes import Node, Leaf
from eml.importances.mdi import MeanDecreaseImpurity


@pytest.fixture
def iris():
    return load_iris()


@pytest.fixture
def decision_tree():
    return DecisionTreeClassifier()


@pytest.fixture
def fitted_decision_tree(iris, decision_tree):
    return decision_tree.fit(iris.data, iris.target)


@pytest.fixture
def random_forest():
    return RandomForestClassifier()


@pytest.fixture
def fitted_random_forest(iris, random_forest):
    return random_forest.fit(iris.data, iris.target)


@pytest.fixture
def ada_boost():
    return AdaBoostClassifier()


@pytest.fixture
def fitted_ada_boost(iris, ada_boost):
    return ada_boost.fit(iris.data, iris.target)


@pytest.fixture
def gradient_boosting():
    return GradientBoostingClassifier()


@pytest.fixture
def fitted_gradient_boosting(iris, gradient_boosting):
    return gradient_boosting.fit(iris.data, iris.target)


@pytest.fixture
def pseudo_tree():
    """PseudoTree used on binary features (reconstruction of a scikit-learn work around)"""

    class PseudoTreeAttributes(object):
        def __init__(self):
            self.feature = [0, 1, -1, -1, 2, -1, -1]
            self.value = [[.7, .3], [.7, .3], [1., 0.], [0., 1.], [.7, .3], [1., 0.], [0., 1.]]
            self.impurity = [.42, .42, 0., 0., .42, 0., 0.]
            self.children_left = [1, 2, -1, -1, 5, -1, -1]
            self.children_right = [4, 3, -1, -1, 6, -1, -1]
            self.n_node_samples = [100, 50, 35, 15, 50, 35, 15]
            self.weighted_n_node_samples = [130., 65., 35., 30., 65., 35., 30.]
            self.node_count = 7

    class PseudoTree(BaseDecisionTree):
        def __init__(self):
            self.tree_ = PseudoTreeAttributes()

        def decision_path(self, X):
            current_node = 0
            decisions_paths = np.zeros((X.shape[0], len(self.tree_.value)), dtype=int)
            decisions_paths[:, 0] = 1
            for idx, x in enumerate(X):
                current_feature = self.tree_.feature[current_node]
                while current_feature != -1:
                    if x[current_feature] == 1:
                        current_node = self.tree_.children_right[current_node]
                    else:
                        current_node = self.tree_.children_left[current_node]
                    decisions_paths[idx, current_node] += 1
                    current_feature = self.tree_.feature[current_node]
                current_node = 0
            return csr_matrix(decisions_paths)

    return PseudoTree()


@pytest.fixture
def pseudo_forest(pseudo_tree):
    """PseudoForest used on binary features (reconstruction of a scikit-learn work around)"""

    class PseudoForest(BaseForest):
        def __init__(self):
            self.estimators_ = [pseudo_tree]

        def _set_oob_score(self):
            pass

    return PseudoForest()


@pytest.fixture
def pseudo_gradient_boosting(pseudo_tree):
    """PseudoForest used on binary features (reconstruction of a scikit-learn work around)"""

    class PseudoGradientBoosting(BaseGradientBoosting):
        def __init__(self):
            self.estimators_ = [np.array([pseudo_tree])]

    return PseudoGradientBoosting()


def test_mdi_initialization():
    mdi = MeanDecreaseImpurity(use_precompute=False)

    assert hasattr(mdi, 'use_precompute')
    assert hasattr(mdi, 'estimator')
    assert hasattr(mdi, 'n_features_')
    assert hasattr(mdi, '_base')
    assert hasattr(mdi, '_name')
    assert hasattr(mdi, '_is_forest')


def test_mdi_initialization_attributes():
    mdi = MeanDecreaseImpurity(use_precompute=False)

    assert mdi.use_precompute is False
    assert mdi.estimator is None
    assert mdi.n_features_ is None
    assert mdi._base is None
    assert mdi._name is None
    assert mdi._is_forest is None


def test_fit_tree(fitted_decision_tree):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_decision_tree)
    assert mdi.estimator == fitted_decision_tree
    assert mdi.estimator is fitted_decision_tree

    assert mdi._base == 'scikit-learn'
    assert mdi._name == 'feature_importances_'
    assert mdi._is_forest is False

    assert mdi.n_features_ == fitted_decision_tree.n_features_
    assert mdi.n_features_ is fitted_decision_tree.n_features_


def test_fit_forest(fitted_random_forest):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_random_forest)
    assert mdi.estimator == fitted_random_forest
    assert mdi.estimator is fitted_random_forest

    assert mdi._base == 'scikit-learn'
    assert mdi._name == 'feature_importances_'
    assert mdi._is_forest is True

    assert mdi.n_features_ == fitted_random_forest.n_features_
    assert mdi.n_features_ is fitted_random_forest.n_features_


def test_fit_ada_boost(fitted_ada_boost):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_ada_boost)
    assert mdi.estimator == fitted_ada_boost
    assert mdi.estimator is fitted_ada_boost

    assert mdi._base == 'scikit-learn'
    assert mdi._name == 'feature_importances_'
    assert mdi._is_forest is True

    assert mdi.n_features_ == fitted_ada_boost.estimators_[0].n_features_
    assert mdi.n_features_ is fitted_ada_boost.estimators_[0].n_features_
    

def test_fit_raise_error(iris, random_forest):
    mdi = MeanDecreaseImpurity(use_precompute=False)

    with pytest.raises(NotFittedError):
        mdi.fit(random_forest)

    svc = SVC(kernel='linear', C=1)
    svc.fit(iris.data, iris.target)

    with pytest.raises(TypeError):
        mdi.fit(svc)


def test__compute_impurity_importance_from():
    node = Node(index=1, left=2, right=3, feature=0, value=[.7, .3], impurity=.42, n_node_samples=100)
    left = Leaf(index=2, value=[1., 0.], impurity=0., n_node_samples=70)
    right = Leaf(index=3, value=[0., 1.], impurity=0., n_node_samples=30)

    impurity_importance = MeanDecreaseImpurity._compute_impurity_importance_from(node, left, right)
    expected_impurity_importance = 42.
    assert math.isclose(impurity_importance, expected_impurity_importance)


def test__compute_sklearn_tree_importances(pseudo_tree):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.n_features_ = 4

    importances1 = mdi._compute_sklearn_tree_importances(pseudo_tree, X=None, weighted=False, normalize=False)
    importances2 = mdi._compute_sklearn_tree_importances(pseudo_tree, X=None, weighted=False, normalize=True)
    importances3 = mdi._compute_sklearn_tree_importances(pseudo_tree, X=None, weighted=True, normalize=False)
    importances4 = mdi._compute_sklearn_tree_importances(pseudo_tree, X=None, weighted=True, normalize=True)

    expected_importances1 = np.array([0., .21, .21, 0.])
    expected_importances2 = np.array([0., .5, .5, 0.])
    expected_importances3 = np.array([0., .21, .21, 0.])
    expected_importances4 = np.array([0., .5, .5, 0.])

    np.testing.assert_allclose(importances1, expected_importances1)
    np.testing.assert_allclose(importances2, expected_importances2)
    np.testing.assert_allclose(importances3, expected_importances3)
    np.testing.assert_allclose(importances4, expected_importances4)


def test__compute_sklearn_forest_importances(pseudo_forest):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.estimator = pseudo_forest
    mdi.n_features_ = 4

    importances1 = mdi._compute_sklearn_forest_importances(X=None, weighted=False, normalize=False)
    importances2 = mdi._compute_sklearn_forest_importances(X=None, weighted=False, normalize=True)
    importances3 = mdi._compute_sklearn_forest_importances(X=None, weighted=True, normalize=False)
    importances4 = mdi._compute_sklearn_forest_importances(X=None, weighted=True, normalize=True)

    expected_importances1 = np.array([0., .21, .21, 0.])
    expected_importances2 = np.array([0., .5, .5, 0.])
    expected_importances3 = np.array([0., .21, .21, 0.])
    expected_importances4 = np.array([0., .5, .5, 0.])

    np.testing.assert_allclose(importances1, expected_importances1)
    np.testing.assert_allclose(importances2, expected_importances2)
    np.testing.assert_allclose(importances3, expected_importances3)
    np.testing.assert_allclose(importances4, expected_importances4)


def test__compute_sklearn_forest_importances_for_gradient_boosting(pseudo_gradient_boosting):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.estimator = pseudo_gradient_boosting
    mdi.n_features_ = 4

    importances1 = mdi._compute_sklearn_forest_importances(X=None, weighted=False, normalize=False)
    importances2 = mdi._compute_sklearn_forest_importances(X=None, weighted=False, normalize=True)
    importances3 = mdi._compute_sklearn_forest_importances(X=None, weighted=True, normalize=False)
    importances4 = mdi._compute_sklearn_forest_importances(X=None, weighted=True, normalize=True)

    expected_importances1 = np.array([0., .21, .21, 0.])
    expected_importances2 = np.array([0., .5, .5, 0.])
    expected_importances3 = np.array([0., .21, .21, 0.])
    expected_importances4 = np.array([0., .5, .5, 0.])

    np.testing.assert_allclose(importances1, expected_importances1)
    np.testing.assert_allclose(importances2, expected_importances2)
    np.testing.assert_allclose(importances3, expected_importances3)
    np.testing.assert_allclose(importances4, expected_importances4)


def test__compute_sklearn_importances_forest(fitted_random_forest, mocker):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_random_forest)

    spy1 = mocker.spy(mdi, '_compute_sklearn_tree_importances')
    spy2 = mocker.spy(mdi, '_compute_sklearn_forest_importances')

    mdi._compute_sklearn_importances(X=None, normalize=True, weighted=True)

    last_estimator = mdi.estimator.estimators_[-1]
    spy1.assert_called_with(last_estimator, X=None, weighted=True, normalize=True)
    spy2.assert_called_once_with(X=None, weighted=True, normalize=True)


def test__compute_sklearn_importances_tree(fitted_decision_tree, mocker):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_decision_tree)

    spy1 = mocker.spy(mdi, '_compute_sklearn_tree_importances')
    spy2 = mocker.spy(mdi, '_compute_sklearn_forest_importances')

    mdi._compute_sklearn_importances(X=None, normalize=True, weighted=True)

    spy1.assert_called_once_with(fitted_decision_tree, X=None, weighted=True, normalize=True)
    spy2.assert_not_called()


def test__compute_importances(fitted_random_forest, mocker):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_random_forest)

    spy = mocker.spy(mdi, '_compute_sklearn_importances')

    mdi._compute_importances(X=None, weighted=True, normalize=True)

    spy.assert_called_once_with(X=None, weighted=True, normalize=True)


def test__compute_importances_error(iris, fitted_random_forest):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_random_forest)
    mdi._base = 'xgboost'

    with pytest.raises(ValueError):
        mdi._compute_importances(X=None, weighted=True, normalize=True)


def test_interpret_values(fitted_decision_tree, fitted_random_forest, fitted_gradient_boosting, fitted_ada_boost):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    for e in (fitted_decision_tree, fitted_random_forest, fitted_gradient_boosting, fitted_ada_boost):
        mdi.fit(e)
        importances = mdi.interpret(X=None, weighted=True, normalize=True)
        np.testing.assert_allclose(e.feature_importances_, importances)


def test_interpret_not_precomputed(fitted_random_forest, mocker):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_random_forest)

    spy = mocker.spy(mdi, '_compute_importances')
    importances = mdi.interpret(X=None, weighted=True, normalize=True)
    spy.assert_called_once_with(X=None, weighted=True, normalize=True)

    np.testing.assert_allclose(fitted_random_forest.feature_importances_, importances)


def test_interpret_precomputed(fitted_random_forest, mocker):
    mdi = MeanDecreaseImpurity(use_precompute=True)
    mdi.fit(fitted_random_forest)

    spy = mocker.spy(mdi, '_compute_importances')
    importances = mdi.interpret(X=None, weighted=True, normalize=True)
    spy.assert_not_called()

    assert np.array_equal(mdi.estimator.feature_importances_, importances)


def test_predict(fitted_random_forest, mocker):
    mdi = MeanDecreaseImpurity(use_precompute=False)
    mdi.fit(fitted_random_forest)

    spy = mocker.spy(mdi, 'interpret')
    _ = mdi.predict(X=None, weighted=True, normalize=True)
    spy.assert_called_once_with(X=None, weighted=True, normalize=True)
