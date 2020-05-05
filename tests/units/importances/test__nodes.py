import math
import copy

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.tree import BaseDecisionTree

from eml.importances._nodes import Leaf, Node, _get_node_from, get_sklearn_nodes_from


@pytest.fixture
def leaf_initializers():
    initializers = dict(index=1, value=1, impurity=.3, n_node_samples=1900.)
    return initializers


@pytest.fixture
def leaf_initializers_for_inequality():
    initializers = dict(index=2, value=1, impurity=.3, n_node_samples=1900.)
    return initializers


@pytest.fixture
def node_initializers():
    initializers = dict(index=1, left=2, right=3, feature=1, value=1, impurity=.3, n_node_samples=1900.)
    return initializers


@pytest.fixture
def node_initializers_for_inequality():
    initializers = dict(index=2, left=3, right=-1, feature=1, value=1, impurity=.3, n_node_samples=1900.)
    return initializers


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


def test_leaf_initialization(leaf_initializers):
    leaf = Leaf(**leaf_initializers)
    assert hasattr(leaf, 'index')
    assert hasattr(leaf, 'value')
    assert hasattr(leaf, 'impurity')
    assert hasattr(leaf, 'n_node_samples')
    assert hasattr(leaf, 'weighted_impurity')


def test_leaf_initialization_attributes(leaf_initializers):
    leaf = Leaf(**leaf_initializers)
    for attribute, value in leaf_initializers.items():
        assert getattr(leaf, attribute) == value
    assert math.isclose(leaf.weighted_impurity, leaf_initializers['n_node_samples'] * leaf_initializers['impurity'])


def test_leaf_equality(leaf_initializers):
    leaf1, leaf2 = Leaf(**leaf_initializers), Leaf(**leaf_initializers)
    assert leaf1 == leaf2
    assert not (leaf1 is leaf2)


def test_leaf_inequality(leaf_initializers, leaf_initializers_for_inequality):
    leaf1, leaf2 = Leaf(**leaf_initializers), Leaf(**leaf_initializers_for_inequality)
    assert leaf1 != leaf2
    assert not (leaf1 is leaf2)


def test_node_initialization(node_initializers):
    node = Node(**node_initializers)
    assert hasattr(node, 'index')
    assert hasattr(node, 'left')
    assert hasattr(node, 'right')
    assert hasattr(node, 'feature')
    assert hasattr(node, 'value')
    assert hasattr(node, 'impurity')
    assert hasattr(node, 'n_node_samples')
    assert hasattr(node, 'weighted_impurity')


def test_node_initialization_attributes(node_initializers):
    node = Node(**node_initializers)
    for attribute, value in node_initializers.items():
        assert getattr(node, attribute) == value
    assert math.isclose(node.weighted_impurity, node_initializers['n_node_samples'] * node_initializers['impurity'])


def test_node_equality(node_initializers):
    node1, node2 = Node(**node_initializers), Node(**node_initializers)
    assert node1 == node2
    assert not (node1 is node2)


def test_node_inequality(node_initializers, node_initializers_for_inequality):
    node1, node2 = Node(**node_initializers), Node(**node_initializers_for_inequality)
    assert node1 != node2
    assert not (node1 is node2)


def test__get_node_from(node_initializers):
    node = _get_node_from(**node_initializers)
    expected_node = Node(**node_initializers)
    assert node == expected_node
    assert not (node is expected_node)

    leaf_initializers = copy.deepcopy(node_initializers)
    leaf_initializers['left'], leaf_initializers['right'] = -1, -1
    leaf = _get_node_from(**leaf_initializers)
    del leaf_initializers['left']; del leaf_initializers['right']; del leaf_initializers['feature']
    expected_leaf = Leaf(**leaf_initializers)
    assert leaf == expected_leaf
    assert not (leaf is expected_leaf)


def test_get_sklearn_nodes_from(pseudo_tree):
    nodes = get_sklearn_nodes_from(pseudo_tree)
    expected_nodes = [
        Node(index=0, left=1, right=4, feature=0, value=[.7, .3], impurity=.42, n_node_samples=130.),
        Node(index=1, left=2, right=3, feature=1, value=[.7, .3], impurity=.42, n_node_samples=65.),
        Leaf(index=2, value=[1., 0.], impurity=0., n_node_samples=35.),
        Leaf(index=3, value=[0., 1.], impurity=0., n_node_samples=30.),
        Node(index=4, left=5, right=6, feature=2, value=[.7, .3], impurity=.42, n_node_samples=65.),
        Leaf(index=5, value=[1., 0.], impurity=0., n_node_samples=35.),
        Leaf(index=6, value=[0., 1.], impurity=0., n_node_samples=30.)
    ]

    assert nodes == expected_nodes


def test_get_sklearn_nodes_from_non_weighted(pseudo_tree):
    nodes = get_sklearn_nodes_from(pseudo_tree, weighted=False)
    expected_nodes = [
        Node(index=0, left=1, right=4, feature=0, value=[.7, .3], impurity=.42, n_node_samples=100),
        Node(index=1, left=2, right=3, feature=1, value=[.7, .3], impurity=.42, n_node_samples=50),
        Leaf(index=2, value=[1., 0.], impurity=0., n_node_samples=35),
        Leaf(index=3, value=[0., 1.], impurity=0., n_node_samples=15),
        Node(index=4, left=5, right=6, feature=2, value=[.7, .3], impurity=.42, n_node_samples=50),
        Leaf(index=5, value=[1., 0.], impurity=0., n_node_samples=35),
        Leaf(index=6, value=[0., 1.], impurity=0., n_node_samples=15)
    ]

    assert nodes == expected_nodes


def test_get_sklearn_nodes_from_non_weighted_X(pseudo_tree):
    X = np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1]
    ])
    nodes = get_sklearn_nodes_from(pseudo_tree, X=X, weighted=False)
    expected_nodes = [
        Node(index=0, left=1, right=4, feature=0, value=[.7, .3], impurity=.42, n_node_samples=8),
        Node(index=1, left=2, right=3, feature=1, value=[.7, .3], impurity=.42, n_node_samples=4),
        Leaf(index=2, value=[1., 0.], impurity=0., n_node_samples=2),
        Leaf(index=3, value=[0., 1.], impurity=0., n_node_samples=2),
        Node(index=4, left=5, right=6, feature=2, value=[.7, .3], impurity=.42, n_node_samples=4),
        Leaf(index=5, value=[1., 0.], impurity=0., n_node_samples=0),
        Leaf(index=6, value=[0., 1.], impurity=0., n_node_samples=4)
    ]

    assert nodes == expected_nodes


def test_get_sklearn_nodes_from_weighted_X(pseudo_tree):
    X = np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1]
    ])
    nodes = get_sklearn_nodes_from(pseudo_tree, X=X, weighted=True)
    expected_nodes = [
        Node(index=0, left=1, right=4, feature=0, value=[.7, .3], impurity=.42, n_node_samples=10.4),
        Node(index=1, left=2, right=3, feature=1, value=[.7, .3], impurity=.42, n_node_samples=5.2),
        Leaf(index=2, value=[1., 0.], impurity=0., n_node_samples=2.),
        Leaf(index=3, value=[0., 1.], impurity=0., n_node_samples=4.),
        Node(index=4, left=5, right=6, feature=2, value=[.7, .3], impurity=.42, n_node_samples=5.2),
        Leaf(index=5, value=[1., 0.], impurity=0., n_node_samples=0.),
        Leaf(index=6, value=[0., 1.], impurity=0., n_node_samples=8.)
    ]

    assert nodes == expected_nodes
