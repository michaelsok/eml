from eml.utils.helpers import are_equal


class Leaf:
    """Leaf node from a tree like model"""
    def __init__(self, index, value, impurity, n_node_samples):
        self.index = index
        self.value = value
        self.impurity = impurity
        self.n_node_samples = n_node_samples

    def __eq__(self, obj):
        return are_equal(self, obj)

    def __ne__(self, obj):
        return not self.__eq__(obj)

    @property
    def weighted_impurity(self):
        """Weighted impurity by node samples

        Returns
        -------
        float
            weighted impurity by node samples

        """
        return self.n_node_samples * self.impurity


class Node(Leaf):
    """Node from a tree like model. Subclass of Leaf since a node is a leaf for a subtree"""
    def __init__(self, index, left, right, feature, value, impurity, n_node_samples):
        super().__init__(index=index, value=value, impurity=impurity, n_node_samples=n_node_samples)
        self.left = left
        self.right = right
        self.feature = feature


def _get_node_from(index, left, right, feature, value, impurity, n_node_samples):
    """Return a Node (or a Leaf) from its attributes

    Parameters
    ----------
    index : int
        node identifier (between 0 and n_nodes)
    left : int
        identifier of left child of the node
    right : int
        identifier of right child of the node
    feature : int
        index
    value : float or list
        value present in node, float if regression, list for classification
    impurity : float
        impurity present in node
    n_node_samples : float
        number of samples in node

    Returns
    -------
    Node or Leaf
        Node instantiated if children, Leaf otherwise

    """
    if (left == -1) & (right == -1):
        return Leaf(index, value, impurity, n_node_samples)
    return Node(index, left, right, feature, value, impurity, n_node_samples)


def get_sklearn_nodes_from(tree, X=None, weighted=True):
    """Return sklearn instantiated nodes from a tree
    TODO: X & weighted : two n_node_samples from two children don't sum up to n_node_samples from their parent
    TODO: it should be fixed by introducing a y and use it to obtain true weight

    Parameters
    ----------
    tree : instance of BaseDecisionTree
        tree from which nodes should be obtained
    X : np.ndarray, pd.DataFrame or None, optional
        data used to replace node samples
    weighted : bool, optional
        if weighted node samples should be used, by default True

    Returns
    -------
    list of Node or Leaf
        nodes and leaves present in tree

    """
    tree_ = tree.tree_
    features = tree_.feature
    impurities = tree_.impurity
    values = tree_.value
    left_children = tree_.children_left
    right_children = tree_.children_right

    n_node_samples_name = 'weighted_n_node_samples' if weighted else 'n_node_samples'
    n_node_samples = getattr(tree_, n_node_samples_name)
    if X is not None:
        activated_nodes = tree.decision_path(X).toarray()
        n_node_samples = (activated_nodes * n_node_samples / tree_.n_node_samples).sum(axis=0)

    attributes = zip(left_children, right_children, features, values, impurities, n_node_samples)
    return [_get_node_from(i, *initializers) for i, initializers in enumerate(attributes)]
