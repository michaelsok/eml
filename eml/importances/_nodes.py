class Leaf:
    def __init__(self, index, feature, value, impurity, n_node_samples):
        self.index = index
        self.feature = feature
        self.value = value
        self.impurity = impurity
        self.n_node_samples = n_node_samples

    @property
    def weighted_impurity(self):
        return self.n_node_samples * self.impurity


class Node(Leaf):
    def __init__(self, index, left, right, feature, value, impurity, n_node_samples):
        super().__init__(index=index, feature=feature, value=value, impurity=impurity, n_node_samples=n_node_samples)
        self.left = left
        self.right = right


def _get_node_from(index, left, right, feature, value, impurity, n_node_samples):
    if (left == -1) & (right == -1):
        return Leaf(index, feature, value, impurity, n_node_samples)
    return Node(index, left, right, feature, value, impurity, n_node_samples)


def get_sklearn_nodes_from(tree, weighted=True):
    tree_ = tree.tree_
    features = tree_.feature
    impurities = tree_.impurity
    n_node_samples_name = 'weighted_n_node_samples' if weighted else 'n_node_samples'
    n_node_samples = getattr(tree_, n_node_samples_name)
    values = tree_.value
    left_children = tree_.children_left
    right_children = tree_.children_right
    attributes = zip(left_children, right_children, features, values, impurities, n_node_samples)
    return [_get_node_from(i, *initializers) for i, initializers in enumerate(attributes)]
