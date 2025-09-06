class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = set(y)

        # If only one class left or max depth reached, return a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._most_common_class(y)
            return self.Node(value=leaf_value)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, num_features)
        if best_feature is None:
            leaf_value = self._most_common_class(y)
            return self.Node(value=leaf_value)

        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return self.Node(feature=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(num_features):
            thresholds = set(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, X, y, feature, threshold):
        parent_entropy = self._entropy(y)
        left_indices = X[:, feature] < threshold
        right_indices = X[:, feature] >= threshold
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0

        n = len(y)
        n_left = len(y[left_indices])
        n_right = len(y[right_indices])
        child_entropy = (n_left / n) * self._entropy(y[left_indices]) + (n_right / n) * self._entropy(y[right_indices])

        return parent_entropy - child_entropy

    def _entropy(self, y):
        from collections import Counter
        import math
        count = Counter(y)
        probabilities = [count[c] / len(y) for c in count]
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    def _most_common_class(self, y):
        from collections import Counter
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return [self._predict(sample) for sample in X]

    def _predict(self, sample):
        node = self.tree
        while node.value is None:
            if sample[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value