class Dataset:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class Prediction:
    def __init__(self, class_label, probability):
        self.class_label = class_label
        self.probability = probability