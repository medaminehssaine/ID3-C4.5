"""id3 decision tree classifier"""
from collections import Counter
from .entropy import entropy, information_gain
from .node import Node


class ID3Classifier:
    """
    id3 decision tree classifier
    
    the classic algorithm by quinlan (1986)
    uses information gain to select splitting features
    """
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
        self.classes_ = None
        self.n_features_ = None
    
    def fit(self, X, y, feature_names=None):
        """
        build decision tree from training data
        
        X: list of samples (list of lists)
        y: list of labels
        feature_names: optional names for features
        """
        self.n_features_ = len(X[0]) if X else 0
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features_)]
        self.classes_ = list(set(y))
        
        # available features for splitting
        available = set(range(self.n_features_))
        
        self.root = self._build_tree(X, y, available, depth=0)
        return self
    
    def _build_tree(self, X, y, available_features, depth):
        """recursively build the tree"""
        node = Node()
        node.samples = len(y)
        node.depth = depth
        
        # count class distribution
        counts = Counter(y)
        most_common = counts.most_common(1)[0][0]
        node.label = most_common  # default prediction
        
        # stopping conditions
        if len(counts) == 1:
            # pure node
            node.is_leaf = True
            return node
        
        if not available_features:
            # no more features to split
            node.is_leaf = True
            return node
        
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return node
        
        if len(y) < self.min_samples_split:
            node.is_leaf = True
            return node
        
        # find best feature to split on
        best_feature = None
        best_gain = -1
        
        for f in available_features:
            gain = information_gain(X, y, f)
            if gain > best_gain:
                best_gain = gain
                best_feature = f
        
        # no information gain possible
        if best_gain <= 0:
            node.is_leaf = True
            return node
        
        # create internal node
        node.feature = best_feature
        node.feature_name = self.feature_names[best_feature]
        node.is_leaf = False
        
        # split data by feature value
        splits = {}
        for i, sample in enumerate(X):
            val = sample[best_feature]
            if val not in splits:
                splits[val] = ([], [])
            splits[val][0].append(sample)
            splits[val][1].append(y[i])
        
        # remove used feature (id3 doesn't reuse features)
        remaining = available_features - {best_feature}
        
        # recursively build children
        for val, (X_subset, y_subset) in splits.items():
            child = self._build_tree(X_subset, y_subset, remaining, depth + 1)
            node.children[val] = child
        
        return node
    
    def predict(self, X):
        """predict class labels for samples"""
        if self.root is None:
            raise ValueError("tree not fitted yet, call fit() first")
        
        return [self.root.predict_one(sample) for sample in X]
    
    def predict_one(self, sample):
        """predict class for single sample"""
        if self.root is None:
            raise ValueError("tree not fitted yet")
        return self.root.predict_one(sample)
    
    def get_depth(self):
        """return max depth of tree"""
        if self.root is None:
            return 0
        return self._get_depth(self.root)
    
    def _get_depth(self, node):
        if node.is_leaf:
            return 0
        return 1 + max(self._get_depth(child) for child in node.children.values())
    
    def get_n_leaves(self):
        """count leaf nodes"""
        if self.root is None:
            return 0
        return self._count_leaves(self.root)
    
    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        return sum(self._count_leaves(child) for child in node.children.values())
    
    def __repr__(self):
        if self.root is None:
            return "ID3Classifier(not fitted)"
        return f"ID3Classifier(depth={self.get_depth()}, leaves={self.get_n_leaves()})"
