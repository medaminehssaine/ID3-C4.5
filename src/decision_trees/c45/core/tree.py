"""c4.5 decision tree classifier"""
from collections import Counter
from .gain_ratio import (
    gain_ratio, information_gain, split_info,
    is_continuous, best_threshold, entropy
)
from .node import Node


class C45Classifier:
    """
    c4.5 decision tree classifier
    
    improvements over id3:
    - gain ratio instead of info gain
    - handles continuous attributes
    - handles missing values
    - optional post-pruning
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, 
                 min_gain_ratio=0.01, confidence=0.25):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain_ratio = min_gain_ratio
        self.confidence = confidence
        
        self.root = None
        self.feature_names = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_types_ = None  # 'continuous' or 'categorical'
    
    def fit(self, X, y, feature_names=None):
        """
        build decision tree from training data
        
        X: list of samples (list of lists/tuples)
        y: list of labels
        feature_names: optional names for features
        """
        self.n_features_ = len(X[0]) if X else 0
        self.feature_names = feature_names or [f"f{i}" for i in range(self.n_features_)]
        self.classes_ = list(set(y))
        
        # detect feature types
        self.feature_types_ = []
        for i in range(self.n_features_):
            if is_continuous(X, i):
                self.feature_types_.append('continuous')
            else:
                self.feature_types_.append('categorical')
        
        # available features (can be reused for continuous in c4.5)
        available = set(range(self.n_features_))
        
        # Build initial tree
        self.root = self._build_tree(X, y, available, depth=0)
        
        # Apply Pessimistic Error Pruning (Wilson Score)
        from .pruning import pessimistic_prune
        if self.root:
            pessimistic_prune(self.root, self.confidence)
            
        return self
    
    def _build_tree(self, X, y, available_features, depth):
        """recursively build tree"""
        node = Node()
        node.samples = len(y)
        node.depth = depth
        
        # class distribution for pruning
        counts = Counter(y)
        node.class_distribution = dict(counts)
        most_common = counts.most_common(1)[0][0]
        node.label = most_common
        
        # stopping conditions
        if len(counts) == 1:
            node.is_leaf = True
            return node
        
        if not available_features:
            node.is_leaf = True
            return node
        
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return node
        
        if len(y) < self.min_samples_split:
            node.is_leaf = True
            return node
        
        # find best split
        best_feature = None
        best_gr = -1
        best_threshold_val = None
        
        for f in available_features:
            if self.feature_types_[f] == 'continuous':
                # find best threshold
                t, gr = best_threshold(X, y, f)
                if gr > best_gr:
                    best_gr = gr
                    best_feature = f
                    best_threshold_val = t
            else:
                # categorical
                gr = gain_ratio(X, y, f)
                if gr > best_gr:
                    best_gr = gr
                    best_feature = f
                    best_threshold_val = None
        
        # check minimum gain
        if best_gr < self.min_gain_ratio:
            node.is_leaf = True
            return node
        
        if best_feature is None:
            node.is_leaf = True
            return node
        
        # create split node
        node.feature = best_feature
        node.feature_name = self.feature_names[best_feature]
        node.is_leaf = False
        
        if best_threshold_val is not None:
            # continuous split
            node.threshold = best_threshold_val
            node.is_continuous = True
            
            # split data
            left_idx = [i for i, s in enumerate(X) if s[best_feature] is not None 
                       and float(s[best_feature]) <= best_threshold_val]
            right_idx = [i for i, s in enumerate(X) if s[best_feature] is not None 
                        and float(s[best_feature]) > best_threshold_val]
            
            X_left = [X[i] for i in left_idx]
            y_left = [y[i] for i in left_idx]
            X_right = [X[i] for i in right_idx]
            y_right = [y[i] for i in right_idx]
            
            # continuous features can be reused
            if X_left:
                node.left = self._build_tree(X_left, y_left, available_features, depth + 1)
            if X_right:
                node.right = self._build_tree(X_right, y_right, available_features, depth + 1)
        else:
            # categorical split
            node.is_continuous = False
            
            # group by value
            splits = {}
            for i, sample in enumerate(X):
                val = sample[best_feature]
                if val is not None:
                    if val not in splits:
                        splits[val] = ([], [])
                    splits[val][0].append(sample)
                    splits[val][1].append(y[i])
            
            # remove feature for categorical (id3 style)
            remaining = available_features - {best_feature}
            
            for val, (X_sub, y_sub) in splits.items():
                child = self._build_tree(X_sub, y_sub, remaining, depth + 1)
                node.children[val] = child
        
        return node
    
    def predict(self, X):
        """predict class labels for samples"""
        if self.root is None:
            raise ValueError("tree not fitted")
        return [self.root.predict_one(sample) for sample in X]
    
    def predict_one(self, sample):
        """predict single sample"""
        if self.root is None:
            raise ValueError("tree not fitted")
        return self.root.predict_one(sample)
    
    def get_depth(self):
        """max depth of tree"""
        if self.root is None:
            return 0
        return self._get_depth(self.root)
    
    def _get_depth(self, node):
        if node.is_leaf:
            return 0
        
        depths = []
        if node.is_continuous:
            if node.left:
                depths.append(self._get_depth(node.left))
            if node.right:
                depths.append(self._get_depth(node.right))
        else:
            for child in node.children.values():
                depths.append(self._get_depth(child))
        
        return 1 + max(depths) if depths else 0
    
    def get_n_leaves(self):
        """count leaf nodes"""
        if self.root is None:
            return 0
        return self._count_leaves(self.root)
    
    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        
        count = 0
        if node.is_continuous:
            if node.left:
                count += self._count_leaves(node.left)
            if node.right:
                count += self._count_leaves(node.right)
        else:
            for child in node.children.values():
                count += self._count_leaves(child)
        
        return count
    
    def __repr__(self):
        if self.root is None:
            return "C45Classifier(not fitted)"
        return f"C45Classifier(depth={self.get_depth()}, leaves={self.get_n_leaves()})"
