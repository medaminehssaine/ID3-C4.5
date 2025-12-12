"""c4.5 decision tree node with threshold support"""


class Node:
    """
    decision tree node for c4.5
    supports both categorical and continuous splits
    """
    
    def __init__(self, feature=None, feature_name=None, threshold=None,
                 children=None, label=None, is_leaf=False):
        # split attributes
        self.feature = feature
        self.feature_name = feature_name
        self.threshold = threshold      # for continuous: split point
        self.is_continuous = threshold is not None
        
        # children nodes
        self.children = children or {}  # categorical: {value: node}
        self.left = None               # continuous: <= threshold
        self.right = None              # continuous: > threshold
        
        # leaf attributes
        self.label = label
        self.is_leaf = is_leaf
        
        # stats
        self.samples = 0
        self.depth = 0
        self.class_distribution = {}   # for pruning decisions
    
    def predict_one(self, sample, default=None):
        """predict class for single sample"""
        if self.is_leaf:
            return self.label
        
        val = sample[self.feature]
        
        # handle missing value
        if val is None:
            return self.label or default
        
        if self.is_continuous:
            # binary split on the threshold
            try:
                val = float(val)
            except (ValueError, TypeError):
                return self.label or default
            
            if val <= self.threshold:
                if self.left:
                    return self.left.predict_one(sample, default)
            else:
                if self.right:
                    return self.right.predict_one(sample, default)
            return self.label or default
        else:
            # categorical split
            if val in self.children:
                return self.children[val].predict_one(sample, default)
            else:
                return self.label or default
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.label})"
        if self.is_continuous:
            return f"Node({self.feature_name} <= {self.threshold:.2f})"
        return f"Node({self.feature_name}, children={list(self.children.keys())})"
