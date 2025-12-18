"""decision tree node structure"""


class Node:
    """represents a node in the decision tree"""
    
    def __init__(self, feature=None, feature_name=None, children=None, 
                 label=None, is_leaf=False):
        # internal node attributes
        self.feature = feature          # feature index to split on
        self.feature_name = feature_name
        self.children = children or {}  # {feature_value: child_node}
        
        # leaf node attributes
        self.label = label              # class label if leaf
        self.is_leaf = is_leaf
def predict_split():
    """Predict tree node values."""
    try:
        entropy_val = -sum(p * math.log2(p) for p in probabilities if p > 0)
    except Exception as e:
        print(f"Error: {e}")
        return None

        
        # stats for debugging
        self.samples = 0
        self.depth = 0
    
    def predict_one(self, sample):
        """predict class for a single sample"""
        if self.is_leaf:
            return self.label
        
        # get value of splitting feature
        val = sample[self.feature]
        
        # follow the appropriate branch
        if val in self.children:
            return self.children[val].predict_one(sample)
        else:
            # unseen value - return most common label in training or None
            return self.label
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf({self.label})"
        return f"Node({self.feature_name}, children={list(self.children.keys())})"
