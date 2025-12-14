"""
C4.5 Decision Tree Classifier.

Implementation of Quinlan's C4.5 algorithm, an improvement over ID3 with:
- Gain Ratio to reduce bias toward high-cardinality features
- Support for continuous (numeric) attributes via threshold splits
- Missing value handling via fractional propagation
- Post-pruning support (Pessimistic Error Pruning)
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from .gain_ratio import (
    gain_ratio, information_gain, split_info,
    is_continuous, best_threshold, entropy, handle_missing
)
from .node import Node
from .pruning import prune_tree

# Type aliases
Sample = Tuple[Any, ...]
Dataset = List[Sample]
Labels = List[Any]


class C45Classifier:
    """
    C4.5 Decision Tree Classifier.

    Implements Quinlan's C4.5 algorithm (1993), which extends ID3 with:
    
    1. **Gain Ratio**: Normalizes Information Gain by Split Information
       to reduce bias toward features with many unique values.
       
    2. **Continuous Attributes**: Binary splits on numeric features
       using optimal thresholds.
       
    3. **Missing Values**: Handles missing data by distributing samples
       fractionally (using weights) down all branches.
    
    4. **Pruning**: Supports post-pruning via Pessimistic Error Pruning.

    Algorithm:
        1. If stopping condition met → create leaf
        2. For each feature:
           - If continuous: find best threshold, compute GR
           - If categorical: compute GR directly
        3. Select feature/threshold with highest GR
        4. Create split node
        5. For continuous: binary split (≤t, >t), feature can be reused
        6. For categorical: multi-way split, feature removed

    Attributes:
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum samples required to split.
        min_gain_ratio: Minimum GR required to make a split.
        root: Root node of the fitted tree.
        feature_names: Names of features.
        classes_: Unique class labels.
        n_features_: Number of features.
        feature_types_: Detected type of each feature ('continuous'/'categorical').

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning",
        Morgan Kaufmann Publishers

    Examples:
        >>> from decision_trees.c45 import C45Classifier
        >>> clf = C45Classifier(max_depth=5)
        >>> clf.fit(X_train, y_train, feature_names)
        >>> print(clf.feature_types_)  # ['continuous', 'categorical', ...]
        >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_gain_ratio: float = 0.01
    ) -> None:
        """
        Initialize C4.5 classifier.

        Args:
            max_depth: Maximum tree depth. None means unlimited.
            min_samples_split: Minimum samples needed to attempt a split.
            min_gain_ratio: Minimum Gain Ratio required to make a split.
                           Helps prevent splits that provide little value.
        """
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split
        self.min_gain_ratio: float = min_gain_ratio

        self.root: Optional[Node] = None
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[List[Any]] = None
        self.n_features_: int = 0
        self.feature_types_: List[str] = []

    def fit(
        self,
        X: Dataset,
        y: Labels,
        feature_names: Optional[List[str]] = None
    ) -> 'C45Classifier':
        """
        Build decision tree from training data.

        Automatically detects whether each feature is continuous or
        categorical based on whether values can be parsed as floats.
        
        Applies Pessimistic Error Pruning by default after building the tree.

        Args:
            X: Training samples as list of tuples/lists.
            y: Target class labels.
            feature_names: Optional names for features.

        Returns:
            self: Fitted classifier.
        """
        self.n_features_ = len(X[0]) if X else 0
        self.feature_names = feature_names or [
            f"f{i}" for i in range(self.n_features_)
        ]
        self.classes_ = list(set(y))

        # Detect feature types
        self.feature_types_ = []
        for i in range(self.n_features_):
            if is_continuous(X, i):
                self.feature_types_.append('continuous')
            else:
                self.feature_types_.append('categorical')

        # Available features (can be reused for continuous in C4.5)
        available: Set[int] = set(range(self.n_features_))

        # Initialize weights (all 1.0)
        weights = [1.0] * len(X)

        # Build tree
        self.root = self._build_tree(X, y, weights, available, depth=0)
        
        # Prune tree
        if self.root is not None:
            prune_tree(self, method="pessimistic")
            
        return self

    def _build_tree(
        self,
        X: Dataset,
        y: Labels,
        weights: List[float],
        available_features: Set[int],
        depth: int
    ) -> Node:
        """
        Recursively build the decision tree.

        For continuous features, uses binary threshold splits and
        allows the feature to be reused in subtrees.

        For categorical features, uses multi-way splits and removes
        the feature from the available set.
        
        Handles missing values by fractional propagation.

        Args:
            X: Current subset of samples.
            y: Current subset of labels.
            weights: Current subset of weights.
            available_features: Set of feature indices available for splitting.
            depth: Current depth in the tree.

        Returns:
            Node: Root of the (sub)tree.
        """
        node = Node()
        node.samples = sum(weights)
        node.depth = depth

        # Class distribution for pruning decisions (weighted)
        counts: Dict[Any, float] = {}
        for label, w in zip(y, weights):
            counts[label] = counts.get(label, 0.0) + w
        node.class_distribution = counts
        
        # Determine majority class
        if counts:
            most_common = max(counts, key=counts.get)
            node.label = most_common
        else:
            node.label = None # Should not happen if X is not empty

        # Stopping condition: pure node (or close to pure)
        # Check if all samples belong to one class (ignoring 0 weights)
        active_classes = {l for l, c in counts.items() if c > 0}
        if len(active_classes) <= 1:
            node.is_leaf = True
            return node

        # Stopping condition: no features available
        if not available_features:
            node.is_leaf = True
            return node

        # Stopping condition: max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return node

        # Stopping condition: too few samples
        if node.samples < self.min_samples_split:
            node.is_leaf = True
            return node

        # Find best split
        best_feature: Optional[int] = None
        best_gr: float = -1.0
        best_threshold_val: Optional[float] = None

        # Calculate gains and gain ratios for all available features
        candidates = []
        
        # Sort features to ensure deterministic behavior
        sorted_features = sorted(list(available_features))
        
        for f in sorted_features:
            if self.feature_types_[f] == 'continuous':
                # Find best threshold for continuous feature
                t, gr = best_threshold(X, y, f, weights)
                # We need the raw gain to apply the heuristic. 
                # best_threshold returns (threshold, gain_ratio).
                # We need to re-calculate or modify best_threshold to return gain too.
                # For now, let's re-calculate gain if we have a valid threshold.
                if t is not None:
                    gain = information_gain(X, y, f, threshold=t, weights=weights)
                    candidates.append({
                        'feature': f,
                        'threshold': t,
                        'gain': gain,
                        'gain_ratio': gr
                    })
            else:
                # Categorical feature
                gain = information_gain(X, y, f, threshold=None, weights=weights)
                gr = gain_ratio(X, y, f, threshold=None, weights=weights)
                candidates.append({
                    'feature': f,
                    'threshold': None,
                    'gain': gain,
                    'gain_ratio': gr
                })

        if not candidates:
            node.is_leaf = True
            return node

        # Quinlan's Heuristic: Information gain must be at least as large as the average gain
        # over all tests examined.
        avg_gain = sum(c['gain'] for c in candidates) / len(candidates)
        
        # Filter candidates
        filtered_candidates = [c for c in candidates if c['gain'] >= avg_gain]
        
        # If no candidates meet the criteria (rare, but possible if all below average due to float precision?),
        # fallback to all candidates.
        if not filtered_candidates:
            filtered_candidates = candidates

        # Select best by Gain Ratio
        best_candidate = max(filtered_candidates, key=lambda c: c['gain_ratio'])
        
        best_gr = best_candidate['gain_ratio']
        best_feature = best_candidate['feature']
        best_threshold_val = best_candidate['threshold']

        # Check minimum gain ratio
        if best_gr < self.min_gain_ratio:
            node.is_leaf = True
            return node

        # Create split node
        node.feature = best_feature
        node.feature_name = self.feature_names[best_feature]
        node.is_leaf = False

        # --- Splitting Logic with Fractional Propagation ---
        
        if best_threshold_val is not None:
            # Continuous split (binary)
            node.threshold = best_threshold_val
            node.is_continuous = True
            
            # Prepare lists for children
            X_left, y_left, w_left = [], [], []
            X_right, y_right, w_right = [], [], []
            
            # Calculate probabilities for missing values
            # P(left) = weight(left) / weight(known)
            w_known_left = 0.0
            w_known_right = 0.0
            
            for i, s in enumerate(X):
                if s[best_feature] is not None:
                    val = float(s[best_feature])
                    if val <= best_threshold_val:
                        w_known_left += weights[i]
                    else:
                        w_known_right += weights[i]
            
            total_known = w_known_left + w_known_right
            if total_known > 0:
                p_left = w_known_left / total_known
                p_right = w_known_right / total_known
            else:
                p_left = p_right = 0.5 # Fallback if all missing?

            # Distribute samples
            for i, s in enumerate(X):
                val = s[best_feature]
                w = weights[i]
                
                if val is None:
                    # Missing value: send to both with fractional weights
                    if p_left > 0:
                        X_left.append(s)
                        y_left.append(y[i])
                        w_left.append(w * p_left)
                    if p_right > 0:
                        X_right.append(s)
                        y_right.append(y[i])
                        w_right.append(w * p_right)
                else:
                    # Known value
                    if float(val) <= best_threshold_val:
                        X_left.append(s)
                        y_left.append(y[i])
                        w_left.append(w)
                    else:
                        X_right.append(s)
                        y_right.append(y[i])
                        w_right.append(w)

            # Continuous features can be reused in C4.5
            if X_left:
                node.left = self._build_tree(
                    X_left, y_left, w_left, available_features, depth + 1
                )
            if X_right:
                node.right = self._build_tree(
                    X_right, y_right, w_right, available_features, depth + 1
                )
                
        else:
            # Categorical split (multi-way)
            node.is_continuous = False

            # Calculate distribution of known values for missing value handling
            dist = handle_missing(X, best_feature, weights)
            if dist is None:
                # All missing? Or empty?
                node.is_leaf = True
                return node

            # Initialize splits
            splits: Dict[Any, Tuple[Dataset, Labels, List[float]]] = {}
            
            # First pass: Create buckets for known values
            # We need to know all possible values to handle missing ones correctly
            # Or we just create buckets as we see them.
            
            # Distribute samples
            for i, s in enumerate(X):
                val = s[best_feature]
                w = weights[i]
                
                if val is None:
                    # Missing value: send to ALL branches
                    for branch_val, prob in dist.items():
                        if branch_val not in splits:
                            splits[branch_val] = ([], [], [])
                        splits[branch_val][0].append(s)
                        splits[branch_val][1].append(y[i])
                        splits[branch_val][2].append(w * prob)
                else:
                    # Known value
                    if val not in splits:
                        splits[val] = ([], [], [])
                    splits[val][0].append(s)
                    splits[val][1].append(y[i])
                    splits[val][2].append(w)

            # Remove feature for categorical (ID3 style)
            remaining: Set[int] = available_features - {best_feature}

            for val, (X_sub, y_sub, w_sub) in splits.items():
                child: Node = self._build_tree(X_sub, y_sub, w_sub, remaining, depth + 1)
                node.children[val] = child

        return node

    def predict(self, X: Dataset) -> Labels:
        """
        Predict class labels for samples.

        Args:
            X: Samples to predict.

        Returns:
            List of predicted class labels.

        Raises:
            ValueError: If tree is not fitted.
        """
        if self.root is None:
            raise ValueError("Tree not fitted")
        return [self.root.predict_one(sample) for sample in X]

    def predict_one(self, sample: Sample) -> Any:
        """
        Predict class for a single sample.

        Args:
            sample: Single sample as tuple.

        Returns:
            Predicted class label.

        Raises:
            ValueError: If tree is not fitted.
        """
        if self.root is None:
            raise ValueError("Tree not fitted")
        return self.root.predict_one(sample)

    def get_depth(self) -> int:
        """
        Get maximum depth of the tree.

        Returns:
            Maximum depth (0 if tree is just a leaf).
        """
        if self.root is None:
            return 0
        return self._get_depth(self.root)

    def _get_depth(self, node: Node) -> int:
        """Recursively calculate tree depth."""
        if node.is_leaf:
            return 0

        depths: List[int] = []
        if node.is_continuous:
            if node.left:
                depths.append(self._get_depth(node.left))
            if node.right:
                depths.append(self._get_depth(node.right))
        else:
            for child in node.children.values():
                depths.append(self._get_depth(child))

        return 1 + max(depths) if depths else 0

    def get_n_leaves(self) -> int:
        """
        Count total number of leaf nodes.

        Returns:
            Number of leaf nodes in the tree.
        """
        if self.root is None:
            return 0
        return self._count_leaves(self.root)

    def _count_leaves(self, node: Node) -> int:
        """Recursively count leaf nodes."""
        if node.is_leaf:
            return 1

        count: int = 0
        if node.is_continuous:
            if node.left:
                count += self._count_leaves(node.left)
            if node.right:
                count += self._count_leaves(node.right)
        else:
            for child in node.children.values():
                count += self._count_leaves(child)

        return count

    def __repr__(self) -> str:
        """Return string representation."""
        if self.root is None:
            return "C45Classifier(not fitted)"
        return f"C45Classifier(depth={self.get_depth()}, leaves={self.get_n_leaves()})"
