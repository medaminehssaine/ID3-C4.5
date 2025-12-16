"""
C4.5 Decision Tree Classifier.

Implementation of Quinlan's C4.5 algorithm, an improvement over ID3 with:
- Gain Ratio to reduce bias toward high-cardinality features
- Support for continuous (numeric) attributes via threshold splits
- Missing value handling
- Post-pruning support
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from ...base import DecisionTreeBase
from .gain_ratio import (
    gain_ratio, information_gain, split_info,
    is_continuous, best_threshold, entropy
)
from .node import Node

# Type aliases
Sample = Tuple[Any, ...]
Dataset = List[Sample]
Labels = List[Any]


class C45Classifier(DecisionTreeBase):
    """
    C4.5 Decision Tree Classifier.

    Implements Quinlan's C4.5 algorithm (1993), which extends ID3 with:
    
    1. **Gain Ratio**: Normalizes Information Gain by Split Information
       to reduce bias toward features with many unique values.
       
    2. **Continuous Attributes**: Binary splits on numeric features
       using optimal thresholds.
       
    3. **Missing Values**: Handles missing data by distributing samples.
    
    4. **Pruning**: Supports post-pruning via the pruning module.

    Inherits from DecisionTreeBase, providing:
        - Shared `fit`, `predict`, `predict_one` methods
        - `_calculate_entropy` static method

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
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        self.min_gain_ratio: float = min_gain_ratio
        self.feature_types_: List[str] = []

    def _prepare_fit(self, X: Dataset, y: Labels) -> None:
        """
        Detect feature types before tree building.

        C4.5 automatically determines whether each feature is continuous
        or categorical based on whether values can be parsed as floats.

        Args:
            X: Training samples.
            y: Target labels (unused, required by interface).
        """
        self.feature_types_ = []
        for i in range(self.n_features_):
            if is_continuous(X, i):
                self.feature_types_.append('continuous')
            else:
                self.feature_types_.append('categorical')

    def _build_tree(
        self,
        X: Dataset,
        y: Labels,
        available_features: Set[int],
        depth: int
    ) -> Node:
        """
        Recursively build the decision tree.

        For continuous features, uses binary threshold splits and
        allows the feature to be reused in subtrees.

        For categorical features, uses multi-way splits and removes
        the feature from the available set.

        Args:
            X: Current subset of samples.
            y: Current subset of labels.
            available_features: Set of feature indices available for splitting.
            depth: Current depth in the tree.

        Returns:
            Node: Root of the (sub)tree.
        """
        node = Node()
        node.samples = len(y)
        node.depth = depth

        # Class distribution for pruning decisions
        counts: Counter = Counter(y)
        node.class_distribution = dict(counts)
        most_common: Any = counts.most_common(1)[0][0]
        node.label = most_common

        # Stopping condition: pure node
        if len(counts) == 1:
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
        if len(y) < self.min_samples_split:
            node.is_leaf = True
            return node

        # Find best split
        best_feature: Optional[int] = None
        best_gr: float = -1.0
        best_threshold_val: Optional[float] = None

        for f in available_features:
            if self.feature_types_[f] == 'continuous':
                # Find best threshold for continuous feature
                t, gr = best_threshold(X, y, f)
                if gr > best_gr:
                    best_gr = gr
                    best_feature = f
                    best_threshold_val = t
            else:
                # Categorical feature
                gr: float = gain_ratio(X, y, f)
                if gr > best_gr:
                    best_gr = gr
                    best_feature = f
                    best_threshold_val = None

        # Check minimum gain ratio
        if best_gr < self.min_gain_ratio:
            node.is_leaf = True
            return node

        if best_feature is None:
            node.is_leaf = True
            return node

        # Create split node
        node.feature = best_feature
        node.feature_name = self.feature_names[best_feature]
        node.is_leaf = False

        if best_threshold_val is not None:
            # Continuous split (binary)
            node.threshold = best_threshold_val
            node.is_continuous = True

            # Split data
            left_idx: List[int] = [
                i for i, s in enumerate(X)
                if s[best_feature] is not None
                and float(s[best_feature]) <= best_threshold_val
            ]
            right_idx: List[int] = [
                i for i, s in enumerate(X)
                if s[best_feature] is not None
                and float(s[best_feature]) > best_threshold_val
            ]

            X_left: Dataset = [X[i] for i in left_idx]
            y_left: Labels = [y[i] for i in left_idx]
            X_right: Dataset = [X[i] for i in right_idx]
            y_right: Labels = [y[i] for i in right_idx]

            # Continuous features can be reused in C4.5
            if X_left:
                node.left = self._build_tree(
                    X_left, y_left, available_features, depth + 1
                )
            if X_right:
                node.right = self._build_tree(
                    X_right, y_right, available_features, depth + 1
                )
        else:
            # Categorical split (multi-way)
            node.is_continuous = False

            # Group by value
            splits: Dict[Any, Tuple[Dataset, Labels]] = {}
            for i, sample in enumerate(X):
                val = sample[best_feature]
                if val is not None:
                    if val not in splits:
                        splits[val] = ([], [])
                    splits[val][0].append(sample)
                    splits[val][1].append(y[i])

            # Remove feature for categorical (ID3 style)
            remaining: Set[int] = available_features - {best_feature}

            for val, (X_sub, y_sub) in splits.items():
                child: Node = self._build_tree(X_sub, y_sub, remaining, depth + 1)
                node.children[val] = child

        return node

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

