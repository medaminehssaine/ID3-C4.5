"""
ID3 Decision Tree Classifier.

Implementation of Quinlan's ID3 (Iterative Dichotomiser 3) algorithm
for classification using Information Gain as the splitting criterion.

ID3 is designed for categorical features only and creates multi-way
splits (one branch per unique feature value).
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from ...base import DecisionTreeBase
from .entropy import entropy, information_gain
from .node import Node

# Type aliases
Sample = Tuple[Any, ...]
Dataset = List[Sample]
Labels = List[Any]


class ID3Classifier(DecisionTreeBase):
    """
    ID3 Decision Tree Classifier.

    Implements the classic ID3 algorithm by Quinlan (1986) using
    Information Gain to select splitting features. Best suited for
    categorical/discrete features.

    Inherits from DecisionTreeBase, providing:
        - Shared `fit`, `predict`, `predict_one` methods
        - `_calculate_entropy` static method

    Algorithm:
        1. If all samples belong to same class → create leaf
        2. If no features remaining → create leaf with majority class
        3. Select feature with highest Information Gain
        4. Create internal node splitting on that feature
        5. Recursively build subtrees for each feature value
        6. Remove used feature from available set (no reuse in ID3)

    Attributes:
        max_depth: Maximum depth of the tree (None = unlimited).
        min_samples_split: Minimum samples required to split a node.
        root: Root node of the fitted tree.
        feature_names: Names of the features.
        classes_: Unique class labels.
        n_features_: Number of features.

    Reference:
        Quinlan, J.R. (1986). "Induction of Decision Trees",
        Machine Learning 1:81-106

    Examples:
        >>> from decision_trees.id3 import ID3Classifier
        >>> clf = ID3Classifier()
        >>> clf.fit(X_train, y_train, feature_names)
        >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2
    ) -> None:
        """
        Initialize ID3 classifier.

        Args:
            max_depth: Maximum tree depth. None means unlimited.
            min_samples_split: Minimum samples needed to attempt a split.
        """
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    def _build_tree(
        self,
        X: Dataset,
        y: Labels,
        available_features: Set[int],
        depth: int
    ) -> Node:
        """
        Recursively build the decision tree.

        Args:
            X: Current subset of samples.
            y: Current subset of labels.
            available_features: Set of feature indices still available.
            depth: Current depth in the tree.

        Returns:
            Node: Root of the (sub)tree.
        """
        node = Node()
        node.samples = len(y)
        node.depth = depth

        # Count class distribution
        counts: Counter = Counter(y)
        most_common: Any = counts.most_common(1)[0][0]
        node.label = most_common  # Default prediction

        # Stopping condition: pure node
        if len(counts) == 1:
            node.is_leaf = True
            return node

        # Stopping condition: no features left
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

        # Find best feature to split on
        best_feature: Optional[int] = None
        best_gain: float = -1.0

        for f in available_features:
            gain: float = information_gain(X, y, f)
            if gain > best_gain:
                best_gain = gain
                best_feature = f

        # No information gain possible
        if best_gain <= 0 or best_feature is None:
            node.is_leaf = True
            return node

        # Create internal node
        node.feature = best_feature
        node.feature_name = self.feature_names[best_feature]
        node.is_leaf = False

        # Split data by feature value
        splits: Dict[Any, Tuple[Dataset, Labels]] = {}
        for i, sample in enumerate(X):
            val = sample[best_feature]
            if val not in splits:
                splits[val] = ([], [])
            splits[val][0].append(sample)
            splits[val][1].append(y[i])

        # Remove used feature (ID3 doesn't reuse features)
        remaining: Set[int] = available_features - {best_feature}

        # Recursively build children
        for val, (X_subset, y_subset) in splits.items():
            child: Node = self._build_tree(X_subset, y_subset, remaining, depth + 1)
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
        if not node.children:
            return 0
        return 1 + max(self._get_depth(child) for child in node.children.values())

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
        return sum(self._count_leaves(child) for child in node.children.values())

