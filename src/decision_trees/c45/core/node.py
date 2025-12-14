"""
Decision Tree Node structure for C4.5 algorithm.

This module defines the Node class for C4.5 decision trees, which extends
ID3 with support for:
- Continuous attributes via threshold-based binary splits
- Missing value handling
- Pruning metadata (class distribution, sample counts)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union


class Node:
    """
    Represents a single node in a C4.5 decision tree.

    C4.5 supports two types of splits:
    - Categorical: Multi-way split (like ID3), one branch per unique value
    - Continuous: Binary split on threshold (≤ threshold vs > threshold)

    Attributes:
        feature (Optional[int]): Index of the splitting feature.
        feature_name (Optional[str]): Human-readable name of the split feature.
        threshold (Optional[float]): Split threshold for continuous features.
        is_continuous (bool): True if this is a continuous (threshold) split.
        children (Dict[Any, Node]): Child nodes for categorical splits.
        left (Optional[Node]): Left child (≤ threshold) for continuous splits.
        right (Optional[Node]): Right child (> threshold) for continuous splits.
        label (Optional[Any]): Predicted class label.
        is_leaf (bool): True if this is a terminal node.
        samples (float): Weighted number of training samples that reached this node.
        depth (int): Depth of this node in the tree.
        class_distribution (Dict[Any, float]): Weighted count of each class at this node.
            Used for pruning decisions and handling missing values.

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning"
    """

    def __init__(
        self,
        feature: Optional[int] = None,
        feature_name: Optional[str] = None,
        threshold: Optional[float] = None,
        children: Optional[Dict[Any, 'Node']] = None,
        label: Optional[Any] = None,
        is_leaf: bool = False
    ) -> None:
        """
        Initialize a C4.5 decision tree node.

        Args:
            feature: Index of the splitting feature (None for leaves).
            feature_name: Human-readable name of the splitting feature.
            threshold: Split threshold for continuous features.
            children: Dictionary mapping values to child nodes (categorical).
            label: Class prediction for this node.
            is_leaf: Whether this is a leaf node.
        """
        # Split attributes
        self.feature: Optional[int] = feature
        self.feature_name: Optional[str] = feature_name
        self.threshold: Optional[float] = threshold
        self.is_continuous: bool = threshold is not None

        # Children for categorical splits
        self.children: Dict[Any, Node] = children or {}

        # Children for continuous splits (binary)
        self.left: Optional[Node] = None   # ≤ threshold
        self.right: Optional[Node] = None  # > threshold

        # Leaf attributes
        self.label: Optional[Any] = label
        self.is_leaf: bool = is_leaf

        # Statistics for pruning and missing values
        self.samples: float = 0.0
        self.depth: int = 0
        self.class_distribution: Dict[Any, float] = {}

    def predict_one(
        self,
        sample: tuple,
        default: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Predict class label for a single sample.

        Handles both categorical and continuous splits, and gracefully
        deals with missing values by returning the node's majority class.

        Algorithm:
            1. If leaf node → return label
            2. Get sample's value for splitting feature
            3. If value is None (missing) → return this node's label
            4. For continuous: compare to threshold, go left/right
            5. For categorical: look up child by value
            6. If branch not found → return fallback label

        Args:
            sample: Tuple of feature values.
            default: Default label if prediction fails.

        Returns:
            Predicted class label.

        Examples:
            >>> # Continuous split
            >>> node = Node(feature=0, feature_name='temp', threshold=25.0)
            >>> node.left = Node(label='cold', is_leaf=True)
            >>> node.right = Node(label='hot', is_leaf=True)
            >>> node.is_continuous = True
            >>> node.predict_one((20.0,))
            'cold'
            >>> node.predict_one((30.0,))
            'hot'
        """
        if self.is_leaf:
            return self.label

        val: Any = sample[self.feature]

        # Handle missing values
        if val is None:
            return self.label or default

        if self.is_continuous:
            # Binary split on threshold
            try:
                numeric_val: float = float(val)
            except (ValueError, TypeError):
                return self.label or default

            if numeric_val <= self.threshold:
                if self.left is not None:
                    return self.left.predict_one(sample, default)
            else:
                if self.right is not None:
                    return self.right.predict_one(sample, default)
            return self.label or default
        else:
            # Categorical split
            if val in self.children:
                return self.children[val].predict_one(sample, default)
            else:
                # Unseen value - return majority class
                return self.label or default

    def get_majority_class(self) -> Optional[Any]:
        """
        Get the majority class from the class distribution.

        Returns:
            The class with highest count, or None if no distribution.
        """
        if not self.class_distribution:
            return self.label
        return max(self.class_distribution, key=self.class_distribution.get)

    def __repr__(self) -> str:
        """Return string representation of the node."""
        if self.is_leaf:
            return f"Leaf({self.label})"
        if self.is_continuous:
            return f"Node({self.feature_name} <= {self.threshold:.2f})"
        return f"Node({self.feature_name}, children={list(self.children.keys())})"
