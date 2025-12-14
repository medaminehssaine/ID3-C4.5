"""
Decision Tree Node structure for ID3 algorithm.

This module defines the Node class used to represent nodes in an ID3
decision tree. Each node is either:
- An internal node: splits on a feature, has children for each value
- A leaf node: contains a class prediction
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class Node:
    """
    Represents a single node in an ID3 decision tree.

    ID3 creates multi-way splits on categorical features; each unique
    feature value leads to a different child node.

    Attributes:
        feature (Optional[int]): Index of the feature used for splitting.
            None for leaf nodes.
        feature_name (Optional[str]): Human-readable name of the split feature.
        children (Dict[Any, Node]): Mapping from feature value to child node.
            Empty for leaf nodes.
        label (Optional[Any]): Predicted class for this node.
            Used for leaf nodes and as fallback for unseen values.
        is_leaf (bool): True if this is a terminal (leaf) node.
        samples (int): Number of training samples that reached this node.
            Useful for debugging and pruning.
        depth (int): Depth of this node in the tree (root = 0).

    Reference:
        Quinlan, J.R. (1986). "Induction of Decision Trees", Machine Learning 1:81-106
    """

    def __init__(
        self,
        feature: Optional[int] = None,
        feature_name: Optional[str] = None,
        children: Optional[Dict[Any, 'Node']] = None,
        label: Optional[Any] = None,
        is_leaf: bool = False
    ) -> None:
        """
        Initialize a decision tree node.

        Args:
            feature: Index of the splitting feature (None for leaves).
            feature_name: Human-readable name of the splitting feature.
            children: Dictionary mapping feature values to child nodes.
            label: Class prediction (for leaves or as fallback).
            is_leaf: Whether this is a leaf node.
        """
        # Internal node attributes
        self.feature: Optional[int] = feature
        self.feature_name: Optional[str] = feature_name
        self.children: Dict[Any, Node] = children or {}

        # Leaf node attributes
        self.label: Optional[Any] = label
        self.is_leaf: bool = is_leaf

        # Statistics for debugging and pruning
        self.samples: int = 0
        self.depth: int = 0

    def predict_one(self, sample: tuple) -> Optional[Any]:
        """
        Predict class label for a single sample.

        Traverses the tree from this node following the appropriate
        branch based on feature values until reaching a leaf.

        Algorithm:
            1. If this is a leaf → return label
            2. Get the sample's value for the splitting feature
            3. If value exists in children → recurse to that child
            4. If value unseen → return this node's label as fallback

        Args:
            sample: Tuple of feature values (must match training feature order).

        Returns:
            Predicted class label, or None if tree is malformed.

        Examples:
            >>> leaf = Node(label='yes', is_leaf=True)
            >>> leaf.predict_one(('sunny', 'hot'))
            'yes'
        """
        if self.is_leaf:
            return self.label

        # Get value of the splitting feature
        val: Any = sample[self.feature]

        # Follow the appropriate branch
        if val in self.children:
            return self.children[val].predict_one(sample)
        else:
            # Unseen value - return most common class from training
            return self.label

    def __repr__(self) -> str:
        """Return string representation of the node."""
        if self.is_leaf:
            return f"Leaf({self.label})"
        return f"Node({self.feature_name}, children={list(self.children.keys())})"
