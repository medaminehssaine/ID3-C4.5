"""
Abstract Base Class for Decision Tree Classifiers.

This module provides the common foundation for ID3 and C4.5 decision tree
implementations, implementing shared functionality while allowing algorithm-
specific overrides.

Reference:
    Quinlan, J.R. (1986). "Induction of Decision Trees", Machine Learning 1:81-106
    Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", Morgan Kaufmann
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

# Type aliases for clarity
Sample = Tuple[Any, ...]
Dataset = List[Sample]
Labels = List[Any]


class DecisionTreeBase(ABC):
    """
    Abstract Base Class for Decision Tree Classifiers.

    Provides the common interface and shared functionality for decision tree
    algorithms including ID3 and C4.5. Subclasses must implement the
    algorithm-specific `_build_tree` method.

    Shared Functionality:
        - `fit`: Train the decision tree
        - `predict`: Predict class labels for multiple samples
        - `predict_one`: Predict class for a single sample
        - `get_depth`: Get maximum tree depth
        - `get_n_leaves`: Count leaf nodes
        - `_calculate_entropy`: Shannon entropy calculation

    Abstract Methods (must be implemented by subclasses):
        - `_build_tree`: Algorithm-specific tree construction

    Attributes:
        max_depth (Optional[int]): Maximum depth of the tree (None = unlimited).
        min_samples_split (int): Minimum samples required to attempt a split.
        root: Root node of the fitted tree.
        feature_names (List[str]): Names of the features.
        classes_ (List[Any]): Unique class labels.
        n_features_ (int): Number of features.

    Reference:
        Shannon, C.E. (1948). "A Mathematical Theory of Communication"
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2
    ) -> None:
        """
        Initialize base decision tree classifier.

        Args:
            max_depth: Maximum tree depth. None means unlimited.
            min_samples_split: Minimum samples needed to attempt a split.
        """
        self.max_depth: Optional[int] = max_depth
        self.min_samples_split: int = min_samples_split

        self.root: Optional[Any] = None  # Node type from subclass
        self.feature_names: Optional[List[str]] = None
        self.classes_: Optional[List[Any]] = None
        self.n_features_: int = 0

    def fit(
        self,
        X: Dataset,
        y: Labels,
        feature_names: Optional[List[str]] = None
    ) -> 'DecisionTreeBase':
        """
        Build decision tree from training data.

        Args:
            X: Training samples as list of tuples/lists.
            y: Target class labels.
            feature_names: Optional names for features (for visualization).

        Returns:
            self: Fitted classifier.

        Raises:
            ValueError: If X and y have different lengths.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.n_features_ = len(X[0]) if X else 0
        self.feature_names = feature_names or [
            f"f{i}" for i in range(self.n_features_)
        ]
        self.classes_ = list(set(y))

        # Subclass-specific initialization
        self._prepare_fit(X, y)

        # Available features for splitting
        available: Set[int] = set(range(self.n_features_))

        self.root = self._build_tree(X, y, available, depth=0)
        return self

    def _prepare_fit(self, X: Dataset, y: Labels) -> None:
        """
        Hook for subclass-specific initialization before tree building.

        Override in subclasses that need additional setup (e.g., C4.5
        needs to detect feature types).

        Args:
            X: Training samples.
            y: Target labels.
        """
        pass  # Default: no additional preparation

    @abstractmethod
    def _build_tree(
        self,
        X: Dataset,
        y: Labels,
        available_features: Set[int],
        depth: int
    ) -> Any:
        """
        Recursively build the decision tree.

        This is the core algorithm-specific method that must be implemented
        by each decision tree variant (ID3, C4.5, etc.).

        Args:
            X: Current subset of samples.
            y: Current subset of labels.
            available_features: Set of feature indices still available.
            depth: Current depth in the tree.

        Returns:
            Node: Root node of the (sub)tree.
        """
        pass

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
            raise ValueError("Tree not fitted yet, call fit() first")

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
            raise ValueError("Tree not fitted yet")
        return self.root.predict_one(sample)

    @abstractmethod
    def get_depth(self) -> int:
        """
        Get maximum depth of the tree.

        Returns:
            Maximum depth (0 if tree is just a leaf).
        """
        pass

    @abstractmethod
    def get_n_leaves(self) -> int:
        """
        Count total number of leaf nodes.

        Returns:
            Number of leaf nodes in the tree.
        """
        pass

    @staticmethod
    def _calculate_entropy(y: Labels) -> float:
        """
        Calculate Shannon entropy of a label distribution.

        Shannon entropy measures the uncertainty or impurity in a dataset.
        Higher entropy indicates more mixed classes (harder to predict).

        Mathematical Formula:
            H(S) = -Σᵢ p(cᵢ) × log₂(p(cᵢ))

        Where:
            - S is the set of samples
            - cᵢ is each unique class
            - p(cᵢ) = count(cᵢ) / |S| is the proportion of class cᵢ

        Properties:
            - H(S) = 0 when all samples belong to one class (pure node)
            - H(S) = 1 for binary classification with 50/50 split
            - H(S) = log₂(k) for k equally distributed classes

        Convention:
            0 × log₂(0) is treated as 0 (limit as p→0⁺ of p×log₂(p) = 0)

        Reference:
            Shannon, C.E. (1948). "A Mathematical Theory of Communication"

        Args:
            y: List of class labels. Can be any hashable type.

        Returns:
            float: Entropy value in range [0, log₂(num_classes)].
                   Returns 0.0 for empty input.

        Examples:
            >>> DecisionTreeBase._calculate_entropy(['yes']*4)  # Pure
            0.0
            >>> DecisionTreeBase._calculate_entropy(['yes', 'no'])  # Balanced
            1.0
        """
        if not y:
            return 0.0

        counts: Counter = Counter(y)
        total: int = len(y)

        ent: float = 0.0
        for count in counts.values():
            if count > 0:
                p: float = count / total
                ent -= p * math.log2(p)

        return ent

    def __repr__(self) -> str:
        """Return string representation."""
        class_name = self.__class__.__name__
        if self.root is None:
            return f"{class_name}(not fitted)"
        return f"{class_name}(depth={self.get_depth()}, leaves={self.get_n_leaves()})"
