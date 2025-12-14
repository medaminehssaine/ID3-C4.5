"""
Entropy and Information Gain calculations for ID3 Decision Trees.

This module implements the core splitting criteria for ID3 as described in
J.R. Quinlan's "Induction of Decision Trees" (1986).

ID3 (Iterative Dichotomiser 3) uses Information Gain to select the best
feature for splitting at each node. The feature with highest Information
Gain is chosen as the splitting attribute.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# Type aliases for clarity
Sample = Tuple[Any, ...]
Dataset = List[Sample]
Labels = List[Any]


def entropy(y: Labels) -> float:
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
        Quinlan, J.R. (1986). "Induction of Decision Trees", Machine Learning 1:81-106

    Args:
        y: List of class labels. Can be any hashable type.

    Returns:
        float: Entropy value in range [0, log₂(num_classes)].
               Returns 0.0 for empty input.

    Examples:
        >>> entropy(['yes', 'yes', 'yes', 'yes'])  # Pure set
        0.0
        >>> entropy(['yes', 'no'])  # Balanced binary
        1.0
        >>> round(entropy(['yes']*9 + ['no']*5), 4)  # Classic [9+, 5-]
        0.9403
        >>> entropy([])  # Empty set
        0.0
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


def information_gain(
    X: Dataset,
    y: Labels,
    feature_idx: int,
    feature_names: Optional[List[str]] = None
) -> float:
    """
    Calculate Information Gain for splitting on a categorical feature.

    Information Gain measures the reduction in entropy achieved by
    partitioning the dataset based on a feature's values. ID3 selects
    the feature with highest Information Gain at each node.

    Mathematical Formula:
        IG(S, A) = H(S) - Σᵥ (|Sᵥ| / |S|) × H(Sᵥ)

    Where:
        - H(S) is the entropy of the parent node
        - A is the attribute being evaluated
        - v iterates over all unique values of attribute A
        - Sᵥ = {x ∈ S : A(x) = v} is the subset with value v
        - |Sᵥ|/|S| is the fraction of samples with value v

    Interpretation:
        - IG = 0: Feature provides no information (random split)
        - IG = H(S): Feature perfectly separates all classes
        - Higher IG → more useful feature for classification

    Limitation (addressed by C4.5's Gain Ratio):
        Information Gain is biased toward features with many unique values.
        A feature with n unique values can have IG up to log₂(n), even if
        it's not genuinely useful (e.g., unique ID columns).

    Reference:
        Quinlan, J.R. (1986). "Induction of Decision Trees", Machine Learning 1:81-106

    Args:
        X: Dataset as list of samples (tuples/lists of feature values).
        y: Corresponding class labels.
        feature_idx: Index of the feature to evaluate.
        feature_names: Optional list of feature names (unused, for API compatibility).

    Returns:
        float: Information gain value in range [0, H(S)].

    Examples:
        >>> X = [('a', 'x'), ('a', 'y'), ('b', 'x'), ('b', 'y')]
        >>> y = ['yes', 'yes', 'no', 'no']
        >>> information_gain(X, y, 0)  # Feature 0 perfectly splits
        1.0
        >>> information_gain(X, y, 1)  # Feature 1 provides no info
        0.0
    """
    parent_entropy: float = entropy(y)

    # Group samples by feature value
    splits: Dict[Any, Labels] = {}
    for i, sample in enumerate(X):
        val = sample[feature_idx]
        if val not in splits:
            splits[val] = []
        splits[val].append(y[i])

    # Calculate weighted average of child entropies
    total: int = len(y)
    weighted_child_entropy: float = 0.0

    for subset_labels in splits.values():
        weight: float = len(subset_labels) / total
        weighted_child_entropy += weight * entropy(subset_labels)

    return parent_entropy - weighted_child_entropy


def gain_ratio(X: Dataset, y: Labels, feature_idx: int) -> float:
    """
    Calculate Gain Ratio (C4.5-style, for comparison purposes).

    Gain Ratio normalizes Information Gain by the intrinsic value
    (Split Information) of the feature. This reduces bias toward
    features with many unique values.

    Mathematical Formula:
        GR(S, A) = IG(S, A) / IV(S, A)

        IV(S, A) = -Σᵥ (|Sᵥ| / |S|) × log₂(|Sᵥ| / |S|)

    Note:
        This is provided for comparison with C4.5. ID3 uses pure
        Information Gain without normalization.

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning"

    Args:
        X: Dataset as list of samples.
        y: Corresponding class labels.
        feature_idx: Index of the feature to evaluate.

    Returns:
        float: Gain ratio value. Returns 0.0 if intrinsic value is 0.
    """
    ig: float = information_gain(X, y, feature_idx)

    # Calculate intrinsic value (split info)
    counts: Counter = Counter(sample[feature_idx] for sample in X)
    total: int = len(X)

    iv: float = 0.0
    for count in counts.values():
        if count > 0:
            p: float = count / total
            iv -= p * math.log2(p)

    if iv == 0:
        return 0.0

    return ig / iv
