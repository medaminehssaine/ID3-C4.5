"""
Gain Ratio and Continuous Attribute Handling for C4.5 Decision Trees.

This module implements the core splitting criteria for C4.5 as described in
J.R. Quinlan's "C4.5: Programs for Machine Learning" (1993).

C4.5 improves upon ID3 by:
1. Using Gain Ratio instead of Information Gain to reduce bias toward
   high-cardinality features
2. Supporting continuous (numeric) attributes via binary threshold splits
3. Handling missing values through weighted distribution
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

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
        - p(cᵢ) = |{x ∈ S : class(x) = cᵢ}| / |S|

    Properties:
        - H(S) = 0 when all samples belong to one class (pure node)
        - H(S) = 1 for binary classification with 50/50 split (maximum uncertainty)
        - H(S) = log₂(k) for k equally distributed classes

    Reference:
        Shannon, C.E. (1948). "A Mathematical Theory of Communication"

    Args:
        y: List of class labels.

    Returns:
        float: Entropy value in range [0, log₂(num_classes)].
               Returns 0.0 for empty input.

    Examples:
        >>> entropy(['yes', 'yes', 'yes'])  # Pure set
        0.0
        >>> entropy(['yes', 'no'])  # Binary balanced
        1.0
        >>> abs(entropy(['yes']*9 + ['no']*5) - 0.9403) < 0.001  # Classic [9+, 5-]
        True
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


def split_info(X: Dataset, feature_idx: int, threshold: Optional[float] = None) -> float:
    """
    Calculate Split Information (Intrinsic Value) for a feature.

    Split Information penalizes features with many distinct values,
    preventing C4.5 from being biased toward high-cardinality features
    like unique identifiers.

    Mathematical Formula:
        SI(S, A) = -Σᵥ (|Sᵥ| / |S|) × log₂(|Sᵥ| / |S|)

    Where:
        - S is the dataset
        - A is the attribute being split on
        - Sᵥ is the subset where attribute A has value v

    For continuous attributes with threshold t:
        SI(S, A, t) = -(|S≤t|/|S|)×log₂(|S≤t|/|S|) - (|S>t|/|S|)×log₂(|S>t|/|S|)

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", Chapter 2

    Args:
        X: Dataset as list of samples (tuples/lists of feature values).
        feature_idx: Index of the feature to evaluate.
        threshold: Optional threshold for continuous features.
                   If provided, performs binary split (≤ threshold, > threshold).

    Returns:
        float: Split information value. Returns 0.0 if split produces
               empty partitions (which would cause division by zero in GR).

    Examples:
        >>> X = [('a',), ('b',), ('c',), ('d',)]  # 4 unique values
        >>> split_info(X, 0)  # log₂(4) = 2.0
        2.0
        >>> X = [('a',), ('a',), ('b',), ('b',)]  # 2 unique values, equal split
        >>> split_info(X, 0)  # log₂(2) = 1.0
        1.0
    """
    if threshold is not None:
        # Binary split for continuous attributes
        left: int = sum(1 for s in X if s[feature_idx] is not None 
                        and float(s[feature_idx]) <= threshold)
        right: int = len(X) - left
        total: int = len(X)

        if left == 0 or right == 0:
            return 0.0

        p_left: float = left / total
        p_right: float = right / total
        return -(p_left * math.log2(p_left) + p_right * math.log2(p_right))
    else:
        # Multi-way split for categorical attributes
        counts: Counter = Counter(sample[feature_idx] for sample in X 
                                  if sample[feature_idx] is not None)
        total: int = sum(counts.values())

        if total == 0:
            return 0.0

        si: float = 0.0
        for count in counts.values():
            if count > 0:
                p: float = count / total
                si -= p * math.log2(p)

        return si


def information_gain(
    X: Dataset,
    y: Labels,
    feature_idx: int,
    threshold: Optional[float] = None
) -> float:
    """
    Calculate Information Gain for splitting on a feature.

    Information Gain measures the reduction in entropy achieved by
    partitioning the dataset based on a feature. Higher gain indicates
    the feature provides more information for classification.

    Mathematical Formula:
        IG(S, A) = H(S) - Σᵥ (|Sᵥ| / |S|) × H(Sᵥ)

    Where:
        - H(S) is the entropy of the parent node
        - Sᵥ is the subset of samples where feature A has value v
        - The sum is over all unique values v of feature A

    For continuous attributes with threshold t:
        IG(S, A, t) = H(S) - (|S≤t|/|S|)×H(S≤t) - (|S>t|/|S|)×H(S>t)

    Reference:
        Quinlan, J.R. (1986). "Induction of Decision Trees", Machine Learning 1:81-106

    Args:
        X: Dataset as list of samples.
        y: Corresponding class labels.
        feature_idx: Index of the feature to split on.
        threshold: Optional threshold for continuous features.

    Returns:
        float: Information gain value in range [0, H(S)].
               Returns 0.0 if the split provides no information.

    Examples:
        >>> X = [('a',), ('a',), ('b',), ('b',)]
        >>> y = ['yes', 'yes', 'no', 'no']
        >>> information_gain(X, y, 0)  # Perfect split
        1.0
    """
    parent_entropy: float = entropy(y)

    if threshold is not None:
        # Binary split for continuous feature
        left_y: Labels = [y[i] for i, s in enumerate(X)
                          if s[feature_idx] is not None
                          and float(s[feature_idx]) <= threshold]
        right_y: Labels = [y[i] for i, s in enumerate(X)
                           if s[feature_idx] is not None
                           and float(s[feature_idx]) > threshold]

        total: int = len(left_y) + len(right_y)
        if total == 0:
            return 0.0

        weighted: float = (
            (len(left_y) / total) * entropy(left_y) +
            (len(right_y) / total) * entropy(right_y)
        )
    else:
        # Multi-way split for categorical feature
        splits: Dict[Any, Labels] = {}
        for i, sample in enumerate(X):
            val = sample[feature_idx]
            if val is not None:
                if val not in splits:
                    splits[val] = []
                splits[val].append(y[i])

        total: int = sum(len(subset) for subset in splits.values())
        if total == 0:
            return 0.0

        weighted: float = sum(
            (len(subset) / total) * entropy(subset)
            for subset in splits.values()
        )

    return parent_entropy - weighted


def gain_ratio(
    X: Dataset,
    y: Labels,
    feature_idx: int,
    threshold: Optional[float] = None
) -> float:
    """
    Calculate Gain Ratio for C4.5 splitting criterion.

    Gain Ratio normalizes Information Gain by Split Information to
    reduce bias toward features with many distinct values. This is
    the primary splitting criterion in C4.5.

    Mathematical Formula:
        GR(S, A) = IG(S, A) / SI(S, A)

    Where:
        - IG(S, A) is the Information Gain
        - SI(S, A) is the Split Information (Intrinsic Value)

    Properties:
        - GR(S, A) ≤ IG(S, A) always (since SI ≥ 1 for 2+ partitions)
        - GR penalizes high-cardinality features (high SI → low GR)
        - GR = 0 when SI = 0 (undefined, but handled as 0)

    Edge Case:
        When SI = 0 (all samples have same value), returns 0.0 to avoid
        division by zero. This correctly indicates no splitting value.

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", Chapter 2

    Args:
        X: Dataset as list of samples.
        y: Corresponding class labels.
        feature_idx: Index of the feature to evaluate.
        threshold: Optional threshold for continuous features.

    Returns:
        float: Gain ratio value. Returns 0.0 if Split Information is 0.

    Examples:
        >>> X = [('a', 'x'), ('b', 'x'), ('c', 'y'), ('d', 'y')]
        >>> y = ['yes', 'yes', 'no', 'no']
        >>> gr_high_card = gain_ratio(X, y, 0)  # 4 unique values
        >>> gr_low_card = gain_ratio(X, y, 1)   # 2 unique values
        >>> gr_low_card > gr_high_card  # Low cardinality preferred
        True
    """
    ig: float = information_gain(X, y, feature_idx, threshold)
    si: float = split_info(X, feature_idx, threshold)

    if si == 0:
        return 0.0

    return ig / si


def is_continuous(X: Dataset, feature_idx: int) -> bool:
    """
    Detect whether a feature contains continuous (numeric) values.

    Examines sample values to determine if they can be parsed as floats.
    Missing values (None) are skipped during detection.

    Args:
        X: Dataset as list of samples.
        feature_idx: Index of the feature to check.

    Returns:
        bool: True if feature values are numeric, False if categorical.

    Note:
        Only checks first 10 non-None samples for efficiency.
        String representations of numbers (e.g., "3.14") are treated as continuous.
    """
    checked: int = 0
    for sample in X:
        if checked >= 10:
            break
        val = sample[feature_idx]
        if val is None:
            continue
        try:
            float(val)
            checked += 1
        except (ValueError, TypeError):
            return False
    return checked > 0


def best_threshold(
    X: Dataset,
    y: Labels,
    feature_idx: int
) -> Tuple[Optional[float], float]:
    """
    Find optimal split threshold for a continuous feature.

    Uses Quinlan's midpoint method: candidate thresholds are placed at
    midpoints between consecutive sorted values where the class label
    changes. This ensures the threshold lies between actual data points.

    Algorithm:
        1. Collect (value, label) pairs, excluding missing values
        2. Sort by value
        3. Find midpoints between consecutive samples with different labels
        4. Evaluate Gain Ratio for each candidate threshold
        5. Return threshold with highest Gain Ratio

    Optimization:
        The algorithm prioritizes class boundaries (points where label changes)
        as candidates, as these are most likely to produce good splits.
        Falls back to all midpoints if no class changes exist.

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", Chapter 2.4

    Args:
        X: Dataset as list of samples.
        y: Corresponding class labels.
        feature_idx: Index of the continuous feature.

    Returns:
        Tuple[Optional[float], float]: (best_threshold, best_gain_ratio)
            Returns (None, 0.0) if no valid threshold found.

    Examples:
        >>> X = [(1.0,), (2.0,), (3.0,), (4.0,)]
        >>> y = ['no', 'no', 'yes', 'yes']
        >>> t, gr = best_threshold(X, y, 0)
        >>> 2.0 < t < 3.0  # Threshold at class boundary
        True
    """
    # Collect (value, label) pairs, skip missing values
    pairs: List[Tuple[float, Any]] = []
    for i in range(len(X)):
        val = X[i][feature_idx]
        if val is not None:
            try:
                pairs.append((float(val), y[i]))
            except (ValueError, TypeError):
                continue

    if len(pairs) < 2:
        return None, 0.0

    # Sort by value
    pairs.sort(key=lambda x: x[0])

    # Find candidate thresholds at class boundaries
    candidates: List[float] = []
    for i in range(len(pairs) - 1):
        if pairs[i][1] != pairs[i + 1][1]:
            # Class label changes - good split candidate
            midpoint: float = (pairs[i][0] + pairs[i + 1][0]) / 2
            candidates.append(midpoint)

    # If no class boundaries, try all unique midpoints
    if not candidates:
        values: List[float] = sorted(set(p[0] for p in pairs))
        candidates = [
            (values[i] + values[i + 1]) / 2
            for i in range(len(values) - 1)
        ]

    if not candidates:
        return None, 0.0

    # Find threshold with best Gain Ratio
    best_t: Optional[float] = None
    best_gr: float = -1.0

    for t in candidates:
        gr: float = gain_ratio(X, y, feature_idx, threshold=t)
        if gr > best_gr:
            best_gr = gr
            best_t = t

    return best_t, best_gr


def handle_missing(
    X: Dataset,
    feature_idx: int
) -> Optional[Dict[Any, float]]:
    """
    Calculate value distribution for handling missing values.

    C4.5 handles missing values by distributing samples with unknown
    values across all branches proportionally to the distribution of
    known values. This function computes the distribution weights.

    Algorithm:
        1. Count occurrences of each known value
        2. Compute probability: P(v) = count(v) / total_known
        3. Return distribution dictionary

    Usage:
        When a sample has a missing value for the split feature,
        it is sent down ALL branches with fractional weight equal
        to the proportion of training samples with that value.

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", Chapter 2.5

    Args:
        X: Dataset as list of samples.
        feature_idx: Index of the feature with missing values.

    Returns:
        Optional[Dict[Any, float]]: Mapping from value to probability.
            Returns None if no known values exist.

    Examples:
        >>> X = [('a',), ('a',), ('b',), (None,)]
        >>> dist = handle_missing(X, 0)
        >>> dist['a']  # 2/3 probability
        0.6666666666666666
        >>> dist['b']  # 1/3 probability
        0.3333333333333333
    """
    known: Counter = Counter()
    for sample in X:
        val = sample[feature_idx]
        if val is not None:
            known[val] += 1

    total_known: int = sum(known.values())
    if total_known == 0:
        return None

    distribution: Dict[Any, float] = {
        v: c / total_known for v, c in known.items()
    }

    return distribution
