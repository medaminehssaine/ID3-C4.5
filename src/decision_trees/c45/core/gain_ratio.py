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


def entropy(y: Labels, weights: Optional[List[float]] = None) -> float:
    """
    Calculate Shannon entropy of a label distribution.

    Supports weighted samples for C4.5 missing value handling.

    Mathematical Formula:
        H(S) = -Σᵢ p(cᵢ) × log₂(p(cᵢ))

    Where:
        - S is the set of samples
        - cᵢ is each unique class
        - p(cᵢ) = sum(weights for class cᵢ) / total_weight

    Args:
        y: List of class labels.
        weights: Optional list of weights for each sample.
                 If None, assumes all weights are 1.0.

    Returns:
        float: Entropy value in range [0, log₂(num_classes)].
               Returns 0.0 for empty input.
    """
    if not y:
        return 0.0

    if weights is None:
        # Unweighted case (standard)
        counts = Counter(y)
        total = len(y)
    else:
        # Weighted case
        counts = {}
        total = 0.0
        for label, w in zip(y, weights):
            counts[label] = counts.get(label, 0.0) + w
            total += w
            
        if total == 0:
            return 0.0

    ent: float = 0.0
    for count in counts.values():
        if count > 0:
            p: float = count / total
            ent -= p * math.log2(p)

    return ent


def split_info(
    X: Dataset, 
    feature_idx: int, 
    threshold: Optional[float] = None,
    weights: Optional[List[float]] = None
) -> float:
    """
    Calculate Split Information (Intrinsic Value) for a feature.

    Split Information penalizes features with many distinct values.
    Supports weighted samples.

    Mathematical Formula:
        SI(S, A) = -Σᵥ (|Sᵥ| / |S|) × log₂(|Sᵥ| / |S|)

    Args:
        X: Dataset as list of samples.
        feature_idx: Index of the feature to evaluate.
        threshold: Optional threshold for continuous features.
        weights: Optional list of weights for each sample.

    Returns:
        float: Split information value.
    """
    if weights is None:
        weights = [1.0] * len(X)
        
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    if threshold is not None:
        # Binary split for continuous attributes
        left_weight = 0.0
        for i, s in enumerate(X):
            if s[feature_idx] is not None and float(s[feature_idx]) <= threshold:
                left_weight += weights[i]
                
        right_weight = total_weight - left_weight

        if left_weight == 0 or right_weight == 0:
            return 0.0

        p_left = left_weight / total_weight
        p_right = right_weight / total_weight
        return -(p_left * math.log2(p_left) + p_right * math.log2(p_right))
    else:
        # Multi-way split for categorical attributes
        counts = {}
        for i, sample in enumerate(X):
            val = sample[feature_idx]
            if val is not None:
                counts[val] = counts.get(val, 0.0) + weights[i]
        
        si = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total_weight
                si -= p * math.log2(p)

        return si


def information_gain(
    X: Dataset,
    y: Labels,
    feature_idx: int,
    threshold: Optional[float] = None,
    weights: Optional[List[float]] = None
) -> float:
    """
    Calculate Information Gain for splitting on a feature.

    Supports weighted samples and C4.5 missing value penalty.
    
    Algorithm:
    1. Filter to samples with KNOWN values for the feature.
    2. Calculate Gain on this subset.
    3. Multiply by F = (total weight of known) / (total weight of all).

    Args:
        X: Dataset as list of samples.
        y: Corresponding class labels.
        feature_idx: Index of the feature to split on.
        threshold: Optional threshold for continuous features.
        weights: Optional list of weights for each sample.

    Returns:
        float: Information gain value.
    """
    if weights is None:
        weights = [1.0] * len(y)

    total_weight_all = sum(weights)
    if total_weight_all == 0:
        return 0.0

    # 1. Filter to known values
    known_X = []
    known_y = []
    known_weights = []
    total_weight_known = 0.0

    for i, s in enumerate(X):
        if s[feature_idx] is not None:
            known_X.append(s)
            known_y.append(y[i])
            known_weights.append(weights[i])
            total_weight_known += weights[i]

    if total_weight_known == 0:
        return 0.0

    # 2. Calculate Gain on known subset
    parent_entropy = entropy(known_y, known_weights)
    weighted_child_entropy = 0.0

    if threshold is not None:
        # Binary split for continuous feature
        left_y = []
        left_weights = []
        right_y = []
        right_weights = []
        
        for i, s in enumerate(known_X):
            val = float(s[feature_idx])
            if val <= threshold:
                left_y.append(known_y[i])
                left_weights.append(known_weights[i])
            else:
                right_y.append(known_y[i])
                right_weights.append(known_weights[i])

        # Calculate weighted average of child entropies
        for subset_y, subset_weights in [(left_y, left_weights), (right_y, right_weights)]:
            subset_total = sum(subset_weights)
            if subset_total > 0:
                weighted_child_entropy += (subset_total / total_weight_known) * entropy(subset_y, subset_weights)
                
    else:
        # Multi-way split for categorical feature
        splits = {} # val -> (y_subset, weights_subset)
        
        for i, sample in enumerate(known_X):
            val = sample[feature_idx]
            if val not in splits:
                splits[val] = ([], [])
            splits[val][0].append(known_y[i])
            splits[val][1].append(known_weights[i])

        for subset_y, subset_weights in splits.values():
            subset_total = sum(subset_weights)
            if subset_total > 0:
                weighted_child_entropy += (subset_total / total_weight_known) * entropy(subset_y, subset_weights)

    gain_known = parent_entropy - weighted_child_entropy
    
    # 3. Apply Penalty F
    F = total_weight_known / total_weight_all
    return F * gain_known


def gain_ratio(
    X: Dataset,
    y: Labels,
    feature_idx: int,
    threshold: Optional[float] = None,
    weights: Optional[List[float]] = None
) -> float:
    """
    Calculate Gain Ratio for C4.5 splitting criterion.

    Supports weighted samples.

    Args:
        X: Dataset as list of samples.
        y: Corresponding class labels.
        feature_idx: Index of the feature to evaluate.
        threshold: Optional threshold for continuous features.
        weights: Optional list of weights for each sample.

    Returns:
        float: Gain ratio value.
    """
    ig = information_gain(X, y, feature_idx, threshold, weights)
    si = split_info(X, feature_idx, threshold, weights)

    if si == 0:
        return 0.0

    return ig / si


def is_continuous(X: Dataset, feature_idx: int) -> bool:
    """
    Detect whether a feature contains continuous (numeric) values.
    """
    checked = 0
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
    feature_idx: int,
    weights: Optional[List[float]] = None
) -> Tuple[Optional[float], float]:
    """
    Find optimal split threshold for a continuous feature.

    Uses Quinlan's C4.5 method:
    1. Sort instances by attribute value.
    2. Only test split points where class label changes.
    3. Threshold is set to v_i (largest value in lower partition), NOT midpoint.

    Args:
        X: Dataset as list of samples.
        y: Corresponding class labels.
        feature_idx: Index of the continuous feature.
        weights: Optional list of weights for each sample.

    Returns:
        Tuple[Optional[float], float]: (best_threshold, best_gain_ratio)
    """
    if weights is None:
        weights = [1.0] * len(y)

    # Collect (value, label, weight) triplets, skip missing values
    triplets = []
    for i in range(len(X)):
        val = X[i][feature_idx]
        if val is not None:
            try:
                triplets.append((float(val), y[i], weights[i]))
            except (ValueError, TypeError):
                continue

    if len(triplets) < 2:
        return None, 0.0

    # Sort by value
    triplets.sort(key=lambda x: x[0])

    # Find candidate thresholds (where class changes)
    candidates = []
    
    for i in range(len(triplets) - 1):
        if triplets[i][1] != triplets[i + 1][1]:
            # Class label changes - good split candidate
            # C4.5 uses the value itself (v_i) as the threshold
            threshold = triplets[i][0]
            candidates.append(threshold)

    # Remove duplicates
    candidates = sorted(list(set(candidates)))

    if not candidates:
        # Fallback
        values = sorted(list(set(t[0] for t in triplets)))
        if len(values) > 1:
            candidates = values[:-1]
        else:
            return None, 0.0

    # Find threshold with best Gain Ratio
    best_t = None
    best_gr = -1.0

    for t in candidates:
        gr = gain_ratio(X, y, feature_idx, threshold=t, weights=weights)
        if gr > best_gr:
            best_gr = gr
            best_t = t

    return best_t, best_gr


def handle_missing(
    X: Dataset,
    feature_idx: int,
    weights: Optional[List[float]] = None
) -> Optional[Dict[Any, float]]:
    """
    Calculate value distribution for handling missing values.
    
    Returns weighted probability distribution of known values.
    """
    if weights is None:
        weights = [1.0] * len(X)
        
    known_counts = {}
    total_known_weight = 0.0
    
    for i, sample in enumerate(X):
        val = sample[feature_idx]
        if val is not None:
            w = weights[i]
            known_counts[val] = known_counts.get(val, 0.0) + w
            total_known_weight += w

    if total_known_weight == 0:
        return None

    distribution = {
        v: c / total_known_weight for v, c in known_counts.items()
    }

    return distribution
