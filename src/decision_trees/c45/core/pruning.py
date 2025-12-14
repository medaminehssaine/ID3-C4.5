"""
Pruning algorithms for C4.5 Decision Trees.

This module implements post-pruning strategies for C4.5 trees:
1. Reduced Error Pruning (REP) - requires validation set
2. Pessimistic Error Pruning (PEP) - Quinlan's original C4.5 method

Pruning reduces overfitting by removing subtrees that don't improve
generalization performance.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


def prune_tree(
    tree: Any,
    X_val: List[tuple],
    y_val: List[Any],
    method: str = "reduced_error"
) -> None:
    """
    Apply post-pruning to a fitted C4.5 tree.

    Pruning removes subtrees that don't improve validation accuracy,
    resulting in a simpler, more generalizable model.

    Args:
        tree: Fitted C45Classifier instance.
        X_val: Validation dataset.
        y_val: Validation labels.
        method: Pruning method - "reduced_error" or "pessimistic".

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", Chapter 4
    """
    if method == "pessimistic":
        pessimistic_prune(tree)
    else:
        reduced_error_prune(tree, X_val, y_val)


def reduced_error_prune(
    tree: Any,
    X_val: List[tuple],
    y_val: List[Any]
) -> None:
    """
    Reduced Error Pruning using a validation set.

    Algorithm:
        1. Compute baseline accuracy on validation set
        2. For each internal node (bottom-up):
           a. Temporarily convert to leaf (majority class)
           b. If accuracy >= baseline: make permanent, update baseline
           c. Otherwise: restore subtree
        3. Repeat until no improvements

    This is a greedy algorithm that may not find the optimal tree,
    but works well in practice and is computationally efficient.

    Pros:
        - Simple and intuitive
        - Guaranteed to not hurt validation accuracy
    
    Cons:
        - Requires holdout validation set (reduces training data)
        
    Reference:
        Quinlan, J.R. (1987). "Simplifying Decision Trees"

    Args:
        tree: Fitted C45Classifier instance.
        X_val: Validation dataset.
        y_val: Validation labels.
    """
    if tree.root is None:
        return

    # Get baseline accuracy
    baseline: float = _accuracy(tree, X_val, y_val)

    # Iterate until no changes
    changed: bool = True
    while changed:
        changed = False
        nodes: List = _get_internal_nodes(tree.root)

        for node in nodes:
            if node.is_leaf:
                continue

            # Save original state
            was_leaf: bool = node.is_leaf
            old_children: Dict = node.children
            old_left = node.left
            old_right = node.right
            old_continuous: bool = node.is_continuous

            # Try making it a leaf
            node.is_leaf = True

            new_acc: float = _accuracy(tree, X_val, y_val)

            if new_acc >= baseline:
                # Pruning helped or maintained accuracy
                baseline = new_acc
                node.children = {}
                node.left = None
                node.right = None
                changed = True
            else:
                # Restore original subtree
                node.is_leaf = was_leaf
                node.children = old_children
                node.left = old_left
                node.right = old_right
                node.is_continuous = old_continuous


def pessimistic_prune(tree: Any, confidence: float = 0.25) -> None:
    """
    Pessimistic Error Pruning (C4.5's default method).

    This method doesn't require a validation set. Instead, it uses
    a pessimistic estimate of error rate based on training data,
    adding a correction factor to account for overfitting.

    Algorithm:
        For each internal node (bottom-up):
            1. Calculate pessimistic error if pruned (leaf + correction)
            2. Calculate sum of pessimistic errors of subtree leaves
            3. If pruned_error ≤ subtree_error: prune

    Mathematical Formula:
        Pessimistic Error = (e + 0.5) / N

    Where:
        - e = number of training errors at this node
        - N = number of samples at this node
        - 0.5 is the continuity correction (Quinlan's choice)

    For tighter bounds, use upper confidence limit:
        UCB = f + (z²/2N) + z×√(f/N - f²/N + z²/4N²) / (1 + z²/N)

    Where:
        - f = e/N (observed error rate)
        - z = z-score for confidence level (z=0.69 for 25% confidence)

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", pp. 37-42

    Args:
        tree: Fitted C45Classifier instance.
        confidence: Confidence level for error bound (default 0.25 = 25%).
    """
    if tree.root is None:
        return

    # z-score for confidence level (using normal approximation)
    # For 25% confidence (one-tailed): z ≈ 0.6745
    z: float = _z_score(confidence)

    _pessimistic_prune_recursive(tree.root, z)


def _pessimistic_prune_recursive(node: Any, z: float) -> float:
    """
    Recursively apply pessimistic pruning bottom-up.

    Args:
        node: Current node to evaluate.
        z: z-score for confidence bound.

    Returns:
        Pessimistic error for this subtree.
    """
    if node.is_leaf:
        return _upper_confidence_error(node, z)

    # Recurse to children first (bottom-up)
    subtree_error: float = 0.0

    if node.is_continuous:
        if node.left:
            subtree_error += _pessimistic_prune_recursive(node.left, z)
        if node.right:
            subtree_error += _pessimistic_prune_recursive(node.right, z)
    else:
        for child in node.children.values():
            subtree_error += _pessimistic_prune_recursive(child, z)

    # Calculate error if we prune here
    leaf_error: float = _upper_confidence_error(node, z)

    # Prune if leaf error is less than or equal to subtree error
    if leaf_error <= subtree_error:
        node.is_leaf = True
        node.children = {}
        node.left = None
        node.right = None
        return leaf_error
    else:
        return subtree_error


def _upper_confidence_error(node: Any, z: float) -> float:
    """
    Calculate upper confidence bound for error rate.

    Uses Wilson score interval for binomial proportion.

    Formula:
        UCB = (f + z²/(2n) + z×√(f(1-f)/n + z²/(4n²))) / (1 + z²/n)

    Where:
        - f = observed error rate = e/n
        - n = number of samples
        - z = z-score for desired confidence

    Simplified for small samples (Quinlan's approximation):
        error = (e + 0.5) / n × n = e + 0.5

    Args:
        node: Node to calculate error for.
        z: z-score for confidence level.

    Returns:
        Upper confidence bound for number of errors.
    """
    n: int = node.samples
    if n == 0:
        return 0.0

    # Number of errors = samples - majority class count
    if node.class_distribution:
        majority: int = max(node.class_distribution.values())
        e: int = n - majority
    else:
        e: int = 0

    f: float = e / n  # Observed error rate

    # Wilson score interval upper bound
    z2: float = z * z
    numerator: float = (
        f + z2 / (2 * n) + z * math.sqrt(f * (1 - f) / n + z2 / (4 * n * n))
    )
    denominator: float = 1 + z2 / n

    ucb_rate: float = numerator / denominator

    # Return expected number of errors
    return ucb_rate * n


def _z_score(confidence: float) -> float:
    """
    Get z-score for given confidence level (one-tailed).

    Uses common values; for other levels, uses rough approximation.

    Args:
        confidence: Confidence level (e.g., 0.25 for 25%).

    Returns:
        Corresponding z-score.
    """
    # Common z-scores for one-tailed tests
    z_table: Dict[float, float] = {
        0.01: 2.326,
        0.05: 1.645,
        0.10: 1.282,
        0.25: 0.6745,
        0.50: 0.0,
    }

    if confidence in z_table:
        return z_table[confidence]

    # Simple approximation for other values
    return 0.6745  # Default to 25% confidence


def pessimistic_error_rate(node: Any, z: float = 0.6745) -> float:
    """
    Calculate pessimistic error rate for a node.

    Uses continuity correction as per Quinlan's C4.5.

    Mathematical Formula:
        Pessimistic Error Rate = (e + 0.5) / N

    Where:
        - e = number of misclassified samples (based on majority class)
        - N = total samples at this node
        - 0.5 = continuity correction factor

    Reference:
        Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", p. 39

    Args:
        node: Node to calculate error for.
        z: z-score for confidence bound (unused in simple formula).

    Returns:
        Pessimistic error rate in [0, 1].
    """
    n: int = node.samples
    if n == 0:
        return 0.0

    # Errors = total - majority class count
    if node.class_distribution:
        majority: int = max(node.class_distribution.values())
        e: int = n - majority
    else:
        e: int = 0

    # Add pessimistic correction
    return (e + 0.5) / n


def subtree_error(node: Any) -> float:
    """
    Compute total pessimistic error across all leaves of a subtree.

    Recursively sums (errors + 0.5) for all leaf nodes.

    Args:
        node: Root of the subtree.

    Returns:
        Total pessimistic error for the subtree.
    """
    if node.is_leaf:
        n: int = node.samples
        if n == 0:
            return 0.0

        if node.class_distribution:
            majority: int = max(node.class_distribution.values())
            e: int = n - majority
        else:
            e: int = 0

        # Pessimistic correction
        return e + 0.5

    # Sum children
    total: float = 0.0

    if node.is_continuous:
        if node.left:
            total += subtree_error(node.left)
        if node.right:
            total += subtree_error(node.right)
    else:
        for child in node.children.values():
            total += subtree_error(child)

    return total


def _accuracy(tree: Any, X: List[tuple], y: List[Any]) -> float:
    """
    Calculate accuracy of tree on given dataset.

    Args:
        tree: Fitted classifier with predict() method.
        X: Dataset.
        y: True labels.

    Returns:
        Accuracy in [0, 1].
    """
    if not X:
        return 1.0

    preds: List[Any] = tree.predict(X)
    correct: int = sum(1 for t, p in zip(y, preds) if t == p)
    return correct / len(y)


def _get_internal_nodes(node: Any, nodes: Optional[List] = None) -> List:
    """
    Collect all internal (non-leaf) nodes in the tree.

    Traverses in pre-order, collecting internal nodes.

    Args:
        node: Current node.
        nodes: Accumulator list (for recursion).

    Returns:
        List of internal nodes.
    """
    if nodes is None:
        nodes = []

    if not node.is_leaf:
        nodes.append(node)

        if node.is_continuous:
            if node.left:
                _get_internal_nodes(node.left, nodes)
            if node.right:
                _get_internal_nodes(node.right, nodes)
        else:
            for child in node.children.values():
                _get_internal_nodes(child, nodes)

    return nodes
