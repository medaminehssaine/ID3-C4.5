#!/usr/bin/env python3
"""
Unit tests for C4.5 Decision Tree implementation.

Tests verify:
- Gain Ratio calculation (corrects ID3's bias)
- Continuous attribute handling (threshold selection)
- Mixed categorical/continuous features
- Missing value handling
- Pruning functionality

Run with: python tests/test_c45.py
"""
import sys
import os
from typing import List, Tuple, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_trees.c45 import C45Classifier
from decision_trees.c45.core.gain_ratio import (
    entropy, split_info, information_gain, gain_ratio,
    is_continuous, best_threshold, handle_missing
)
from decision_trees.c45.core.node import Node
from decision_trees.c45.data.loader import load_iris, load_golf, load_with_missing
from decision_trees.c45.core.pruning import prune_tree, pessimistic_error_rate, subtree_error


# =============================================================================
# ENTROPY TESTS (same as ID3, for completeness)
# =============================================================================

def test_entropy() -> None:
    """
    Basic entropy test: balanced binary should equal 1.0.
    
    H([+5, -5]) = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1.0
    """
    y: List[str] = ["yes"] * 5 + ["no"] * 5
    assert abs(entropy(y) - 1.0) < 0.001


def test_entropy_edge_cases() -> None:
    """Test entropy edge cases."""
    # Empty set
    assert entropy([]) == 0.0
    
    # Single element
    assert entropy(["a"]) == 0.0
    
    # Pure set
    assert entropy(["x"] * 100) == 0.0


# =============================================================================
# SPLIT INFO TESTS
# =============================================================================

def test_split_info() -> None:
    """
    Split Info should be higher for more values (more splits).
    
    SI measures the intrinsic information in the split itself,
    independent of class labels.
    """
    X_4vals: List[Tuple[str]] = [("a",), ("b",), ("c",), ("d",)]
    si_4: float = split_info(X_4vals, 0)

    X_2vals: List[Tuple[str]] = [("a",), ("a",), ("b",), ("b",)]
    si_2: float = split_info(X_2vals, 0)

    assert si_4 > si_2  # 4 values = more split info


def test_split_info_binary() -> None:
    """
    Split Info for balanced binary split should equal 1.0.
    
    SI = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1.0
    """
    X: List[Tuple[str]] = [("a",), ("a",), ("b",), ("b",)]
    si: float = split_info(X, 0)
    assert abs(si - 1.0) < 0.001


def test_split_info_continuous() -> None:
    """Test Split Info with threshold (binary split)."""
    X: List[Tuple[float]] = [(1.0,), (2.0,), (3.0,), (4.0,)]
    
    # Threshold at 2.5 splits 2/2
    si: float = split_info(X, 0, threshold=2.5)
    assert abs(si - 1.0) < 0.001  # Perfect binary split


# =============================================================================
# GAIN RATIO TESTS
# =============================================================================

def test_gain_ratio_less_than_ig() -> None:
    """
    Gain Ratio should always be ≤ Information Gain.
    
    GR = IG / SI, where SI ≥ 0 (typically ≥ 1.0 for multi-value splits)
    """
    X: List[Tuple[str, str]] = [
        ("a", "x"),
        ("b", "x"),
        ("c", "y"),
        ("d", "y"),
    ]
    y: List[str] = ["yes", "yes", "no", "no"]

    ig: float = information_gain(X, y, 0)
    gr: float = gain_ratio(X, y, 0)

    assert gr <= ig + 0.001  # Allow small floating point error


def test_gain_ratio_prefers_simple_splits() -> None:
    """
    Gain Ratio should prefer simpler splits with fewer branches.
    
    This corrects ID3's bias toward high-cardinality features.
    """
    X: List[Tuple[str, str]] = [
        ("a", "x"),
        ("b", "x"),
        ("c", "y"),
        ("d", "y"),
    ]
    y: List[str] = ["yes", "yes", "no", "no"]

    gr_high_card: float = gain_ratio(X, y, 0)  # 4 unique values
    gr_low_card: float = gain_ratio(X, y, 1)   # 2 unique values

    # Low cardinality should be preferred
    assert gr_low_card > gr_high_card


# =============================================================================
# CONTINUOUS ATTRIBUTE TESTS
# =============================================================================

def test_is_continuous() -> None:
    """Test continuous feature detection."""
    X_numeric: List[Tuple[float]] = [(1.5,), (2.3,), (3.1,)]
    assert is_continuous(X_numeric, 0) is True

    X_categorical: List[Tuple[str]] = [("a",), ("b",), ("c",)]
    assert is_continuous(X_categorical, 0) is False

    X_string_numeric: List[Tuple[str]] = [("1.5",), ("2.3",), ("3.1",)]
    assert is_continuous(X_string_numeric, 0) is True


def test_best_threshold() -> None:
    """
    Test threshold finding for continuous features.
    
    Threshold should be placed at class boundary midpoint.
    """
    X: List[Tuple[float]] = [(1.0,), (2.0,), (3.0,), (4.0,)]
    y: List[str] = ["no", "no", "yes", "yes"]

    t, gr = best_threshold(X, y, 0)
    
    assert t is not None
    assert 2.0 < t < 3.0  # Threshold between class change


def test_best_threshold_multiple_boundaries() -> None:
    """Test with multiple class change points."""
    X: List[Tuple[float]] = [
        (1.0,), (2.0,), (3.0,), (4.0,), (5.0,), (6.0,)
    ]
    y: List[str] = ["a", "a", "b", "b", "a", "a"]

    t, gr = best_threshold(X, y, 0)
    
    # Should find one of the boundaries
    assert t is not None
    assert gr > 0.0


def test_best_threshold_no_split() -> None:
    """Test when no useful threshold exists."""
    X: List[Tuple[float]] = [(1.0,), (2.0,), (3.0,)]
    y: List[str] = ["yes", "yes", "yes"]  # All same class

    t, gr = best_threshold(X, y, 0)
    
    # No useful split, but may still return a threshold
    assert gr < 0.001  # Very low or zero gain


# =============================================================================
# NODE TESTS
# =============================================================================

def test_node_continuous() -> None:
    """Test node with threshold-based split."""
    left = Node(label="no", is_leaf=True)
    right = Node(label="yes", is_leaf=True)

    node = Node(feature=0, feature_name="temp", threshold=25.0)
    node.left = left
    node.right = right
    node.is_continuous = True

    assert node.predict_one((20.0,)) == "no"   # 20 <= 25
    assert node.predict_one((30.0,)) == "yes"  # 30 > 25


def test_node_missing_value() -> None:
    """Test prediction with missing value falls back to node label."""
    left = Node(label="no", is_leaf=True)
    right = Node(label="yes", is_leaf=True)

    node = Node(feature=0, feature_name="temp", threshold=25.0)
    node.left = left
    node.right = right
    node.is_continuous = True
    node.label = "maybe"  # Fallback for missing

    # None value should return fallback
    assert node.predict_one((None,)) == "maybe"


# =============================================================================
# CLASSIFIER TESTS
# =============================================================================

def test_classifier_continuous() -> None:
    """Test classifier on all-continuous data (Iris dataset)."""
    X, y, names = load_iris()

    clf = C45Classifier()
    clf.fit(X, y, names)

    # Check feature types detected correctly
    assert all(t == 'continuous' for t in clf.feature_types_)

    y_pred = clf.predict(X)
    acc: float = sum(1 for t, p in zip(y, y_pred) if t == p) / len(y)
    
    assert acc >= 0.8


def test_classifier_mixed() -> None:
    """Test on mixed categorical/continuous data (Golf dataset)."""
    X, y, names = load_golf()

    clf = C45Classifier()
    clf.fit(X, y, names)

    # Outlook and windy = categorical, temp and humidity = continuous
    assert clf.feature_types_[0] == 'categorical'  # outlook
    assert clf.feature_types_[1] == 'continuous'   # temperature

    y_pred = clf.predict(X)
    acc: float = sum(1 for t, p in zip(y, y_pred) if t == p) / len(y)
    
    assert acc >= 0.7


def test_depth_limit() -> None:
    """Test max_depth constraint."""
    X, y, names = load_iris()

    clf = C45Classifier(max_depth=1)
    clf.fit(X, y, names)

    assert clf.get_depth() <= 1


def test_min_gain() -> None:
    """Test min_gain_ratio stopping criterion."""
    X, y, names = load_iris()

    clf = C45Classifier(min_gain_ratio=0.5)  # High threshold
    clf.fit(X, y, names)

    # Should stop early with fewer leaves
    assert clf.get_n_leaves() < 10


def test_missing_value_prediction() -> None:
    """Test prediction with missing feature value."""
    X: List[Tuple[float, str]] = [(1.0, "a"), (2.0, "b"), (3.0, "a")]
    y: List[str] = ["no", "no", "yes"]

    clf = C45Classifier()
    clf.fit(X, y)

    # Sample with missing value should still work
    pred = clf.predict_one((None, "a"))
    assert pred in ["yes", "no"]


# =============================================================================
# MISSING VALUE HANDLING TESTS
# =============================================================================

def test_handle_missing() -> None:
    """Test missing value distribution calculation."""
    X: List[Tuple[Optional[str]]] = [("a",), ("a",), ("b",), (None,)]
    
    dist = handle_missing(X, 0)
    
    assert dist is not None
    assert abs(dist["a"] - 2/3) < 0.001
    assert abs(dist["b"] - 1/3) < 0.001


def test_handle_missing_all_none() -> None:
    """Test when all values are missing."""
    X: List[Tuple[Optional[str]]] = [(None,), (None,)]
    
    dist = handle_missing(X, 0)
    
    assert dist is None


# =============================================================================
# PRUNING TESTS
# =============================================================================

def test_pessimistic_error_rate() -> None:
    """
    Test pessimistic error calculation.
    
    Pessimistic error = (errors + 0.5) / total
    This adds a penalty to prevent overfitting.
    """
    node = Node()
    node.samples = 10
    node.class_distribution = {"yes": 8, "no": 2}  # 2 errors
    
    per: float = pessimistic_error_rate(node)
    
    # (2 + 0.5) / 10 = 0.25
    assert abs(per - 0.25) < 0.001


def test_subtree_error() -> None:
    """Test subtree error calculation for pruning."""
    leaf = Node(label="yes", is_leaf=True)
    leaf.samples = 10
    leaf.class_distribution = {"yes": 8, "no": 2}
    
    err: float = subtree_error(leaf)
    
    # errors + 0.5 = 2.5
    assert abs(err - 2.5) < 0.001


def test_prune_tree_reduces_size() -> None:
    """Test that pruning reduces tree complexity."""
    X, y, names = load_iris()
    
    # Split data
    split: int = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    clf = C45Classifier()
    clf.fit(X_train, y_train, names)
    
    leaves_before: int = clf.get_n_leaves()
    
    prune_tree(clf, X_val, y_val)
    
    leaves_after: int = clf.get_n_leaves()
    
    # Pruning should reduce or maintain leaf count
    assert leaves_after <= leaves_before


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests() -> bool:
    """Run all tests manually and report results."""
    tests = [
        test_entropy,
        test_entropy_edge_cases,
        test_split_info,
        test_split_info_binary,
        test_split_info_continuous,
        test_gain_ratio_less_than_ig,
        test_gain_ratio_prefers_simple_splits,
        test_is_continuous,
        test_best_threshold,
        test_best_threshold_multiple_boundaries,
        test_best_threshold_no_split,
        test_node_continuous,
        test_node_missing_value,
        test_classifier_continuous,
        test_classifier_mixed,
        test_depth_limit,
        test_min_gain,
        test_missing_value_prediction,
        test_handle_missing,
        test_handle_missing_all_none,
        test_pessimistic_error_rate,
        test_subtree_error,
        test_prune_tree_reduces_size,
    ]

    passed: int = 0
    failed: int = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{len(tests)} tests passed")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
