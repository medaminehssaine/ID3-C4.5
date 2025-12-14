#!/usr/bin/env python3
"""
Unit tests for ID3 Decision Tree implementation.

Tests verify mathematical correctness of entropy and information gain
calculations, as well as classifier functionality.

Run with: python -m pytest tests/ -v
or simply: python tests/test_id3.py
"""
import sys
import os
from typing import List, Tuple, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_trees.id3 import ID3Classifier
from decision_trees.id3.core.entropy import entropy, information_gain, gain_ratio
from decision_trees.id3.core.node import Node
from decision_trees.id3.data.loader import load_play_tennis, load_mushroom_sample
from decision_trees.id3.utils.validation import accuracy, train_test_split


# =============================================================================
# ENTROPY TESTS
# =============================================================================

def test_entropy_pure() -> None:
    """
    Entropy of a pure set should be 0.
    
    Mathematical verification:
        H(S) = -Σ p_i × log₂(p_i)
        For pure set: p = 1.0, so H = -1 × log₂(1) = 0
    """
    y: List[str] = ["yes"] * 10
    assert entropy(y) == 0.0


def test_entropy_balanced() -> None:
    """
    Entropy of balanced binary set should be 1.0.
    
    Mathematical verification:
        H(S) = -0.5 × log₂(0.5) - 0.5 × log₂(0.5)
             = -0.5 × (-1) - 0.5 × (-1)
             = 0.5 + 0.5 = 1.0
    """
    y: List[str] = ["yes"] * 5 + ["no"] * 5
    assert abs(entropy(y) - 1.0) < 0.001


def test_entropy_classic() -> None:
    """
    Classic example from Quinlan's paper: H([9+, 5-]) ≈ 0.9403.
    
    Mathematical verification:
        p(+) = 9/14, p(-) = 5/14
        H = -(9/14)×log₂(9/14) - (5/14)×log₂(5/14)
          ≈ -(0.643 × -0.637) - (0.357 × -1.485)
          ≈ 0.410 + 0.530 = 0.940
    """
    y: List[str] = ["yes"] * 9 + ["no"] * 5
    h: float = entropy(y)
    assert abs(h - 0.9403) < 0.001


def test_entropy_empty() -> None:
    """Entropy of empty set should be 0 (by convention)."""
    assert entropy([]) == 0.0


def test_entropy_multiclass() -> None:
    """
    Entropy with 4 equally distributed classes should be log₂(4) = 2.0.
    
    Mathematical verification:
        H = -4 × (0.25 × log₂(0.25)) = -4 × 0.25 × (-2) = 2.0
    """
    y: List[str] = ["a", "b", "c", "d"]
    assert abs(entropy(y) - 2.0) < 0.001


# =============================================================================
# INFORMATION GAIN TESTS
# =============================================================================

def test_information_gain_perfect() -> None:
    """
    Feature that perfectly separates classes should give maximum gain.
    
    IG = H(parent) - weighted H(children) = 1.0 - 0 = 1.0
    """
    X: List[Tuple[str, str]] = [
        ("a", "x"),
        ("a", "y"),
        ("b", "x"),
        ("b", "y"),
    ]
    y: List[str] = ["yes", "yes", "no", "no"]

    # Feature 0 should give max info gain (perfect split)
    ig: float = information_gain(X, y, 0)
    assert abs(ig - 1.0) < 0.001


def test_information_gain_zero() -> None:
    """
    Feature that doesn't help should give 0 gain.
    
    When splitting doesn't change class distribution in children,
    weighted child entropy equals parent entropy, so IG = 0.
    """
    X: List[Tuple[str, str]] = [
        ("a", "x"),
        ("a", "x"),
        ("b", "x"),
        ("b", "x"),
    ]
    y: List[str] = ["yes", "no", "yes", "no"]

    # Feature 1 (all same value) gives no information
    ig: float = information_gain(X, y, 1)
    assert ig == 0.0


def test_information_gain_partial() -> None:
    """
    Partial split should give partial information gain.
    """
    X: List[Tuple[str, str]] = [
        ("a", "x"),
        ("a", "y"),
        ("b", "x"),
        ("b", "x"),
    ]
    y: List[str] = ["yes", "yes", "no", "no"]

    ig: float = information_gain(X, y, 0)
    # Should be between 0 and parent entropy
    assert 0 < ig <= 1.0


# =============================================================================
# GAIN RATIO TESTS
# =============================================================================

def test_gain_ratio_less_than_ig() -> None:
    """
    Gain Ratio should always be ≤ Information Gain.
    
    GR = IG / SI, and SI ≥ 0 (typically > 1 for multiple values).
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

    assert gr <= ig


def test_gain_ratio_penalizes_cardinality() -> None:
    """
    Gain Ratio should penalize high-cardinality features.
    
    Feature with 4 values should have lower GR than feature with 2 values,
    even if both separate classes equally well.
    """
    X: List[Tuple[str, str]] = [
        ("a", "x"),
        ("b", "x"),
        ("c", "y"),
        ("d", "y"),
    ]
    y: List[str] = ["yes", "yes", "no", "no"]

    gr_high: float = gain_ratio(X, y, 0)  # 4 unique values
    gr_low: float = gain_ratio(X, y, 1)   # 2 unique values

    assert gr_low > gr_high


# =============================================================================
# NODE TESTS
# =============================================================================

def test_node_leaf() -> None:
    """Test leaf node creation and prediction."""
    node = Node(label="yes", is_leaf=True)
    assert node.is_leaf
    assert node.label == "yes"
    assert node.predict_one(("a", "b")) == "yes"


def test_node_internal() -> None:
    """Test internal node with children."""
    leaf_yes = Node(label="yes", is_leaf=True)
    leaf_no = Node(label="no", is_leaf=True)

    node = Node(
        feature=0,
        feature_name="color",
        children={"red": leaf_yes, "blue": leaf_no}
    )

    assert not node.is_leaf
    assert node.predict_one(("red", "small")) == "yes"
    assert node.predict_one(("blue", "large")) == "no"


def test_node_unseen_value() -> None:
    """Test prediction with unseen feature value falls back to node label."""
    leaf_yes = Node(label="yes", is_leaf=True)
    
    node = Node(
        feature=0,
        feature_name="color",
        children={"red": leaf_yes},
        label="no"  # Fallback for unseen values
    )

    # Unseen value "green" should return fallback label
    assert node.predict_one(("green", "small")) == "no"


# =============================================================================
# CLASSIFIER TESTS
# =============================================================================

def test_classifier_fit_predict() -> None:
    """Test basic fit and predict functionality."""
    X, y, names = load_play_tennis()

    clf = ID3Classifier()
    clf.fit(X, y, names)

    # ID3 should achieve 100% on training data (no pruning)
    y_pred = clf.predict(X)
    assert accuracy(y, y_pred) == 1.0


def test_classifier_depth() -> None:
    """Test tree depth calculation."""
    X, y, names = load_play_tennis()

    clf = ID3Classifier()
    clf.fit(X, y, names)

    depth = clf.get_depth()
    assert depth > 0
    assert depth <= len(names)  # Can't be deeper than number of features


def test_classifier_max_depth() -> None:
    """Test max_depth constraint."""
    X, y, names = load_play_tennis()

    clf = ID3Classifier(max_depth=1)
    clf.fit(X, y, names)

    assert clf.get_depth() <= 1


def test_classifier_min_samples() -> None:
    """Test min_samples_split constraint."""
    X, y, names = load_play_tennis()

    # With high min_samples, tree should be shallow
    clf = ID3Classifier(min_samples_split=10)
    clf.fit(X, y, names)

    # Should stop early due to sample threshold
    assert clf.get_n_leaves() < 10


def test_classifier_repr() -> None:
    """Test string representation of classifier."""
    clf = ID3Classifier()
    assert "not fitted" in repr(clf)

    X, y, _ = load_play_tennis()
    clf.fit(X, y)
    assert "depth" in repr(clf)
    assert "leaves" in repr(clf)


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def test_train_test_split() -> None:
    """Test data splitting functionality."""
    X: List[int] = list(range(100))
    y: List[str] = ["a"] * 50 + ["b"] * 50

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=0.2, random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_unseen_feature_value_inference() -> None:
    """Test prediction with unseen feature value during inference."""
    X: List[Tuple[str, str]] = [("a", "x"), ("b", "x")]
    y: List[str] = ["yes", "no"]

    clf = ID3Classifier()
    clf.fit(X, y)

    # Unseen value "c" - should return some reasonable prediction
    pred = clf.predict_one(("c", "x"))
    assert pred in ["yes", "no"]


def test_mushroom_dataset() -> None:
    """Test on mushroom dataset - safety critical classification."""
    X, y, names = load_mushroom_sample()

    clf = ID3Classifier()
    clf.fit(X, y, names)

    y_pred = clf.predict(X)
    acc = accuracy(y, y_pred)
    
    # Should learn well on training data
    assert acc >= 0.9


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests() -> bool:
    """Run all tests manually and report results."""
    tests = [
        test_entropy_pure,
        test_entropy_balanced,
        test_entropy_classic,
        test_entropy_empty,
        test_entropy_multiclass,
        test_information_gain_perfect,
        test_information_gain_zero,
        test_information_gain_partial,
        test_gain_ratio_less_than_ig,
        test_gain_ratio_penalizes_cardinality,
        test_node_leaf,
        test_node_internal,
        test_node_unseen_value,
        test_classifier_fit_predict,
        test_classifier_depth,
        test_classifier_max_depth,
        test_classifier_min_samples,
        test_classifier_repr,
        test_train_test_split,
        test_unseen_feature_value_inference,
        test_mushroom_dataset,
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
