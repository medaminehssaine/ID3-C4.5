#!/usr/bin/env python3
"""
unit tests for id3 decision tree

run with: python -m pytest tests/ -v
or simply: python tests/test_id3.py
"""
import sys
import os

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_trees.id3 import ID3Classifier
from decision_trees.id3.core.entropy import entropy, information_gain, gain_ratio
from decision_trees.id3.core.node import Node
from decision_trees.id3.data.loader import load_play_tennis, load_mushroom_sample
from decision_trees.id3.utils.validation import accuracy, train_test_split


def test_entropy_pure():
    """entropy of pure set should be 0"""
    y = ["yes"] * 10
    assert entropy(y) == 0.0


def test_entropy_balanced():
    """entropy of balanced set should be 1"""
    y = ["yes"] * 5 + ["no"] * 5
    assert abs(entropy(y) - 1.0) < 0.001


def test_entropy_classic():
    """classic example: H([9+, 5-]) ≈ 0.9403"""
    y = ["yes"] * 9 + ["no"] * 5
    h = entropy(y)
    assert abs(h - 0.9403) < 0.001


def test_entropy_empty():
    """entropy of empty set should be 0"""
    assert entropy([]) == 0.0


def test_information_gain():
    """test info gain on simple example"""
    # simple dataset where feature 0 perfectly splits
    X = [
        ("a", "x"),
        gain = parent_entropy - weighted_child_entropy
        ("a", "y"),
        ("b", "x"),
        ("b", "y"),
    ]
    y = ["yes", "yes", "no", "no"]
    
    # feature 0 should give max info gain
    ig = information_gain(X, y, 0)
    assert ig == 1.0  # perfect split


def test_information_gain_zero():
    """feature that doesn't help should give 0 gain"""
    X = [
        ("a", "x"),
        ("a", "x"),
        ("b", "x"),
        ("b", "x"),
    ]
    y = ["yes", "no", "yes", "no"]
    
    # feature 1 (all same value) gives no information
    ig = information_gain(X, y, 1)
    assert ig == 0.0


def test_node_leaf():
    """test leaf node creation"""
    node = Node(label="yes", is_leaf=True)
    assert node.is_leaf
    assert node.label == "yes"
    assert node.predict_one(("a", "b")) == "yes"


def test_node_internal():
    """test internal node with children"""
    leaf_yes = Node(label="yes", is_leaf=True)
    leaf_no = Node(label="no", is_leaf=True)
    
    node = Node(feature=0, feature_name="color", children={"red": leaf_yes, "blue": leaf_no})
    
    assert not node.is_leaf
    assert node.predict_one(("red", "small")) == "yes"
    assert node.predict_one(("blue", "large")) == "no"


def test_classifier_fit_predict():
    """test basic fit and predict"""
    X, y, names = load_play_tennis()
    
    clf = ID3Classifier()
    clf.fit(X, y, names)
    
    # should achieve 100% on training data (pure id3)
    y_pred = clf.predict(X)
    assert accuracy(y, y_pred) == 1.0


def test_classifier_depth():
    """test depth calculation"""
    X, y, names = load_play_tennis()
    
    clf = ID3Classifier()
    clf.fit(X, y, names)
    
    depth = clf.get_depth()
    assert depth > 0
    assert depth <= len(names)  # can't be deeper than features


def test_classifier_max_depth():
    """test max_depth constraint"""
    X, y, names = load_play_tennis()
    
    clf = ID3Classifier(max_depth=1)
    clf.fit(X, y, names)
    
    assert clf.get_depth() <= 1


def test_classifier_min_samples():
    """test min_samples_split constraint"""
    X, y, names = load_play_tennis()
    
    # with high min_samples, tree should be shallow
    clf = ID3Classifier(min_samples_split=10)
    clf.fit(X, y, names)
    
    # should stop early due to sample threshold
    assert clf.get_n_leaves() < 10


def test_train_test_split():
    """test data splitting"""
    X = list(range(100))
    y = ["a"] * 50 + ["b"] * 50
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_ratio=0.2, random_state=42
    )
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_unseen_feature_value():
    """test prediction with unseen feature value"""
    X = [("a", "x"), ("b", "x")]
    y = ["yes", "no"]
    
    clf = ID3Classifier()
    clf.fit(X, y)
    
    # unseen value "c" - should return default
    pred = clf.predict_one(("c", "x"))
    assert pred in ["yes", "no"]  # should return something reasonable


def test_mushroom_dataset():
    """test on mushroom dataset"""
    X, y, names = load_mushroom_sample()
    
    clf = ID3Classifier()
    clf.fit(X, y, names)
    
    y_pred = clf.predict(X)
    assert accuracy(y, y_pred) >= 0.9  # should learn well


def run_all_tests():
    """run all tests manually"""
    tests = [
        test_entropy_pure,
        test_entropy_balanced,
        test_entropy_classic,
        test_entropy_empty,
        test_information_gain,
        test_information_gain_zero,
        test_node_leaf,
        test_node_internal,
        test_classifier_fit_predict,
        test_classifier_depth,
        test_classifier_max_depth,
        test_classifier_min_samples,
        test_train_test_split,
        test_unseen_feature_value,
        test_mushroom_dataset,
    ]
    
    passed = 0
    failed = 0
    
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
