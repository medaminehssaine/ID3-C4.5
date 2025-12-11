#!/usr/bin/env python3
"""
unit tests for c4.5 decision tree

run with: python tests/test_c45.py
"""
import sys
import os

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_trees.c45 import C45Classifier
from decision_trees.c45.core.gain_ratio import (
    entropy, split_info, information_gain, gain_ratio,
    is_continuous, best_threshold
)
from decision_trees.c45.core.node import Node
from decision_trees.c45.data.loader import load_iris, load_golf, load_with_missing


def test_entropy():
    """basic entropy test"""
    y = ["yes"] * 5 + ["no"] * 5
    assert abs(entropy(y) - 1.0) < 0.001


def test_split_info():
    """split info should be higher for more values"""
    X = [("a",), ("b",), ("c",), ("d",)]
    si_4 = split_info(X, 0)
    
    X2 = [("a",), ("a",), ("b",), ("b",)]
    si_2 = split_info(X2, 0)
    
    assert si_4 > si_2  # 4 values = more split info


def test_gain_ratio_less_than_ig():
    """gain ratio <= info gain always"""
    X = [("a", "x"), ("b", "x"), ("c", "y"), ("d", "y")]
    y = ["yes", "yes", "no", "no"]
    
    ig = information_gain(X, y, 0)
    gr = gain_ratio(X, y, 0)
    
    assert gr <= ig


def test_is_continuous():
    """test continuous detection"""
    X = [(1.5,), (2.3,), (3.1,)]
    assert is_continuous(X, 0) == True
    
    X2 = [("a",), ("b",), ("c",)]
    assert is_continuous(X2, 0) == False


def test_best_threshold():
    """test threshold finding"""
    X = [(1.0,), (2.0,), (3.0,), (4.0,)]
    y = ["no", "no", "yes", "yes"]
    
    t, gr = best_threshold(X, y, 0)
    assert t is not None
    assert 2.0 < t < 3.0  # threshold between class change


def test_node_continuous():
    """test node with threshold"""
    left = Node(label="no", is_leaf=True)
    right = Node(label="yes", is_leaf=True)
    
    node = Node(feature=0, feature_name="temp", threshold=25.0)
    node.left = left
    node.right = right
    node.is_continuous = True
    
    assert node.predict_one((20.0,)) == "no"
    assert node.predict_one((30.0,)) == "yes"


def test_classifier_continuous():
    """test on continuous data"""
    X, y, names = load_iris()
    
    clf = C45Classifier()
    clf.fit(X, y, names)
    
    # check feature types detected
    assert all(t == 'continuous' for t in clf.feature_types_)
    
    y_pred = clf.predict(X)
    acc = sum(1 for t, p in zip(y, y_pred) if t == p) / len(y)
    assert acc >= 0.8


def test_classifier_mixed():
    """test on mixed categorical/continuous"""
    X, y, names = load_golf()
    
    clf = C45Classifier()
    clf.fit(X, y, names)
    
    # outlook and windy = categorical, temp and humidity = continuous
    assert clf.feature_types_[0] == 'categorical'  # outlook
    assert clf.feature_types_[1] == 'continuous'   # temperature
    
    y_pred = clf.predict(X)
    acc = sum(1 for t, p in zip(y, y_pred) if t == p) / len(y)
    assert acc >= 0.7


def test_depth_limit():
    """test max_depth constraint"""
    X, y, names = load_iris()
    
    clf = C45Classifier(max_depth=1)
    clf.fit(X, y, names)
    
    assert clf.get_depth() <= 1


def test_min_gain():
    """test min_gain_ratio stopping"""
    X, y, names = load_iris()
    
    clf = C45Classifier(min_gain_ratio=0.5)  # high threshold
    clf.fit(X, y, names)
    
    # should stop early
    assert clf.get_n_leaves() < 10


def test_missing_value_prediction():
    """test prediction with missing value"""
    X = [(1.0, "a"), (2.0, "b"), (3.0, "a")]
    y = ["no", "no", "yes"]
    
    clf = C45Classifier()
    clf.fit(X, y)
    
    # sample with missing value
    pred = clf.predict_one((None, "a"))
    assert pred in ["yes", "no"]


def run_all_tests():
    """run all tests"""
    tests = [
        test_entropy,
        test_split_info,
        test_gain_ratio_less_than_ig,
        test_is_continuous,
        test_best_threshold,
        test_node_continuous,
        test_classifier_continuous,
        test_classifier_mixed,
        test_depth_limit,
        test_min_gain,
        test_missing_value_prediction,
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
