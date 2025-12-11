"""entropy and information gain calculations for id3"""
from collections import Counter
import math


def entropy(y):
    """shannon entropy of a label list"""
    if not y:
        return 0.0
    
    counts = Counter(y)
    total = len(y)
    
    ent = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            ent -= p * math.log2(p)
    
    return ent


def information_gain(X, y, feature_idx, feature_names=None):
    """
    compute information gain for splitting on a feature
    X: list of samples (each sample is a list/tuple of feature values)
    y: list of labels
    feature_idx: index of feature to split on
    """
    parent_entropy = entropy(y)
    
    # group samples by feature value
    splits = {}
    for i, sample in enumerate(X):
        val = sample[feature_idx]
        if val not in splits:
            splits[val] = []
        splits[val].append(y[i])
    
    # weighted average of child entropies
    total = len(y)
    weighted_child_entropy = 0.0
    
    for subset_labels in splits.values():
        weight = len(subset_labels) / total
        weighted_child_entropy += weight * entropy(subset_labels)
    
    return parent_entropy - weighted_child_entropy


def gain_ratio(X, y, feature_idx):
    """c4.5 style gain ratio (for future use)"""
    ig = information_gain(X, y, feature_idx)
    
    # intrinsic value (split info)
    counts = Counter(sample[feature_idx] for sample in X)
    total = len(X)
    
    iv = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            iv -= p * math.log2(p)
    
    if iv == 0:
        return 0.0
    
    return ig / iv
