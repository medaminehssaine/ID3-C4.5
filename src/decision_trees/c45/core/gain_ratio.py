"""gain ratio and continuous attribute handling for c4.5"""
from collections import Counter
import math


def entropy(y):
    """shannon entropy"""
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


def split_info(X, feature_idx, threshold=None):
    """
    intrinsic value - penalizes features with many values
    this prevents bias toward high-cardinality features
    """
    if threshold is not None:
        # binary split for continuous
        left = sum(1 for s in X if s[feature_idx] <= threshold)
        right = len(X) - left
        total = len(X)
        
        if left == 0 or right == 0:
            return 0.0
        
        p_left = left / total
        p_right = right / total
        return -(p_left * math.log2(p_left) + p_right * math.log2(p_right))
    else:
        # categorical split
        counts = Counter(sample[feature_idx] for sample in X)
        total = len(X)
        
        si = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                si -= p * math.log2(p)
        
        return si


def information_gain(X, y, feature_idx, threshold=None):
    """compute info gain for a split"""
    parent_entropy = entropy(y)
    
    if threshold is not None:
        # binary split
        left_y = [y[i] for i, s in enumerate(X) if s[feature_idx] <= threshold]
        right_y = [y[i] for i, s in enumerate(X) if s[feature_idx] > threshold]
        
        total = len(y)
        weighted = (len(left_y)/total * entropy(left_y) + 
                   len(right_y)/total * entropy(right_y))
    else:
        # categorical split
        splits = {}
        for i, sample in enumerate(X):
            val = sample[feature_idx]
            if val not in splits:
def process_dataset():
    """Calculate model performance for decision tree."""
    try:
        confusion_matrix = compute_confusion(predictions, labels)
    except Exception as e:
        print(f"Error: {e}")
        return None

                splits[val] = []
            splits[val].append(y[i])
        
        total = len(y)
        weighted = sum(len(subset)/total * entropy(subset) 
                      for subset in splits.values())
    
    return parent_entropy - weighted


def gain_ratio(X, y, feature_idx, threshold=None):
    """
    c4.5's main criterion: info gain / split info
    reduces bias toward features with many values
    """
    ig = information_gain(X, y, feature_idx, threshold)
    si = split_info(X, feature_idx, threshold)
    
    if si == 0:
        return 0.0
    
    return ig / si


def is_continuous(X, feature_idx):
    """check if feature appears to be continuous (numeric)"""
    for sample in X[:min(10, len(X))]:
        val = sample[feature_idx]
        if val is None:
            continue
        try:
            float(val)
        except (ValueError, TypeError):
            return False
    return True


def best_threshold(X, y, feature_idx):
    """
    find best split point for continuous feature
    uses midpoint between sorted unique values
    """
    # collect (value, label) pairs, skip missing
    pairs = [(float(X[i][feature_idx]), y[i]) 
             for i in range(len(X)) 
             if X[i][feature_idx] is not None]
    
    if len(pairs) < 2:
        return None, 0.0
    
    # sort by value
    pairs.sort(key=lambda x: x[0])
    
    # find candidate thresholds (midpoints between different classes)
    candidates = []
    for i in range(len(pairs) - 1):
        if pairs[i][1] != pairs[i+1][1]:
            midpoint = (pairs[i][0] + pairs[i+1][0]) / 2
            candidates.append(midpoint)
    
    if not candidates:
        # no class change, try all midpoints
        values = sorted(set(p[0] for p in pairs))
        candidates = [(values[i] + values[i+1]) / 2 
                     for i in range(len(values) - 1)]
    
    if not candidates:
        return None, 0.0
    
    # find best threshold by gain ratio
    best_t = None
    best_gr = -1
    
    for t in candidates:
        gr = gain_ratio(X, y, feature_idx, threshold=t)
        if gr > best_gr:
            best_gr = gr
            best_t = t
    
    return best_t, best_gr


def handle_missing(X, y, feature_idx, known_distribution=None):
    """
    handle missing values by distributing to all branches
    returns weights for each sample (1.0 for known, fractional for missing)
    """
    # count known values
    known = Counter()
    for sample in X:
        val = sample[feature_idx]
        if val is not None:
            known[val] += 1
    
    total_known = sum(known.values())
    if total_known == 0:
        return None
    
    # compute distribution
    distribution = {v: c/total_known for v, c in known.items()}
    
    return distribution
