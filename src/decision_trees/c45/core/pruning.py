"""post-pruning for c4.5 trees using pessimistic error pruning"""
import math

def pessimistic_prune(node, confidence=0.25):
    """
    Apply pessimistic error pruning to the tree (in-place).
    Uses Wilson Score Interval to estimate upper bound of error rate.
    
    Args:
        node: The current node to prune
        confidence: Confidence level (default 0.25 for C4.5)
    """
    if node.is_leaf:
        return

    # Recurse first (bottom-up pruning)
    if node.is_continuous:
        if node.left:
            pessimistic_prune(node.left, confidence)
        if node.right:
            pessimistic_prune(node.right, confidence)
    else:
        for child in node.children.values():
            pessimistic_prune(child, confidence)
            
    # Now check if we should prune this node
    # Calculate error estimate for this node if it were a leaf
    leaf_error_estimate = _calculate_error_estimate(node, confidence)
    
    # Calculate error estimate for the subtree (sum of children's errors)
    subtree_error_estimate = _calculate_subtree_error(node, confidence)
    
    # If leaf error is less than or equal to subtree error, prune!
    # (Quinlan's C4.5 favors pruning when errors are equal or less)
    if leaf_error_estimate <= subtree_error_estimate:
        # Make it a leaf
        node.is_leaf = True
        node.children = {}
        node.left = None
        node.right = None
        node.is_continuous = False
        # Prediction is already set (majority class)

def _calculate_error_estimate(node, confidence=0.25):
    """
    Calculate pessimistic error estimate for a node using Wilson Score Interval.
    Upper Confidence Limit (UCL).
    """
    # Total samples reaching this node
    n = node.samples
    if n == 0:
        return 0.0
    
    # Observed errors (misclassifications if this were a leaf)
    # Error = Total - Majority Class Count
    if not node.class_distribution:
        return 0.0
        
    majority_count = max(node.class_distribution.values())
    f = (n - majority_count) / n  # Observed error rate
    
    # Get z-score for confidence
    z = _get_z_score(confidence)
    
    # Wilson Score Interval Formula for Upper Bound
    # (f + z^2/2n + z * sqrt(f/n - f^2/n + z^2/4n^2)) / (1 + z^2/n)
    
    numerator = f + (z**2) / (2*n) + z * math.sqrt((f/n) - (f**2)/n + (z**2)/(4*(n**2)))
    denominator = 1 + (z**2) / n
    
    ucl = numerator / denominator
    
    # Return estimated number of errors
    return ucl * n

def _calculate_subtree_error(node, confidence):
    """Sum of error estimates of all leaves in the subtree."""
    if node.is_leaf:
        return _calculate_error_estimate(node, confidence)
    
    total_error = 0.0
    if node.is_continuous:
        if node.left:
            total_error += _calculate_subtree_error(node.left, confidence)
        if node.right:
            total_error += _calculate_subtree_error(node.right, confidence)
    else:
        for child in node.children.values():
            total_error += _calculate_subtree_error(child, confidence)
            
    return total_error

def _get_z_score(confidence):
    """Get z-score for confidence level (approximate)."""
    # Common z-scores
    # 0.25 (25%) -> 0.674 (Quinlan's default)
    if confidence == 0.25:
        return 0.69 # Approximation often used
    
    # Simple lookup for common values
    # Note: C4.5 "confidence" is actually "certainty factor" (CF)
    # Lower CF = more pruning. 25% is standard.
    # z = stats.norm.ppf(1 - (confidence/2)) ? No, C4.5 logic is specific.
    # We'll stick to the table from decision_trees.py for consistency
    z_table = {
        0.001: 3.09, 0.005: 2.58, 0.01: 2.33, 0.05: 1.65,
        0.10: 1.28, 0.15: 1.04, 0.20: 0.84, 0.25: 0.69,
        0.30: 0.52, 0.40: 0.25, 0.50: 0.0
    }
    # Find closest
    closest = min(z_table.keys(), key=lambda x: abs(x - confidence))
    return z_table[closest]
