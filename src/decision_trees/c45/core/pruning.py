"""post-pruning for c4.5 trees"""
from collections import Counter


def reduced_error_prune(tree, X_val, y_val):
    """
    reduced error pruning using validation set
    replaces subtrees with leaves if it improves validation accuracy
    """
    if tree.root is None:
        return
    
    # get baseline accuracy
    baseline = _accuracy(tree, X_val, y_val)
    
    # try pruning each internal node
    changed = True
    while changed:
        changed = False
        nodes = _get_internal_nodes(tree.root)
        
        for node in nodes:
            if node.is_leaf:
                continue
            
            # save original state
            was_leaf = node.is_leaf
            old_children = node.children
            old_left = node.left
            old_right = node.right
            old_continuous = node.is_continuous
            
            # try making it a leaf
            node.is_leaf = True
            
            new_acc = _accuracy(tree, X_val, y_val)
            
            if new_acc >= baseline:
                # pruning helped or maintained accuracy
                baseline = new_acc
                node.children = {}
                node.left = None
                node.right = None
                changed = True
            else:
                # restore
                node.is_leaf = was_leaf
                node.children = old_children
                node.left = old_left
                node.right = old_right
                node.is_continuous = old_continuous


def prune_tree(tree, X_val, y_val):
    """main pruning interface"""
    reduced_error_prune(tree, X_val, y_val)


def _accuracy(tree, X, y):
    """helper to compute accuracy"""
    if not X:
        return 1.0
    preds = tree.predict(X)
    return sum(1 for t, p in zip(y, preds) if t == p) / len(y)


def _get_internal_nodes(node, nodes=None):
    """collect all internal (non-leaf) nodes"""
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


def pessimistic_error_rate(node, z=1.0):
    """
    pessimistic error estimate for pruning decisions
    uses continuity correction (quinlan's c4.5)
    """
    n = node.samples
    e = n - max(node.class_distribution.values()) if node.class_distribution else 0
    
    # add pessimistic correction
    return (e + 0.5) / n if n > 0 else 0


def subtree_error(node):
    """compute total error across all leaves of subtree"""
    if node.is_leaf:
        n = node.samples
        e = n - max(node.class_distribution.values()) if node.class_distribution else 0
        return e + 0.5  # pessimistic correction
    
    total = 0.0
    if node.is_continuous:
        if node.left:
            total += subtree_error(node.left)
        if node.right:
            total += subtree_error(node.right)
    else:
        for child in node.children.values():
            total += subtree_error(child)
    
    return total
