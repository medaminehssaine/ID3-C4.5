"""
Model Serialization for Decision Trees.

This module provides save/load functionality for fitted decision tree
models using JSON serialization.

Features:
    - Save fitted trees to JSON format
    - Load trees back for prediction
    - Human-readable format for inspection
    - Version compatibility checking

Usage:
    >>> save_model(classifier, 'model.json')
    >>> loaded = load_model('model.json')
    >>> predictions = loaded.predict(X_test)
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

# Type alias
ModelDict = Dict[str, Any]


def save_model(
    model: Any,
    filepath: str,
    include_metadata: bool = True
) -> None:
    """
    Save a fitted decision tree model to a JSON file.

    The model is serialized to a human-readable JSON format that
    can be inspected, edited, or loaded back for prediction.

    Args:
        model: Fitted ID3Classifier or C45Classifier instance.
        filepath: Path to save the JSON file.
        include_metadata: Whether to include training metadata.

    Raises:
        ValueError: If model is not fitted.

    Examples:
        >>> clf = ID3Classifier()
        >>> clf.fit(X, y, feature_names)
        >>> save_model(clf, 'my_tree.json')
    """
    if model.root is None:
        raise ValueError("Model not fitted. Call fit() first.")

    model_dict: ModelDict = {
        'version': '1.0',
        'model_type': model.__class__.__name__,
    }

    if include_metadata:
        model_dict['metadata'] = {
            'saved_at': datetime.now().isoformat(),
            'n_features': model.n_features_,
            'feature_names': model.feature_names,
            'classes': list(model.classes_) if model.classes_ else None,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
        }

        # C4.5-specific
        if hasattr(model, 'min_gain_ratio'):
            model_dict['metadata']['min_gain_ratio'] = model.min_gain_ratio
        if hasattr(model, 'feature_types_'):
            model_dict['metadata']['feature_types'] = model.feature_types_

    # Serialize tree structure
    model_dict['tree'] = _node_to_dict(model.root)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(model_dict, f, indent=2, default=str)


def load_model(filepath: str) -> Any:
    """
    Load a decision tree model from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Fitted classifier instance.

    Raises:
        ValueError: If file format is invalid.

    Examples:
        >>> clf = load_model('my_tree.json')
        >>> predictions = clf.predict(X_test)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        model_dict = json.load(f)

    if 'version' not in model_dict or 'model_type' not in model_dict:
        raise ValueError("Invalid model file format")

    model_type = model_dict['model_type']
    metadata = model_dict.get('metadata', {})

    # Create appropriate classifier
    if model_type == 'ID3Classifier':
        from .id3 import ID3Classifier
        classifier = ID3Classifier(
            max_depth=metadata.get('max_depth'),
            min_samples_split=metadata.get('min_samples_split', 2)
        )
    elif model_type == 'C45Classifier':
        from .c45 import C45Classifier
        classifier = C45Classifier(
            max_depth=metadata.get('max_depth'),
            min_samples_split=metadata.get('min_samples_split', 2),
            min_gain_ratio=metadata.get('min_gain_ratio', 0.01)
        )
        if 'feature_types' in metadata:
            classifier.feature_types_ = metadata['feature_types']
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Restore metadata
    classifier.n_features_ = metadata.get('n_features', 0)
    classifier.feature_names = metadata.get('feature_names')
    classifier.classes_ = metadata.get('classes')

    # Reconstruct tree
    tree_dict = model_dict['tree']
    classifier.root = _dict_to_node(tree_dict, model_type)

    return classifier


def _node_to_dict(node: Any) -> ModelDict:
    """Recursively convert node to dictionary."""
    if node is None:
        return None

    d: ModelDict = {
        'is_leaf': node.is_leaf,
        'label': node.label,
        'samples': getattr(node, 'samples', 0),
        'depth': getattr(node, 'depth', 0),
    }

    if hasattr(node, 'class_distribution') and node.class_distribution:
        d['class_distribution'] = node.class_distribution

    if not node.is_leaf:
        d['feature'] = node.feature
        d['feature_name'] = node.feature_name

        # Check for continuous split (C4.5)
        if hasattr(node, 'is_continuous') and node.is_continuous:
            d['is_continuous'] = True
            d['threshold'] = node.threshold
            d['left'] = _node_to_dict(node.left) if node.left else None
            d['right'] = _node_to_dict(node.right) if node.right else None
        else:
            d['is_continuous'] = False
            d['children'] = {
                str(k): _node_to_dict(v)
                for k, v in node.children.items()
            }

    return d


def _dict_to_node(d: ModelDict, model_type: str) -> Any:
    """Recursively convert dictionary to node."""
    if d is None:
        return None

    # Import appropriate node class
    if model_type == 'ID3Classifier':
        from .id3.core.node import Node
    else:
        from .c45.core.node import Node

    node = Node()
    node.is_leaf = d.get('is_leaf', False)
    node.label = d.get('label')
    node.samples = d.get('samples', 0)
    node.depth = d.get('depth', 0)

    if 'class_distribution' in d:
        node.class_distribution = d['class_distribution']

    if not node.is_leaf:
        node.feature = d.get('feature')
        node.feature_name = d.get('feature_name')

        if d.get('is_continuous', False):
            node.is_continuous = True
            node.threshold = d.get('threshold')
            node.left = _dict_to_node(d.get('left'), model_type)
            node.right = _dict_to_node(d.get('right'), model_type)
        else:
            node.is_continuous = False
            children_dict = d.get('children', {})
            for k, v in children_dict.items():
                # Try to convert key back to original type
                try:
                    key = eval(k) if k not in ('True', 'False', 'None') else k
                except:
                    key = k
                node.children[key] = _dict_to_node(v, model_type)

    return node


def model_to_json(model: Any) -> str:
    """
    Convert model to JSON string (for API responses, etc.)

    Args:
        model: Fitted classifier.

    Returns:
        str: JSON string representation.
    """
    if model.root is None:
        raise ValueError("Model not fitted")

    model_dict = {
        'model_type': model.__class__.__name__,
        'tree': _node_to_dict(model.root)
    }

    return json.dumps(model_dict, indent=2, default=str)


def model_summary(model: Any) -> str:
    """
    Generate a text summary of the model.

    Args:
        model: Fitted classifier.

    Returns:
        str: Human-readable summary.
    """
    if model.root is None:
        return f"{model.__class__.__name__}: Not fitted"

    lines = [
        f"{'='*50}",
        f" {model.__class__.__name__} Summary",
        f"{'='*50}",
        f"",
        f"Structure:",
        f"  - Depth: {model.get_depth()}",
        f"  - Leaf nodes: {model.get_n_leaves()}",
        f"  - Features: {model.n_features_}",
        f"",
        f"Configuration:",
        f"  - Max depth: {model.max_depth or 'unlimited'}",
        f"  - Min samples split: {model.min_samples_split}",
    ]

    if hasattr(model, 'min_gain_ratio'):
        lines.append(f"  - Min gain ratio: {model.min_gain_ratio}")

    if hasattr(model, 'feature_types_') and model.feature_types_:
        lines.append(f"")
        lines.append(f"Feature Types:")
        for name, ftype in zip(model.feature_names or [], model.feature_types_):
            lines.append(f"  - {name}: {ftype}")

    if model.classes_:
        lines.append(f"")
        lines.append(f"Classes: {model.classes_}")

    lines.append(f"{'='*50}")

    return '\n'.join(lines)
