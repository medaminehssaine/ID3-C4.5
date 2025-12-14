"""
Tree visualization utilities for ID3 decision trees.

Provides functions to:
- Print tree structure to console
- Export tree to Graphviz DOT format
- Convert tree to dictionary for serialization
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tree import ID3Classifier
    from ..core.node import Node


def print_tree(tree: 'ID3Classifier', indent: str = "") -> None:
    """
    Print tree structure to console in a readable format.

    Args:
        tree: Fitted ID3Classifier instance.
        indent: Initial indentation (for nested calls).
    """
    if tree.root is not None:
        _print_node(tree.root, indent)


def _print_node(node: 'Node', indent: str = "", branch: str = "") -> None:
    """
    Recursively print node and its children.

    Args:
        node: Current node to print.
        indent: Current indentation level.
        branch: Branch label leading to this node.
    """
    if node is None:
        return

    prefix: str = indent + branch

    if node.is_leaf:
        print(f"{prefix}-> [{node.label}]")
    else:
        print(f"{prefix}[{node.feature_name}?]")
        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last: bool = (i == len(children) - 1)
            new_indent: str = indent + ("    " if is_last else "│   ")
            _print_node(child, new_indent, f"= {value} ")


def tree_to_string(tree: 'ID3Classifier') -> str:
    """
    Get tree as string instead of printing.

    Args:
        tree: Fitted ID3Classifier instance.

    Returns:
        String representation of the tree structure.
    """
    lines: List[str] = []
    if tree.root is not None:
        _node_to_string(tree.root, lines, "", "")
    return "\n".join(lines)


def _node_to_string(
    node: 'Node',
    lines: List[str],
    indent: str = "",
    branch: str = ""
) -> None:
    """
    Recursively build string representation of node.

    Args:
        node: Current node.
        lines: Accumulator list for output lines.
        indent: Current indentation.
        branch: Branch label.
    """
    if node is None:
        return

    prefix: str = indent + branch

    if node.is_leaf:
        lines.append(f"{prefix}-> [{node.label}]")
    else:
        lines.append(f"{prefix}[{node.feature_name}?]")
        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last: bool = (i == len(children) - 1)
            new_indent: str = indent + ("    " if is_last else "│   ")
            _node_to_string(child, lines, new_indent, f"= {value} ")


def tree_to_dict(tree: 'ID3Classifier') -> Optional[Dict[str, Any]]:
    """
    Convert tree to dictionary for serialization.

    Args:
        tree: Fitted ID3Classifier instance.

    Returns:
        Dictionary representation of the tree, or None if not fitted.
    """
    if tree.root is None:
        return None
    return _node_to_dict(tree.root)


def _node_to_dict(node: 'Node') -> Optional[Dict[str, Any]]:
    """
    Recursively convert node to dictionary.

    Args:
        node: Current node.

    Returns:
        Dictionary representation of the node.
    """
    if node is None:
        return None

    d: Dict[str, Any] = {
        "is_leaf": node.is_leaf,
        "samples": node.samples
    }

    if node.is_leaf:
        d["label"] = node.label
    else:
        d["feature"] = node.feature
        d["feature_name"] = node.feature_name
        d["children"] = {
            str(value): _node_to_dict(child)
            for value, child in node.children.items()
        }

    return d


def export_graphviz(
    tree: 'ID3Classifier',
    filename: Optional[str] = None
) -> str:
    """
    Export tree to Graphviz DOT format.

    Args:
        tree: Fitted ID3Classifier instance.
        filename: Optional path to write DOT file.

    Returns:
        DOT format string representation.
    """
    lines: List[str] = ["digraph Tree {"]
    lines.append('    node [shape=box, style="rounded,filled"];')

    node_id: List[int] = [0]  # Mutable counter for unique node IDs

    def add_node(
        node: 'Node',
        parent_id: Optional[int] = None,
        edge_label: Optional[str] = None
    ) -> None:
        """Add node and its children to the graph."""
        if node is None:
            return

        current_id: int = node_id[0]
        node_id[0] += 1

        if node.is_leaf:
            label: str = f"{node.label}\\n({node.samples} samples)"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#90EE90"];')
        else:
            label = f"{node.feature_name}?\\n({node.samples} samples)"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#ADD8E6"];')

        if parent_id is not None and edge_label is not None:
            lines.append(f'    n{parent_id} -> n{current_id} [label="{edge_label}"];')

        if not node.is_leaf:
            for value, child in node.children.items():
                add_node(child, current_id, str(value))

    if tree.root is not None:
        add_node(tree.root)

    lines.append("}")

    dot: str = "\n".join(lines)

    if filename:
        with open(filename, 'w') as f:
            f.write(dot)

    return dot
