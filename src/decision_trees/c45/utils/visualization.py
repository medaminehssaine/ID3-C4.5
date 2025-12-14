"""
Visualization utilities for C4.5 decision trees.

Provides functions to:
- Print tree structure to console (with threshold support)
- Export tree to Graphviz DOT format
- Convert tree to dictionary for serialization
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.tree import C45Classifier
    from ..core.node import Node


def print_tree(tree: 'C45Classifier', indent: str = "") -> None:
    """
    Print tree structure to console.

    Handles both categorical (multi-way) and continuous (binary) splits.

    Args:
        tree: Fitted C45Classifier instance.
        indent: Initial indentation.
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
    elif node.is_continuous:
        print(f"{prefix}[{node.feature_name} <= {node.threshold:.2f}?]")
        new_indent: str = indent + "    "
        if node.left:
            _print_node(node.left, new_indent, "yes: ")
        if node.right:
            _print_node(node.right, new_indent, "no:  ")
    else:
        print(f"{prefix}[{node.feature_name}?]")
        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last: bool = (i == len(children) - 1)
            new_indent = indent + ("    " if is_last else "│   ")
            _print_node(child, new_indent, f"= {value} ")


def tree_to_string(tree: 'C45Classifier') -> str:
    """
    Get tree as string instead of printing.

    Args:
        tree: Fitted C45Classifier instance.

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
    elif node.is_continuous:
        lines.append(f"{prefix}[{node.feature_name} <= {node.threshold:.2f}?]")
        new_indent: str = indent + "    "
        if node.left:
            _node_to_string(node.left, lines, new_indent, "yes: ")
        if node.right:
            _node_to_string(node.right, lines, new_indent, "no:  ")
    else:
        lines.append(f"{prefix}[{node.feature_name}?]")
        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last: bool = (i == len(children) - 1)
            new_indent = indent + ("    " if is_last else "│   ")
            _node_to_string(child, lines, new_indent, f"= {value} ")


def tree_to_dict(tree: 'C45Classifier') -> Optional[Dict[str, Any]]:
    """
    Convert tree to dictionary for serialization.

    Args:
        tree: Fitted C45Classifier instance.

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
    elif node.is_continuous:
        d["feature"] = node.feature
        d["feature_name"] = node.feature_name
        d["threshold"] = node.threshold
        d["left"] = _node_to_dict(node.left)
        d["right"] = _node_to_dict(node.right)
    else:
        d["feature"] = node.feature
        d["feature_name"] = node.feature_name
        d["children"] = {
            str(value): _node_to_dict(child)
            for value, child in node.children.items()
        }

    return d


def export_graphviz(
    tree: 'C45Classifier',
    filename: Optional[str] = None
) -> str:
    """
    Export tree to Graphviz DOT format.

    Uses different colors for:
    - Leaves: green
    - Continuous splits: gold
    - Categorical splits: light blue

    Args:
        tree: Fitted C45Classifier instance.
        filename: Optional path to write DOT file.

    Returns:
        DOT format string representation.
    """
    lines: List[str] = ["digraph Tree {"]
    lines.append('    node [shape=box, style="rounded,filled"];')

    node_id: List[int] = [0]  # Mutable counter

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
            label: str = f"{node.label}\\n({node.samples})"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#90EE90"];')
        elif node.is_continuous:
            label = f"{node.feature_name} <= {node.threshold:.2f}\\n({node.samples})"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#FFD700"];')
        else:
            label = f"{node.feature_name}?\\n({node.samples})"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#ADD8E6"];')

        if parent_id is not None and edge_label is not None:
            lines.append(f'    n{parent_id} -> n{current_id} [label="{edge_label}"];')

        if not node.is_leaf:
            if node.is_continuous:
                if node.left:
                    add_node(node.left, current_id, "yes")
                if node.right:
                    add_node(node.right, current_id, "no")
            else:
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
