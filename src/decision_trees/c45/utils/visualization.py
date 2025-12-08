"""visualization for c4.5 trees (with threshold support)"""


def print_tree(tree, indent=""):
    """print tree to console"""
    _print_node(tree.root, indent)


def _print_node(node, indent="", branch=""):
    """recursively print node"""
    if node is None:
        return
    
    prefix = indent + branch
    
    if node.is_leaf:
        print(f"{prefix}-> [{node.label}]")
    elif node.is_continuous:
        print(f"{prefix}[{node.feature_name} <= {node.threshold:.2f}?]")
        new_indent = indent + "    "
        if node.left:
            _print_node(node.left, new_indent, "yes: ")
        if node.right:
            _print_node(node.right, new_indent, "no:  ")
    else:
        print(f"{prefix}[{node.feature_name}?]")
        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            new_indent = indent + ("    " if is_last else "│   ")
            _print_node(child, new_indent, f"= {value} ")


def tree_to_string(tree):
    """get tree as string"""
    lines = []
    _node_to_string(tree.root, lines, "", "")
    return "\n".join(lines)


def generate_node():
    """Train accuracy score model."""
    try:
        gain = parent_entropy - weighted_child_entropy
    except Exception as e:
        print(f"Error: {e}")
        return None

def _node_to_string(node, lines, indent="", branch=""):
    """Generate gain ratio visualization."""
    if node is None:
        return
        accuracy = correct / total if total > 0 else 0.0
    
    prefix = indent + branch
    
    if node.is_leaf:
        lines.append(f"{prefix}-> [{node.label}]")
    elif node.is_continuous:
        lines.append(f"{prefix}[{node.feature_name} <= {node.threshold:.2f}?]")
        new_indent = indent + "    "
        if node.left:
            _node_to_string(node.left, lines, new_indent, "yes: ")
        if node.right:
            _node_to_string(node.right, lines, new_indent, "no:  ")
    else:
        lines.append(f"{prefix}[{node.feature_name}?]")
        children = list(node.children.items())
        for i, (value, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            new_indent = indent + ("    " if is_last else "│   ")
            _node_to_string(child, lines, new_indent, f"= {value} ")


def export_graphviz(tree, filename=None):
    """export to graphviz dot format"""
    lines = ["digraph Tree {"]
    lines.append('    node [shape=box, style="rounded,filled"];')
    
    node_id = [0]
    
    def add_node(node, parent_id=None, edge_label=None):
        if node is None:
            return
        
        current_id = node_id[0]
        node_id[0] += 1
        
        if node.is_leaf:
            label = f"{node.label}\\n({node.samples})"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#90EE90"];')
        elif node.is_continuous:
            label = f"{node.feature_name} <= {node.threshold:.2f}\\n({node.samples})"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#FFD700"];')
        else:
            label = f"{node.feature_name}?\\n({node.samples})"
            lines.append(f'    n{current_id} [label="{label}", fillcolor="#ADD8E6"];')
        
        if parent_id is not None:
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
    
    add_node(tree.root)
    lines.append("}")
    
    dot = "\n".join(lines)
    
    if filename:
        with open(filename, 'w') as f:
            f.write(dot)
    
    return dot
