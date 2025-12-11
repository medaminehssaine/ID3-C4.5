# c4.5 decision tree package
from .core.tree import C45Classifier
from .core.gain_ratio import gain_ratio, split_info, best_threshold
from .core.node import Node

__version__ = "1.0.0"
__all__ = ["C45Classifier", "gain_ratio", "split_info", "best_threshold", "Node"]
