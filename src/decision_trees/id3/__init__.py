# id3 decision tree package
from .core.tree import ID3Classifier
from .core.entropy import entropy, information_gain
from .core.node import Node

__version__ = "1.0.0"
__all__ = ["ID3Classifier", "entropy", "information_gain", "Node"]
