# decision trees package
# unified access to id3 and c4.5 implementations

from .id3 import ID3Classifier
from .c45 import C45Classifier

__version__ = "1.0.0"
__all__ = ["ID3Classifier", "C45Classifier"]
