"""
Decision Trees Package.

A from-scratch implementation of ID3, C4.5, and Ensemble methods.
"""
from .base import DecisionTreeBase
from .id3 import ID3Classifier
from .c45 import C45Classifier

# Ensemble Methods
from .ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

# Utilities
from .metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ClassificationMetrics
)
from .tuning import GridSearchCV, RandomizedSearchCV
from .serialization import save_model, load_model, model_summary
from .optimized import (
    entropy_fast,
    information_gain_fast,
    gini_impurity,
    gain_ratio_fast,
    FeatureImportance
)

# Benchmarks
from .benchmarks import BenchmarkSuite

__version__ = "3.0.0"
__author__ = "Hssaine, Benalla, El Younsi"

__all__ = [
    # Core
    "DecisionTreeBase",
    "ID3Classifier",
    "C45Classifier",
    # Ensemble
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
    # Metrics
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "classification_report",
    "ClassificationMetrics",
    # Tuning
    "GridSearchCV",
    "RandomizedSearchCV",
    # Serialization
    "save_model",
    "load_model",
    "model_summary",
    # Optimized
    "entropy_fast",
    "information_gain_fast",
    "gini_impurity",
    "gain_ratio_fast",
    "FeatureImportance",
    # Benchmarks
    "BenchmarkSuite",
]
