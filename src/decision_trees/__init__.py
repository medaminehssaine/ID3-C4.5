"""
Decision Trees Package.

A from-scratch implementation of ID3 and C4.5 decision tree algorithms
following Quinlan's original papers, with professional OOP architecture.

Core Classifiers:
    - ID3Classifier: Classic ID3 with Information Gain (categorical only)
    - C45Classifier: Extended C4.5 with Gain Ratio, continuous features, pruning

Utilities:
    - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
    - Evaluation metrics (confusion matrix, F1, precision, recall)
    - Model serialization (save/load JSON)
    - NumPy-optimized computations

Reference:
    Quinlan, J.R. (1986). "Induction of Decision Trees", Machine Learning 1:81-106
    Quinlan, J.R. (1993). "C4.5: Programs for Machine Learning", Morgan Kaufmann
"""
from .base import DecisionTreeBase
from .id3 import ID3Classifier
from .c45 import C45Classifier

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

__version__ = "2.0.0"
__author__ = "Decision Tree Research Team"

__all__ = [
    # Core
    "DecisionTreeBase",
    "ID3Classifier",
    "C45Classifier",
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
]
