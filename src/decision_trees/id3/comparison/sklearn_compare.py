"""
Comparison utilities for benchmarking ID3 against scikit-learn.

This module provides functions to compare our from-scratch ID3 implementation
with scikit-learn's DecisionTreeClassifier.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Type

# Type aliases
Dataset = List[Tuple[Any, ...]]
Labels = List[Any]
ComparisonResults = Dict[str, Dict[str, Any]]


def compare_with_sklearn(
    our_model: Any,
    X: Dataset,
    y: Labels,
    X_test: Optional[Dataset] = None,
    y_test: Optional[Labels] = None
) -> Optional[ComparisonResults]:
    """
    Compare our ID3 implementation with sklearn's DecisionTreeClassifier.

    Encodes categorical features for sklearn compatibility and computes
    accuracy, tree depth, and leaf count for both models.

    Args:
        our_model: Fitted ID3Classifier instance.
        X: Training dataset.
        y: Training labels.
        X_test: Optional test dataset.
        y_test: Optional test labels.

    Returns:
        Dictionary with comparison metrics, or None if sklearn not installed.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("sklearn not installed, skipping comparison")
        return None

    # Encode labels for sklearn
    le = LabelEncoder()
    y_encoded: List[int] = le.fit_transform(y).tolist()

    # Encode categorical features for sklearn
    feature_encoders: List[Dict[Any, int]] = []
    X_encoded: List[List[int]] = []

    for sample in X:
        encoded_sample: List[int] = []
        for i, val in enumerate(sample):
            if i >= len(feature_encoders):
                feature_encoders.append({})
            if val not in feature_encoders[i]:
                feature_encoders[i][val] = len(feature_encoders[i])
            encoded_sample.append(feature_encoders[i][val])
        X_encoded.append(encoded_sample)

    # Train sklearn model
    sklearn_model = DecisionTreeClassifier(criterion='entropy', random_state=42)

    start: float = time.time()
    sklearn_model.fit(X_encoded, y_encoded)
    sklearn_train_time: float = time.time() - start

    # Predictions on training data
    our_pred: Labels = our_model.predict(X)
    sklearn_pred: Labels = le.inverse_transform(
        sklearn_model.predict(X_encoded)
    ).tolist()

    our_acc: float = sum(1 for t, p in zip(y, our_pred) if t == p) / len(y)
    sklearn_acc: float = sum(1 for t, p in zip(y, sklearn_pred) if t == p) / len(y)

    results: ComparisonResults = {
        "our_model": {
            "train_accuracy": our_acc,
            "depth": our_model.get_depth(),
            "n_leaves": our_model.get_n_leaves()
        },
        "sklearn": {
            "train_accuracy": sklearn_acc,
            "depth": sklearn_model.get_depth(),
            "n_leaves": sklearn_model.get_n_leaves(),
            "train_time": sklearn_train_time
        }
    }

    # Test set evaluation if provided
    if X_test is not None and y_test is not None:
        X_test_encoded: List[List[int]] = []
        for sample in X_test:
            encoded_sample: List[int] = []
            for i, val in enumerate(sample):
                if val in feature_encoders[i]:
                    encoded_sample.append(feature_encoders[i][val])
                else:
                    encoded_sample.append(-1)  # Unknown value
            X_test_encoded.append(encoded_sample)

        our_test_pred: Labels = our_model.predict(X_test)
        sklearn_test_pred: Labels = le.inverse_transform(
            sklearn_model.predict(X_test_encoded)
        ).tolist()

        results["our_model"]["test_accuracy"] = (
            sum(1 for t, p in zip(y_test, our_test_pred) if t == p) / len(y_test)
        )
        results["sklearn"]["test_accuracy"] = (
            sum(1 for t, p in zip(y_test, sklearn_test_pred) if t == p) / len(y_test)
        )

    return results


def compute_confusion_matrix(
    y_true: Labels,
    y_pred: Labels
) -> Tuple[Dict[Tuple[Any, Any], int], List[Any]]:
    """
    Compute confusion matrix from true and predicted labels.

    Mathematical Definition:
        CM[i,j] = count of samples with true label i predicted as j

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Tuple of (confusion_dict, classes) where:
            - confusion_dict: {(true, pred): count}
            - classes: sorted list of unique classes
    """
    classes: List[Any] = sorted(set(y_true) | set(y_pred))
    cm: Dict[Tuple[Any, Any], int] = {}

    for t, p in zip(y_true, y_pred):
        key = (t, p)
        cm[key] = cm.get(key, 0) + 1

    return cm, classes


def compute_metrics(
    y_true: Labels,
    y_pred: Labels,
    positive_class: Optional[Any] = None
) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, precision, recall, F1-score.

    Mathematical Formulas:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = 2 × (Precision × Recall) / (Precision + Recall)

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        positive_class: Class to treat as positive (for binary metrics).
                       If None, uses first class alphabetically.

    Returns:
        Dictionary with 'accuracy', 'precision', 'recall', 'f1' scores.
    """
    classes: List[Any] = sorted(set(y_true) | set(y_pred))
    
    if positive_class is None:
        positive_class = classes[0] if classes else None

    if positive_class is None:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Compute confusion matrix components
    tp: int = sum(1 for t, p in zip(y_true, y_pred) 
                  if t == positive_class and p == positive_class)
    fp: int = sum(1 for t, p in zip(y_true, y_pred) 
                  if t != positive_class and p == positive_class)
    fn: int = sum(1 for t, p in zip(y_true, y_pred) 
                  if t == positive_class and p != positive_class)
    tn: int = sum(1 for t, p in zip(y_true, y_pred) 
                  if t != positive_class and p != positive_class)

    # Calculate metrics
    accuracy: float = (tp + tn) / len(y_true) if y_true else 0.0
    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def print_comparison(results: Optional[ComparisonResults]) -> None:
    """
    Pretty-print comparison results between our model and sklearn.

    Args:
        results: Comparison results dictionary from compare_with_sklearn().
    """
    if results is None:
        return

    print("\n" + "=" * 50)
    print("COMPARISON: Our ID3 vs sklearn DecisionTree")
    print("=" * 50)

    print(f"\n{'Metric':<25} {'Our ID3':>12} {'sklearn':>12}")
    print("-" * 50)

    our = results["our_model"]
    sk = results["sklearn"]

    print(f"{'Train Accuracy':<25} {our['train_accuracy']:>12.4f} {sk['train_accuracy']:>12.4f}")

    if "test_accuracy" in our:
        print(f"{'Test Accuracy':<25} {our['test_accuracy']:>12.4f} {sk['test_accuracy']:>12.4f}")

    print(f"{'Tree Depth':<25} {our['depth']:>12} {sk['depth']:>12}")
    print(f"{'Number of Leaves':<25} {our['n_leaves']:>12} {sk['n_leaves']:>12}")
    print()


def print_confusion_matrix(
    y_true: Labels,
    y_pred: Labels,
    title: str = "Confusion Matrix"
) -> None:
    """
    Print a formatted confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        title: Title for the matrix display.
    """
    cm, classes = compute_confusion_matrix(y_true, y_pred)

    print(f"\n{title}")
    print("=" * (12 + 10 * len(classes)))

    # Header
    header: str = f"{'True \\ Pred':<12}"
    for c in classes:
        header += f"{str(c):>10}"
    print(header)
    print("-" * (12 + 10 * len(classes)))

    # Rows
    for true_class in classes:
        row: str = f"{str(true_class):<12}"
        for pred_class in classes:
            count: int = cm.get((true_class, pred_class), 0)
            row += f"{count:>10}"
        print(row)
    print()


def benchmark(
    model_class: Type[Any],
    X: Dataset,
    y: Labels,
    n_runs: int = 5,
    **model_params: Any
) -> Dict[str, float]:
    """
    Benchmark training time over multiple runs.

    Args:
        model_class: Classifier class to benchmark.
        X: Training dataset.
        y: Training labels.
        n_runs: Number of training runs.
        **model_params: Parameters passed to the classifier constructor.

    Returns:
        Dictionary with mean, min, max times and number of runs.
    """
    times: List[float] = []

    for _ in range(n_runs):
        model = model_class(**model_params)
        start: float = time.time()
        model.fit(X, y)
        times.append(time.time() - start)

    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "n_runs": n_runs
    }
