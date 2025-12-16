"""
Evaluation Metrics for Decision Trees.

This module provides comprehensive classification metrics including
confusion matrix, precision, recall, F1-score, and more.

Metrics Included:
    - Accuracy
    - Precision (per-class and macro/micro)
    - Recall (per-class and macro/micro)
    - F1-Score (per-class and macro/micro)
    - Confusion Matrix
    - Classification Report

Reference:
    Powers, D.M.W. (2011). "Evaluation: From Precision, Recall and
    F-Measure to ROC, Informedness, Markedness & Correlation"
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

Labels = List[Any]


def accuracy_score(y_true: Labels, y_pred: Labels) -> float:
    """
    Calculate classification accuracy.

    Mathematical Formula:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
                 = Correct / Total

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        float: Accuracy in range [0, 1].

    Examples:
        >>> accuracy_score(['a', 'b', 'a'], ['a', 'b', 'b'])
        0.6666666666666666
    """
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(
    y_true: Labels,
    y_pred: Labels,
    labels: Optional[List[Any]] = None
) -> Tuple[List[List[int]], List[Any]]:
    """
    Compute confusion matrix.

    The confusion matrix C[i, j] contains the count of samples with
    true label i that were predicted as label j.

    Mathematical Definition:
        C[i, j] = |{x : true(x) = label_i ∧ pred(x) = label_j}|

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Optional list of label names in desired order.

    Returns:
        Tuple of (matrix, labels) where matrix is a 2D list and
        labels is the list of class labels.

    Examples:
        >>> cm, labels = confusion_matrix(['a', 'b', 'a'], ['a', 'b', 'b'])
        >>> cm
        [[1, 1], [0, 1]]
        >>> labels
        ['a', 'b']
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)

    matrix = [[0] * n_labels for _ in range(n_labels)]

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1

    return matrix, labels


def precision_score(
    y_true: Labels,
    y_pred: Labels,
    average: str = 'macro'
) -> Union[float, Dict[Any, float]]:
    """
    Calculate precision (positive predictive value).

    Mathematical Formula:
        Precision(c) = TP(c) / (TP(c) + FP(c))
                     = TP(c) / Predicted(c)

    Where TP(c) is true positives for class c.

    Averaging Methods:
        - 'macro': Unweighted mean of per-class precision
        - 'micro': TP_total / (TP_total + FP_total)
        - 'weighted': Weighted by class support
        - None: Return per-class dictionary

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method.

    Returns:
        float or Dict: Precision score(s).
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n = len(labels)

    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(n)) - tp
        per_class[label] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    if average is None:
        return per_class
    elif average == 'macro':
        return sum(per_class.values()) / len(per_class) if per_class else 0.0
    elif average == 'micro':
        tp_total = sum(cm[i][i] for i in range(n))
        fp_total = sum(sum(cm[j][i] for j in range(n)) - cm[i][i] for i in range(n))
        return tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    elif average == 'weighted':
        support = Counter(y_true)
        total = len(y_true)
        return sum(
            per_class[label] * support[label] / total
            for label in labels if label in support
        )
    else:
        raise ValueError(f"Unknown average: {average}")


def recall_score(
    y_true: Labels,
    y_pred: Labels,
    average: str = 'macro'
) -> Union[float, Dict[Any, float]]:
    """
    Calculate recall (sensitivity, true positive rate).

    Mathematical Formula:
        Recall(c) = TP(c) / (TP(c) + FN(c))
                  = TP(c) / Actual(c)

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method ('macro', 'micro', 'weighted', or None).

    Returns:
        float or Dict: Recall score(s).
    """
    cm, labels = confusion_matrix(y_true, y_pred)
    n = len(labels)

    per_class = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fn = sum(cm[i][j] for j in range(n)) - tp
        per_class[label] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if average is None:
        return per_class
    elif average == 'macro':
        return sum(per_class.values()) / len(per_class) if per_class else 0.0
    elif average == 'micro':
        tp_total = sum(cm[i][i] for i in range(n))
        fn_total = sum(sum(cm[i][j] for j in range(n)) - cm[i][i] for i in range(n))
        return tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    elif average == 'weighted':
        support = Counter(y_true)
        total = len(y_true)
        return sum(
            per_class[label] * support[label] / total
            for label in labels if label in support
        )
    else:
        raise ValueError(f"Unknown average: {average}")


def f1_score(
    y_true: Labels,
    y_pred: Labels,
    average: str = 'macro'
) -> Union[float, Dict[Any, float]]:
    """
    Calculate F1 score (harmonic mean of precision and recall).

    Mathematical Formula:
        F1 = 2 × (Precision × Recall) / (Precision + Recall)

    The F1 score balances precision and recall, giving both equal
    weight. It's useful when you care about both false positives
    and false negatives.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging method.

    Returns:
        float or Dict: F1 score(s).
    """
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)

    per_class = {}
    for label in prec:
        p, r = prec[label], rec[label]
        per_class[label] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    if average is None:
        return per_class
    elif average == 'macro':
        return sum(per_class.values()) / len(per_class) if per_class else 0.0
    elif average == 'weighted':
        support = Counter(y_true)
        total = len(y_true)
        return sum(
            per_class[label] * support.get(label, 0) / total
            for label in per_class
        )
    else:
        raise ValueError(f"Unknown average: {average}")


def classification_report(
    y_true: Labels,
    y_pred: Labels,
    digits: int = 4
) -> str:
    """
    Generate a text report showing main classification metrics.

    Similar to sklearn's classification_report, shows precision,
    recall, F1-score, and support for each class.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        digits: Number of decimal places.

    Returns:
        str: Formatted classification report.

    Examples:
        >>> print(classification_report(['a', 'b', 'a'], ['a', 'b', 'b']))
                      precision    recall  f1-score   support
        ...
    """
    labels = sorted(set(y_true) | set(y_pred))
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    support = Counter(y_true)

    # Header
    header = f"{'':>15} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    lines = [header, '-' * len(header)]

    # Per-class rows
    for label in labels:
        p = prec.get(label, 0.0)
        r = rec.get(label, 0.0)
        f = f1.get(label, 0.0)
        s = support.get(label, 0)
        lines.append(
            f"{str(label):>15} {p:>10.{digits}f} {r:>10.{digits}f} "
            f"{f:>10.{digits}f} {s:>10}"
        )

    lines.append('')

    # Averages
    total = len(y_true)
    for avg_name in ['macro avg', 'weighted avg']:
        avg = 'macro' if avg_name.startswith('macro') else 'weighted'
        p = precision_score(y_true, y_pred, average=avg)
        r = recall_score(y_true, y_pred, average=avg)
        f = f1_score(y_true, y_pred, average=avg)
        lines.append(
            f"{avg_name:>15} {p:>10.{digits}f} {r:>10.{digits}f} "
            f"{f:>10.{digits}f} {total:>10}"
        )

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    lines.insert(-2, f"{'accuracy':>15} {'':>10} {'':>10} {acc:>10.{digits}f} {total:>10}")

    return '\n'.join(lines)


def print_confusion_matrix(
    y_true: Labels,
    y_pred: Labels,
    labels: Optional[List[Any]] = None
) -> str:
    """
    Generate a formatted confusion matrix string.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: Optional label order.

    Returns:
        str: Formatted confusion matrix.
    """
    cm, label_names = confusion_matrix(y_true, y_pred, labels)

    # Calculate column widths
    max_label = max(len(str(l)) for l in label_names)
    max_val = max(max(row) for row in cm) if cm else 0
    val_width = max(len(str(max_val)), 4)

    # Header row
    header = ' ' * (max_label + 2) + '  '.join(
        f"{str(l):>{val_width}}" for l in label_names
    )
    lines = ['Confusion Matrix:', header, '-' * len(header)]

    # Data rows
    for i, label in enumerate(label_names):
        row_str = '  '.join(f"{cm[i][j]:>{val_width}}" for j in range(len(label_names)))
        lines.append(f"{str(label):>{max_label}}  {row_str}")

    return '\n'.join(lines)


class ClassificationMetrics:
    """
    Container for all classification metrics.

    Computes all metrics once and provides easy access to results.

    Attributes:
        accuracy: Overall accuracy.
        precision: Per-class and average precision.
        recall: Per-class and average recall.
        f1: Per-class and average F1.
        confusion_matrix: The confusion matrix.
        support: Class support counts.
    """

    def __init__(self, y_true: Labels, y_pred: Labels) -> None:
        """
        Compute all metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        """
        self.y_true = y_true
        self.y_pred = y_pred

        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision_per_class = precision_score(y_true, y_pred, average=None)
        self.recall_per_class = recall_score(y_true, y_pred, average=None)
        self.f1_per_class = f1_score(y_true, y_pred, average=None)

        self.precision_macro = precision_score(y_true, y_pred, average='macro')
        self.recall_macro = recall_score(y_true, y_pred, average='macro')
        self.f1_macro = f1_score(y_true, y_pred, average='macro')

        self.cm, self.labels = confusion_matrix(y_true, y_pred)
        self.support = Counter(y_true)

    def summary(self) -> str:
        """Get formatted summary of all metrics."""
        return classification_report(self.y_true, self.y_pred)

    def __repr__(self) -> str:
        return (
            f"ClassificationMetrics(accuracy={self.accuracy:.4f}, "
            f"f1_macro={self.f1_macro:.4f})"
        )
