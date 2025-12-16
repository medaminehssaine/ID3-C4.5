"""
AdaBoost Classifier.

Adaptive Boosting using decision stumps.
"""
import math
from collections import Counter
from typing import Any, List, Optional, Tuple

from ..id3.core.tree import ID3Classifier

Labels = List[Any]
Dataset = List[Tuple[Any, ...]]


class AdaBoostClassifier:
    """
    AdaBoost using decision stumps.
    
    Formula:
        α_t = 0.5 × ln((1 - ε_t) / ε_t)
        w_i^(t+1) = w_i^(t) × exp(-α_t × y_i × h_t(x_i))
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.estimators_: List[Tuple[ID3Classifier, float]] = []
        self.classes_: List[Any] = []
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X: Dataset,
        y: Labels,
        feature_names: Optional[List[str]] = None
    ) -> 'AdaBoostClassifier':
        """Fit AdaBoost ensemble."""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        self.feature_names_ = feature_names or [f"f{i}" for i in range(n_features)]
        self.classes_ = list(set(y))

        # Convert to binary {-1, 1}
        class_map = {self.classes_[0]: -1, self.classes_[1]: 1} if len(self.classes_) == 2 else {}
        y_binary = [class_map.get(label, 1) for label in y]

        # Initialize weights
        weights = [1.0 / n_samples] * n_samples

        for t in range(self.n_estimators):
            # Train weak learner (stump)
            stump = ID3Classifier(max_depth=1)
            stump.fit(X, y, feature_names=self.feature_names_)

            # Get predictions
            y_pred = stump.predict(X)
            y_pred_binary = [class_map.get(p, 1) for p in y_pred]

            # Compute weighted error
            error = sum(
                w for w, yp, yt in zip(weights, y_pred_binary, y_binary) if yp != yt
            )
            error = max(error, 1e-10)  # Avoid division by zero
            error = min(error, 1 - 1e-10)  # Avoid log(0)

            # Compute alpha
            alpha = self.learning_rate * 0.5 * math.log((1 - error) / error)

            # Update weights
            new_weights = []
            for i, (w, yp, yt) in enumerate(zip(weights, y_pred_binary, y_binary)):
                new_w = w * math.exp(-alpha * yp * yt)
                new_weights.append(new_w)

            # Normalize
            total = sum(new_weights)
            weights = [w / total for w in new_weights]

            self.estimators_.append((stump, alpha))

        return self

    def predict(self, X: Dataset) -> Labels:
        """Predict using weighted voting."""
        predictions = []
        for sample in X:
            score = 0.0
            for stump, alpha in self.estimators_:
                pred = stump.predict_one(sample)
                if len(self.classes_) == 2:
                    sign = 1 if pred == self.classes_[1] else -1
                else:
                    sign = 1
                score += alpha * sign

            # Map back to class
            if len(self.classes_) == 2:
                predictions.append(self.classes_[1] if score >= 0 else self.classes_[0])
            else:
                predictions.append(self.classes_[0])

        return predictions

    def __repr__(self) -> str:
        return f"AdaBoostClassifier(n_estimators={self.n_estimators})"
