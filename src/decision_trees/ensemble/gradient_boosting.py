"""
Gradient Boosting Classifier.

Gradient Boosting Decision Trees (GBDT) using C4.5.
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from ..c45.core.tree import C45Classifier

Labels = List[Any]
Dataset = List[Tuple[Any, ...]]


class GradientBoostingClassifier:
    """
    Gradient Boosting using C4.5 trees.
    
    Fits a sequence of C4.5 trees to the negative gradients of the loss function.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state

        self.estimators_: List[C45Classifier] = []
        self.init_score_: float = 0.0
        self.classes_: List[Any] = []
        self.feature_names_: Optional[List[str]] = None

    def fit(
        self,
        X: Dataset,
        y: Labels,
        feature_names: Optional[List[str]] = None
    ) -> 'GradientBoostingClassifier':
        """Fit gradient boosting ensemble."""
        if self.random_state is not None:
            random.seed(self.random_state)

        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        self.feature_names_ = feature_names or [f"f{i}" for i in range(n_features)]
        self.classes_ = list(set(y))

        # Binary classification check
        if len(self.classes_) != 2:
            raise ValueError("Currently supports binary classification only")

        class_map = {self.classes_[0]: 0, self.classes_[1]: 1}
        y_binary = [class_map[label] for label in y]

        # Initialize with log-odds
        pos_count = sum(y_binary)
        neg_count = n_samples - pos_count
        self.init_score_ = math.log((pos_count + 1e-10) / (neg_count + 1e-10))

        # Current predictions (log-odds)
        F = [self.init_score_] * n_samples

        for _ in range(self.n_estimators):
            # Compute gradients (negative gradient for binary cross-entropy)
            gradients = []
            for i in range(n_samples):
                p = 1 / (1 + math.exp(-F[i]))  # Sigmoid
                gradients.append(y_binary[i] - p)

            # Subsample
            if self.subsample < 1.0:
                n_sub = int(n_samples * self.subsample)
                indices = random.sample(range(n_samples), n_sub)
                X_sub = [X[i] for i in indices]
                g_sub = [gradients[i] for i in indices]
            else:
                X_sub = X
                g_sub = gradients
                indices = list(range(n_samples))

            # Fit tree to gradients
            # We treat gradients as class labels: 'pos' if > 0 else 'neg'
            # This is a simplification for using C4.5 (which is a classifier) as a regressor
            g_labels = ['pos' if g > 0 else 'neg' for g in g_sub]

            tree = C45Classifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sub, g_labels, feature_names=self.feature_names_)
            
            # Update predictions
            for i in range(n_samples):
                pred = tree.predict_one(X[i])
                # Simple step: move in direction of gradient sign
                step = self.learning_rate * (1 if pred == 'pos' else -1)
                F[i] += step

            self.estimators_.append(tree)

        return self

    def _predict_raw(self, X: Dataset) -> List[float]:
        """Get raw predictions (log-odds)."""
        F = [self.init_score_] * len(X)
        for tree in self.estimators_:
            for i, sample in enumerate(X):
                pred = tree.predict_one(sample)
                step = self.learning_rate * (1 if pred == 'pos' else -1)
                F[i] += step
        return F

    def predict_proba(self, X: Dataset) -> List[Dict[Any, float]]:
        """Predict class probabilities."""
        F = self._predict_raw(X)
        probas = []
        for f in F:
            p = 1 / (1 + math.exp(-f))
            probas.append({
                self.classes_[0]: 1 - p,
                self.classes_[1]: p
            })
        return probas

    def predict(self, X: Dataset) -> Labels:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return [
            self.classes_[1] if p[self.classes_[1]] >= 0.5 else self.classes_[0]
            for p in probas
        ]

    def __repr__(self) -> str:
        return f"GradientBoostingClassifier(n_estimators={self.n_estimators})"
