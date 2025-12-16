"""
Random Forest Classifier.

Bagging ensemble using C4.5 decision trees with feature subsampling.
"""
import random
from collections import Counter
from typing import Any, List, Optional, Tuple

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from ..c45.core.tree import C45Classifier

Labels = List[Any]
Dataset = List[Tuple[Any, ...]]


class RandomForestClassifier:
    """
    Random Forest using C4.5 trees.
    
    Features:
    - Bootstrap sampling
    - Feature bagging (sqrt or log2)
    - OOB score estimation
    - Parallel tree construction
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.estimators_: List[C45Classifier] = []
        self.feature_names_: Optional[List[str]] = None
        self.oob_score_: float = 0.0

    def fit(
        self,
        X: Dataset,
        y: Labels,
        feature_names: Optional[List[str]] = None
    ) -> 'RandomForestClassifier':
        """Fit the random forest."""
        if self.random_state is not None:
            random.seed(self.random_state)

        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        self.feature_names_ = feature_names or [f"f{i}" for i in range(n_features)]

        # Determine max features to consider
        if self.max_features == 'sqrt':
            max_feat = max(1, int(n_features ** 0.5))
        elif self.max_features == 'log2':
            max_feat = max(1, int(n_features ** 0.5))
        elif isinstance(self.max_features, int):
            max_feat = self.max_features
        else:
            max_feat = n_features

        # Build trees
        def build_tree(seed: int) -> Tuple[C45Classifier, List[int]]:
            rng = random.Random(seed)
            
            # Bootstrap sample
            if self.bootstrap:
                indices = [rng.randint(0, n_samples - 1) for _ in range(n_samples)]
            else:
                indices = list(range(n_samples))
                
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]

            # Feature subsampling
            features = rng.sample(range(n_features), min(max_feat, n_features))
            X_sub = [tuple(sample[f] for f in features) for sample in X_boot]
            sub_names = [self.feature_names_[f] for f in features]

            tree = C45Classifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sub, y_boot, feature_names=sub_names)
            tree._feature_indices = features  # Store for prediction
            return tree, indices

        if HAS_JOBLIB and self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(build_tree)(i) for i in range(self.n_estimators)
            )
        else:
            results = [build_tree(i) for i in range(self.n_estimators)]

        self.estimators_ = [r[0] for r in results]
        oob_indices = [set(range(n_samples)) - set(r[1]) for r in results]

        # OOB score
        if self.oob_score and self.bootstrap:
            self.oob_score_ = self._compute_oob_score(X, y, oob_indices)

        return self

    def predict(self, X: Dataset) -> Labels:
        """Predict using majority voting."""
        predictions = []
        for sample in X:
            votes = []
            for tree in self.estimators_:
                # Extract relevant features
                indices = getattr(tree, '_feature_indices', list(range(len(sample))))
                sub_sample = tuple(sample[f] for f in indices)
                votes.append(tree.predict_one(sub_sample))
            predictions.append(Counter(votes).most_common(1)[0][0])
        return predictions

    def predict_proba(self, X: Dataset) -> List[dict]:
        """Predict class probabilities."""
        probas = []
        for sample in X:
            votes = Counter()
            for tree in self.estimators_:
                indices = getattr(tree, '_feature_indices', list(range(len(sample))))
                sub_sample = tuple(sample[f] for f in indices)
                votes[tree.predict_one(sub_sample)] += 1
            total = sum(votes.values())
            probas.append({k: v / total for k, v in votes.items()})
        return probas

    def _compute_oob_score(self, X: Dataset, y: Labels, oob_indices: List[set]) -> float:
        """Compute out-of-bag accuracy."""
        oob_predictions = {}
        
        for i, tree in enumerate(self.estimators_):
            for idx in oob_indices[i]:
                sample = X[idx]
                indices = getattr(tree, '_feature_indices', list(range(len(sample))))
                sub_sample = tuple(sample[f] for f in indices)
                pred = tree.predict_one(sub_sample)
                
                if idx not in oob_predictions:
                    oob_predictions[idx] = []
                oob_predictions[idx].append(pred)

        correct = 0
        total = 0
        for idx, preds in oob_predictions.items():
            if preds:
                majority = Counter(preds).most_common(1)[0][0]
                if majority == y[idx]:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return f"RandomForestClassifier(n_estimators={self.n_estimators})"
