"""
Feature Importance Analysis.

Multiple methods: Gini importance, Permutation importance, Split count.
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import random


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods."""

    def __init__(self, model: Any):
        self.model = model
        self.importances_: Dict[str, float] = {}

    def compute_gini_importance(self) -> Dict[str, float]:
        """
        Compute importance as total impurity reduction per feature.
        
        This is the default sklearn method.
        """
        importances = defaultdict(float)
        total_samples = self._get_total_samples()

        def traverse(node, n_samples_parent):
            if node is None or getattr(node, 'is_leaf', True):
                return

            feature_name = getattr(node, 'feature_name', None)
            if feature_name is None:
                return

            # Estimate impurity reduction (simplified)
            n_samples = getattr(node, 'samples', n_samples_parent)
            weight = n_samples / total_samples
            importances[feature_name] += weight

            # Traverse children
            if hasattr(node, 'children'):
                for child in node.children.values():
                    traverse(child, n_samples)
            if hasattr(node, 'left'):
                traverse(node.left, n_samples)
            if hasattr(node, 'right'):
                traverse(node.right, n_samples)

        root = getattr(self.model, 'root', None)
        if root:
            traverse(root, total_samples)

        # Normalize
        total = sum(importances.values()) or 1
        self.importances_ = {k: v / total for k, v in importances.items()}
        return self.importances_

    def compute_split_count(self) -> Dict[str, int]:
        """Count how many times each feature is used for splitting."""
        counts = defaultdict(int)

        def traverse(node):
            if node is None or getattr(node, 'is_leaf', True):
                return

            feature_name = getattr(node, 'feature_name', None)
            if feature_name:
                counts[feature_name] += 1

            if hasattr(node, 'children'):
                for child in node.children.values():
                    traverse(child)
            if hasattr(node, 'left'):
                traverse(node.left)
            if hasattr(node, 'right'):
                traverse(node.right)

        root = getattr(self.model, 'root', None)
        if root:
            traverse(root)

        return dict(counts)

    def compute_permutation_importance(
        self,
        X: List[Tuple],
        y: List[Any],
        n_repeats: int = 5,
        random_state: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute permutation importance.
        
        Measures accuracy drop when feature values are shuffled.
        """
        if random_state is not None:
            random.seed(random_state)

        from decision_trees import accuracy_score

        # Baseline accuracy
        y_pred = self.model.predict(X)
        baseline = accuracy_score(y, y_pred)

        n_features = len(X[0])
        feature_names = getattr(self.model, 'feature_names', 
                                [f'f{i}' for i in range(n_features)])

        importances = {}
        for f_idx in range(n_features):
            drops = []
            for _ in range(n_repeats):
                # Permute feature
                X_perm = [list(sample) for sample in X]
                values = [sample[f_idx] for sample in X]
                random.shuffle(values)
                for i, sample in enumerate(X_perm):
                    sample[f_idx] = values[i]
                X_perm = [tuple(s) for s in X_perm]

                # Compute accuracy
                y_pred_perm = self.model.predict(X_perm)
                acc_perm = accuracy_score(y, y_pred_perm)
                drops.append(baseline - acc_perm)

            importances[feature_names[f_idx]] = sum(drops) / len(drops)

        self.importances_ = importances
        return importances

    def to_ranked_list(self) -> List[Tuple[str, float]]:
        """Return features ranked by importance."""
        return sorted(self.importances_.items(), key=lambda x: -x[1])

    def _get_total_samples(self) -> int:
        """Get total training samples from model."""
        root = getattr(self.model, 'root', None)
        if root and hasattr(root, 'samples'):
            return root.samples
        return 100  # Fallback


def feature_importance(model: Any, method: str = 'gini') -> Dict[str, float]:
    """Quick function to compute feature importance."""
    analyzer = FeatureImportanceAnalyzer(model)
    if method == 'gini':
        return analyzer.compute_gini_importance()
    elif method == 'count':
        return analyzer.compute_split_count()
    else:
        raise ValueError(f"Unknown method: {method}")
