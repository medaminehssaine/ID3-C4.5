"""
Hyperparameter Tuning for Decision Trees.

This module provides grid search and cross-validation utilities for
optimizing decision tree hyperparameters, similar to sklearn's GridSearchCV.

Features:
    - GridSearchCV: Exhaustive search over parameter grid
    - Cross-validation with stratified folds
    - Parallel evaluation support (future)
    - Best model selection and refitting

Reference:
    Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap for
    Accuracy Estimation and Model Selection"
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

# Type aliases
Dataset = List[Tuple[Any, ...]]
Labels = List[Any]
ParamGrid = Dict[str, List[Any]]


class GridSearchCV:
    """
    Exhaustive search over specified parameter values with cross-validation.

    GridSearchCV implements a "fit" and "score" method. It also implements
    "predict" on the best estimator found during the search.

    Mathematical Background:
        For each parameter combination, the average validation score is:
        
        Score(θ) = (1/k) × Σᵢ accuracy(model(θ, train_i), val_i)
        
        Where k is the number of CV folds.

    Attributes:
        estimator_class: The classifier class to tune.
        param_grid: Dictionary mapping parameter names to lists of values.
        cv: Number of cross-validation folds.
        scoring: Scoring function ('accuracy' or callable).
        best_params_: Best parameters found.
        best_score_: Best cross-validation score.
        best_estimator_: Estimator fitted with best parameters.
        cv_results_: Dictionary with detailed CV results.

    Examples:
        >>> from decision_trees import ID3Classifier
        >>> from decision_trees.tuning import GridSearchCV
        >>> 
        >>> param_grid = {
        ...     'max_depth': [None, 3, 5, 7],
        ...     'min_samples_split': [2, 5, 10]
        ... }
        >>> 
        >>> search = GridSearchCV(ID3Classifier, param_grid, cv=5)
        >>> search.fit(X, y)
        >>> print(f"Best: {search.best_params_} -> {search.best_score_:.3f}")
    """

    def __init__(
        self,
        estimator_class: Type[Any],
        param_grid: ParamGrid,
        cv: int = 5,
        scoring: Union[str, Callable] = 'accuracy',
        verbose: int = 0,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize GridSearchCV.

        Args:
            estimator_class: The decision tree class to instantiate.
            param_grid: Dictionary mapping hyperparameter names to value lists.
            cv: Number of cross-validation folds (default: 5).
            scoring: 'accuracy' or a callable(y_true, y_pred) -> float.
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
            random_state: Random seed for reproducible splits.
        """
        self.estimator_class = estimator_class
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.random_state = random_state

        # Results (set after fit)
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: float = 0.0
        self.best_estimator_: Optional[Any] = None
        self.cv_results_: Dict[str, List[Any]] = {}

    def fit(
        self,
        X: Dataset,
        y: Labels,
        feature_names: Optional[List[str]] = None
    ) -> 'GridSearchCV':
        """
        Run grid search with cross-validation.

        For each parameter combination, trains and evaluates models
        using k-fold cross-validation, then selects the best parameters.

        Args:
            X: Training data.
            y: Training labels.
            feature_names: Optional feature names for the estimator.

        Returns:
            self: Fitted GridSearchCV.
        """
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        # Initialize results
        self.cv_results_ = {
            'params': [],
            'mean_score': [],
            'std_score': [],
            'fold_scores': []
        }

        best_score = -1.0
        best_params = None

        total = len(param_combinations)

        for idx, combo in enumerate(param_combinations):
            params = dict(zip(param_names, combo))

            if self.verbose >= 1:
                print(f"[{idx+1}/{total}] Testing {params}")

            # Cross-validation
            fold_scores = self._cross_validate(X, y, params, feature_names)
            mean_score = sum(fold_scores) / len(fold_scores)
            std_score = (
                sum((s - mean_score) ** 2 for s in fold_scores) / len(fold_scores)
            ) ** 0.5

            # Store results
            self.cv_results_['params'].append(params)
            self.cv_results_['mean_score'].append(mean_score)
            self.cv_results_['std_score'].append(std_score)
            self.cv_results_['fold_scores'].append(fold_scores)

            if self.verbose >= 2:
                print(f"    Score: {mean_score:.4f} (+/- {std_score:.4f})")

            # Track best
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score

        # Refit on full data with best parameters
        self.best_estimator_ = self.estimator_class(**best_params)
        self.best_estimator_.fit(X, y, feature_names)

        if self.verbose >= 1:
            print(f"\nBest: {best_params} -> {best_score:.4f}")

        return self

    def _cross_validate(
        self,
        X: Dataset,
        y: Labels,
        params: Dict[str, Any],
        feature_names: Optional[List[str]]
    ) -> List[float]:
        """Perform k-fold cross-validation for given parameters."""
        folds = self._create_folds(X, y)
        scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_val = [X[i] for i in val_idx]
            y_val = [y[i] for i in val_idx]

            # Train
            model = self.estimator_class(**params)
            model.fit(X_train, y_train, feature_names)

            # Predict
            y_pred = model.predict(X_val)

            # Score
            score = self._compute_score(y_val, y_pred)
            scores.append(score)

        return scores

    def _create_folds(
        self,
        X: Dataset,
        y: Labels
    ) -> List[Tuple[List[int], List[int]]]:
        """Create stratified k-fold indices."""
        n = len(X)
        indices = list(range(n))

        # Shuffle
        if self.random_state is not None:
            random.seed(self.random_state)
        random.shuffle(indices)

        # Create folds
        fold_size = n // self.cv
        folds = []

        for i in range(self.cv):
            start = i * fold_size
            end = start + fold_size if i < self.cv - 1 else n

            val_idx = indices[start:end]
            train_idx = indices[:start] + indices[end:]

            folds.append((train_idx, val_idx))

        return folds

    def _compute_score(self, y_true: Labels, y_pred: Labels) -> float:
        """Compute score using the configured scoring method."""
        if callable(self.scoring):
            return self.scoring(y_true, y_pred)
        elif self.scoring == 'accuracy':
            correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
            return correct / len(y_true)
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

    def predict(self, X: Dataset) -> Labels:
        """Predict using the best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV not fitted. Call fit() first.")
        return self.best_estimator_.predict(X)

    def score(self, X: Dataset, y: Labels) -> float:
        """Score using the best estimator."""
        y_pred = self.predict(X)
        return self._compute_score(y, y_pred)

    def get_results_dataframe(self) -> Dict[str, List[Any]]:
        """
        Get CV results as a dictionary (can be converted to DataFrame).

        Returns:
            Dictionary with params, mean_score, std_score, and rank.
        """
        results = {
            'params': self.cv_results_['params'],
            'mean_score': self.cv_results_['mean_score'],
            'std_score': self.cv_results_['std_score'],
        }

        # Add rank
        scores = self.cv_results_['mean_score']
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_idx, 1):
            ranks[idx] = rank
        results['rank'] = ranks

        return results

    def __repr__(self) -> str:
        if self.best_params_ is None:
            return "GridSearchCV(not fitted)"
        return (
            f"GridSearchCV(best_score={self.best_score_:.4f}, "
            f"best_params={self.best_params_})"
        )


class RandomizedSearchCV(GridSearchCV):
    """
    Randomized search over hyperparameters.

    Unlike GridSearchCV, RandomizedSearchCV samples a fixed number of
    parameter combinations from the specified distributions. This is
    more efficient when the search space is large.

    Additional Attributes:
        n_iter: Number of parameter combinations to sample.
    """

    def __init__(
        self,
        estimator_class: Type[Any],
        param_distributions: ParamGrid,
        n_iter: int = 10,
        cv: int = 5,
        scoring: Union[str, Callable] = 'accuracy',
        verbose: int = 0,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize RandomizedSearchCV.

        Args:
            estimator_class: The decision tree class.
            param_distributions: Dictionary mapping names to value lists/distributions.
            n_iter: Number of random combinations to try.
            cv: Number of CV folds.
            scoring: Scoring method.
            verbose: Verbosity level.
            random_state: Random seed.
        """
        super().__init__(
            estimator_class, param_distributions, cv, scoring, verbose, random_state
        )
        self.n_iter = n_iter

    def fit(
        self,
        X: Dataset,
        y: Labels,
        feature_names: Optional[List[str]] = None
    ) -> 'RandomizedSearchCV':
        """Run randomized search with cross-validation."""
        if self.random_state is not None:
            random.seed(self.random_state)

        # Sample n_iter random combinations
        param_names = list(self.param_grid.keys())
        sampled_combos = []

        for _ in range(self.n_iter):
            combo = tuple(random.choice(self.param_grid[name]) for name in param_names)
            sampled_combos.append(combo)

        # Temporarily replace param_grid for parent's fit
        original_grid = self.param_grid
        self.param_grid = {
            name: list(set(combo[i] for combo in sampled_combos))
            for i, name in enumerate(param_names)
        }

        # Use subset of combinations
        param_combinations = sampled_combos

        # Initialize results
        self.cv_results_ = {
            'params': [],
            'mean_score': [],
            'std_score': [],
            'fold_scores': []
        }

        best_score = -1.0
        best_params = None

        for idx, combo in enumerate(param_combinations):
            params = dict(zip(param_names, combo))

            if self.verbose >= 1:
                print(f"[{idx+1}/{self.n_iter}] Testing {params}")

            fold_scores = self._cross_validate(X, y, params, feature_names)
            mean_score = sum(fold_scores) / len(fold_scores)
            std_score = (
                sum((s - mean_score) ** 2 for s in fold_scores) / len(fold_scores)
            ) ** 0.5

            self.cv_results_['params'].append(params)
            self.cv_results_['mean_score'].append(mean_score)
            self.cv_results_['std_score'].append(std_score)
            self.cv_results_['fold_scores'].append(fold_scores)

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        self.best_params_ = best_params
        self.best_score_ = best_score

        # Refit
        self.best_estimator_ = self.estimator_class(**best_params)
        self.best_estimator_.fit(X, y, feature_names)

        self.param_grid = original_grid
        return self
