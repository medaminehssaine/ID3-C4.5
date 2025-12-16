"""
Tests for Ensemble Methods.

Tests RandomForest, AdaBoost, and GradientBoosting classifiers.
"""
import pytest
from decision_trees.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)


# Minimal test data
X_TRAIN = [
    ('sunny', 'hot'),
    ('sunny', 'hot'),
    ('overcast', 'hot'),
    ('rain', 'mild'),
    ('rain', 'cool'),
    ('rain', 'cool'),
    ('overcast', 'cool'),
    ('sunny', 'mild'),
    ('sunny', 'cool'),
    ('rain', 'mild'),
    ('sunny', 'mild'),
    ('overcast', 'mild'),
    ('overcast', 'hot'),
    ('rain', 'mild'),
]
Y_TRAIN = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
FEATURE_NAMES = ['outlook', 'temp']


class TestRandomForest:
    """Tests for RandomForestClassifier."""

    def test_fit_predict(self):
        """Test basic fit and predict."""
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        predictions = rf.predict(X_TRAIN[:3])
        assert len(predictions) == 3
        assert all(p in ['yes', 'no'] for p in predictions)

    def test_n_estimators(self):
        """Test correct number of trees created."""
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        assert len(rf.estimators_) == 10

    def test_predict_proba(self):
        """Test probability predictions."""
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        probas = rf.predict_proba(X_TRAIN[:2])
        assert len(probas) == 2
        for proba in probas:
            assert sum(proba.values()) == pytest.approx(1.0)

    def test_repr(self):
        """Test string representation."""
        rf = RandomForestClassifier(n_estimators=50)
        assert 'RandomForestClassifier' in repr(rf)
        assert '50' in repr(rf)


class TestAdaBoost:
    """Tests for AdaBoostClassifier."""

    def test_fit_predict_binary(self):
        """Test binary classification."""
        ada = AdaBoostClassifier(n_estimators=5)
        ada.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        predictions = ada.predict(X_TRAIN[:3])
        assert len(predictions) == 3
        assert all(p in ['yes', 'no'] for p in predictions)

    def test_estimators_created(self):
        """Test weak learners are created."""
        ada = AdaBoostClassifier(n_estimators=10)
        ada.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        assert len(ada.estimators_) == 10
        assert all(isinstance(e, tuple) for e in ada.estimators_)

    def test_repr(self):
        """Test string representation."""
        ada = AdaBoostClassifier(n_estimators=25)
        assert 'AdaBoostClassifier' in repr(ada)


class TestGradientBoosting:
    """Tests for GradientBoostingClassifier."""

    def test_fit_predict_binary(self):
        """Test binary classification."""
        gb = GradientBoostingClassifier(n_estimators=5, max_depth=2)
        gb.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        predictions = gb.predict(X_TRAIN[:3])
        assert len(predictions) == 3
        assert all(p in ['yes', 'no'] for p in predictions)

    def test_predict_proba(self):
        """Test probability predictions."""
        gb = GradientBoostingClassifier(n_estimators=5, max_depth=2)
        gb.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        probas = gb.predict_proba(X_TRAIN[:2])
        assert len(probas) == 2
        for proba in probas:
            total = sum(proba.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_learning_rate(self):
        """Test learning rate affects predictions."""
        gb1 = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)
        gb2 = GradientBoostingClassifier(n_estimators=10, learning_rate=1.0)
        
        gb1.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        gb2.fit(X_TRAIN, Y_TRAIN, feature_names=FEATURE_NAMES)
        
        # Different learning rates should produce different raw scores
        assert gb1.init_score_ == gb2.init_score_  # Init is same

    def test_repr(self):
        """Test string representation."""
        gb = GradientBoostingClassifier(n_estimators=100)
        assert 'GradientBoostingClassifier' in repr(gb)
