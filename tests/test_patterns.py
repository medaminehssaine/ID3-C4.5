"""
Tests for Design Patterns (Criteria, Callbacks, Builders).
"""
import pytest
from decision_trees.criteria import (
    InformationGainCriterion,
    GainRatioCriterion,
    GiniImpurityCriterion,
    VarianceReductionCriterion,
    get_criterion
)
from decision_trees.callbacks import (
    EarlyStopping,
    ProgressBar,
    MetricsLogger,
    CallbackList
)
from decision_trees.builders import TreeBuilderFactory, ExactGreedyBuilder


class TestCriteria:
    """Tests for split criteria."""

    def test_entropy_pure(self):
        """Entropy of pure set is 0."""
        criterion = InformationGainCriterion()
        assert criterion.impurity(['yes'] * 10) == 0.0

    def test_entropy_balanced(self):
        """Entropy of balanced binary is 1."""
        criterion = InformationGainCriterion()
        assert criterion.impurity(['yes', 'no']) == pytest.approx(1.0)

    def test_gini_pure(self):
        """Gini of pure set is 0."""
        criterion = GiniImpurityCriterion()
        assert criterion.impurity(['yes'] * 10) == 0.0

    def test_gini_balanced(self):
        """Gini of balanced binary is 0.5."""
        criterion = GiniImpurityCriterion()
        assert criterion.impurity(['yes', 'no']) == pytest.approx(0.5)

    def test_gain_ratio(self):
        """Gain Ratio should be <= Information Gain."""
        y = ['yes'] * 5 + ['no'] * 5
        y_splits = [['yes'] * 3 + ['no'] * 2, ['yes'] * 2 + ['no'] * 3]
        
        ig = InformationGainCriterion()
        gr = GainRatioCriterion()
        
        ig_score = ig.split_score(y, y_splits)
        gr_score = gr.split_score(y, y_splits)
        
        assert gr_score <= ig_score

    def test_variance_pure(self):
        """Variance of identical values is 0."""
        criterion = VarianceReductionCriterion()
        assert criterion.impurity([5, 5, 5, 5]) == 0.0

    def test_variance_spread(self):
        """Variance of spread values is positive."""
        criterion = VarianceReductionCriterion()
        assert criterion.impurity([1, 2, 3, 4, 5]) > 0.0

    def test_get_criterion(self):
        """Test criterion registry."""
        assert isinstance(get_criterion('gini'), GiniImpurityCriterion)
        assert isinstance(get_criterion('gain_ratio'), GainRatioCriterion)


class TestCallbacks:
    """Tests for training callbacks."""

    def test_early_stopping(self):
        """Test early stopping triggers."""
        es = EarlyStopping(monitor='val_loss', patience=2)
        
        es.on_epoch_end(0, {'val_loss': 1.0})
        assert not es.stop_training
        
        es.on_epoch_end(1, {'val_loss': 1.1})
        assert not es.stop_training
        
        es.on_epoch_end(2, {'val_loss': 1.2})
        assert es.stop_training

    def test_metrics_logger(self):
        """Test metrics are logged."""
        logger = MetricsLogger()
        logger.on_train_begin()
        logger.on_epoch_end(0, {'loss': 0.5})
        logger.on_epoch_end(1, {'loss': 0.3})
        
        assert logger.history['loss'] == [0.5, 0.3]

    def test_callback_list(self):
        """Test callback list propagation."""
        es = EarlyStopping(patience=1)
        cb_list = CallbackList([es])
        
        cb_list.on_epoch_end(0, {'val_loss': 1.0})
        cb_list.on_epoch_end(1, {'val_loss': 1.1})
        
        assert cb_list.stop_training


class TestBuilders:
    """Tests for tree builders."""

    def test_factory_exact_greedy(self):
        """Test factory creates exact greedy builder."""
        builder = TreeBuilderFactory.create('exact_greedy')
        assert isinstance(builder, ExactGreedyBuilder)

    def test_factory_histogram(self):
        """Test factory creates histogram builder."""
        builder = TreeBuilderFactory.create('histogram', bins=128)
        assert builder.bins == 128

    def test_builder_builds_tree(self):
        """Test builder creates tree structure."""
        builder = ExactGreedyBuilder(criterion='gini', max_depth=2)
        
        X = [('a',), ('a',), ('b',), ('b',)]
        y = ['yes', 'yes', 'no', 'no']
        
        tree = builder.build(X, y, ['feature'])
        assert tree['type'] in ['node', 'leaf']
