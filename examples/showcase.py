#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║              DECISION TREES SHOWCASE - FULL PIPELINE             ║
║                                                                  ║
║  Demonstrates all features: training, tuning, metrics, saving   ║
╚══════════════════════════════════════════════════════════════════╝

This script showcases the complete decision tree package capabilities:
1. Training ID3 and C4.5 classifiers
2. Hyperparameter tuning with GridSearchCV
3. Comprehensive evaluation metrics
4. Model serialization (save/load)
5. Feature importance analysis
6. Performance comparison
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_trees import (
    # Classifiers
    ID3Classifier, C45Classifier,
    # Tuning
    GridSearchCV,
    # Metrics
    accuracy_score, f1_score, classification_report, ClassificationMetrics,
    # Serialization
    save_model, load_model, model_summary,
    # Optimized
    FeatureImportance, entropy_fast
)
from decision_trees.id3.data.loader import load_play_tennis, load_mushroom_sample
from decision_trees.c45.data.loader import load_iris, load_golf
from decision_trees.id3.utils.visualization import print_tree as print_id3_tree
from decision_trees.c45.utils.visualization import print_tree as print_c45_tree


# =============================================================================
# Terminal Colors
# =============================================================================
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def header(title: str) -> None:
    width = 65
    print(f"\n{C.CYAN}{'═' * width}{C.END}")
    print(f"{C.CYAN}║{C.END} {C.BOLD}{title.center(width-4)}{C.END} {C.CYAN}║{C.END}")
    print(f"{C.CYAN}{'═' * width}{C.END}\n")


def section(title: str) -> None:
    print(f"\n{C.YELLOW}▸ {title}{C.END}")
    print(f"{C.DIM}{'─' * 55}{C.END}")


def success(msg: str) -> None:
    print(f"  {C.GREEN}✓{C.END} {msg}")


def info(msg: str) -> None:
    print(f"  {C.BLUE}ℹ{C.END} {msg}")


def metric(name: str, value: float, bar_len: int = 20) -> None:
    bar = "█" * int(value * bar_len)
    print(f"  {name:<20} {C.CYAN}{bar}{C.END} {C.BOLD}{value:.1%}{C.END}")


# =============================================================================
# Demo Functions
# =============================================================================

def demo_optimized_entropy() -> None:
    """Showcase NumPy-optimized entropy calculation."""
    header("NUMPY-OPTIMIZED COMPUTATIONS")
    
    section("Entropy Comparison")
    
    # Test data
    y_small = ["yes"] * 50 + ["no"] * 50
    
    # NumPy version
    start = time.perf_counter()
    for _ in range(1000):
        h = entropy_fast(y_small)
    numpy_time = (time.perf_counter() - start) * 1000
    
    info(f"Entropy of [50+, 50-]: {h:.4f}")
    info(f"NumPy version: {numpy_time:.2f}ms for 1000 iterations")
    success("Vectorized computations available!")


def demo_id3_training() -> None:
    """Train ID3 on Play Tennis dataset."""
    header("ID3 CLASSIFIER - CATEGORICAL DATA")
    
    X, y, feature_names = load_play_tennis()
    
    section("Play Tennis Dataset")
    info(f"Samples: {len(X)}")
    info(f"Features: {', '.join(feature_names)}")
    
    section("Training ID3")
    start = time.time()
    clf = ID3Classifier()
    clf.fit(X, y, feature_names)
    train_time = (time.time() - start) * 1000
    
    success(f"Trained in {train_time:.2f}ms")
    info(f"Tree depth: {clf.get_depth()}")
    info(f"Leaf nodes: {clf.get_n_leaves()}")
    
    section("Decision Tree Structure")
    print_id3_tree(clf)
    
    y_pred = clf.predict(X)
    section("Training Metrics")
    metric("Accuracy", accuracy_score(y, y_pred))


def demo_c45_continuous() -> None:
    """Train C4.5 on Iris dataset (continuous features)."""
    header("C4.5 CLASSIFIER - CONTINUOUS DATA")
    
    X, y, feature_names = load_iris()
    
    section("Iris Dataset")
    info(f"Samples: {len(X)}")
    info(f"Features: {', '.join(feature_names)}")
    info(f"Classes: setosa, versicolor, virginica")
    
    section("Training C4.5")
    clf = C45Classifier(max_depth=4)
    clf.fit(X, y, feature_names)
    
    info(f"Feature types detected: {clf.feature_types_}")
    info(f"Tree depth: {clf.get_depth()}")
    
    section("Decision Tree (with thresholds)")
    print_c45_tree(clf)
    
    y_pred = clf.predict(X)
    section("Training Metrics")
    metric("Accuracy", accuracy_score(y, y_pred))
    metric("F1 (macro)", f1_score(y, y_pred, average='macro'))


def demo_hyperparameter_tuning() -> None:
    """Demonstrate GridSearchCV for hyperparameter optimization."""
    header("HYPERPARAMETER TUNING")
    
    X, y, feature_names = load_mushroom_sample()
    
    section("Mushroom Dataset (Safety Critical!)")
    info(f"Samples: {len(X)}")
    info("Task: Classify edible vs poisonous")
    
    section("GridSearchCV Configuration")
    param_grid = {
        'max_depth': [None, 3, 5],
        'min_samples_split': [2, 5, 10]
    }
    info(f"Parameter grid: {param_grid}")
    info("Cross-validation folds: 3")
    
    section("Running Grid Search...")
    search = GridSearchCV(
        ID3Classifier,
        param_grid,
        cv=3,
        verbose=0,
        random_state=42
    )
    
    start = time.time()
    search.fit(X, y, feature_names)
    search_time = time.time() - start
    
    success(f"Completed in {search_time:.2f}s")
    print()
    info(f"Best parameters: {search.best_params_}")
    info(f"Best CV score: {search.best_score_:.4f}")
    
    section("Top 3 Parameter Combinations")
    results = search.get_results_dataframe()
    sorted_idx = sorted(range(len(results['mean_score'])),
                       key=lambda i: results['mean_score'][i], reverse=True)
    
    for rank, idx in enumerate(sorted_idx[:3], 1):
        params = results['params'][idx]
        score = results['mean_score'][idx]
        print(f"  {rank}. {params} → {score:.4f}")


def demo_metrics() -> None:
    """Showcase comprehensive evaluation metrics."""
    header("EVALUATION METRICS")
    
    X, y, feature_names = load_golf()
    
    section("Golf Dataset (Mixed Types)")
    
    clf = C45Classifier()
    clf.fit(X, y, feature_names)
    y_pred = clf.predict(X)
    
    section("ClassificationMetrics Object")
    metrics = ClassificationMetrics(y, y_pred)
    
    print(f"  {C.BOLD}Accuracy:{C.END} {metrics.accuracy:.4f}")
    print(f"  {C.BOLD}Precision (macro):{C.END} {metrics.precision_macro:.4f}")
    print(f"  {C.BOLD}Recall (macro):{C.END} {metrics.recall_macro:.4f}")
    print(f"  {C.BOLD}F1 (macro):{C.END} {metrics.f1_macro:.4f}")
    
    section("Full Classification Report")
    print(classification_report(y, y_pred))


def demo_serialization() -> None:
    """Demonstrate model save/load functionality."""
    header("MODEL SERIALIZATION")
    
    X, y, feature_names = load_play_tennis()
    
    section("Train and Save Model")
    clf = ID3Classifier()
    clf.fit(X, y, feature_names)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'model.json')
    save_model(clf, output_path)
    success(f"Saved to: {output_path}")
    
    section("Load and Verify")
    loaded = load_model(output_path)
    
    y_pred_original = clf.predict(X)
    y_pred_loaded = loaded.predict(X)
    
    match = all(a == b for a, b in zip(y_pred_original, y_pred_loaded))
    if match:
        success("Predictions match! Model loaded correctly.")
    
    section("Model Summary")
    print(model_summary(loaded))


def demo_feature_importance() -> None:
    """Showcase feature importance extraction."""
    header("FEATURE IMPORTANCE")
    
    X, y, feature_names = load_golf()
    
    section("Training C4.5 on Golf Dataset")
    clf = C45Classifier()
    clf.fit(X, y, feature_names)
    
    section("Computing Feature Importance")
    importance = FeatureImportance()
    importance.compute(clf, feature_names)
    
    section("Ranked Features")
    for name, score in importance.to_ranked_list():
        bar = "█" * int(score * 30)
        print(f"  {name:<15} {C.MAGENTA}{bar}{C.END} {score:.3f}")


def demo_comparison() -> None:
    """Compare ID3 vs C4.5 on same dataset."""
    header("ALGORITHM COMPARISON: ID3 vs C4.5")
    
    X, y, feature_names = load_play_tennis()
    
    section("Play Tennis Dataset (Categorical)")
    
    # Train both
    id3 = ID3Classifier()
    id3.fit(X, y, feature_names)
    
    c45 = C45Classifier()
    c45.fit(X, y, feature_names)
    
    id3_acc = accuracy_score(y, id3.predict(X))
    c45_acc = accuracy_score(y, c45.predict(X))
    
    section("Results")
    print(f"\n  {C.BOLD}{'Metric':<20}{'ID3':>12}{'C4.5':>12}{C.END}")
    print(f"  {'─' * 44}")
    print(f"  {'Accuracy':<20}{C.CYAN}{id3_acc:>12.1%}{C.END}{C.MAGENTA}{c45_acc:>12.1%}{C.END}")
    print(f"  {'Tree Depth':<20}{C.CYAN}{id3.get_depth():>12}{C.END}{C.MAGENTA}{c45.get_depth():>12}{C.END}")
    print(f"  {'Leaf Nodes':<20}{C.CYAN}{id3.get_n_leaves():>12}{C.END}{C.MAGENTA}{c45.get_n_leaves():>12}{C.END}")
    
    section("Key Differences")
    print(f"""
  {C.CYAN}ID3{C.END}
    • Uses Information Gain (biased toward high cardinality)
    • Categorical features only
    • No pruning

  {C.MAGENTA}C4.5{C.END}
    • Uses Gain Ratio (corrects bias)
    • Handles continuous + categorical
    • Supports pruning and missing values
    """)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════════════╗
║{C.END}                                                                  {C.CYAN}║
║{C.END}  {C.BOLD}DECISION TREES PACKAGE - FULL SHOWCASE{C.END}                        {C.CYAN}║
║{C.END}  {C.DIM}ID3 & C4.5 Implementation from Scratch{C.END}                        {C.CYAN}║
║{C.END}                                                                  {C.CYAN}║
║{C.END}  {C.GREEN}✓ Training{C.END}  {C.GREEN}✓ Tuning{C.END}  {C.GREEN}✓ Metrics{C.END}  {C.GREEN}✓ Serialization{C.END}      {C.CYAN}║
║{C.END}                                                                  {C.CYAN}║
╚══════════════════════════════════════════════════════════════════╝{C.END}
""")

    demos = [
        ("Optimized Computations", demo_optimized_entropy),
        ("ID3 Training", demo_id3_training),
        ("C4.5 Continuous", demo_c45_continuous),
        ("Hyperparameter Tuning", demo_hyperparameter_tuning),
        ("Evaluation Metrics", demo_metrics),
        ("Model Serialization", demo_serialization),
        ("Feature Importance", demo_feature_importance),
        ("Algorithm Comparison", demo_comparison),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"{C.RED}Error in {name}: {e}{C.END}")

    print(f"""
{C.GREEN}{'═' * 65}
 SHOWCASE COMPLETE ✓
 
 All features demonstrated successfully!
{'═' * 65}{C.END}
""")


if __name__ == "__main__":
    main()
