#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║                    C4.5 DECISION TREE DEMO                       ║
║                                                                  ║
║  quinlan's improved algorithm - continuous + categorical        ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_trees.c45 import C45Classifier
from decision_trees.c45.core.gain_ratio import (
    gain_ratio, split_info, best_threshold, entropy, information_gain
)
from decision_trees.c45.data.loader import load_iris, load_wine_sample, load_golf
from decision_trees.c45.utils.visualization import print_tree, export_graphviz
from decision_trees.c45.core.pruning import prune_tree


# colors
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


def header(title):
    width = 60
    print(f"\n{C.MAGENTA}{'═' * width}{C.END}")
    print(f"{C.MAGENTA}║{C.END} {C.BOLD}{title.center(width-4)}{C.END} {C.MAGENTA}║{C.END}")
    print(f"{C.MAGENTA}{'═' * width}{C.END}\n")


def subheader(title):
    print(f"\n{C.YELLOW}▸ {title}{C.END}")
    print(f"{C.DIM}{'─' * 50}{C.END}")


def success(msg):
    print(f"{C.GREEN}✓{C.END} {msg}")


def info(msg):
    print(f"{C.BLUE}ℹ{C.END} {msg}")


def warn(msg):
    print(f"{C.YELLOW}⚠{C.END} {msg}")


def demo_id3_problem():
    """show the problem with information gain"""
    header("THE PROBLEM WITH ID3")
    
    print(f"{C.DIM}id3 uses information gain, which has a bias...{C.END}\n")
    
    # high cardinality example
    X = [("a", "x"), ("b", "x"), ("c", "y"), ("d", "y")]
    y = ["yes", "yes", "no", "no"]
    
    subheader("example: high vs low cardinality")
    print(f"  feature 0: {C.CYAN}4 unique values{C.END} (a, b, c, d)")
    print(f"  feature 1: {C.CYAN}2 unique values{C.END} (x, y)")
    print(f"  {C.DIM}both separate classes equally well{C.END}")
    
    ig_0 = information_gain(X, y, 0)
    ig_1 = information_gain(X, y, 1)
    
    subheader("information gain (id3)")
    bar0 = "█" * int(ig_0 * 20)
    bar1 = "█" * int(ig_1 * 20)
    print(f"  feature 0 (4 vals): {C.CYAN}{bar0}{C.END} {ig_0:.3f}")
    print(f"  feature 1 (2 vals): {C.CYAN}{bar1}{C.END} {ig_1:.3f}")
    
    warn("id3 is biased toward high-cardinality features!")
    print(f"{C.DIM}  extreme case: unique ID column would always be chosen{C.END}")


def demo_gain_ratio():
    """demonstrate gain ratio solution"""
    header("GAIN RATIO: THE C4.5 SOLUTION")
    
    print(f"{C.DIM}gain ratio = info gain / split info{C.END}")
    print(f"{C.DIM}penalizes features with many values{C.END}\n")
    
    X = [("a", "x"), ("b", "x"), ("c", "y"), ("d", "y")]
    y = ["yes", "yes", "no", "no"]
    
    subheader("split info calculation")
    si_0 = split_info(X, 0)
    si_1 = split_info(X, 1)
    print(f"  feature 0: SI = {C.BOLD}{si_0:.3f}{C.END} {C.DIM}(4 values = high){C.END}")
    print(f"  feature 1: SI = {C.BOLD}{si_1:.3f}{C.END} {C.DIM}(2 values = low){C.END}")
    
    subheader("gain ratio comparison")
    gr_0 = gain_ratio(X, y, 0)
    gr_1 = gain_ratio(X, y, 1)
    
    bar0 = "█" * int(gr_0 * 20)
    bar1 = "█" * int(gr_1 * 20)
    print(f"  feature 0: GR = {C.CYAN}{bar0}{C.END} {gr_0:.3f}")
    print(f"  feature 1: GR = {C.MAGENTA}{bar1}{C.END} {gr_1:.3f}")
    
    success("gain ratio correctly prefers the simpler split!")


def demo_continuous():
    """continuous attribute handling"""
    header("CONTINUOUS ATTRIBUTES")
    
    print(f"{C.DIM}c4.5 finds optimal thresholds for numeric features{C.END}\n")
    
    # simple continuous example
    X = [(1.0,), (2.0,), (3.0,), (4.0,), (5.0,), (6.0,)]
    y = ["no", "no", "no", "yes", "yes", "yes"]
    
    subheader("example: single continuous feature")
    print(f"  {C.BOLD}value   class{C.END}")
    for (val,), label in zip(X, y):
        color = C.GREEN if label == "yes" else C.RED
        print(f"   {val:.1f}    {color}{label}{C.END}")
    
    subheader("finding optimal threshold")
    t, gr = best_threshold(X, y, 0)
    print(f"  best threshold: {C.BOLD}{t:.1f}{C.END}")
    print(f"  gain ratio: {gr:.3f}")
    print(f"\n  {C.DIM}→ split at {t} perfectly separates classes{C.END}")


def demo_iris():
    """iris dataset - classic ML benchmark"""
    header("IRIS DATASET: ALL CONTINUOUS")
    
    X, y, feature_names = load_iris()
    
    subheader("dataset overview")
    print(f"  samples:  {C.BOLD}{len(X)}{C.END}")
    print(f"  features: {C.CYAN}{', '.join(feature_names)}{C.END}")
    print(f"  classes:  {C.GREEN}setosa{C.END}, {C.YELLOW}versicolor{C.END}, {C.MAGENTA}virginica{C.END}")
    
    subheader("sample data (first 3 per class)")
    print(f"  {C.BOLD}{'sepal_l':<8}{'sepal_w':<8}{'petal_l':<8}{'petal_w':<8}{'class'}{C.END}")
    
    # show diverse samples
    classes = {}
    for x, label in zip(X, y):
        if label not in classes:
            classes[label] = []
        if len(classes[label]) < 2:
            classes[label].append(x)
    
    colors = {"setosa": C.GREEN, "versicolor": C.YELLOW, "virginica": C.MAGENTA}
    for label, samples in classes.items():
        for x in samples:
            c = colors[label]
            print(f"  {x[0]:<8.1f}{x[1]:<8.1f}{x[2]:<8.1f}{x[3]:<8.1f}{c}{label}{C.END}")
    
    # train
    subheader("training c4.5 classifier")
    start = time.time()
    clf = C45Classifier(max_depth=3)
    clf.fit(X, y, feature_names)
    train_time = (time.time() - start) * 1000
    
    success(f"trained in {train_time:.2f}ms")
    info(f"detected types: {clf.feature_types_}")
    info(f"tree depth: {clf.get_depth()}")
    
    subheader("learned decision tree (with thresholds)")
    print_tree(clf)
    
    y_pred = clf.predict(X)
    acc = sum(1 for t, p in zip(y, y_pred) if t == p) / len(y)
    
    subheader("performance")
    bar = "█" * int(acc * 30)
    print(f"  {C.GREEN}{bar}{C.END} {C.BOLD}{acc:.1%}{C.END}")


def demo_mixed():
    """mixed categorical + continuous"""
    header("MIXED TYPES: GOLF DATASET")
    
    print(f"{C.DIM}c4.5 auto-detects and handles both types{C.END}\n")
    
    X, y, feature_names = load_golf()
    
    subheader("dataset")
    print(f"  features: {C.CYAN}{', '.join(feature_names)}{C.END}")
    print(f"  {C.DIM}outlook: categorical{C.END}")
    print(f"  {C.DIM}temperature, humidity: continuous{C.END}")
    print(f"  {C.DIM}windy: categorical{C.END}")
    
    clf = C45Classifier()
    clf.fit(X, y, feature_names)
    
    subheader("auto-detected types")
    for name, ftype in zip(feature_names, clf.feature_types_):
        icon = "📊" if ftype == "continuous" else "📋"
        print(f"  {icon} {name}: {C.CYAN}{ftype}{C.END}")
    
    subheader("learned tree (mixed splits)")
    print_tree(clf)


def demo_pruning():
    """demonstrate reduced error pruning"""
    header("PRUNING: AVOIDING OVERFITTING")
    
    print(f"{C.DIM}prune leaves that don't improve validation accuracy{C.END}\n")
    
    X, y, feature_names = load_iris()
    
    # split data
    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    subheader(f"data split: {len(X_train)} train, {len(X_val)} validation")
    
    # train without pruning
    clf = C45Classifier()
    clf.fit(X_train, y_train, feature_names)
    
    subheader("before pruning")
    print(f"  tree depth:  {C.BOLD}{clf.get_depth()}{C.END}")
    print(f"  leaf nodes:  {C.BOLD}{clf.get_n_leaves()}{C.END}")
    val_acc = sum(1 for t, p in zip(y_val, clf.predict(X_val)) if t == p) / len(y_val)
    print(f"  val accuracy: {C.BOLD}{val_acc:.1%}{C.END}")
    
    # prune
    prune_tree(clf, X_val, y_val)
    
    subheader("after pruning")
    print(f"  tree depth:  {C.BOLD}{clf.get_depth()}{C.END}")
    print(f"  leaf nodes:  {C.BOLD}{clf.get_n_leaves()}{C.END}")
    val_acc_pruned = sum(1 for t, p in zip(y_val, clf.predict(X_val)) if t == p) / len(y_val)
    print(f"  val accuracy: {C.BOLD}{val_acc_pruned:.1%}{C.END}")
    
    if clf.get_n_leaves() == 1:
        info("tree pruned to single node (small validation set)")


def demo_compare():
    """id3 vs c4.5 comparison"""
    header("HEAD TO HEAD: ID3 vs C4.5")
    
    try:
        from decision_trees.id3 import ID3Classifier
        from decision_trees.id3.data.loader import load_play_tennis
    except ImportError:
        warn("id3 package not found")
        return
    
    X, y, feature_names = load_play_tennis()
    
    subheader("play tennis dataset (categorical)")
    
    # train both
    id3 = ID3Classifier()
    id3.fit(X, y, feature_names)
    
    c45 = C45Classifier()
    c45.fit(X, y, feature_names)
    
    # compare
    id3_acc = sum(1 for t, p in zip(y, id3.predict(X)) if t == p) / len(y)
    c45_acc = sum(1 for t, p in zip(y, c45.predict(X)) if t == p) / len(y)
    
    print(f"\n  {C.BOLD}{'metric':<20}{'ID3':>12}{'C4.5':>12}{C.END}")
    print(f"  {'─' * 44}")
    print(f"  {'accuracy':<20}{C.CYAN}{id3_acc:>12.1%}{C.END}{C.MAGENTA}{c45_acc:>12.1%}{C.END}")
    print(f"  {'tree depth':<20}{C.CYAN}{id3.get_depth():>12}{C.END}{C.MAGENTA}{c45.get_depth():>12}{C.END}")
    print(f"  {'leaf nodes':<20}{C.CYAN}{id3.get_n_leaves():>12}{C.END}{C.MAGENTA}{c45.get_n_leaves():>12}{C.END}")
    
    subheader("key differences")
    print(f"  {C.CYAN}ID3{C.END}")
    print(f"    • uses information gain")
    print(f"    • categorical features only")
    print(f"    • no pruning")
    print(f"\n  {C.MAGENTA}C4.5{C.END}")
    print(f"    • uses gain ratio (less bias)")
    print(f"    • handles continuous features")
    print(f"    • supports pruning")
    print(f"    • handles missing values")


def demo_export():
    """export to graphviz"""
    header("EXPORT: VISUALIZATION")
    
    X, y, feature_names = load_iris()
    clf = C45Classifier(max_depth=3)
    clf.fit(X, y, feature_names)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'tree_c45.dot')
    dot = export_graphviz(clf, output_path)
    
    success(f"exported to: {output_path}")
    
    subheader("graphviz preview")
    print(f"{C.DIM}{dot[:300]}...{C.END}")
    
    print(f"\n{C.MAGENTA}tip:{C.END} visualize with: dot -Tpng tree_c45.dot -o tree.png")


def main():
    print(f"""
{C.MAGENTA}╔══════════════════════════════════════════════════════════════════╗
║{C.END}                                                                  {C.MAGENTA}║
║{C.END}  {C.BOLD}C4.5 DECISION TREE{C.END}                                           {C.MAGENTA}║
║{C.END}  {C.DIM}quinlan's improved algorithm (1993){C.END}                           {C.MAGENTA}║
║{C.END}                                                                  {C.MAGENTA}║
║{C.END}  {C.GREEN}✓ gain ratio{C.END}  {C.GREEN}✓ continuous{C.END}  {C.GREEN}✓ pruning{C.END}  {C.GREEN}✓ missing values{C.END}   {C.MAGENTA}║
║{C.END}                                                                  {C.MAGENTA}║
╚══════════════════════════════════════════════════════════════════╝{C.END}
""")
    
    demos = [
        ("id3 problem", demo_id3_problem),
        ("gain ratio", demo_gain_ratio),
        ("continuous", demo_continuous),
        ("iris", demo_iris),
        ("mixed types", demo_mixed),
        ("pruning", demo_pruning),
        ("comparison", demo_compare),
        ("export", demo_export),
    ]
    
    for name, demo in demos:
        demo()
    
    print(f"""
{C.MAGENTA}{'═' * 60}
 DEMO COMPLETE ✓
{'═' * 60}{C.END}
""")


if __name__ == "__main__":
    main()
