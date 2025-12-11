#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║                    ID3 DECISION TREE DEMO                        ║
║                                                                  ║
║  quinlan's classic algorithm (1986) - categorical features      ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from decision_trees.id3 import ID3Classifier
from decision_trees.id3.core.entropy import entropy, information_gain
from decision_trees.id3.data.loader import load_play_tennis, load_mushroom_sample, load_iris_categorical
from decision_trees.id3.utils.validation import accuracy, train_test_split, cross_validate
from decision_trees.id3.utils.visualization import print_tree, export_graphviz
from decision_trees.id3.comparison.sklearn_compare import compare_with_sklearn, print_comparison


# colors for terminal output
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def header(title):
    width = 60
    print(f"\n{C.CYAN}{'═' * width}{C.END}")
    print(f"{C.CYAN}║{C.END} {C.BOLD}{title.center(width-4)}{C.END} {C.CYAN}║{C.END}")
    print(f"{C.CYAN}{'═' * width}{C.END}\n")


def subheader(title):
    print(f"\n{C.YELLOW}▸ {title}{C.END}")
    print(f"{C.DIM}{'─' * 50}{C.END}")


def success(msg):
    print(f"{C.GREEN}✓{C.END} {msg}")


def info(msg):
    print(f"{C.BLUE}ℹ{C.END} {msg}")


def demo_entropy():
    """demonstrate entropy calculation - the heart of id3"""
    header("ENTROPY: THE FOUNDATION")
    
    print(f"{C.DIM}shannon entropy measures uncertainty in a dataset.{C.END}")
    print(f"{C.DIM}higher entropy = more mixed classes = harder to predict{C.END}\n")
    
    subheader("pure set (all same class)")
    y_pure = ["yes"] * 10
    h = entropy(y_pure)
    print(f"  labels: {C.GREEN}[yes×10]{C.END}")
    print(f"  H = {C.BOLD}{h:.4f}{C.END} {C.DIM}← minimum entropy, no uncertainty{C.END}")
    
    subheader("balanced set (50/50 split)")
    y_balanced = ["yes"] * 5 + ["no"] * 5
    h = entropy(y_balanced)
    print(f"  labels: {C.GREEN}[yes×5]{C.END}, {C.RED}[no×5]{C.END}")
    print(f"  H = {C.BOLD}{h:.4f}{C.END} {C.DIM}← maximum entropy, coin flip{C.END}")
    
    subheader("classic example: [9+, 5-]")
    y_classic = ["yes"] * 9 + ["no"] * 5
    h = entropy(y_classic)
    print(f"  labels: {C.GREEN}[yes×9]{C.END}, {C.RED}[no×5]{C.END}")
    print(f"  H = {C.BOLD}{h:.4f}{C.END} {C.DIM}← from quinlan's original paper{C.END}")


def demo_information_gain():
    """demonstrate information gain calculation"""
    header("INFORMATION GAIN: CHOOSING SPLITS")
    
    print(f"{C.DIM}info gain = parent entropy - weighted avg of child entropies{C.END}")
    print(f"{C.DIM}higher gain = more useful feature for classification{C.END}\n")
    
    # simple example
    X = [
        ("sunny", "hot"),
        ("sunny", "mild"),
        ("rain", "hot"),
        ("rain", "mild"),
    ]
    y = ["no", "no", "yes", "yes"]
    
    subheader("example dataset")
    print(f"  {C.BOLD}weather{C.END}   {C.BOLD}temp{C.END}    {C.BOLD}play?{C.END}")
    for i, (sample, label) in enumerate(zip(X, y)):
        color = C.GREEN if label == "yes" else C.RED
        print(f"  {sample[0]:<8} {sample[1]:<7} {color}{label}{C.END}")
    
    subheader("information gain per feature")
    for i, name in enumerate(["weather", "temp"]):
        ig = information_gain(X, y, i)
        bar = "█" * int(ig * 20)
        print(f"  {name:<10} IG = {C.BOLD}{ig:.3f}{C.END}  {C.CYAN}{bar}{C.END}")
    
    print(f"\n  {C.DIM}→ 'weather' perfectly separates the classes!{C.END}")


def demo_play_tennis():
    """classic play tennis example"""
    header("TRAINING: PLAY TENNIS DATASET")
    
    X, y, feature_names = load_play_tennis()
    
    subheader("dataset overview")
    print(f"  samples:  {C.BOLD}{len(X)}{C.END}")
    print(f"  features: {C.CYAN}{', '.join(feature_names)}{C.END}")
    print(f"  classes:  {C.GREEN}yes{C.END}, {C.RED}no{C.END}")
    
    # show sample data
    subheader("sample data (first 5)")
    print(f"  {C.BOLD}{'outlook':<10}{'temp':<8}{'humidity':<10}{'wind':<8}{'play'}{C.END}")
    for i in range(5):
        color = C.GREEN if y[i] == "yes" else C.RED
        print(f"  {X[i][0]:<10}{X[i][1]:<8}{X[i][2]:<10}{X[i][3]:<8}{color}{y[i]}{C.END}")
    
    # train
    subheader("training id3 classifier")
    start = time.time()
    clf = ID3Classifier()
    clf.fit(X, y, feature_names)
    train_time = (time.time() - start) * 1000
    
    success(f"trained in {train_time:.2f}ms")
    info(f"tree depth: {clf.get_depth()}")
    info(f"leaf nodes: {clf.get_n_leaves()}")
    
    # show tree
    subheader("learned decision tree")
    print_tree(clf)
    
    # accuracy
    y_pred = clf.predict(X)
    acc = accuracy(y, y_pred)
    subheader("training accuracy")
    bar = "█" * int(acc * 30)
    print(f"  {C.GREEN}{bar}{C.END} {C.BOLD}{acc:.1%}{C.END}")


def demo_prediction():
    """demonstrate prediction on new samples"""
    header("PREDICTION: NEW SAMPLES")
    
    X, y, feature_names = load_play_tennis()
    clf = ID3Classifier()
    clf.fit(X, y, feature_names)
    
    # test cases
    test_cases = [
        (("sunny", "cool", "high", "strong"), "should we play?"),
        (("overcast", "hot", "high", "weak"), "overcast day"),
        (("rain", "mild", "normal", "weak"), "light rain"),
    ]
    
    subheader("predicting new cases")
    for sample, desc in test_cases:
        pred = clf.predict_one(sample)
        color = C.GREEN if pred == "yes" else C.RED
        print(f"\n  {C.DIM}{desc}{C.END}")
        print(f"  input:  {sample}")
        print(f"  output: {color}{C.BOLD}{pred.upper()}{C.END}")


def demo_mushroom():
    """mushroom classification - safety critical!"""
    header("REAL WORLD: MUSHROOM CLASSIFICATION")
    
    print(f"{C.YELLOW}⚠  edible vs poisonous - lives depend on accuracy!{C.END}\n")
    
    X, y, feature_names = load_mushroom_sample()
    
    subheader("dataset")
    print(f"  samples:  {C.BOLD}{len(X)}{C.END}")
    print(f"  features: {C.CYAN}{', '.join(feature_names)}{C.END}")
    
    edible = sum(1 for l in y if l == "edible")
    poison = len(y) - edible
    print(f"  classes:  {C.GREEN}edible ({edible}){C.END}, {C.RED}poisonous ({poison}){C.END}")
    
    # train
    clf = ID3Classifier()
    clf.fit(X, y, feature_names)
    
    subheader("learned decision tree")
    print_tree(clf)
    
    y_pred = clf.predict(X)
    acc = accuracy(y, y_pred)
    
    subheader("performance")
    if acc == 1.0:
        print(f"  {C.GREEN}★ PERFECT ACCURACY: {acc:.1%}{C.END}")
        print(f"  {C.DIM}  the tree learned a perfect rule!{C.END}")
    else:
        print(f"  accuracy: {acc:.1%}")


def demo_cross_validation():
    """k-fold cross-validation"""
    header("VALIDATION: K-FOLD CROSS-VALIDATION")
    
    print(f"{C.DIM}test generalization by training on k-1 folds, testing on 1{C.END}\n")
    
    X, y, _ = load_iris_categorical()
    
    subheader(f"3-fold cv on iris (n={len(X)})")
    scores = cross_validate(ID3Classifier, X, y, k=3)
    
    for i, score in enumerate(scores):
        bar = "█" * int(score * 20)
        print(f"  fold {i+1}: {C.CYAN}{bar}{C.END} {score:.1%}")
    
    mean = sum(scores) / len(scores)
    print(f"\n  {C.BOLD}mean: {mean:.1%}{C.END}")


def demo_sklearn_comparison():
    """compare with sklearn"""
    header("COMPARISON: VS SKLEARN")
    
    try:
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        print(f"{C.YELLOW}sklearn not installed, skipping{C.END}")
        return
    
    X, y, feature_names = load_play_tennis()
    
    clf = ID3Classifier()
    clf.fit(X, y, feature_names)
    
    results = compare_with_sklearn(clf, X, y)
    if results:
        print_comparison(results)


def demo_export():
    """export to graphviz"""
    header("EXPORT: GRAPHVIZ")
    
    X, y, feature_names = load_play_tennis()
    clf = ID3Classifier()
    clf.fit(X, y, feature_names)
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'tree_id3.dot')
    dot = export_graphviz(clf, output_path)
    
    success(f"exported to: {output_path}")
    
    subheader("dot format preview")
    print(f"{C.DIM}{dot[:300]}...{C.END}")
    
    print(f"\n{C.CYAN}tip:{C.END} visualize with: dot -Tpng tree_id3.dot -o tree.png")


def main():
    print(f"""
{C.CYAN}╔══════════════════════════════════════════════════════════════════╗
║{C.END}                                                                  {C.CYAN}║
║{C.END}  {C.BOLD}ID3 DECISION TREE{C.END}                                            {C.CYAN}║
║{C.END}  {C.DIM}quinlan's classic algorithm for classification{C.END}                {C.CYAN}║
║{C.END}                                                                  {C.CYAN}║
╚══════════════════════════════════════════════════════════════════╝{C.END}
""")
    
    demos = [
        ("entropy", demo_entropy),
        ("information gain", demo_information_gain),
        ("training", demo_play_tennis),
        ("prediction", demo_prediction),
        ("mushroom", demo_mushroom),
        ("cross-validation", demo_cross_validation),
        ("sklearn comparison", demo_sklearn_comparison),
        ("export", demo_export),
    ]
    
    for name, demo in demos:
        demo()
    
    print(f"""
{C.GREEN}{'═' * 60}
 DEMO COMPLETE ✓
{'═' * 60}{C.END}
""")


if __name__ == "__main__":
    main()
