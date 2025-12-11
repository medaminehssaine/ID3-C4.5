# 🌳 ID3 & C4.5 Decision Trees

> Professional implementations of Quinlan's classic decision tree algorithms for the KDD course.

<div align="center">

| Algorithm | Year | Criterion | Continuous | Pruning |
|:---------:|:----:|:---------:|:----------:|:-------:|
| **ID3** | 1986 | Info Gain | ✗ | ✗ |
| **C4.5** | 1993 | Gain Ratio | ✓ | ✓ |

</div>

---

## 📁 Project Structure

```
.
├── src/decision_trees/      # unified package
│   ├── id3/                 # ID3 algorithm
│   │   ├── core/            # entropy, node, tree
│   │   ├── data/            # sample datasets
│   │   ├── utils/           # validation, visualization
│   │   └── comparison/      # sklearn benchmarks
│   └── c45/                 # C4.5 algorithm
│       ├── core/            # gain_ratio, pruning
│       ├── data/            # continuous datasets
│       └── utils/           # visualization
├── tests/                   # unit tests (26 total)
├── examples/                # demo scripts
├── outputs/                 # generated .dot files
└── pyproject.toml           # pip installable
```

---

## 🚀 Quick Start

```python
# === ID3: categorical features ===
from decision_trees.id3 import ID3Classifier
from decision_trees.id3.data import load_play_tennis

X, y, names = load_play_tennis()
clf = ID3Classifier()
clf.fit(X, y, names)
print(clf.predict_one(("sunny", "cool", "normal", "weak")))
# → "yes"


# === C4.5: continuous + categorical ===
from decision_trees.c45 import C45Classifier
from decision_trees.c45.data import load_iris

X, y, names = load_iris()
clf = C45Classifier()
clf.fit(X, y, names)
# auto-detects continuous features and finds thresholds
```

---

## 🧪 Testing

```bash
python tests/test_id3.py   # 15 tests ✓
python tests/test_c45.py   # 11 tests ✓
```

---

## 🎬 Demos

Beautiful terminal demos with colors, progress bars, and educational explanations.

```bash
python examples/demo_id3.py   # entropy, training, prediction, CV
python examples/demo_c45.py   # gain ratio, thresholds, pruning
```

**Demo highlights:**
- 📊 Visual entropy/gain calculations
- 🌳 Tree visualization in console
- 📈 Accuracy bars and metrics
- 🔄 ID3 vs C4.5 comparison

---

## 📚 Algorithm Details

### ID3 (Iterative Dichotomiser 3)

```
function ID3(D, features):
    if all samples same class → return leaf
    if no features left → return majority class
    
    best = argmax(features, key=InformationGain)
    node = new Node(best)
    
    for each value v of best:
        subset = samples where feature[best] = v
        node.children[v] = ID3(subset, features - {best})
    
    return node
```

**Key formula:**
```
H(S) = -Σ p(c) × log₂(p(c))        # entropy
IG(S, A) = H(S) - Σ (|Sᵥ|/|S|) × H(Sᵥ)   # info gain
```

---

### C4.5 Improvements

| Feature | How it works |
|---------|--------------|
| **Gain Ratio** | `GR = IG / SplitInfo` — penalizes high-cardinality |
| **Continuous** | Binary splits at optimal thresholds |
| **Missing** | Distributes samples proportionally |
| **Pruning** | Reduced error pruning on validation set |

---

## 🔧 Installation

```bash
# run directly (no install needed)
python examples/demo_id3.py

# or install as package
pip install -e .

# then import anywhere
from decision_trees import ID3Classifier, C45Classifier
```

---

## 📊 Sample Output

**ID3 Tree (Play Tennis):**
```
[outlook?]
├── sunny [humidity?]
│   ├── high → [no]
│   └── normal → [yes]
├── overcast → [yes]
└── rain [wind?]
    ├── weak → [yes]
    └── strong → [no]
```

**C4.5 Tree (Iris):**
```
[petal_length <= 2.50?]
    yes: → [setosa]
    no:  [petal_width <= 1.65?]
        yes: → [versicolor]
        no:  → [virginica]
```

---

## 👥 Team

KDD Course Project — Decision Tree Algorithms

---

<div align="center">
<sub>Built with ❤️ for learning machine learning fundamentals</sub>
</div>
