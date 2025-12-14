# 🌳 ID3 & C4.5 Decision Trees

> Professional implementations of Quinlan's classic decision tree algorithms for the KDD course.
> Built from scratch with strict adherence to the original research papers.

<div align="center">

| Algorithm | Year | Criterion | Continuous | Missing Values | Pruning |
|:---------:|:----:|:---------:|:----------:|:--------------:|:-------:|
| **ID3** | 1986 | Info Gain | ✗ | ✗ | ✗ |
| **C4.5** | 1993 | Gain Ratio | ✓ | ✓ | ✓ |

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
├── tests/                   # unit tests (40+ tests)
├── examples/                # demo scripts
├── outputs/                 # generated .dot files
├── REPORT_STRUCTURE.md      # theory-to-code mapping
└── pyproject.toml           # pip installable
```

---

## 🔧 Installation

```bash
# Clone and navigate
cd "ID3 & C4.5"

# Install as editable package
pip install -e .

# Optional: install test and comparison dependencies
pip install -e ".[dev,compare]"
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
print(clf.feature_types_)  # ['continuous', 'continuous', ...]
# Auto-detects continuous features and finds thresholds
```

---

## 📐 Mathematical Foundations

### Entropy (Shannon Entropy)

The foundation of both ID3 and C4.5. Measures uncertainty in a dataset.

```
H(S) = -Σᵢ p(cᵢ) × log₂(p(cᵢ))
```

**Properties:**
- `H(S) = 0` for pure sets (all same class)
- `H(S) = 1` for balanced binary (50/50 split)
- `H(S) = log₂(k)` for k equally distributed classes

---

### Information Gain (ID3 Criterion)

Measures reduction in entropy after splitting.

```
IG(S, A) = H(S) - Σᵥ (|Sᵥ|/|S|) × H(Sᵥ)
```

**Problem:** Biased toward high-cardinality features!

*Example:* A unique ID column always has maximum IG but provides no generalization.

---

### Gain Ratio (C4.5 Solution)

Normalizes Information Gain to reduce bias.

```
GR(S, A) = IG(S, A) / SI(S, A)

SI(S, A) = -Σᵥ (|Sᵥ|/|S|) × log₂(|Sᵥ|/|S|)
```

**Why it works:**
- Split Information (SI) is high for features with many values
- Dividing IG by SI penalizes high-cardinality features
- `GR ≤ IG` always holds (SI ≥ 1 for 2+ partitions)

**Mathematical Proof:**
```
SI(S,A) = -Σᵥ pᵥ × log₂(pᵥ)  where pᵥ = |Sᵥ|/|S|

For n equally sized partitions: SI = log₂(n) ≥ 1 when n ≥ 2
Therefore: GR = IG/SI ≤ IG
```

---

### Continuous Attribute Handling

C4.5 finds optimal thresholds for numeric features using binary splits.

**Algorithm:**
1. Sort values
2. Find midpoints where class changes
3. Evaluate GR for each candidate threshold
4. Choose threshold with maximum GR

**Key difference from ID3:** Continuous features can be reused in subtrees!

---

### Pessimistic Error Pruning

C4.5's default pruning method (no validation set required).

```
Pessimistic Error = (errors + 0.5) / N
```

Uses Wilson score interval for tighter bounds:
```
UCB = (f + z²/2n + z×√(f(1-f)/n + z²/4n²)) / (1 + z²/n)
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Or run individually
python tests/test_id3.py   # 21 tests
python tests/test_c45.py   # 23 tests
```

### Key Test Cases

| Test | Formula Verified |
|------|------------------|
| `test_entropy_classic` | H([9+, 5-]) ≈ 0.9403 |
| `test_gain_ratio_less_than_ig` | GR ≤ IG always |
| `test_best_threshold` | Optimal threshold at class boundary |

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

## 📚 References

1. **Quinlan, J.R. (1986).** *"Induction of Decision Trees"*, Machine Learning 1:81-106
   - Original ID3 algorithm

2. **Quinlan, J.R. (1993).** *"C4.5: Programs for Machine Learning"*, Morgan Kaufmann
   - Gain Ratio, continuous attributes, pruning

3. **Shannon, C.E. (1948).** *"A Mathematical Theory of Communication"*
   - Foundation of entropy concept

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

KDD Course Project — ID3 & C4.5 Decision Trees

| Name | Role |
|------|------|
| Mohammed Amine Hssaine | Implementation |
| Ouissam Benalla | Implementation |
| Mohamed Taha El Younsi | Implementation |

---

<div align="center">
<sub>Built with ❤️ for learning machine learning fundamentals</sub>
</div>
