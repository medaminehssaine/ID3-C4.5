# 🌳 Decision Trees from Scratch

<div align="center">

**A professional, research-grade implementation of ID3 and C4.5 Decision Tree algorithms**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 44/44](https://img.shields.io/badge/tests-44%2F44-brightgreen.svg)]()

*Built from scratch following Quinlan's original research papers*

</div>

---

## 📚 Overview

This package provides **production-ready** implementations of two foundational decision tree algorithms:

| Algorithm | Year | Key Feature | Use Case |
|-----------|------|-------------|----------|
| **ID3** | 1986 | Information Gain | Categorical features |
| **C4.5** | 1993 | Gain Ratio + Continuous | Mixed feature types |

### Why This Package?

- ✅ **No sklearn dependency** - Pure Python + NumPy implementation
- ✅ **Academic accuracy** - Follows Quinlan's papers exactly
- ✅ **Production features** - Serialization, tuning, metrics
- ✅ **Fully documented** - Mathematical formulas in docstrings
- ✅ **Type-hinted** - Full typing for IDE support

---

## 🚀 Quick Start

```python
from decision_trees import ID3Classifier, C45Classifier

# Categorical data → ID3
X = [('sunny', 'hot'), ('rain', 'cool'), ('overcast', 'mild')]
y = ['no', 'yes', 'yes']

clf = ID3Classifier()
clf.fit(X, y, feature_names=['weather', 'temp'])
print(clf.predict([('sunny', 'cool')]))  # ['yes']

# Mixed/continuous data → C4.5
X = [(25.0, 'sunny'), (18.0, 'rain'), (22.0, 'overcast')]
y = ['no', 'yes', 'yes']

clf = C45Classifier()
clf.fit(X, y, feature_names=['temperature', 'weather'])
print(clf.predict([(20.0, 'rain')]))  # ['yes']
```

---

## 📖 Mathematical Foundation

### Entropy (Information Content)

The entropy $H(S)$ measures uncertainty in a dataset:

$$H(S) = -\sum_{c \in C} p(c) \cdot \log_2 p(c)$$

where $p(c)$ is the proportion of class $c$ in set $S$.

| Scenario | Entropy | Interpretation |
|----------|---------|----------------|
| Pure (all same class) | 0.0 | No uncertainty |
| Balanced binary | 1.0 | Maximum uncertainty |
| [9+, 5-] (Quinlan's example) | 0.940 | High uncertainty |

### Information Gain (ID3)

ID3 selects the feature with highest Information Gain:

$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot H(S_v)$$

**Limitation**: Biased toward high-cardinality features (e.g., unique IDs).

### Gain Ratio (C4.5)

C4.5 normalizes IG by Split Information to fix the bias:

$$GR(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

where:

$$SplitInfo(S, A) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \cdot \log_2 \frac{|S_v|}{|S|}$$

---

## 🛠️ Features

### Core Classifiers

```python
from decision_trees import ID3Classifier, C45Classifier

# ID3 - Categorical only
id3 = ID3Classifier(
    max_depth=5,           # Limit tree depth
    min_samples_split=2    # Min samples to split
)

# C4.5 - Handles everything
c45 = C45Classifier(
    max_depth=None,        # Unlimited depth
    min_samples_split=2,
    min_gain_ratio=0.01    # Prevents trivial splits
)
```

### Hyperparameter Tuning

```python
from decision_trees import GridSearchCV, ID3Classifier

param_grid = {
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10, 20]
}

search = GridSearchCV(ID3Classifier, param_grid, cv=5, verbose=1)
search.fit(X, y, feature_names)

print(f"Best params: {search.best_params_}")
print(f"Best score: {search.best_score_:.4f}")

# Use best model
predictions = search.predict(X_test)
```

### Evaluation Metrics

```python
from decision_trees import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print()
print(classification_report(y_test, y_pred))
```

### Model Serialization

```python
from decision_trees import save_model, load_model, model_summary

# Save trained model
save_model(clf, 'my_tree.json')

# Load for inference
loaded = load_model('my_tree.json')
predictions = loaded.predict(X_new)

# Model summary
print(model_summary(clf))
```

### Feature Importance

```python
from decision_trees import FeatureImportance

importance = FeatureImportance()
importance.compute(clf, feature_names)

# Ranked list
for name, score in importance.to_ranked_list():
    print(f"{name}: {score:.3f}")
```

### Tree Visualization

```python
from decision_trees.id3.utils.visualization import print_tree, export_graphviz

# Console output
print_tree(clf)

# Graphviz DOT
export_graphviz(clf, 'tree.dot')
# Then: dot -Tpng tree.dot -o tree.png
```

---

## 📁 Project Structure

```
src/decision_trees/
├── __init__.py          # Package exports
├── base.py              # DecisionTreeBase abstract class
├── optimized.py         # NumPy-vectorized computations
├── metrics.py           # Evaluation metrics
├── tuning.py            # GridSearchCV, RandomizedSearchCV
├── serialization.py     # Save/load models
│
├── id3/                 # ID3 Implementation
│   ├── core/
│   │   ├── tree.py      # ID3Classifier
│   │   ├── node.py      # Node structure
│   │   └── entropy.py   # Entropy calculations
│   ├── data/            # Sample datasets
│   └── utils/           # Visualization
│
└── c45/                 # C4.5 Implementation
    ├── core/
    │   ├── tree.py      # C45Classifier
    │   ├── node.py      # Node (+ threshold)
    │   ├── gain_ratio.py # Gain Ratio, thresholds
    │   └── pruning.py   # REP, PEP pruning
    ├── data/
    └── utils/
```

---

## 📊 Algorithm Comparison

| Feature | ID3 | C4.5 |
|---------|-----|------|
| Splitting criterion | Information Gain | Gain Ratio |
| Continuous features | ❌ | ✅ Binary threshold |
| Missing values | ❌ | ✅ Proportional distribution |
| Feature reuse | ❌ Once per path | ✅ Continuous can repeat |
| Pruning | ❌ | ✅ REP, Pessimistic |
| Bias | High cardinality | Corrected |

---

## 🧪 Testing

```bash
# Run all tests
python tests/test_id3.py   # 21 tests
python tests/test_c45.py   # 23 tests

# Run demos
python examples/demo_id3.py
python examples/demo_c45.py
```

---

## 📚 References

1. **Quinlan, J.R. (1986)**. "Induction of Decision Trees", *Machine Learning* 1:81-106
2. **Quinlan, J.R. (1993)**. "C4.5: Programs for Machine Learning", Morgan Kaufmann
3. **Shannon, C.E. (1948)**. "A Mathematical Theory of Communication"
4. **Breiman, L. et al. (1984)**. "Classification and Regression Trees"
---

<div align="center">
<i>Built by Hssaine, Benalla and El Younsi</i>
</div>
