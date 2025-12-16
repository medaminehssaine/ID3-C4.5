# 🌳 Decision Trees from Scratch

<div align="center">

**A professional, research-grade implementation of ID3 and C4.5 Decision Tree algorithms**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 44/44](https://img.shields.io/badge/tests-44%2F44-brightgreen.svg)]()

*Built from scratch following Quinlan's original research papers*

</div>

---

## � Installation

Since this is a local package, you can install it in editable mode:

```bash
pip install -e .
```

Or simply ensure the `src` directory is in your `PYTHONPATH`.

---

## 🚀 Quick Start

Here is how to import and use the classifiers in 3 lines of code:

### 1. ID3 (Categorical Data)

```python
from decision_trees import ID3Classifier

# 1. Prepare Data
X = [('sunny', 'hot'), ('rain', 'cool'), ('overcast', 'mild')]
y = ['no', 'yes', 'yes']

# 2. Train
clf = ID3Classifier()
clf.fit(X, y, feature_names=['weather', 'temp'])

# 3. Predict
print(clf.predict([('sunny', 'cool')]))  # Output: ['yes']
```

### 2. C4.5 (Mixed Data + Pruning)

```python
from decision_trees import C45Classifier

# 1. Prepare Data (can mix numbers and strings)
X = [(25.0, 'sunny'), (18.0, 'rain'), (22.0, 'overcast')]
y = ['no', 'yes', 'yes']

# 2. Train (with pruning enabled by default)
clf = C45Classifier(max_depth=5)
clf.fit(X, y, feature_names=['temperature', 'weather'])

# 3. Predict
print(clf.predict([(20.0, 'rain')]))  # Output: ['yes']
```

---

## �️ Advanced Usage

### Hyperparameter Tuning (GridSearch)

Find the best parameters automatically:

```python
from decision_trees import GridSearchCV, C45Classifier

# Define search space
param_grid = {
    'max_depth': [3, 5, None],
    'min_samples_split': [2, 5]
}

# Run Grid Search
search = GridSearchCV(C45Classifier, param_grid, cv=5)
search.fit(X_train, y_train)

print(f"Best Params: {search.best_params_}")
best_model = search.best_estimator_
```

### Model Serialization

Save your trained model to JSON and load it later:

```python
from decision_trees import save_model, load_model

# Save
save_model(clf, 'my_model.json')

# Load
loaded_clf = load_model('my_model.json')
```

### Evaluation Metrics

Get a full classification report:

```python
from decision_trees import classification_report

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 📊 Algorithm Comparison

| Feature | ID3 | C4.5 |
|---------|-----|------|
| **Splitting** | Information Gain | Gain Ratio |
| **Continuous Data** | ❌ | ✅ (Thresholds) |
| **Missing Values** | ❌ | ✅ (Probabilistic) |
| **Pruning** | ❌ | ✅ (Pessimistic) |
| **Best For** | Simple Categorical | Complex Real-world |

---

## 📁 Project Structure

```
src/decision_trees/
├── __init__.py          # Exports
├── base.py              # Abstract Base Class
├── optimized.py         # NumPy Optimizations
├── metrics.py           # Precision, Recall, F1
├── tuning.py            # GridSearchCV
├── serialization.py     # JSON Save/Load
├── id3/                 # ID3 Implementation
└── c45/                 # C4.5 Implementation
```

---

<div align="center">
<i>Built by Hssaine, Benalla and El Younsi</i>
</div>
