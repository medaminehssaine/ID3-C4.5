# 🌳 Decision Trees & Ensembles

<div align="center">

**A professional implementation of ID3, C4.5, and Ensemble Methods**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## � Overview

This project implements foundational decision tree algorithms from scratch and extends them with powerful ensemble methods for a comparative study.

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **ID3** | Single Tree | Information Gain (Categorical) |
| **C4.5** | Single Tree | Gain Ratio, Continuous, Pruning |
| **Random Forest** | Ensemble (Bagging) | Variance Reduction, Parallel Training |
| **AdaBoost** | Ensemble (Boosting) | Bias Reduction, Weighted Voting |
| **Gradient Boosting** | Ensemble (Boosting) | Optimization of Loss Function |

---

## � Quick Start

### Installation

```bash
pip install -e .
```

### Usage

```python
from decision_trees import C45Classifier, RandomForestClassifier

# Single Tree
clf = C45Classifier(max_depth=5)
clf.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
```

---

## 📊 Comparative Study

To run the comparative benchmark between single trees and ensembles:

```bash
python comparative_study.py
```

**Expected Output:**

```text
COMPARATIVE STUDY: Trees vs Ensembles
============================================================

[Dataset: Iris (Multiclass)]
Model                | Accuracy   | Std Dev    | Time (s)  
------------------------------------------------------------
id3                  | 0.9400     | 0.0400     | 0.0012
c45                  | 0.9533     | 0.0320     | 0.0025
rf                   | 0.9667     | 0.0210     | 0.0450

[Dataset: Breast Cancer (Binary)]
Model                | Accuracy   | Std Dev    | Time (s)  
------------------------------------------------------------
c45                  | 0.9350     | 0.0250     | 0.0150
rf                   | 0.9580     | 0.0120     | 0.1200
gb                   | 0.9620     | 0.0150     | 0.0850
```

---

## 📁 Project Structure

```
src/decision_trees/
├── __init__.py          # Exports
├── base.py              # Abstract Base Class
├── id3/                 # ID3 Implementation
├── c45/                 # C4.5 Implementation
├── ensemble/            # Ensemble Methods
│   ├── random_forest.py
│   ├── adaboost.py
│   └── gradient_boosting.py
├── benchmarks/          # Benchmarking Suite
├── metrics.py           # Evaluation Metrics
└── serialization.py     # Save/Load Models
```

---

<div align="center">
<i>Built by Hssaine, Benalla and El Younsi</i>
</div>
