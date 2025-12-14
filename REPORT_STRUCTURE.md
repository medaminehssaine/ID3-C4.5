# Report Structure: Theory-to-Code Mapping

This document maps theoretical concepts from Quinlan's papers to their implementation in our codebase. Use this for exam preparation and writing your KDD report.

---

## 📚 Core References

1. **Quinlan, J.R. (1986).** *"Induction of Decision Trees"*, Machine Learning 1:81-106
   - Original ID3 algorithm
   
2. **Quinlan, J.R. (1993).** *"C4.5: Programs for Machine Learning"*, Morgan Kaufmann
   - C4.5 improvements: gain ratio, continuous attributes, pruning

---

## 🔢 Mathematical Concepts → Code Mapping

### 1. Entropy (Shannon Entropy)

| Concept | Formula | Code Location |
|---------|---------|---------------|
| Entropy | `H(S) = -Σᵢ p(cᵢ) × log₂(p(cᵢ))` | `id3/core/entropy.py::entropy()` |
| | | `c45/core/gain_ratio.py::entropy()` |

**Properties to mention:**
- H(S) = 0 for pure sets (all same class)
- H(S) = 1 for balanced binary (coin flip uncertainty)
- H(S) = log₂(k) for k equally distributed classes

**Edge case:** `0 × log₂(0) = 0` by convention (limit as p→0⁺)

---

### 2. Information Gain (ID3 Criterion)

| Concept | Formula | Code Location |
|---------|---------|---------------|
| Info Gain | `IG(S,A) = H(S) - Σᵥ (|Sᵥ|/|S|) × H(Sᵥ)` | `id3/core/entropy.py::information_gain()` |

**Interpretation:**
- Measures reduction in uncertainty after split
- Higher IG = more useful feature
- ID3 chooses feature with max IG at each node

**Limitation (key exam point):**
- Biased toward high-cardinality features
- Extreme case: unique ID column always has max IG

---

### 3. Split Information (Intrinsic Value)

| Concept | Formula | Code Location |
|---------|---------|---------------|
| Split Info | `SI(S,A) = -Σᵥ (|Sᵥ|/|S|) × log₂(|Sᵥ|/|S|)` | `c45/core/gain_ratio.py::split_info()` |

**Purpose:** Penalizes features that produce many small partitions

---

### 4. Gain Ratio (C4.5 Criterion)

| Concept | Formula | Code Location |
|---------|---------|---------------|
| Gain Ratio | `GR(S,A) = IG(S,A) / SI(S,A)` | `c45/core/gain_ratio.py::gain_ratio()` |

**Key properties:**
- GR ≤ IG always (SI ≥ 1 for multi-way splits)
- Reduces bias toward high-cardinality features
- Edge case: SI = 0 → GR = 0 (handled in code)

---

### 5. Continuous Attribute Handling

| Concept | Method | Code Location |
|---------|--------|---------------|
| Threshold Selection | Midpoint at class boundaries | `c45/core/gain_ratio.py::best_threshold()` |
| Binary Split | `value ≤ t` vs `value > t` | `c45/core/node.py::predict_one()` |

**Algorithm:**
1. Sort values
2. Find midpoints where class changes
3. Evaluate GR for each candidate
4. Choose threshold with max GR

**Key difference from ID3:** Continuous features can be reused in subtrees

---

### 6. Missing Value Handling

| Concept | Method | Code Location |
|---------|--------|---------------|
| Distribution | Weighted distribution to branches | `c45/core/gain_ratio.py::handle_missing()` |
| Prediction | Fall back to majority class | `c45/core/node.py::predict_one()` |

**C4.5 approach:** Distribute sample with weight proportional to branch population

---

### 7. Pruning

| Method | Formula | Code Location |
|--------|---------|---------------|
| Reduced Error | Accuracy on validation set | `c45/core/pruning.py::reduced_error_prune()` |
| Pessimistic | `(e + 0.5) / N` correction | `c45/core/pruning.py::pessimistic_prune()` |
| Upper Confidence | Wilson score interval | `c45/core/pruning.py::_upper_confidence_error()` |

**Pessimistic Error Formula:**
```
Pessimistic Rate = (errors + 0.5) / samples
```

**Wilson Score Upper Bound:**
```
UCB = (f + z²/2n + z×√(f(1-f)/n + z²/4n²)) / (1 + z²/n)
```

---

## 🏗️ Algorithm Structure

### ID3 Algorithm (Pseudocode → Code)

```
function ID3(D, features):
    if all samples same class → return Leaf(class)
    if no features left → return Leaf(majority_class)
    
    best = argmax(features, key=InformationGain)
    node = Node(best)
    
    for each value v of best:
        subset = {x ∈ D : x[best] = v}
        node.children[v] = ID3(subset, features - {best})
    
    return node
```

**Code:** `id3/core/tree.py::ID3Classifier._build_tree()`

### C4.5 Extensions

| Extension | Location |
|-----------|----------|
| Use GR instead of IG | `c45/core/tree.py` line 103 |
| Continuous threshold search | `c45/core/tree.py` lines 94-100 |
| Reuse continuous features | `c45/core/tree.py` line 140 |
| Post-pruning | `c45/core/pruning.py` |

---

## 📊 Test Cases for Defense

### Entropy Verification

| Input | Expected | Test |
|-------|----------|------|
| `['yes'] × 10` | 0.0 | `test_entropy_pure()` |
| `['yes'] × 5 + ['no'] × 5` | 1.0 | `test_entropy_balanced()` |
| `['yes'] × 9 + ['no'] × 5` | ≈0.9403 | `test_entropy_classic()` |

### Gain Ratio vs Information Gain

| Test | Assertion |
|------|-----------|
| GR ≤ IG | `test_gain_ratio_less_than_ig()` |
| Low cardinality preferred | `test_gain_ratio_prefers_simple_splits()` |

### Threshold Selection

| Test | Assertion |
|------|-----------|
| Class boundary detection | `test_best_threshold()` |
| Optimal midpoint | Threshold between 2.0 and 3.0 for class change at 2→3 |

---

## 📁 File-to-Concept Quick Reference

| File | Concepts |
|------|----------|
| `id3/core/entropy.py` | H(S), IG(S,A) |
| `c45/core/gain_ratio.py` | H(S), IG(S,A), SI(S,A), GR(S,A), threshold |
| `c45/core/pruning.py` | REP, PEP, Wilson interval |
| `c45/core/node.py` | Binary splits, missing values |
| `id3/core/tree.py` | ID3 algorithm |
| `c45/core/tree.py` | C4.5 algorithm |

---

## ✅ Exam Preparation Checklist

- [ ] Can derive entropy formula from first principles
- [ ] Can explain why ID3 is biased toward high-cardinality
- [ ] Can show Gain Ratio ≤ Information Gain mathematically
- [ ] Can trace threshold selection algorithm by hand
- [ ] Can explain pessimistic vs reduced error pruning
- [ ] Can calculate pessimistic error for a sample node
- [ ] Understand C4.5's advantage for continuous features
