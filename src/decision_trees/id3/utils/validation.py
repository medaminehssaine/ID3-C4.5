"""validation and metrics utilities"""
import random
from collections import defaultdict


def accuracy(y_true, y_pred):
    """calculate accuracy score"""
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def train_test_split(X, y, test_ratio=0.2, random_state=None):
    """
    split data into train and test sets
    returns: X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        random.seed(random_state)
    
    indices = list(range(len(X)))
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    
    return X_train, X_test, y_train, y_test


def cross_validate(model_class, X, y, k=5, **model_params):
    """
    k-fold cross validation
    returns list of accuracy scores for each fold
    """
    n = len(X)
    fold_size = n // k
    indices = list(range(n))
    random.shuffle(indices)
    
    scores = []
    
    for i in range(k):
        # create fold
        test_start = i * fold_size
        test_end = test_start + fold_size if i < k - 1 else n
        test_idx = set(indices[test_start:test_end])
        
        X_train = [X[j] for j in range(n) if j not in test_idx]
        y_train = [y[j] for j in range(n) if j not in test_idx]
        X_test = [X[j] for j in range(n) if j in test_idx]
        y_test = [y[j] for j in range(n) if j in test_idx]
        
        # train and evaluate
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        scores.append(accuracy(y_test, y_pred))
    
    return scores


def confusion_matrix(y_true, y_pred, labels=None):
    """
    compute confusion matrix
    returns dict of dicts: matrix[true_label][pred_label] = count
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    matrix = {t: {p: 0 for p in labels} for t in labels}
    
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    
    return matrix


def print_confusion_matrix(y_true, y_pred, labels=None):
    """pretty print confusion matrix"""
    matrix = confusion_matrix(y_true, y_pred, labels)
    labels = list(matrix.keys())
    
    # header
    max_label_len = max(len(str(l)) for l in labels)
    header = " " * (max_label_len + 2) + "  ".join(f"{l:>{max_label_len}}" for l in labels)
    print("Predicted:")
    print(header)
    print("Actual:")
    
    for true_label in labels:
        row = f"{true_label:>{max_label_len}}: "
        row += "  ".join(f"{matrix[true_label][p]:>{max_label_len}}" for p in labels)
        print(row)


def precision_recall(y_true, y_pred, positive_label):
    """compute precision and recall for a specific class"""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    return precision, recall
