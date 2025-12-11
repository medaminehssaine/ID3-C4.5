"""comparison with sklearn decision tree"""
import time


def compare_with_sklearn(our_model, X, y, X_test=None, y_test=None):
    """
    compare our id3 with sklearn's decision tree
    requires sklearn to be installed
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print("sklearn not installed, skipping comparison")
        return None
    
    # encode labels for sklearn
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # encode features (sklearn needs numeric)
    feature_encoders = []
    X_encoded = []
    
    for sample in X:
        encoded_sample = []
        for i, val in enumerate(sample):
            if i >= len(feature_encoders):
                feature_encoders.append({})
            if val not in feature_encoders[i]:
                feature_encoders[i][val] = len(feature_encoders[i])
            encoded_sample.append(feature_encoders[i][val])
        X_encoded.append(encoded_sample)
    
    # train sklearn model
    sklearn_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    
    start = time.time()
    sklearn_model.fit(X_encoded, y_encoded)
    sklearn_train_time = time.time() - start
    
    # train our model (already trained, just measure prediction)
    
    # predictions on training data
    our_pred = our_model.predict(X)
    sklearn_pred = le.inverse_transform(sklearn_model.predict(X_encoded))
    
    our_acc = sum(1 for t, p in zip(y, our_pred) if t == p) / len(y)
    sklearn_acc = sum(1 for t, p in zip(y, sklearn_pred) if t == p) / len(y)
    
    results = {
        "our_model": {
            "train_accuracy": our_acc,
            "depth": our_model.get_depth(),
            "n_leaves": our_model.get_n_leaves()
        },
        "sklearn": {
            "train_accuracy": sklearn_acc,
            "depth": sklearn_model.get_depth(),
            "n_leaves": sklearn_model.get_n_leaves(),
            "train_time": sklearn_train_time
        }
    }
    
    # test set if provided
    if X_test is not None and y_test is not None:
        X_test_encoded = []
        for sample in X_test:
            encoded_sample = []
            for i, val in enumerate(sample):
                if val in feature_encoders[i]:
                    encoded_sample.append(feature_encoders[i][val])
                else:
                    encoded_sample.append(-1)  # unknown value
            X_test_encoded.append(encoded_sample)
        
        our_test_pred = our_model.predict(X_test)
        sklearn_test_pred = le.inverse_transform(sklearn_model.predict(X_test_encoded))
        
        results["our_model"]["test_accuracy"] = sum(1 for t, p in zip(y_test, our_test_pred) if t == p) / len(y_test)
        results["sklearn"]["test_accuracy"] = sum(1 for t, p in zip(y_test, sklearn_test_pred) if t == p) / len(y_test)
    
    return results


def print_comparison(results):
    """pretty print comparison results"""
    if results is None:
        return
    
    print("\n" + "=" * 50)
    print("COMPARISON: Our ID3 vs sklearn DecisionTree")
    print("=" * 50)
    
    print(f"\n{'Metric':<25} {'Our ID3':>12} {'sklearn':>12}")
    print("-" * 50)
    
    our = results["our_model"]
    sk = results["sklearn"]
    
    print(f"{'Train Accuracy':<25} {our['train_accuracy']:>12.4f} {sk['train_accuracy']:>12.4f}")
    
    if "test_accuracy" in our:
        print(f"{'Test Accuracy':<25} {our['test_accuracy']:>12.4f} {sk['test_accuracy']:>12.4f}")
    
    print(f"{'Tree Depth':<25} {our['depth']:>12} {sk['depth']:>12}")
    print(f"{'Number of Leaves':<25} {our['n_leaves']:>12} {sk['n_leaves']:>12}")
    print()


def benchmark(model_class, X, y, n_runs=5, **model_params):
    """
    benchmark training time over multiple runs
    returns average training time in seconds
    """
    times = []
    
    for _ in range(n_runs):
        model = model_class(**model_params)
        start = time.time()
        model.fit(X, y)
        times.append(time.time() - start)
    
    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "n_runs": n_runs
    }
