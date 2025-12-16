"""
Benchmarking Suite.

Compare models against sklearn, XGBoost, and LightGBM.
"""
import time
from typing import Any, Dict, List, Optional, Tuple

Dataset = List[Tuple[Any, ...]]
Labels = List[Any]


class BenchmarkSuite:
    """Automated benchmarking for decision tree models."""

    DATASETS = {
        'iris': '_load_iris',
        'wine': '_load_wine',
        'breast_cancer': '_load_breast_cancer',
    }

    def run(
        self,
        dataset: str = 'iris',
        models: Optional[List[str]] = None,
        cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Run benchmark on specified dataset."""
        models = models or ['id3', 'c45', 'rf']
        
        X, y, feature_names = self._load_dataset(dataset)
        results = {}

        for model_name in models:
            model = self._get_model(model_name)
            if model is None:
                continue

            metrics = self._evaluate(model, X, y, feature_names, cv)
            results[model_name] = metrics

        return results

    def _load_dataset(self, name: str) -> Tuple[Dataset, Labels, List[str]]:
        """Load a built-in dataset."""
        if name == 'iris':
            return self._load_iris()
        elif name == 'wine':
            return self._load_wine()
        elif name == 'breast_cancer':
            return self._load_breast_cancer()
        else:
            raise ValueError(f"Unknown dataset: {name}")

    def _load_iris(self) -> Tuple[Dataset, Labels, List[str]]:
        """Load Iris dataset."""
        try:
            from sklearn.datasets import load_iris
            data = load_iris()
            X = [tuple(row) for row in data.data.tolist()]
            y = [data.target_names[i] for i in data.target]
            feature_names = list(data.feature_names)
            return X, y, feature_names
        except ImportError:
            # Fallback: minimal iris data
            return self._minimal_iris()

    def _load_wine(self) -> Tuple[Dataset, Labels, List[str]]:
        """Load Wine dataset."""
        try:
            from sklearn.datasets import load_wine
            data = load_wine()
            X = [tuple(row) for row in data.data.tolist()]
            y = [str(i) for i in data.target]
            feature_names = list(data.feature_names)
            return X, y, feature_names
        except ImportError:
            raise ValueError("sklearn required for wine dataset")

    def _load_breast_cancer(self) -> Tuple[Dataset, Labels, List[str]]:
        """Load Breast Cancer dataset."""
        try:
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            X = [tuple(row) for row in data.data.tolist()]
            y = [data.target_names[i] for i in data.target]
            feature_names = list(data.feature_names)
            return X, y, feature_names
        except ImportError:
            # Fallback: minimal synthetic data
            print("Warning: sklearn not found, using minimal synthetic data for breast_cancer")
            return self._minimal_breast_cancer()

    def _minimal_breast_cancer(self) -> Tuple[Dataset, Labels, List[str]]:
        """Minimal synthetic binary dataset."""
        import random
        random.seed(42)
        X = []
        y = []
        feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area']
        for _ in range(50):
            # Malignant-like
            X.append((
                random.uniform(15, 25),
                random.uniform(20, 30),
                random.uniform(100, 150),
                random.uniform(800, 1500)
            ))
            y.append('malignant')
            # Benign-like
            X.append((
                random.uniform(6, 14),
                random.uniform(10, 20),
                random.uniform(40, 90),
                random.uniform(200, 600)
            ))
            y.append('benign')
        return X, y, feature_names

    def _minimal_iris(self) -> Tuple[Dataset, Labels, List[str]]:
        """Minimal Iris data for testing."""
        X = [
            (5.1, 3.5, 1.4, 0.2), (4.9, 3.0, 1.4, 0.2),
            (7.0, 3.2, 4.7, 1.4), (6.4, 3.2, 4.5, 1.5),
            (6.3, 3.3, 6.0, 2.5), (5.8, 2.7, 5.1, 1.9),
        ]
        y = ['setosa', 'setosa', 'versicolor', 'versicolor', 'virginica', 'virginica']
        feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        return X, y, feature_names

    def _get_model(self, name: str) -> Any:
        """Get model instance by name."""
        if name == 'id3':
            from decision_trees import ID3Classifier
            return ID3Classifier()
        elif name == 'c45':
            from decision_trees import C45Classifier
            return C45Classifier()
        elif name == 'rf':
            from decision_trees.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=10)
        elif name == 'gb':
            from decision_trees.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(n_estimators=10)
        else:
            return None

    def _evaluate(
        self,
        model: Any,
        X: Dataset,
        y: Labels,
        feature_names: List[str],
        cv: int
    ) -> Dict[str, float]:
        """Evaluate model with cross-validation."""
        from decision_trees import accuracy_score

        n = len(X)
        fold_size = n // cv
        scores = []
        times = []

        for i in range(cv):
            start = i * fold_size
            end = start + fold_size if i < cv - 1 else n

            X_test = X[start:end]
            y_test = y[start:end]
            X_train = X[:start] + X[end:]
            y_train = y[:start] + y[end:]

            t0 = time.time()
            model.fit(X_train, y_train, feature_names=feature_names)
            train_time = time.time() - t0

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            scores.append(acc)
            times.append(train_time)

        return {
            'accuracy_mean': sum(scores) / len(scores),
            'accuracy_std': (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
            'train_time_mean': sum(times) / len(times)
        }

    def compare_html(self, results: Dict[str, Dict[str, float]], output: str = 'benchmark.html'):
        """Generate HTML benchmark report."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Decision Trees Benchmark</title>
    <style>
        body { font-family: Arial; padding: 20px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Benchmark Results</h1>
    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Std</th>
            <th>Train Time (s)</th>
        </tr>
"""
        for model, metrics in results.items():
            html += f"""
        <tr>
            <td>{model}</td>
            <td>{metrics['accuracy_mean']:.4f}</td>
            <td>±{metrics['accuracy_std']:.4f}</td>
            <td>{metrics['train_time_mean']:.4f}</td>
        </tr>
"""
        html += """
    </table>
</body>
</html>
"""
        with open(output, 'w') as f:
            f.write(html)
        return output
