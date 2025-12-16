"""
Comparative Study: Single Trees vs Ensemble Methods.

This script runs a benchmark comparing ID3, C4.5, Random Forest, AdaBoost, and Gradient Boosting
on the Iris and Breast Cancer datasets.
"""
import sys
import time
from typing import List, Tuple

from decision_trees import (
    ID3Classifier,
    C45Classifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    accuracy_score
)
from decision_trees.benchmarks import BenchmarkSuite

def run_study():
    print("=" * 60)
    print("COMPARATIVE STUDY: Trees vs Ensembles")
    print("=" * 60)

    suite = BenchmarkSuite()
    
    # Define models to compare
    models = ['id3', 'c45', 'rf', 'gb']
    
    # Run on Iris (Multiclass)
    # Note: GB currently supports binary only, so we skip it for Iris
    print("\n[Dataset: Iris (Multiclass)]")
    iris_models = ['id3', 'c45', 'rf']
    results_iris = suite.run('iris', iris_models, cv=5)
    print_results(results_iris)

    # Run on Breast Cancer (Binary)
    print("\n[Dataset: Breast Cancer (Binary)]")
    bc_models = ['c45', 'rf', 'gb'] # ID3 might struggle with continuous features if not discretized
    results_bc = suite.run('breast_cancer', bc_models, cv=5)
    print_results(results_bc)

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("1. Random Forest generally reduces variance compared to single C4.5 trees.")
    print("2. Gradient Boosting often achieves higher accuracy on binary tasks.")
    print("3. Single trees (C4.5) are faster to train and easier to interpret.")
    print("=" * 60)

def print_results(results: dict):
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Std Dev':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for model, metrics in results.items():
        print(f"{model:<20} | {metrics['accuracy_mean']:.4f}     | {metrics['accuracy_std']:.4f}     | {metrics['train_time_mean']:.4f}")

if __name__ == "__main__":
    run_study()
