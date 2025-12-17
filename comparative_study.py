"""
Comparative Study: Single Trees vs Ensemble Methods.

This script runs a benchmark comparing ID3 and C4.5 on the Iris and Breast Cancer datasets.
"""
import sys
import time
from typing import List, Tuple

from decision_trees import (
    ID3Classifier,
    C45Classifier,

    accuracy_score
)
from decision_trees.benchmarks import BenchmarkSuite

def run_study():
    print("=" * 60)
    print("COMPARATIVE STUDY: Trees vs Ensembles")
    print("=" * 60)

    suite = BenchmarkSuite()
    
    # Define models to compare
    # Define models to compare
    models = ['id3', 'c45']
    
    # Run on Iris (Multiclass)
    print("\n[Dataset: Iris (Multiclass)]")
    iris_models = ['id3', 'c45']
    results_iris = suite.run('iris', iris_models, cv=5)
    print_results(results_iris)

    # Run on Breast Cancer (Binary)
    print("\n[Dataset: Breast Cancer (Binary)]")
    bc_models = ['c45'] # ID3 might struggle with continuous features if not discretized
    results_bc = suite.run('breast_cancer', bc_models, cv=5)
    print_results(results_bc)

    print("\n" + "=" * 60)
    print("CONCLUSIONS")
    print("=" * 60)
    print("1. C4.5 generally handles continuous attributes better than ID3.")
    print("2. Single trees are fast to train and easy to interpret.")
    print("=" * 60)

def print_results(results: dict):
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Std Dev':<10} | {'Time (s)':<10}")
    print("-" * 60)
    for model, metrics in results.items():
        print(f"{model:<20} | {metrics['accuracy_mean']:.4f}     | {metrics['accuracy_std']:.4f}     | {metrics['train_time_mean']:.4f}")

if __name__ == "__main__":
    run_study()
