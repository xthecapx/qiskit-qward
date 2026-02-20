"""
Tests for Classical Baselines.

Specification: phase1_evaluation_framework.md, Section 5
              phase2_experimental_design.md, Section 5.5
Author: test-engineer

Classical baselines serve as reference points for quantum encoding comparison:
  - SVM with RBF kernel (primary baseline)
  - Random Forest
  - Logistic Regression

These tests should PASS (classical models from sklearn).
They verify that our baseline implementations produce expected accuracy ranges.
"""

import pytest
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from conftest import RANDOM_SEED, N_FOLDS, RANGE_PI


class TestSVMBaseline:
    """Tests for SVM with RBF kernel baseline.

    Known performance benchmarks from phase1_dataset_selection.md:
    - Iris: ~97%
    - Wine: ~98%
    - Cancer: ~97%
    """

    @pytest.mark.classical_baseline
    def test_iris_svm_accuracy(self, iris_dataset):
        """SVM on Iris should achieve >= 90% accuracy."""
        X, y = iris_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ]
        )
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.90, f"SVM on Iris: {mean_acc:.3f}, expected >= 0.90"

    @pytest.mark.classical_baseline
    def test_wine_svm_accuracy(self, wine_dataset):
        """SVM on Wine should achieve >= 90% accuracy."""
        X, y = wine_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ]
        )
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.90, f"SVM on Wine: {mean_acc:.3f}, expected >= 0.90"

    @pytest.mark.classical_baseline
    def test_cancer_svm_accuracy(self, cancer_dataset):
        """SVM on Cancer should achieve >= 90% accuracy."""
        X, y = cancer_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ]
        )
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.90, f"SVM on Cancer: {mean_acc:.3f}, expected >= 0.90"


class TestRandomForestBaseline:
    """Tests for Random Forest baseline."""

    @pytest.mark.classical_baseline
    def test_iris_rf_accuracy(self, iris_dataset):
        """Random Forest on Iris should achieve >= 90% accuracy."""
        X, y = iris_dataset
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.90, f"RF on Iris: {mean_acc:.3f}, expected >= 0.90"

    @pytest.mark.classical_baseline
    def test_cancer_rf_accuracy(self, cancer_dataset):
        """Random Forest on Cancer should achieve >= 90% accuracy."""
        X, y = cancer_dataset
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(rf, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.90, f"RF on Cancer: {mean_acc:.3f}, expected >= 0.90"


class TestLogisticRegressionBaseline:
    """Tests for Logistic Regression baseline."""

    @pytest.mark.classical_baseline
    def test_iris_lr_accuracy(self, iris_dataset):
        """Logistic Regression on Iris should achieve >= 85% accuracy."""
        X, y = iris_dataset
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
            ]
        )
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.85, f"LR on Iris: {mean_acc:.3f}, expected >= 0.85"


class TestBaselineReproducibility:
    """Tests for classical baseline reproducibility."""

    @pytest.mark.classical_baseline
    def test_same_seed_same_results(self, iris_dataset):
        """Same random seed should produce identical cross-val scores."""
        X, y = iris_dataset

        results = []
        for _ in range(2):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svm", SVC(kernel="rbf", random_state=RANDOM_SEED)),
                ]
            )
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
            results.append(scores)

        np.testing.assert_array_almost_equal(results[0], results[1])


class TestBaselineWithQuantumPreprocessing:
    """Tests that classical baselines use the same preprocessing as quantum.

    From experimental_design.md Section 5.5:
    Classical baselines must be run on the same preprocessing.
    """

    @pytest.mark.classical_baseline
    def test_svm_with_minmax_pi(self, iris_dataset):
        """SVM with MinMax[0,pi] preprocessing should still perform well."""
        X, y = iris_dataset
        pipe = Pipeline(
            [
                ("scaler", MinMaxScaler(feature_range=(0, RANGE_PI))),
                ("svm", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ]
        )
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.85, f"SVM with MinMax[0,pi] on Iris: {mean_acc:.3f}, expected >= 0.85"

    @pytest.mark.classical_baseline
    def test_svm_with_pca(self, cancer_dataset):
        """SVM with PCA to 8 components should still perform well on Cancer."""
        from sklearn.decomposition import PCA

        X, y = cancer_dataset
        pipe = Pipeline(
            [
                ("pca", PCA(n_components=8, random_state=RANDOM_SEED)),
                ("scaler", MinMaxScaler(feature_range=(0, RANGE_PI))),
                ("svm", SVC(kernel="rbf", random_state=RANDOM_SEED)),
            ]
        )
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        mean_acc = scores.mean()
        assert mean_acc >= 0.85, f"SVM with PCA+MinMax on Cancer: {mean_acc:.3f}, expected >= 0.85"


class TestQuantumClassicalComparison:
    """Tests for the quantum vs classical comparison metric.

    From experimental_design.md Section 5.5:
    Delta_QC = Acc_quantum - Acc_classical_best
    """

    def test_delta_qc_computation(self):
        """Delta_QC should be computable from quantum and classical accuracies."""
        acc_quantum = 0.85
        acc_classical_best = 0.90
        delta_qc = acc_quantum - acc_classical_best
        assert np.isclose(delta_qc, -0.05)

    def test_delta_qc_negative_expected(self):
        """For most datasets, quantum accuracy <= classical accuracy.

        This is the expected baseline -- quantum advantage is rare for
        tabular data classification on current NISQ hardware.
        """
        # Document the expectation that Delta_QC is typically <= 0
        # The experiment should reveal if any encoding achieves Delta_QC > 0
        assert True  # Placeholder for documentation
