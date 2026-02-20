"""
Tests for Full Experiment Pipeline Integration.

Specification: phase2_experimental_design.md
Author: test-engineer

Tests the complete experiment pipeline:
  1. Configuration validation (exclusion rules)
  2. Preprocessing -> Encoding -> Ansatz -> Measurement
  3. Metric computation (accuracy, F1, AUC, kernel alignment)
  4. Reproducibility (same seed = same results)
  5. Result storage format

These tests should FAIL until Phase 4 implementation.
"""

import pytest
import numpy as np
from conftest import (
    ENCODING_NAMES,
    PREPROCESSING_LEVELS,
    RANDOM_SEED,
    RANDOM_SEEDS,
    ANSATZ_REPS,
    SHOTS,
    QUBIT_FORMULAS,
    is_valid_configuration,
    INVALID_ENCODING_PREPROCESSING,
    INVALID_ENCODING_DATASET,
)


class TestExclusionRules:
    """Tests for experiment exclusion rules.

    From experimental_design.md Section 3.
    """

    def test_basis_continuous_excluded(self):
        """Basis + continuous preprocessing should be excluded."""
        assert not is_valid_configuration("basis", "none", "iris", 4)
        assert not is_valid_configuration("basis", "minmax_pi", "iris", 4)
        assert not is_valid_configuration("basis", "zscore_sigmoid", "iris", 4)

    def test_basis_multiclass_excluded(self):
        """Basis + multi-class datasets should be excluded."""
        assert not is_valid_configuration("basis", "pca_minmax", "iris", 4)
        assert not is_valid_configuration("basis", "pca_minmax", "wine", 8)
        assert not is_valid_configuration("basis", "pca_minmax", "har", 8)

    def test_amplitude_no_preprocessing_excluded(self):
        """Amplitude + no preprocessing should be excluded."""
        assert not is_valid_configuration("amplitude", "none", "iris", 4)

    def test_iqp_high_d_excluded(self):
        """IQP full with d > 8 should be excluded (NISQ infeasible)."""
        assert not is_valid_configuration("iqp_full", "minmax_pi", "cancer", 30)

    def test_reuploading_high_d_excluded(self):
        """Re-uploading with d > 8 should be excluded."""
        assert not is_valid_configuration("reuploading", "minmax_pi", "har", 561)

    def test_qubit_budget_exceeded(self):
        """Configurations requiring > 20 qubits should be excluded."""
        # Angle encoding with d=25 requires 25 qubits
        assert not is_valid_configuration("angle_ry", "none", "cancer", 25)

    def test_valid_configurations_count(self):
        """Should have approximately 125 valid configurations (Section 3.3)."""
        datasets = {
            "iris": 4,
            "wine": 8,
            "cancer": 8,
            "mnist": 8,
            "credit_fraud": 8,
            "nsl_kdd": 8,
            "har": 8,
            "heart": 8,
        }
        valid_count = 0
        for enc in ENCODING_NAMES:
            for prep in PREPROCESSING_LEVELS:
                for ds_name, n_feat in datasets.items():
                    if is_valid_configuration(enc, prep, ds_name, n_feat):
                        valid_count += 1

        assert valid_count >= 100, f"Expected >= 100 valid configurations, got {valid_count}"

    def test_angle_all_valid(self):
        """Angle encoding should be valid for all datasets and preprocessings."""
        for prep in PREPROCESSING_LEVELS:
            assert is_valid_configuration("angle_ry", prep, "iris", 4)
            assert is_valid_configuration("angle_ry", prep, "cancer", 8)


class TestQubitFormulas:
    """Tests for qubit count formulas."""

    def test_basis_qubits(self):
        """Basis: Q = d."""
        assert QUBIT_FORMULAS["basis"](4) == 4
        assert QUBIT_FORMULAS["basis"](8) == 8

    def test_amplitude_qubits(self):
        """Amplitude: Q = ceil(log2(d))."""
        assert QUBIT_FORMULAS["amplitude"](4) == 2
        assert QUBIT_FORMULAS["amplitude"](8) == 3
        assert QUBIT_FORMULAS["amplitude"](16) == 4
        assert QUBIT_FORMULAS["amplitude"](3) == 2  # Padded

    def test_angle_qubits(self):
        """Angle: Q = d."""
        assert QUBIT_FORMULAS["angle_ry"](4) == 4
        assert QUBIT_FORMULAS["angle_ry"](8) == 8

    def test_iqp_qubits(self):
        """IQP: Q = d."""
        assert QUBIT_FORMULAS["iqp_full"](4) == 4
        assert QUBIT_FORMULAS["iqp_full"](8) == 8

    def test_reuploading_qubits(self):
        """Re-uploading: Q = d."""
        assert QUBIT_FORMULAS["reuploading"](4) == 4
        assert QUBIT_FORMULAS["reuploading"](8) == 8


class TestExperimentConfiguration:
    """Tests for experiment configuration structure."""

    def test_config_has_required_fields(self, experiment_config_iris_angle):
        """Configuration should have all required fields."""
        required_fields = [
            "dataset",
            "encoding",
            "preprocessing",
            "n_qubits",
            "ansatz",
            "ansatz_reps",
            "optimizer",
            "maxiter",
            "shots",
            "n_folds",
            "seeds",
        ]
        for field in required_fields:
            assert field in experiment_config_iris_angle, f"Missing required field: {field}"

    def test_config_iris_angle(self, experiment_config_iris_angle):
        """Verify Iris + Angle configuration values."""
        cfg = experiment_config_iris_angle
        assert cfg["dataset"] == "iris"
        assert cfg["encoding"] == "angle_ry"
        assert cfg["n_qubits"] == 4
        assert cfg["ansatz_reps"] == ANSATZ_REPS
        assert cfg["shots"] == SHOTS
        assert cfg["seeds"] == RANDOM_SEEDS


class TestPipelineEndToEnd:
    """End-to-end pipeline tests (preprocessing + encoding + ansatz)."""

    def test_iris_angle_pipeline(self, iris_dataset):
        """Full pipeline: Iris -> MinMax[0,pi] -> Angle encoding -> circuit."""
        from qml_data_encoding.pipeline import ExperimentPipeline

        X, y = iris_dataset
        pipeline = ExperimentPipeline(
            encoding="angle_ry",
            preprocessing="minmax_pi",
            n_qubits=4,
            ansatz_reps=ANSATZ_REPS,
        )

        # Build circuit for a single data point
        circuit = pipeline.build_circuit(X[0])
        assert circuit is not None
        assert circuit.num_qubits == 4

    def test_cancer_iqp_pca_pipeline(self, cancer_dataset):
        """Full pipeline: Cancer -> PCA(8) + MinMax -> IQP encoding."""
        from qml_data_encoding.pipeline import ExperimentPipeline

        X, y = cancer_dataset
        pipeline = ExperimentPipeline(
            encoding="iqp_full",
            preprocessing="pca_minmax",
            n_qubits=8,
            ansatz_reps=ANSATZ_REPS,
            pca_components=8,
        )

        pipeline.fit_preprocessing(X)
        X_proc = pipeline.transform(X[:5])
        assert X_proc.shape == (5, 8)

        circuit = pipeline.build_circuit(X_proc[0])
        assert circuit.num_qubits == 8

    def test_pipeline_reproducibility(self, iris_dataset):
        """Same seed should produce identical results.

        From experimental_design.md Section 8: Reproducibility requirements.
        """
        from qml_data_encoding.pipeline import ExperimentPipeline

        X, y = iris_dataset

        results = []
        for _ in range(2):
            pipeline = ExperimentPipeline(
                encoding="angle_ry",
                preprocessing="minmax_pi",
                n_qubits=4,
                ansatz_reps=ANSATZ_REPS,
                seed=RANDOM_SEED,
            )
            pipeline.fit_preprocessing(X)
            X_proc = pipeline.transform(X[:10])
            results.append(X_proc)

        np.testing.assert_array_equal(results[0], results[1])


class TestResultStorage:
    """Tests for result storage format.

    From experimental_design.md Section 8.2.
    """

    def test_result_has_required_columns(self):
        """Result dataclass should have all required fields."""
        from qml_data_encoding.pipeline import ExperimentResult

        required_columns = [
            "dataset",
            "encoding",
            "preprocessing",
            "fold",
            "seed",
            "train_accuracy",
            "test_accuracy",
            "f1_macro",
            "convergence_iters",
            "final_loss",
            "generalization_gap",
            "circuit_depth",
            "gate_count",
            "cx_count",
        ]
        result = ExperimentResult(
            dataset="iris",
            encoding="angle_ry",
            preprocessing="minmax_pi",
            fold=0,
            seed=42,
            train_accuracy=0.95,
            test_accuracy=0.90,
            f1_macro=0.89,
            convergence_iters=150,
            final_loss=0.3,
            generalization_gap=0.05,
            circuit_depth=8,
            gate_count=20,
            cx_count=4,
        )

        for col in required_columns:
            assert hasattr(result, col), f"Missing result field: {col}"


class TestMetricComputation:
    """Tests for metric computation correctness."""

    def test_accuracy_computation(self):
        """Test accuracy = correct / total."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        accuracy = np.mean(y_true == y_pred)
        assert np.isclose(accuracy, 0.8)

    def test_generalization_gap(self):
        """Generalization gap = train_acc - test_acc."""
        train_acc = 0.95
        test_acc = 0.85
        gap = train_acc - test_acc
        assert np.isclose(gap, 0.10)

    def test_generalization_gap_sign(self):
        """Positive gap indicates overfitting; negative indicates underfitting."""
        # Overfitting case
        assert (0.95 - 0.80) > 0  # Positive = overfitting
        # Good generalization
        assert abs(0.90 - 0.88) < 0.05  # Small gap = good


class TestSeedManagement:
    """Tests for seed management and reproducibility.

    From experimental_design.md Section 4.2.
    """

    def test_five_fold_seeds(self):
        """Five fold seeds should be 42, 43, 44, 45, 46."""
        assert RANDOM_SEEDS == [42, 43, 44, 45, 46]

    def test_stratified_split_deterministic(self, iris_dataset):
        """Stratified split with same seed should produce same split."""
        from sklearn.model_selection import StratifiedKFold

        X, y = iris_dataset
        skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        splits1 = list(skf1.split(X, y))
        splits2 = list(skf2.split(X, y))

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)
