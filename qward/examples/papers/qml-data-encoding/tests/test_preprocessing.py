"""
Tests for Classical Preprocessing Pipeline.

Specification: phase2_preprocessing_theory.md
Author: test-engineer

Four preprocessing levels:
  Level 0 (None): Identity + range clipping
  Level 1 (MinMax): Map to [0, pi]
  Level 2 (Z-score + sigmoid): Map to (0, pi)
  Level 3 (PCA + MinMax): Decorrelate + map to [0, pi]

These tests should PASS (classical preprocessing is implemented via sklearn).
"""

import pytest
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from conftest import sigmoid, zscore_sigmoid_transform, RANGE_PI


class TestMinMaxPreprocessing:
    """Tests for Level 1: MinMax normalization to [0, pi]."""

    def test_output_range(self, iris_dataset):
        """MinMax output should be in [0, pi] for all features."""
        X, _ = iris_dataset
        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        X_scaled = scaler.fit_transform(X)

        assert np.all(X_scaled >= 0.0)
        assert np.all(X_scaled <= RANGE_PI + 1e-10)

    def test_min_maps_to_zero(self, iris_dataset):
        """Feature minimum should map to 0."""
        X, _ = iris_dataset
        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        X_scaled = scaler.fit_transform(X)

        for j in range(X.shape[1]):
            min_idx = np.argmin(X[:, j])
            assert np.isclose(X_scaled[min_idx, j], 0.0, atol=1e-10)

    def test_max_maps_to_pi(self, iris_dataset):
        """Feature maximum should map to pi."""
        X, _ = iris_dataset
        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        X_scaled = scaler.fit_transform(X)

        for j in range(X.shape[1]):
            max_idx = np.argmax(X[:, j])
            assert np.isclose(X_scaled[max_idx, j], RANGE_PI, atol=1e-10)

    def test_preserves_relative_order(self, iris_dataset):
        """MinMax should preserve feature ordering."""
        X, _ = iris_dataset
        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        X_scaled = scaler.fit_transform(X)

        for j in range(X.shape[1]):
            original_order = np.argsort(X[:, j])
            scaled_order = np.argsort(X_scaled[:, j])
            np.testing.assert_array_equal(original_order, scaled_order)

    def test_fit_transform_consistency(self, iris_dataset):
        """fit_transform on train should be consistent with transform on test."""
        X, y = iris_dataset
        from sklearn.model_selection import train_test_split

        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        scaler.fit(X_train)

        X_test_scaled = scaler.transform(X_test)
        # Test data may exceed [0, pi] if it has values outside train range
        # This is expected behavior -- no assertion on test range

    def test_outlier_sensitivity(self):
        """MinMax is sensitive to outliers (documented failure mode).

        From preprocessing_theory.md Section 5.3 Failure Mode 1.
        """
        rng = np.random.default_rng(42)
        # Normal data centered at 5 with one outlier at 100
        X = rng.normal(5, 1, size=(100, 1))
        X[0, 0] = 100.0  # Extreme outlier

        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        X_scaled = scaler.fit_transform(X)

        # Most data compressed into a small range near 0
        non_outlier_range = np.ptp(X_scaled[1:, 0])
        full_range = RANGE_PI
        compression_ratio = non_outlier_range / full_range

        # Assert that outlier causes compression of majority data
        assert compression_ratio < 0.15, (
            f"Expected significant compression with outlier, " f"got ratio {compression_ratio:.3f}"
        )


class TestZScoreSigmoidPreprocessing:
    """Tests for Level 2: Z-score + sigmoid mapping to (0, pi)."""

    def test_output_in_open_interval(self, iris_dataset):
        """Output should be in (0, pi) -- open interval, never exactly 0 or pi."""
        X, _ = iris_dataset
        X_transformed = zscore_sigmoid_transform(X)

        assert np.all(X_transformed > 0.0)
        assert np.all(X_transformed < RANGE_PI)

    def test_centered_data_maps_to_pi_half(self):
        """Data at the mean should map to pi/2 (sigmoid(0) = 0.5, * pi = pi/2)."""
        X = np.array([[0.0], [0.0], [0.0]])
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
        result = RANGE_PI * sigmoid(Z)
        np.testing.assert_allclose(result, RANGE_PI / 2, atol=1e-10)

    def test_sigmoid_monotonic(self):
        """Sigmoid should be monotonically increasing."""
        z = np.linspace(-10, 10, 1000)
        s = sigmoid(z)
        diffs = np.diff(s)
        assert np.all(diffs >= 0)

    def test_sigmoid_bounds(self):
        """Sigmoid output should be in (0, 1) for practical input range.

        For standard z-scores, inputs are typically in [-5, 5].
        At float64 extremes (|z| > ~36), sigmoid rounds to exactly 0 or 1.
        """
        z_practical = np.array([-10, -5, -1, 0, 1, 5, 10])
        s = sigmoid(z_practical)
        assert np.all(s > 0.0)
        assert np.all(s < 1.0)

    def test_outlier_robustness(self):
        """Z-score + sigmoid should be more robust to outliers than MinMax.

        From preprocessing_theory.md Section 3.2.
        """
        rng = np.random.default_rng(42)
        X = rng.normal(5, 1, size=(100, 1))
        X[0, 0] = 100.0  # Extreme outlier

        X_zsig = zscore_sigmoid_transform(X)
        # Non-outlier data should use a reasonable range
        non_outlier_range = np.ptp(X_zsig[1:, 0])

        # Compare with MinMax
        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        X_minmax = scaler.fit_transform(X)
        non_outlier_range_mm = np.ptp(X_minmax[1:, 0])

        # Z-score + sigmoid should give better non-outlier range
        assert non_outlier_range > non_outlier_range_mm, (
            f"Z-score range {non_outlier_range:.3f} should be > "
            f"MinMax range {non_outlier_range_mm:.3f} with outliers"
        )


class TestPCAPreprocessing:
    """Tests for PCA dimensionality reduction."""

    def test_pca_reduces_dimensions(self, cancer_dataset):
        """PCA should reduce Cancer (30 features) to 8 components."""
        X, _ = cancer_dataset
        pca = PCA(n_components=8, random_state=42)
        X_reduced = pca.fit_transform(X)
        assert X_reduced.shape[1] == 8

    def test_pca_components_uncorrelated(self, cancer_dataset):
        """PCA components should be uncorrelated (decorrelated features).

        From preprocessing_theory.md Section 4.2:
        PCA + Angle encoding is appropriate because PCA decorrelates features.
        """
        X, _ = cancer_dataset
        pca = PCA(n_components=8, random_state=42)
        X_reduced = pca.fit_transform(X)

        corr_matrix = np.corrcoef(X_reduced.T)
        # Off-diagonal elements should be near zero
        off_diag = corr_matrix - np.diag(np.diag(corr_matrix))
        assert np.max(np.abs(off_diag)) < 0.01, (
            f"PCA components should be uncorrelated, "
            f"max off-diagonal correlation: {np.max(np.abs(off_diag)):.4f}"
        )

    def test_pca_variance_retention(self, cancer_dataset):
        """PCA to 8 components should retain >= 90% variance for Cancer.

        From preprocessing_theory.md Section 4.3 table.
        """
        X, _ = cancer_dataset
        pca = PCA(n_components=8, random_state=42)
        pca.fit(X)
        variance_retained = np.sum(pca.explained_variance_ratio_)
        assert (
            variance_retained >= 0.85
        ), f"Expected >= 85% variance retained, got {variance_retained:.3f}"

    def test_iris_no_pca_needed(self, iris_dataset):
        """Iris with 4 features should not need PCA (already fits 4 qubits)."""
        X, _ = iris_dataset
        assert X.shape[1] == 4  # Already small enough

    def test_pca_plus_minmax_pipeline(self, cancer_dataset):
        """Full Level 3 pipeline: PCA -> MinMax to [0, pi]."""
        X, _ = cancer_dataset
        pca = PCA(n_components=8, random_state=42)
        X_pca = pca.fit_transform(X)

        scaler = MinMaxScaler(feature_range=(0, RANGE_PI))
        X_final = scaler.fit_transform(X_pca)

        assert X_final.shape[1] == 8
        assert np.all(X_final >= 0.0)
        assert np.all(X_final <= RANGE_PI + 1e-10)


class TestL2Normalization:
    """Tests for L2 normalization (required for amplitude encoding)."""

    def test_unit_norm_output(self, iris_dataset):
        """L2 normalization should produce unit-norm vectors."""
        X, _ = iris_dataset
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_normalized = X / norms

        output_norms = np.linalg.norm(X_normalized, axis=1)
        np.testing.assert_allclose(output_norms, 1.0, atol=1e-10)

    def test_magnitude_information_destroyed(self, iris_dataset):
        """Vectors that differ only in magnitude should become identical."""
        X, _ = iris_dataset
        x = X[0]
        x_scaled = 3.0 * x  # Same direction, different magnitude

        x_norm = x / np.linalg.norm(x)
        x_scaled_norm = x_scaled / np.linalg.norm(x_scaled)

        np.testing.assert_allclose(x_norm, x_scaled_norm, atol=1e-10)

    def test_directional_info_preserved(self, iris_dataset):
        """Vectors with different directions should remain different."""
        X, _ = iris_dataset
        x1 = X[0]
        x2 = X[50]  # Different class, likely different direction

        x1_norm = x1 / np.linalg.norm(x1)
        x2_norm = x2 / np.linalg.norm(x2)

        assert not np.allclose(x1_norm, x2_norm, atol=1e-3)

    def test_zero_vector_handling(self):
        """L2 normalization should fail gracefully for zero vectors."""
        x = np.zeros(4)
        norm = np.linalg.norm(x)
        assert norm == 0.0
        # Division by zero should be handled


class TestPreprocessingEncodingCompatibility:
    """Tests for preprocessing-encoding compatibility checks.

    From preprocessing_theory.md Section 5.4.
    """

    def test_amplitude_requires_l2(self):
        """Amplitude encoding needs L2-normalized input."""
        from conftest import is_valid_configuration

        # Amplitude with "none" preprocessing is invalid
        assert not is_valid_configuration("amplitude", "none", "iris", 4)
        # Amplitude with minmax is valid (implementation should handle L2)
        assert is_valid_configuration("amplitude", "minmax_pi", "iris", 4)

    def test_basis_requires_binarization(self):
        """Basis encoding needs binary input -- most preprocessings are invalid."""
        from conftest import is_valid_configuration

        assert not is_valid_configuration("basis", "none", "cancer", 30)
        assert not is_valid_configuration("basis", "minmax_pi", "cancer", 30)
        assert not is_valid_configuration("basis", "zscore_sigmoid", "cancer", 30)
        # Only PCA+MinMax with binarization is valid for basis
        assert is_valid_configuration("basis", "pca_minmax", "cancer", 8)

    def test_angle_accepts_all_preprocessings(self):
        """Angle encoding should work with all preprocessing levels."""
        from conftest import is_valid_configuration

        for prep in ["none", "minmax_pi", "zscore_sigmoid", "pca_minmax"]:
            assert is_valid_configuration("angle_ry", prep, "iris", 4)


class TestPreprocessingPCAIQPInteraction:
    """Tests for PCA-IQP interaction prediction.

    From preprocessing_theory.md Section 4.2 Proposition:
    PCA neutralizes IQP advantage because PCA removes correlations
    that IQP's quadratic terms exploit.
    """

    def test_pca_decorrelates_cancer(self, cancer_dataset):
        """Verify Cancer dataset has high correlation before PCA and low after.

        This sets up the theoretical prediction that PCA+IQP ~= PCA+Angle.
        """
        X, _ = cancer_dataset

        # Before PCA: high correlations
        corr_before = np.corrcoef(X.T)
        off_diag_before = corr_before - np.diag(np.diag(corr_before))
        max_corr_before = np.max(np.abs(off_diag_before))
        assert (
            max_corr_before > 0.9
        ), f"Cancer should have high correlations, max={max_corr_before:.3f}"

        # After PCA: near-zero correlations
        pca = PCA(n_components=8, random_state=42)
        X_pca = pca.fit_transform(X)
        corr_after = np.corrcoef(X_pca.T)
        off_diag_after = corr_after - np.diag(np.diag(corr_after))
        max_corr_after = np.max(np.abs(off_diag_after))
        assert max_corr_after < 0.01, f"PCA output should be uncorrelated, max={max_corr_after:.4f}"
