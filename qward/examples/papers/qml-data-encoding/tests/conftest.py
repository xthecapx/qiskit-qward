"""
Shared fixtures for QML Data Encoding tests.

Specification:
  - phase2_encoding_theory.md (encoding definitions, kernel formulas, gate counts)
  - phase2_expressibility_analysis.md (expressibility bounds, entanglement)
  - phase2_preprocessing_theory.md (normalization, PCA interaction)
  - phase2_experimental_design.md (experiment protocol, exclusion rules)
  - phase1_dataset_selection.md (8 datasets with profiles)
  - phase1_evaluation_framework.md (metrics, statistical tests)
Author: test-engineer

This module provides reusable pytest fixtures for all QML data encoding test
files, including sample datasets, encoding circuits, preprocessing pipelines,
and helper utilities.
"""

import os
import sys

# Set matplotlib backend before any imports that might use it
os.environ["MPLBACKEND"] = "Agg"

# Make conftest importable by test modules for shared constants/helpers
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Add src/ directory so ``from qml_data_encoding import ...`` works
_SRC_DIR = os.path.join(os.path.dirname(_TESTS_DIR), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import pytest
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants from Phase 2 theoretical design
# ---------------------------------------------------------------------------

# Normalization ranges used across experiments
RANGE_PI = np.pi
RANGE_2PI = 2 * np.pi

# Encoding names matching experimental design
ENCODING_NAMES = ["basis", "amplitude", "angle_ry", "iqp_full", "reuploading"]

# Preprocessing level names
PREPROCESSING_LEVELS = ["none", "minmax_pi", "zscore_sigmoid", "pca_minmax"]

# Fixed controlled variables from experimental design (Section 2.2)
ANSATZ_REPS = 2
OPTIMIZER_MAXITER = 200
SHOTS = 1024
N_FOLDS = 5
RANDOM_SEED = 42
RANDOM_SEEDS = [42, 43, 44, 45, 46]

# Expressibility computation parameters (Section 1.3 of expressibility analysis)
EXPR_N_PAIRS = 5000
EXPR_N_BINS = 75
EXPR_EPSILON = 1e-10

# Statistical test significance level
ALPHA = 0.05

# Qubit count formulas per encoding (Section 3.2.1 of evaluation framework)
QUBIT_FORMULAS = {
    "basis": lambda d: d,
    "amplitude": lambda d: int(np.ceil(np.log2(max(d, 2)))),
    "angle_ry": lambda d: d,
    "iqp_full": lambda d: d,
    "reuploading": lambda d: d,
}

# Expected gate counts for d=4 features (Section 7.1 of encoding theory)
# Format: (single_qubit_gates, two_qubit_gates, total_gates)
EXPECTED_GATE_COUNTS_D4 = {
    "basis": (4, 0, 4),
    "amplitude": (6, 2, 8),
    "angle_ry": (4, 0, 4),
    "iqp_full": (12, 12, 24),
    "reuploading_L1": (8, 3, 11),
    "reuploading_L2": (16, 6, 22),
    "reuploading_L3": (24, 9, 33),
}

# Expected gate counts for d=8 features (Section 7.1 of encoding theory)
EXPECTED_GATE_COUNTS_D8 = {
    "basis": (8, 0, 8),
    "amplitude": (14, 10, 24),
    "angle_ry": (8, 0, 8),
    "iqp_full": (24, 56, 80),
    "reuploading_L1": (16, 7, 23),
    "reuploading_L2": (32, 14, 46),
}


# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def iris_dataset():
    """Iris dataset: 4 features, 3 classes, 150 samples.

    Properties (from phase1_dataset_selection.md):
      - Balanced: 50/50/50
      - Distribution: approximately Gaussian per class
      - Intrinsic dimensionality: ~2 (97.8% variance in 2 PCA components)
      - Classical baseline: SVM ~97%
    """
    data = load_iris()
    return data.data, data.target


@pytest.fixture
def wine_dataset():
    """Wine dataset: 13 features, 3 classes, 178 samples.

    Properties:
      - Slightly imbalanced: 59/71/48
      - Feature scales vary by 2 orders of magnitude
      - Intrinsic dimensionality: ~5-6
      - Classical baseline: SVM ~98%
    """
    data = load_wine()
    return data.data, data.target


@pytest.fixture
def cancer_dataset():
    """Breast Cancer dataset: 30 features, 2 classes, 569 samples.

    Properties:
      - Slightly imbalanced: 212/357
      - High inter-feature correlations (many r > 0.9)
      - Intrinsic dimensionality: ~6-7
      - Classical baseline: SVM ~97%
    """
    data = load_breast_cancer()
    return data.data, data.target


@pytest.fixture
def iris_binary():
    """Binary Iris subset (classes 0 and 1 only) for simpler tests."""
    data = load_iris()
    mask = data.target < 2
    return data.data[mask], data.target[mask]


@pytest.fixture
def small_dataset_4d():
    """Small synthetic dataset with 4 features for encoding tests.

    Features are in [0, pi] range, ready for angle encoding.
    Two linearly separable classes.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_per_class = 25
    # Class 0: centered at (0.5, 0.5, 0.5, 0.5) * pi
    X0 = rng.normal(loc=0.5, scale=0.1, size=(n_per_class, 4)) * np.pi
    # Class 1: centered at (1.5, 1.5, 1.5, 1.5) * pi/2
    X1 = rng.normal(loc=1.0, scale=0.1, size=(n_per_class, 4)) * np.pi / 2
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    # Clip to [0, pi]
    X = np.clip(X, 0, np.pi)
    return X, y


@pytest.fixture
def small_dataset_8d():
    """Small synthetic dataset with 8 features for encoding tests.

    Features are in [0, pi] range, ready for angle encoding.
    Two classes with non-linear boundary.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_per_class = 25
    X0 = rng.uniform(0, np.pi / 2, size=(n_per_class, 8))
    X1 = rng.uniform(np.pi / 2, np.pi, size=(n_per_class, 8))
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


@pytest.fixture
def binary_dataset():
    """Small binary dataset suitable for basis encoding.

    Each feature is 0 or 1. 4 features, 2 classes.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = 40
    X = rng.integers(0, 2, size=(n, 4)).astype(float)
    # Label based on parity of first two bits
    y = ((X[:, 0] + X[:, 1]) % 2).astype(int)
    return X, y


@pytest.fixture
def unit_norm_dataset():
    """Dataset with L2-normalized vectors for amplitude encoding tests.

    4 features (maps to 2 qubits via amplitude encoding).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = 30
    X = rng.normal(size=(n, 4))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    y = (X[:, 0] > 0).astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Known data vectors for deterministic encoding tests
# ---------------------------------------------------------------------------


@pytest.fixture
def known_angle_vector():
    """Known 4D vector for angle encoding verification.

    x = [pi/4, pi/2, pi/3, pi/6]

    For Ry angle encoding, the state is:
      |phi(x)> = tensor_i (cos(xi/2)|0> + sin(xi/2)|1>)

    The kernel K(x, x') = prod_i cos^2((xi - xi')/2)
    """
    return np.array([np.pi / 4, np.pi / 2, np.pi / 3, np.pi / 6])


@pytest.fixture
def known_angle_vector_pair():
    """Pair of 4D vectors with known kernel value for angle encoding.

    x  = [pi/4, pi/2, pi/3, pi/6]
    x' = [pi/3, pi/4, pi/6, pi/2]

    K(x, x') = prod_i cos^2((xi - xi')/2)
    """
    x = np.array([np.pi / 4, np.pi / 2, np.pi / 3, np.pi / 6])
    x_prime = np.array([np.pi / 3, np.pi / 4, np.pi / 6, np.pi / 2])
    # Compute expected kernel
    diffs = x - x_prime
    expected_kernel = np.prod(np.cos(diffs / 2) ** 2)
    return x, x_prime, expected_kernel


@pytest.fixture
def known_amplitude_vector():
    """Known 4D vector for amplitude encoding verification.

    x = [1, 2, 3, 4] -> normalized: [1, 2, 3, 4] / sqrt(30)

    Amplitude encoding state: sum_i x_tilde_i |i>
    where x_tilde = x / ||x||
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    x_normalized = x / np.linalg.norm(x)
    return x, x_normalized


@pytest.fixture
def known_basis_vector():
    """Known binary vector for basis encoding verification.

    x = [1, 0, 1, 1] -> state |1011>

    The encoded state is |1>|0>|1>|1> in computational basis.
    Applying X gates to qubits 0, 2, 3.
    """
    return np.array([1.0, 0.0, 1.0, 1.0])


@pytest.fixture
def known_iqp_vector_2d():
    """Known 2D vector for IQP encoding verification.

    x = [pi/4, pi/3]

    IQP circuit:
      H|0> -> Rz(2*x1)|+> -> RZZ(2*x1*x2) -> H
    """
    return np.array([np.pi / 4, np.pi / 3])


@pytest.fixture
def zero_vector_4d():
    """Zero vector for edge case testing."""
    return np.zeros(4)


@pytest.fixture
def identical_vectors_4d():
    """Pair of identical vectors -- kernel should be 1.0 for all encodings."""
    x = np.array([np.pi / 4, np.pi / 2, np.pi / 3, np.pi / 6])
    return x, x.copy()


# ---------------------------------------------------------------------------
# Preprocessing fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minmax_scaler_pi():
    """MinMax scaler mapping features to [0, pi].

    From preprocessing theory, Level 1:
      T1(x)_j = (x_j - x_min) / (x_max - x_min) * pi
    """
    return MinMaxScaler(feature_range=(0, RANGE_PI))


@pytest.fixture
def minmax_scaler_2pi():
    """MinMax scaler mapping features to [0, 2*pi]."""
    return MinMaxScaler(feature_range=(0, RANGE_2PI))


@pytest.fixture
def standard_scaler():
    """Standard (Z-score) scaler for Level 2 preprocessing.

    Note: Z-score alone produces unbounded output.
    Must be followed by sigmoid mapping to (0, pi).
    """
    return StandardScaler()


@pytest.fixture
def pca_4():
    """PCA reducer to 4 components."""
    return PCA(n_components=4, random_state=RANDOM_SEED)


@pytest.fixture
def pca_8():
    """PCA reducer to 8 components."""
    return PCA(n_components=8, random_state=RANDOM_SEED)


# ---------------------------------------------------------------------------
# Preprocessing pipeline helpers
# ---------------------------------------------------------------------------


def sigmoid(z):
    """Sigmoid function: sigma(z) = 1 / (1 + exp(-z))."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def zscore_sigmoid_transform(X, scaler=None):
    """Z-score + sigmoid mapping to (0, pi).

    From preprocessing theory, Level 2:
      T2(x)_j = pi * sigmoid((x_j - mu_j) / sigma_j)

    Args:
        X: Input data array.
        scaler: Fitted StandardScaler. If None, fit a new one.

    Returns:
        Transformed data in (0, pi).
    """
    if scaler is None:
        scaler = StandardScaler()
        Z = scaler.fit_transform(X)
    else:
        Z = scaler.transform(X)
    return RANGE_PI * sigmoid(Z)


# ---------------------------------------------------------------------------
# Experiment configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_config_iris_angle():
    """Sample experiment configuration for Iris + Angle encoding.

    From experimental_design.md, Section 8.1.
    """
    return {
        "dataset": "iris",
        "encoding": "angle_ry",
        "preprocessing": "minmax_pi",
        "n_qubits": 4,
        "ansatz": "RealAmplitudes",
        "ansatz_reps": ANSATZ_REPS,
        "optimizer": "COBYLA",
        "maxiter": OPTIMIZER_MAXITER,
        "shots": SHOTS,
        "n_folds": N_FOLDS,
        "seeds": RANDOM_SEEDS,
        "pca_components": None,
        "subsample_size": None,
    }


@pytest.fixture
def experiment_config_cancer_iqp():
    """Sample experiment configuration for Cancer + IQP encoding with PCA."""
    return {
        "dataset": "cancer",
        "encoding": "iqp_full",
        "preprocessing": "pca_minmax",
        "n_qubits": 8,
        "ansatz": "RealAmplitudes",
        "ansatz_reps": ANSATZ_REPS,
        "optimizer": "COBYLA",
        "maxiter": OPTIMIZER_MAXITER,
        "shots": SHOTS,
        "n_folds": N_FOLDS,
        "seeds": RANDOM_SEEDS,
        "pca_components": 8,
        "subsample_size": 300,
    }


# ---------------------------------------------------------------------------
# Exclusion rules from experimental design (Section 3)
# ---------------------------------------------------------------------------

# Encoding-preprocessing exclusions
INVALID_ENCODING_PREPROCESSING = [
    ("basis", "none"),  # Basis requires binary input
    ("basis", "minmax_pi"),  # MinMax produces continuous output
    ("basis", "zscore_sigmoid"),  # Z-score produces continuous output
    ("amplitude", "none"),  # Amplitude requires normalized input
]

# Encoding-dataset exclusions (multi-class for basis)
INVALID_ENCODING_DATASET = [
    ("basis", "iris"),  # 3 classes
    ("basis", "wine"),  # 3 classes
    ("basis", "har"),  # 6 classes
]

# NISQ feasibility exclusions
NISQ_MAX_QUBITS = 20
NISQ_IQP_MAX_D = 8  # IQP full with d > 8 exceeds fidelity budget
NISQ_REUPLOAD_MAX_D = 8  # Re-uploading L=2 with d > 8 exceeds depth


def is_valid_configuration(encoding, preprocessing, dataset_name, n_features):
    """Check if an experiment configuration is valid per exclusion rules.

    Args:
        encoding: Encoding method name.
        preprocessing: Preprocessing level name.
        dataset_name: Dataset name.
        n_features: Number of features after preprocessing.

    Returns:
        bool: True if configuration is valid.
    """
    # Check encoding-preprocessing exclusions
    if (encoding, preprocessing) in INVALID_ENCODING_PREPROCESSING:
        return False

    # Check encoding-dataset exclusions
    if (encoding, dataset_name) in INVALID_ENCODING_DATASET:
        return False

    # Check qubit budget
    if encoding in QUBIT_FORMULAS:
        n_qubits = QUBIT_FORMULAS[encoding](n_features)
        if n_qubits > NISQ_MAX_QUBITS:
            return False

    # Check IQP feasibility
    if encoding == "iqp_full" and n_features > NISQ_IQP_MAX_D:
        return False

    # Check re-uploading feasibility
    if encoding == "reuploading" and n_features > NISQ_REUPLOAD_MAX_D:
        return False

    return True


# ---------------------------------------------------------------------------
# Tolerances and thresholds
# ---------------------------------------------------------------------------


@pytest.fixture
def kernel_rtol():
    """Relative tolerance for kernel value comparisons.

    For statevector simulation (exact), we expect numerical precision.
    """
    return 1e-10


@pytest.fixture
def kernel_atol_statistical():
    """Absolute tolerance for shot-based kernel estimation.

    With 10000 shots, kernel estimation has standard error ~ 1/sqrt(shots).
    """
    return 0.02


@pytest.fixture
def expressibility_rtol():
    """Relative tolerance for expressibility KL divergence.

    Expressibility is estimated from finite samples (5000 pairs).
    Allow 20% relative error.
    """
    return 0.20


@pytest.fixture
def entanglement_atol():
    """Absolute tolerance for Meyer-Wallach entanglement measure.

    For statevector simulation, MW should be exact up to numerical precision.
    """
    return 1e-10


# ---------------------------------------------------------------------------
# Pytest markers registration
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "noisy: marks tests that use noise models")
    config.addinivalue_line("markers", "statistical: marks tests that use statistical comparisons")
    config.addinivalue_line("markers", "classical_baseline: marks classical baseline tests")
