"""Phase 5 Task #26: Run encoding comparison experiments (~125 configurations).

Implements the full experimental protocol from Phase 2:
- 5 encodings x 4 preprocessors x 8 datasets (minus exclusions)
- 5-fold stratified cross-validation
- VQC with RealAmplitudes(reps=2), COBYLA(maxiter=200)
- Statevector simulation for efficiency
- Classical baselines (SVM, RF, LR)
- Circuit metrics, expressibility, MW entanglement, KTA
"""

import os
import sys
import json
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

os.environ["MPLBACKEND"] = "Agg"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure library imports work
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_BASE_DIR, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RealAmplitudes

try:
    from qiskit.circuit.library import real_amplitudes as real_amplitudes_fn
except ImportError:
    real_amplitudes_fn = None
from qiskit_aer import AerSimulator

from qml_data_encoding.encodings import (
    AngleEncoding,
    IQPEncoding,
    ReuploadingEncoding,
    AmplitudeEncoding,
    BasisEncoding,
)
from qml_data_encoding.metrics import (
    compute_expressibility,
    meyer_wallach_from_statevector,
    kernel_target_alignment,
)

# Paths
RESULTS_DIR = os.path.join(_BASE_DIR, "results", "encoding_comparison")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Statevector backend
SV_BACKEND = AerSimulator(method="statevector")

# ---- Dataset loaders (same as run_data_profiles.py) ----

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits


def load_heart_disease():
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    data = fetch_openml("heart-statlog", version=1, as_frame=False, parser="auto")
    le = LabelEncoder()
    y = le.fit_transform(data.target)
    return data.data.astype(float), y


def load_mnist_subset():
    digits = load_digits()
    mask = (digits.target == 0) | (digits.target == 1)
    return digits.data[mask], digits.target[mask]


def load_credit_fraud():
    rng = np.random.default_rng(42)
    n_normal, n_fraud, d = 1900, 60, 28
    X_normal = rng.standard_normal((n_normal, d))
    X_fraud = rng.standard_t(df=3, size=(n_fraud, d)) * 2.0 + rng.uniform(-1, 1, d)
    X = np.vstack([X_normal, X_fraud])
    y = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx].astype(int)


def load_nsl_kdd():
    rng = np.random.default_rng(42)
    n, d = 2000, 40
    X = np.zeros((n, d))
    y = np.zeros(n, dtype=int)
    sizes = [800, 400, 400, 200, 200]
    idx = 0
    for c, size in enumerate(sizes):
        center = rng.standard_normal(d) * (c + 1)
        X[idx : idx + size] = rng.standard_normal((size, d)) * 0.5 + center
        y[idx : idx + size] = c
        idx += size
    for j in range(30, 40):
        X[:, j] = (rng.random(n) > 0.7).astype(float)
    return X, y


def load_har():
    rng = np.random.default_rng(42)
    n_per_class, n_classes, d = 300, 6, 561
    n = n_per_class * n_classes
    X = np.zeros((n, d))
    y = np.zeros(n, dtype=int)
    for c in range(n_classes):
        s, e = c * n_per_class, (c + 1) * n_per_class
        center = np.zeros(d)
        center[:20] = rng.standard_normal(20) * 3
        X[s:e] = rng.standard_normal((n_per_class, d)) * 0.3 + center
        y[s:e] = c
    return X, y


DATASETS = {
    "iris": lambda: (load_iris().data, load_iris().target),
    "wine": lambda: (load_wine().data, load_wine().target),
    "breast_cancer": lambda: (load_breast_cancer().data, load_breast_cancer().target),
    "mnist_01": load_mnist_subset,
    "credit_fraud": load_credit_fraud,
    "nsl_kdd": load_nsl_kdd,
    "har": load_har,
    "heart_disease": load_heart_disease,
}


# ---- Preprocessing ----


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def make_preprocessor(name, n_features, n_target_qubits, seed=42):
    """Create preprocessing pipeline returning (fit_fn, transform_fn, n_out)."""
    pca = None
    scaler = None
    n_out = n_features

    if name == "none":
        pass
    elif name == "minmax_pi":
        scaler = MinMaxScaler(feature_range=(0, np.pi))
    elif name == "zscore_sigmoid":
        scaler = StandardScaler()
    elif name == "pca_minmax":
        n_components = min(n_features, n_target_qubits)
        pca = PCA(n_components=n_components, random_state=seed)
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        n_out = n_components

    def fit(X_train):
        data = X_train.copy()
        if pca is not None:
            pca.fit(data)
            data = pca.transform(data)
        if scaler is not None:
            scaler.fit(data)

    def transform(X):
        data = X.copy()
        if pca is not None:
            data = pca.transform(data)
        if scaler is not None:
            data = scaler.transform(data)
            if name == "zscore_sigmoid":
                data = np.pi * sigmoid(data)
        return data

    return fit, transform, n_out


# ---- Encoding + VQC ----


def make_encoding(enc_name, n_qubits):
    """Create encoding object."""
    if enc_name == "angle_ry":
        return AngleEncoding(n_features=n_qubits, rotation_axis="y")
    elif enc_name == "iqp_full":
        return IQPEncoding(n_features=n_qubits, interaction="full")
    elif enc_name == "reuploading":
        return ReuploadingEncoding(n_features=n_qubits, n_layers=2)
    elif enc_name == "amplitude":
        return AmplitudeEncoding(n_features=n_qubits)
    elif enc_name == "basis":
        return BasisEncoding(n_features=n_qubits)
    else:
        raise ValueError(f"Unknown encoding: {enc_name}")


def get_statevector(circuit):
    """Get statevector from a circuit."""
    qc = circuit.copy()
    qc.save_statevector()
    result = SV_BACKEND.run(qc).result()
    return result.get_statevector().data


def _make_ansatz(n_qubits):
    """Create and return a RealAmplitudes ansatz, decomposed for simulation."""
    if real_amplitudes_fn is not None:
        ansatz = real_amplitudes_fn(n_qubits, reps=2)
    else:
        ansatz = RealAmplitudes(n_qubits, reps=2)
    return ansatz


def build_vqc_circuit(encoding, x, ansatz_params, n_qubits, n_classes, enc_theta=None):
    """Build full VQC: encoding + RealAmplitudes ansatz."""
    enc_circuit = encoding.encode(x, theta=enc_theta)
    ansatz = _make_ansatz(n_qubits)
    # Bind parameters
    param_dict = dict(zip(ansatz.parameters, ansatz_params))
    bound_ansatz = ansatz.assign_parameters(param_dict)

    # Decompose to base gates so Aer can simulate
    bound_ansatz = bound_ansatz.decompose()

    full = QuantumCircuit(n_qubits)
    full.compose(enc_circuit, inplace=True)
    full.compose(bound_ansatz, inplace=True)
    return full


def predict_proba_sv(encoding, x, ansatz_params, n_qubits, n_classes, enc_theta=None):
    """Get class probabilities from statevector for a single sample."""
    circuit = build_vqc_circuit(encoding, x, ansatz_params, n_qubits, n_classes, enc_theta)
    sv = get_statevector(circuit)
    probs = np.abs(sv) ** 2

    if n_classes == 2:
        # Binary: P(class=1) = sum of probs where qubit 0 = |1>
        p1 = sum(probs[i] for i in range(len(probs)) if i & 1)
        return np.array([1 - p1, p1])
    else:
        # Multi-class: assign bitstrings to classes via modulo
        class_probs = np.zeros(n_classes)
        for i, p in enumerate(probs):
            class_probs[i % n_classes] += p
        return class_probs


def vqc_loss(params, encoding, X_batch, y_batch, n_qubits, n_classes, enc_theta=None):
    """Cross-entropy loss over a mini-batch."""
    loss = 0.0
    for x, y_true in zip(X_batch, y_batch):
        probs = predict_proba_sv(encoding, x, params, n_qubits, n_classes, enc_theta)
        probs = np.clip(probs, 1e-10, 1.0)
        loss -= np.log(probs[int(y_true)])
    return loss / len(X_batch)


def train_vqc(encoding, X_train, y_train, n_qubits, n_classes, maxiter=200, batch_size=20, seed=42):
    """Train VQC using COBYLA with mini-batch sampling."""
    rng = np.random.default_rng(seed)
    n_ansatz_params = 3 * n_qubits  # RealAmplitudes reps=2

    # For reuploading, we also have encoding trainable params
    enc_theta = None
    if hasattr(encoding, "n_trainable_params"):
        enc_theta = rng.uniform(-np.pi, np.pi, encoding.n_trainable_params)

    theta0 = rng.uniform(-np.pi, np.pi, n_ansatz_params)
    loss_history = []
    n_train = len(X_train)

    def objective(params):
        idx = rng.choice(n_train, size=min(batch_size, n_train), replace=False)
        X_batch = X_train[idx]
        y_batch = y_train[idx]
        loss = vqc_loss(params, encoding, X_batch, y_batch, n_qubits, n_classes, enc_theta)
        loss_history.append(float(loss))
        return loss

    result = minimize(
        objective, theta0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.5}
    )

    return result.x, enc_theta, loss_history


def evaluate_vqc(encoding, params, X_test, y_test, n_qubits, n_classes, enc_theta=None):
    """Evaluate trained VQC on test set."""
    preds = []
    for x in X_test:
        probs = predict_proba_sv(encoding, x, params, n_qubits, n_classes, enc_theta)
        preds.append(np.argmax(probs))
    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    return acc, f1, preds


def get_circuit_metrics(encoding, x_sample, n_qubits, enc_theta=None):
    """Compute circuit depth, gate count, CX count for encoding+ansatz."""
    ansatz = _make_ansatz(n_qubits)
    rng = np.random.default_rng(42)
    param_dict = dict(zip(ansatz.parameters, rng.uniform(-np.pi, np.pi, len(ansatz.parameters))))
    bound_ansatz = ansatz.assign_parameters(param_dict).decompose()

    enc_circuit = encoding.encode(x_sample, theta=enc_theta)
    full = QuantumCircuit(n_qubits)
    full.compose(enc_circuit, inplace=True)
    full.compose(bound_ansatz, inplace=True)

    tc = transpile(full, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1)
    depth = tc.depth()
    gate_count = tc.size()
    cx_count = tc.count_ops().get("cx", 0)
    return depth, gate_count, cx_count


# ---- Exclusion rules ----


def is_valid_config(enc_name, preproc_name, dataset_name, n_features, n_classes):
    """Check if a configuration is valid per Phase 2 exclusion rules."""
    # Basis encoding requires binary input - only valid with binarization
    if enc_name == "basis":
        # Only PCA+MinMax can binarize (via thresholding) - skip for now
        # Basis is excluded from continuous data except with binarization
        if preproc_name != "pca_minmax":
            return False
        # Basis + multi-class excluded
        if n_classes > 2:
            return False

    # Amplitude encoding requires normalized input
    if enc_name == "amplitude" and preproc_name == "none":
        return False

    # High-d without PCA: exceeds NISQ budget
    if preproc_name != "pca_minmax" and n_features > 20:
        if enc_name in ("angle_ry", "iqp_full", "reuploading"):
            return False

    # IQP/Reuploading with d>8 (even after PCA to 8): feasibility check
    # PCA reduces to min(n_features, 8), so after PCA these are OK
    if preproc_name != "pca_minmax" and n_features > 8:
        if enc_name in ("iqp_full", "reuploading"):
            return False

    return True


# ---- Classical baselines ----


def run_classical_baselines(X_train, y_train, X_test, y_test):
    """Run SVM, RF, LR baselines. Return dict of accuracies and F1s."""
    results = {}
    models = {
        "svm_rbf": SVC(kernel="rbf", random_state=42, max_iter=5000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=42),
    }
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[f"{name}_acc"] = accuracy_score(y_test, preds)
            results[f"{name}_f1"] = f1_score(y_test, preds, average="macro", zero_division=0)
        except Exception:
            results[f"{name}_acc"] = 0.0
            results[f"{name}_f1"] = 0.0
    return results


# ---- Main experiment ----

ENCODINGS = ["angle_ry", "iqp_full", "reuploading", "amplitude", "basis"]
PREPROCESSINGS = ["none", "minmax_pi", "zscore_sigmoid", "pca_minmax"]
N_FOLDS = 5
SEEDS = [42, 43, 44, 45, 46]
N_TARGET_QUBITS = 4  # Primary analysis at d=4
MAXITER = 200
BATCH_SIZE = 20


def run_single_config(enc_name, preproc_name, dataset_name, X, y, n_classes):
    """Run one encoding-preprocessing-dataset configuration with 5-fold CV."""
    n_features = X.shape[1]

    # Determine qubit count
    if preproc_name == "pca_minmax":
        n_qubits = min(n_features, N_TARGET_QUBITS)
    else:
        n_qubits = min(n_features, 8)

    # For amplitude encoding, n_features for the AmplitudeEncoding constructor
    # is the number of features being encoded; it computes n_qubits internally
    if enc_name == "amplitude":
        n_feat_eff = n_qubits  # features after preprocessing
        # AmplitudeEncoding computes: n_qubits = ceil(log2(n_feat_eff))
        encoding = AmplitudeEncoding(n_features=n_feat_eff)
        n_qubits = encoding.n_qubits
        n_features_eff = n_feat_eff
    elif enc_name == "basis":
        n_qubits = min(n_features, N_TARGET_QUBITS)
        encoding = make_encoding(enc_name, n_qubits)
        n_features_eff = n_qubits
    else:
        encoding = make_encoding(enc_name, n_qubits)
        n_features_eff = n_qubits

    fold_results = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        seed = SEEDS[fold]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Preprocessing
        fit_fn, transform_fn, n_out = make_preprocessor(
            preproc_name, n_features, N_TARGET_QUBITS, seed=seed
        )
        fit_fn(X_train)
        X_train_proc = transform_fn(X_train)
        X_test_proc = transform_fn(X_test)

        # Trim/pad features to match encoding expectation
        if enc_name == "basis":
            # Binarize: threshold at median
            threshold = np.median(X_train_proc, axis=0)
            X_train_proc = (X_train_proc > threshold).astype(float)
            X_test_proc = (X_test_proc > threshold).astype(float)
            X_train_proc = X_train_proc[:, :n_features_eff]
            X_test_proc = X_test_proc[:, :n_features_eff]
        elif enc_name == "amplitude":
            # Amplitude encoding: trim to n_features_eff features
            # AmplitudeEncoding handles L2 normalization + padding internally
            X_train_proc = X_train_proc[:, :n_features_eff]
            X_test_proc = X_test_proc[:, :n_features_eff]
        else:
            X_train_proc = X_train_proc[:, :n_features_eff]
            X_test_proc = X_test_proc[:, :n_features_eff]

        t0 = time.time()

        # Train VQC
        try:
            params, enc_theta, loss_history = train_vqc(
                encoding,
                X_train_proc,
                y_train,
                n_qubits,
                n_classes,
                maxiter=MAXITER,
                batch_size=BATCH_SIZE,
                seed=seed,
            )
        except Exception as e:
            print(f"    [WARN] Training failed: {e}")
            params, enc_theta, loss_history = None, None, []

        wall_time = time.time() - t0

        # Evaluate
        if params is not None:
            train_acc, train_f1, _ = evaluate_vqc(
                encoding,
                params,
                X_train_proc[:50],
                y_train[:50],
                n_qubits,
                n_classes,
                enc_theta,
            )
            test_acc, test_f1, _ = evaluate_vqc(
                encoding,
                params,
                X_test_proc,
                y_test,
                n_qubits,
                n_classes,
                enc_theta,
            )
        else:
            train_acc = test_acc = 0.0
            train_f1 = test_f1 = 0.0

        # Circuit metrics (once per config, fold 0)
        if fold == 0 and params is not None:
            x_sample = X_train_proc[0]
            try:
                depth, gate_count, cx_count = get_circuit_metrics(
                    encoding, x_sample, n_qubits, enc_theta
                )
            except Exception:
                depth = gate_count = cx_count = -1
        elif fold == 0:
            depth = gate_count = cx_count = -1

        # Convergence: iterations to 90% of final loss reduction
        if loss_history:
            final_loss = loss_history[-1]
            initial_loss = loss_history[0]
            target = initial_loss - 0.9 * (initial_loss - final_loss)
            conv_iters = len(loss_history)
            for i, l in enumerate(loss_history):
                if l <= target:
                    conv_iters = i + 1
                    break
        else:
            final_loss = float("nan")
            conv_iters = MAXITER

        gen_gap = train_acc - test_acc

        # Classical baselines (once per fold)
        classical = run_classical_baselines(X_train_proc, y_train, X_test_proc, y_test)

        row = {
            "dataset": dataset_name,
            "encoding": enc_name,
            "preprocessing": preproc_name,
            "fold": fold,
            "seed": seed,
            "n_qubits": n_qubits,
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "f1_macro": round(test_f1, 4),
            "convergence_iters": conv_iters,
            "final_loss": round(final_loss, 6) if not np.isnan(final_loss) else None,
            "generalization_gap": round(gen_gap, 4),
            "wall_time_seconds": round(wall_time, 2),
        }

        if fold == 0:
            row["circuit_depth"] = depth
            row["gate_count"] = gate_count
            row["cx_count"] = cx_count
        else:
            row["circuit_depth"] = None
            row["gate_count"] = None
            row["cx_count"] = None

        row.update(classical)
        fold_results.append(row)

        print(
            f"    Fold {fold}: test_acc={test_acc:.3f}, f1={test_f1:.3f}, "
            f"time={wall_time:.1f}s, "
            f"classical_best={max(classical.get('svm_rbf_acc',0), classical.get('random_forest_acc',0), classical.get('logistic_regression_acc',0)):.3f}"
        )

    return fold_results


def compute_encoding_metrics(enc_name, n_qubits, X_subset, y_subset):
    """Compute expressibility, MW entanglement, and KTA for an encoding."""
    metrics = {}

    # Expressibility (only for encodings that support it)
    if enc_name in ("angle_ry", "iqp_full", "reuploading"):
        try:
            expr = compute_expressibility(enc_name, n_qubits, n_pairs=500, n_bins=50, seed=42)
            metrics["expressibility"] = round(expr, 6)
        except Exception:
            metrics["expressibility"] = None
    else:
        metrics["expressibility"] = None

    # Meyer-Wallach (average over a few samples)
    if enc_name != "basis" and n_qubits >= 2:
        enc = make_encoding(enc_name, n_qubits)
        mw_vals = []
        for x in X_subset[:10]:
            try:
                mw = enc.meyer_wallach(x)
                mw_vals.append(mw)
            except Exception:
                pass
        metrics["entanglement_mw"] = round(float(np.mean(mw_vals)), 4) if mw_vals else None
    else:
        metrics["entanglement_mw"] = None

    # Kernel-target alignment (on small subset)
    if enc_name != "basis" and len(X_subset) >= 5:
        enc = make_encoding(enc_name, n_qubits)
        try:
            K = enc.kernel_matrix(X_subset[:30])
            kta = kernel_target_alignment(K, y_subset[:30])
            metrics["kernel_alignment"] = round(kta, 4)
        except Exception:
            metrics["kernel_alignment"] = None
    else:
        metrics["kernel_alignment"] = None

    return metrics


def main():
    print("=" * 70)
    print("Phase 5 Task #26: Encoding Comparison Experiments")
    print("=" * 70)

    all_results = []
    config_count = 0
    valid_count = 0
    excluded_count = 0

    for dataset_name, loader in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        X, y = loader()
        n_features = X.shape[1]
        n_classes = len(np.unique(y))
        print(f"  Shape: {X.shape}, Classes: {n_classes}")

        for enc_name in ENCODINGS:
            for preproc_name in PREPROCESSINGS:
                config_count += 1

                if not is_valid_config(enc_name, preproc_name, dataset_name, n_features, n_classes):
                    excluded_count += 1
                    continue

                valid_count += 1
                print(f"\n  Config #{valid_count}: {enc_name} + {preproc_name}")

                try:
                    fold_results = run_single_config(
                        enc_name, preproc_name, dataset_name, X, y, n_classes
                    )

                    # Compute encoding-specific metrics (once per config)
                    if preproc_name == "pca_minmax":
                        n_q = min(n_features, N_TARGET_QUBITS)
                    else:
                        n_q = min(n_features, 8)

                    # Prepare a small subset for metrics
                    fit_fn, transform_fn, _ = make_preprocessor(
                        preproc_name, n_features, N_TARGET_QUBITS, seed=42
                    )
                    fit_fn(X[:100])
                    X_sub = transform_fn(X[:30])
                    if enc_name == "basis":
                        threshold = np.median(X_sub, axis=0)
                        X_sub = (X_sub > threshold).astype(float)
                        X_sub = X_sub[:, :n_q]
                    elif enc_name == "amplitude":
                        X_sub = X_sub[:, :n_q]
                        # n_q for metrics = actual encoding qubits
                        n_q = int(np.ceil(np.log2(max(n_q, 2))))
                    else:
                        X_sub = X_sub[:, :n_q]

                    enc_metrics = compute_encoding_metrics(enc_name, n_q, X_sub, y[:30])

                    # Add encoding metrics to fold results
                    for row in fold_results:
                        row.update(enc_metrics)

                    all_results.extend(fold_results)

                except Exception as e:
                    print(f"    [ERROR] {e}")
                    import traceback

                    traceback.print_exc()

                # Save intermediate results
                if valid_count % 5 == 0:
                    df = pd.DataFrame(all_results)
                    df.to_csv(os.path.join(RESULTS_DIR, "encoding_comparison.csv"), index=False)
                    print(f"  [Checkpoint] Saved {len(all_results)} rows")

    # Final save
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, "encoding_comparison.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total configurations checked: {config_count}")
    print(f"Valid configurations run: {valid_count}")
    print(f"Excluded configurations: {excluded_count}")
    print(f"Total fold results: {len(all_results)}")
    print(f"Results saved: {csv_path}")

    # Summary stats
    if len(all_results) > 0:
        df_agg = (
            df.groupby(["dataset", "encoding", "preprocessing"])
            .agg(
                test_acc_mean=("test_accuracy", "mean"),
                test_acc_std=("test_accuracy", "std"),
                f1_mean=("f1_macro", "mean"),
                f1_std=("f1_macro", "std"),
            )
            .reset_index()
        )
        summary_path = os.path.join(RESULTS_DIR, "summary_by_config.csv")
        df_agg.to_csv(summary_path, index=False)
        print(f"Summary saved: {summary_path}")

        print("\nTop 10 configurations by mean test accuracy:")
        top10 = df_agg.nlargest(10, "test_acc_mean")
        print(top10.to_string(index=False))

    print(f"\nTask #26 complete.")


if __name__ == "__main__":
    main()
