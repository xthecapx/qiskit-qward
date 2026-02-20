"""Expressibility metrics: KL divergence from Haar distribution."""

import numpy as np
from qiskit_aer import AerSimulator

# Epsilon to avoid log(0)
_EPS = 1e-10


def _make_encoding(encoding_name: str, n_features: int):
    """Instantiate an encoding by name."""
    from qml_data_encoding.encodings import (
        AngleEncoding,
        IQPEncoding,
        ReuploadingEncoding,
    )

    if encoding_name == "angle_ry":
        return AngleEncoding(n_features=n_features, rotation_axis="y")
    elif encoding_name == "iqp_full":
        return IQPEncoding(n_features=n_features, interaction="full")
    elif encoding_name == "reuploading":
        return ReuploadingEncoding(n_features=n_features, n_layers=2)
    else:
        raise ValueError(f"Unknown encoding: {encoding_name}")


def compute_fidelity_distribution(
    encoding_name: str,
    n_features: int,
    n_pairs: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """Sample fidelity distribution for a given encoding.

    Generates *n_pairs* pairs of random input vectors, encodes each,
    and computes the fidelity |<phi(x)|phi(x')>|^2.

    Args:
        encoding_name: Name of encoding ("angle_ry", "iqp_full", etc.).
        n_features: Number of features.
        n_pairs: Number of random pairs to sample.
        seed: Random seed.

    Returns:
        Array of fidelity values of shape (n_pairs,).
    """
    enc = _make_encoding(encoding_name, n_features)
    backend = AerSimulator(method="statevector")
    rng = np.random.default_rng(seed)

    fidelities = np.empty(n_pairs)
    for k in range(n_pairs):
        x = rng.uniform(0, 2 * np.pi, size=n_features)
        x_prime = rng.uniform(0, 2 * np.pi, size=n_features)

        sv1 = _get_statevector(enc, x, backend)
        sv2 = _get_statevector(enc, x_prime, backend)

        fidelities[k] = float(np.abs(np.dot(sv1.conj(), sv2)) ** 2)

    return fidelities


def compute_expressibility(
    encoding_name: str,
    n_features: int,
    n_pairs: int = 5000,
    n_bins: int = 75,
    seed: int = 42,
) -> float:
    """Compute expressibility as KL divergence from Haar distribution.

    Expr = KL(P_enc || P_Haar) where the distributions are estimated
    using histograms.

    Args:
        encoding_name: Name of encoding.
        n_features: Number of features.
        n_pairs: Number of random pairs.
        n_bins: Number of histogram bins.
        seed: Random seed.

    Returns:
        KL divergence (non-negative float).
    """
    fidelities = compute_fidelity_distribution(encoding_name, n_features, n_pairs, seed)

    enc = _make_encoding(encoding_name, n_features)
    n_qubits = enc.n_qubits
    dim = 2**n_qubits

    # Histogram of observed fidelities
    bin_edges = np.linspace(0, 1, n_bins + 1)
    counts, _ = np.histogram(fidelities, bins=bin_edges)
    P = counts.astype(float) / counts.sum()
    P = np.clip(P, _EPS, None)
    P = P / P.sum()

    # Haar-random fidelity distribution: P_Haar(F) = (dim-1)*(1-F)^{dim-2}
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]
    Q = (dim - 1) * (1 - bin_centers) ** (dim - 2) * bin_width
    Q = np.clip(Q, _EPS, None)
    Q = Q / Q.sum()

    # KL divergence
    kl = float(np.sum(P * np.log(P / Q)))
    return max(kl, 0.0)


def _get_statevector(enc, x, backend):
    """Get statevector from encoding."""
    circuit = enc.encode(x)
    circuit.save_statevector()
    result = backend.run(circuit).result()
    return result.get_statevector().data
