"""Meyer-Wallach entanglement measure for quantum states."""

import numpy as np


def meyer_wallach_from_statevector(
    sv: np.ndarray,
    n_qubits: int,
) -> float:
    """Compute the Meyer-Wallach entanglement measure.

    Q(|psi>) = (2/n) sum_k (1 - tr(rho_k^2))

    where rho_k is the reduced density matrix of qubit k obtained by
    tracing out all other qubits.

    Args:
        sv: Statevector as a 1D complex array of length 2^n_qubits.
        n_qubits: Number of qubits.

    Returns:
        Meyer-Wallach measure in [0, 1].
    """
    sv = np.asarray(sv, dtype=complex)
    dim = 2**n_qubits

    total_linear_entropy = 0.0
    for k in range(n_qubits):
        rho_k = _reduced_density_matrix(sv, n_qubits, k)
        purity = np.real(np.trace(rho_k @ rho_k))
        total_linear_entropy += 1.0 - purity

    return float(2.0 / n_qubits * total_linear_entropy)


def _reduced_density_matrix(
    sv: np.ndarray,
    n_qubits: int,
    qubit: int,
) -> np.ndarray:
    """Compute the reduced density matrix for a single qubit.

    Args:
        sv: Full statevector of length 2^n_qubits.
        n_qubits: Total number of qubits.
        qubit: Index of the qubit to keep (trace out the rest).

    Returns:
        2x2 density matrix for the specified qubit.
    """
    dim = 2**n_qubits
    rho = np.zeros((2, 2), dtype=complex)

    for i in range(dim):
        for j in range(dim):
            # Extract bit value of target qubit
            bi = (i >> qubit) & 1
            bj = (j >> qubit) & 1
            # Check if all OTHER qubits match between i and j
            mask = dim - 1  # all bits
            mask ^= 1 << qubit  # remove target qubit bit
            if (i & mask) == (j & mask):
                rho[bi, bj] += sv[i] * np.conj(sv[j])

    return rho
