"""Kernel-target alignment metric."""

import numpy as np


def kernel_target_alignment(
    K: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute the kernel-target alignment.

    A(K, Y) = <K_r, Y>_F / (||K_r||_F * ||Y||_F)

    where Y[i,j] = y_i * y_j (outer product of labels in {-1, +1})
    and K_r = 2*K - 1 rescales kernel entries from [0,1] to [-1,1].

    Args:
        K: Kernel (Gram) matrix of shape (n, n) with entries in [0, 1].
        y: Label vector of shape (n,) with values in {-1, +1} or
            {0, 1} (converted to {-1, +1} internally).

    Returns:
        Alignment score in [-1, 1].
    """
    y = np.asarray(y, dtype=float)
    # Convert {0, 1} labels to {-1, +1}
    if set(np.unique(y)).issubset({0.0, 1.0}):
        y = 2.0 * y - 1.0

    Y = np.outer(y, y)
    K = np.asarray(K, dtype=float)

    # Rescale K from [0, 1] to [-1, 1] so the perfect kernel
    # (same-class=1, different-class=0) maps to the ideal Y.
    K_r = 2.0 * K - 1.0

    numerator = np.sum(K_r * Y)
    denom = np.linalg.norm(K_r, "fro") * np.linalg.norm(Y, "fro")
    if denom == 0.0:
        return 0.0
    return float(numerator / denom)
