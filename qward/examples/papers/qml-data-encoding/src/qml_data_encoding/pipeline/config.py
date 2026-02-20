"""Experiment result dataclass."""

from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Container for a single experiment fold result.

    Attributes:
        dataset: Dataset name.
        encoding: Encoding method name.
        preprocessing: Preprocessing level name.
        fold: Cross-validation fold index.
        seed: Random seed used.
        train_accuracy: Training set accuracy.
        test_accuracy: Test set accuracy.
        f1_macro: Macro-averaged F1 score.
        convergence_iters: Number of optimizer iterations.
        final_loss: Final loss value.
        generalization_gap: train_accuracy - test_accuracy.
        circuit_depth: Transpiled circuit depth.
        gate_count: Total gate count.
        cx_count: Number of CX (CNOT) gates.
    """

    dataset: str
    encoding: str
    preprocessing: str
    fold: int
    seed: int
    train_accuracy: float
    test_accuracy: float
    f1_macro: float
    convergence_iters: int
    final_loss: float
    generalization_gap: float
    circuit_depth: int
    gate_count: int
    cx_count: int
