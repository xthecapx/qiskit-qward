"""
Coin-Toss (Ry rotation) implementation for qWard.

This module provides a simple Ry-rotation-based "coin toss" experiment.
Each qubit is independently rotated by Ry(theta), modelling a (possibly
biased) coin:

    Ry(theta) |0> = cos(theta/2) |0> + sin(theta/2) |1>

so the per-qubit probability of measuring |1> is sin^2(theta/2).

Two test modes are supported:

- "fair":   theta = pi/2 on every qubit  ->  uniform 1/2^n distribution.
            Used to validate hardware sampling uniformity / randomness.

- "biased": user-supplied theta (scalar applied to every qubit, or a
            per-qubit list).  Useful for "cheating" the coin in a
            controlled way and observing how the QPU recovers the
            expected biased distribution.

Scaling is achieved purely via num_qubits: each additional qubit adds
one more independent Ry rotation.  Depth-scaling layers can be added
later without changing the public API.
"""

import math
from typing import Dict, List, Sequence, Union

from qiskit import QuantumCircuit


# Default theta for fair coin: Ry(pi/2)|0> = (|0> + |1>) / sqrt(2)
FAIR_THETA = math.pi / 2

# Tolerated total-variation gap when treating outcomes as success modes
# in biased mode (probability difference vs. the most-probable outcome).
_BIASED_PEAK_TOL = 1e-9


def _coerce_thetas(
    num_qubits: int,
    theta: Union[float, Sequence[float]],
) -> List[float]:
    """Normalize ``theta`` to a per-qubit list of length ``num_qubits``."""
    if isinstance(theta, (int, float)):
        return [float(theta)] * num_qubits

    thetas = list(theta)
    if len(thetas) != num_qubits:
        raise ValueError(
            f"theta list length ({len(thetas)}) must match num_qubits ({num_qubits})"
        )
    return [float(t) for t in thetas]


def _prob_one(theta: float) -> float:
    """Per-qubit probability of measuring |1> after Ry(theta) on |0>."""
    return math.sin(theta / 2.0) ** 2


class CoinToss:
    """N-qubit Ry-rotation circuit (independent biased coin tosses).

    Each qubit ``q_i`` starts in |0> and is rotated by ``Ry(theta_i)``,
    so that ``P(q_i = 1) = sin^2(theta_i / 2)``.

    Args:
        num_qubits: Number of qubits (= number of coin tosses).
        theta: Rotation angle (radians).  Either a scalar applied to
            every qubit, or a list of length ``num_qubits`` for
            per-qubit angles.  Default: ``pi/2`` (fair coin).
        use_barriers: Whether to add a barrier before measurement
            (visualization only).

    Attributes:
        circuit: The quantum circuit (no measurement, mirrors QFT class).
        num_qubits: Number of qubits.
        thetas: Resolved per-qubit rotation angles.
    """

    def __init__(
        self,
        num_qubits: int,
        *,
        theta: Union[float, Sequence[float]] = FAIR_THETA,
        use_barriers: bool = True,
    ):
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")

        self.num_qubits = num_qubits
        self.thetas = _coerce_thetas(num_qubits, theta)
        self.use_barriers = use_barriers

        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Build the bare Ry rotation circuit (no measurement)."""
        qc = QuantumCircuit(self.num_qubits, name="CoinToss")
        for qubit, angle in enumerate(self.thetas):
            qc.ry(angle, qubit)
        if self.use_barriers:
            qc.barrier()
        return qc

    def draw(self, **kwargs):
        """Draw the circuit (defaults to matplotlib output)."""
        output = kwargs.pop("output", "mpl")
        return self.circuit.draw(output=output, **kwargs)


class CoinTossCircuitGenerator:
    """Test-circuit generator for the coin-toss experiment.

    Wraps :class:`CoinToss`, appends measurements, and exposes a
    ``success_criteria`` callback plus an ``expected_distribution`` so
    the experiment framework can compute DSR / TVD against the
    theoretical product distribution.

    Test modes:

    - ``"fair"``: theta = pi/2 on every qubit.  Expected distribution
      is uniform over all ``2**num_qubits`` outcomes.  Every outcome
      counts as a "success" (success_rate ~ 1.0); the meaningful
      diagnostic is DSR / TVD vs. the uniform distribution.

    - ``"biased"``: user-supplied theta(s).  Success is defined as
      "outcome equals the most-probable outcome (mode of the product
      distribution)".  ``expected_distribution`` returns the full
      analytic product distribution.

    Args:
        num_qubits: Number of qubits.
        test_mode: ``"fair"`` or ``"biased"``.
        theta: For ``"biased"`` mode, the rotation angle (scalar or
            per-qubit list).  Ignored for ``"fair"`` (forced to pi/2).
        use_barriers: Whether to add a barrier before measurement.

    Example (fair):
        >>> gen = CoinTossCircuitGenerator(num_qubits=3, test_mode="fair")
        >>> gen.success_criteria("010")  # True (any outcome is valid)
        True
        >>> gen.expected_distribution()["000"]
        0.125

    Example (biased):
        >>> import math
        >>> # P(1) = 0.25 per qubit  =>  theta = 2 * arcsin(sqrt(0.25)) = pi/3
        >>> gen = CoinTossCircuitGenerator(
        ...     num_qubits=3, test_mode="biased", theta=math.pi / 3
        ... )
        >>> gen.success_criteria("000")  # True (most probable outcome)
        True
    """

    VALID_MODES = ("fair", "biased")

    def __init__(
        self,
        num_qubits: int = 1,
        *,
        test_mode: str = "fair",
        theta: Union[float, Sequence[float], None] = None,
        use_barriers: bool = True,
    ):
        if num_qubits < 1:
            raise ValueError("num_qubits must be at least 1")
        if test_mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown test_mode: {test_mode}. Use one of {self.VALID_MODES}"
            )

        self.num_qubits = num_qubits
        self.test_mode = test_mode
        self.use_barriers = use_barriers

        if test_mode == "fair":
            # Ignore any user-supplied theta in fair mode.
            self.thetas = [FAIR_THETA] * num_qubits
        else:  # biased
            if theta is None:
                raise ValueError("theta must be specified for biased mode")
            self.thetas = _coerce_thetas(num_qubits, theta)

        self.coin_toss = CoinToss(
            num_qubits,
            theta=self.thetas,
            use_barriers=use_barriers,
        )

        self.circuit = self._create_test_circuit()

        # Pre-compute analytic distribution for success_criteria / DSR.
        self._distribution = self._compute_distribution()
        self._max_prob = max(self._distribution.values()) if self._distribution else 0.0
        self._mode_outcomes = {
            outcome
            for outcome, prob in self._distribution.items()
            if abs(prob - self._max_prob) <= _BIASED_PEAK_TOL
        }

    def _create_test_circuit(self) -> QuantumCircuit:
        """Wrap the bare coin-toss circuit with measurements."""
        qc = QuantumCircuit(self.num_qubits, name=f"CoinToss-{self.test_mode}")
        qc.compose(self.coin_toss.circuit, inplace=True)
        qc.measure_all()
        return qc

    def _compute_distribution(self) -> Dict[str, float]:
        """Analytic product distribution over all 2**n outcomes.

        Bitstrings are formatted with the Qiskit convention used
        elsewhere in QWARD (most-significant qubit on the left),
        matching what ``measure_all`` returns.
        """
        probs = [_prob_one(t) for t in self.thetas]
        distribution: Dict[str, float] = {}
        num_states = 2**self.num_qubits

        for value in range(num_states):
            bitstring = format(value, f"0{self.num_qubits}b")
            # Qiskit little-endian: rightmost char is qubit 0.
            prob = 1.0
            for i, bit in enumerate(reversed(bitstring)):
                p1 = probs[i]
                prob *= p1 if bit == "1" else (1.0 - p1)
            distribution[bitstring] = prob

        return distribution

    # =========================================================================
    # Public API expected by the experiment framework
    # =========================================================================

    def success_criteria(self, outcome: str) -> bool:
        """Determine whether a measurement outcome counts as success.

        - Fair mode: every outcome is "valid" (returns True).  The real
          diagnostic is DSR / TVD against the uniform distribution.
        - Biased mode: success = outcome is the most-probable outcome
          (mode of the product distribution).

        Args:
            outcome: Measurement bitstring (whitespace tolerated).

        Returns:
            True if the outcome counts as success under the current mode.
        """
        clean = outcome.replace(" ", "").strip()

        if self.test_mode == "fair":
            return len(clean) == self.num_qubits and all(b in "01" for b in clean)

        return clean in self._mode_outcomes

    def expected_distribution(self) -> Dict[str, float]:
        """Return the analytic expected probability distribution."""
        return dict(self._distribution)

    def expected_outcomes(self) -> List[str]:
        """Return outcomes used for DSR ('expected peaks').

        - Fair mode: all 2**n outcomes (uniform target).
        - Biased mode: the modal outcome(s).
        """
        if self.test_mode == "fair":
            return sorted(self._distribution.keys())
        return sorted(self._mode_outcomes)

    def random_chance(self) -> float:
        """Classical random-guess baseline matching success_criteria.

        - Fair mode: 1.0 (every outcome is success).
        - Biased mode: probability mass on the modal outcome(s) under
          a uniform classical guess => |modes| / 2**n.
        """
        if self.test_mode == "fair":
            return 1.0
        return len(self._mode_outcomes) / (2**self.num_qubits)

    def draw(self, **kwargs):
        """Draw the test circuit (defaults to matplotlib output)."""
        output = kwargs.pop("output", "mpl")
        return self.circuit.draw(output=output, **kwargs)

    def get_coin_toss(self) -> CoinToss:
        """Get the underlying CoinToss instance."""
        return self.coin_toss
