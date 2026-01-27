"""
Noise Model Generator for QWARD Experiments
============================================

This module provides a unified, research-justified noise model generator for
quantum circuit simulations. All noise parameters are based on empirical data
from current NISQ hardware (superconducting qubits, trapped ions) and published
literature.

Noise Model Justification
-------------------------

The noise parameters used in this module are grounded in empirical measurements
from quantum hardware providers and peer-reviewed research. Below is a summary
of typical error rates observed in current NISQ devices:

+----------------------+------------------+------------------+---------------------------+
| Error Type           | Low (Best-case)  | Medium (Typical) | High (Worst-case)         |
+======================+==================+==================+===========================+
| Single-qubit gate    | 0.01% - 0.05%    | 0.1% - 0.5%      | 0.5% - 1%                 |
| (depolarizing p1)    | (1e-4 to 5e-4)   | (1e-3 to 5e-3)   | (5e-3 to 1e-2)            |
+----------------------+------------------+------------------+---------------------------+
| Two-qubit gate       | 0.1% - 0.5%      | 0.5% - 2%        | 2% - 10%                  |
| (depolarizing p2)    | (1e-3 to 5e-3)   | (5e-3 to 2e-2)   | (2e-2 to 1e-1)            |
+----------------------+------------------+------------------+---------------------------+
| Readout error        | 0.1% - 0.5%      | 1% - 2%          | 2% - 5%                   |
| (p01, p10)           | (1e-3 to 5e-3)   | (1e-2 to 2e-2)   | (2e-2 to 5e-2)            |
+----------------------+------------------+------------------+---------------------------+

Hardware References (January 2026)
----------------------------------

**IBM Heron Processors (quantum.cloud.ibm.com/computers):**

Live calibration data from IBM Quantum Platform [11]:

+------------------+------------+-------------------+------------------------+
| QPU              | Type       | 2Q Error (median) | Readout Error (median) |
+==================+============+===================+========================+
| ibm_boston       | Heron r3   | 0.113% (1.13e-3)  | 0.46% (4.6e-3)         |
+------------------+------------+-------------------+------------------------+
| ibm_pittsburgh   | Heron r3   | 0.155% (1.55e-3)  | 0.42% (4.2e-3)         |
+------------------+------------+-------------------+------------------------+
| ibm_kingston     | Heron r2   | 0.195% (1.95e-3)  | 0.95% (9.5e-3)         |
+------------------+------------+-------------------+------------------------+
| ibm_torino       | Heron r1   | 0.247% (2.47e-3)  | 2.93% (2.93e-2)        |
+------------------+------------+-------------------+------------------------+
| ibm_marrakesh    | Heron r2   | 0.263% (2.63e-3)  | 1.27% (1.27e-2)        |
+------------------+------------+-------------------+------------------------+
| ibm_fez          | Heron r2   | 0.277% (2.77e-3)  | 1.00% (1.0e-2)         |
+------------------+------------+-------------------+------------------------+

Key observations:
- Two-qubit gate error rates approaching 0.1% on Heron r3 [12]
- Readout errors range from 0.4% (best) to 3% (worst)
- Daily calibrations ensure T1/T2 and gate fidelities are optimized [13]

**Rigetti Ankaa Processors (qcs.rigetti.com/qpus):**

- Ankaa-3 (84 qubits): 99.5% median two-qubit gate fidelity (0.5% error) [14]
- Significant improvement from previous generations

**Google Sycamore:**

- Single-qubit gate fidelity: ~99.85% (error 0.15%) [2]
- Two-qubit gate fidelity: ~99.4% (error 0.6%)

**Trapped Ion Systems (IonQ, Quantinuum):**

- Single-qubit gate fidelity: 99.99%+ (error < 0.01%)
  - Best reported: 99.9999% (error ~1e-6) [7]

- Two-qubit gate fidelity: 99% - 99.9% (error 0.1% - 1%)
  - IonQ Aria: ~99.4% two-qubit fidelity [8]

- SPAM (State Prep and Measurement): 99.5% - 99.99%
  - IonQ: ~99.6% readout fidelity [8]

Literature References
---------------------

[1] IBM Quantum. "Characterizing errors on qubit operations via iterative
    randomized benchmarking." IBM Research (2016).
    https://research.ibm.com/publications/characterizing-errors-on-qubit-operations-via-iterative-randomized-benchmarking

[2] Arute et al. "Quantum supremacy using a programmable superconducting
    processor." Nature 574, 505-510 (2019). https://www.nature.com/articles/s41586-019-1666-5

[3] IBM Research. "Benchmarking the noise sensitivity of different parametric
    two-qubit gates in a single superconducting quantum computing platform."
    https://research.ibm.com/publications/benchmarking-the-noise-sensitivity-of-different-parametric-two-qubit-gates-in-a-single-superconducting-quantum-computing-platform

[4] Geller & Zhou. "Efficient error models for fault-tolerant architectures."
    Nature Scientific Reports 3, 14670 (2013). https://arxiv.org/abs/1305.2021

[5] Swiadek et al. "Enhancing Dispersive Readout of Superconducting Qubits."
    arXiv:2307.07765 (2023). Achieved 0.25% readout error in 100ns.

[6] Lienhard et al. "Model-based Optimization of Superconducting Qubit Readout."
    arXiv:2308.02079 (2023). ~1.5% error for 17-qubit simultaneous readout.

[7] Wang et al. "High-fidelity gates on atomic qubits." Nature 620, 734-740 (2023).
    Single-qubit error below 1e-4 in trapped ions.

[8] IonQ. "IonQ Aria System Specifications." (2024).
    https://ionq.com/quantum-systems/aria

[9] Escofet et al. "An Accurate Efficient Analytic Model of Fidelity under
    Depolarizing Noise." arXiv:2503.06693 (2025).

[10] Tomita & Svore. "Low-distance surface codes under realistic quantum noise."
     Physical Review A 90, 062320 (2014).

[11] IBM Quantum Platform. "Compute Resources - QPU Specifications."
     https://quantum.cloud.ibm.com/computers
     Live calibration data accessed January 2026.

[12] IBM Research. "Noise characterization and error mitigation on IBM Heron
     processors: Part 1." APS Global Physics Summit 2025.
     https://research.ibm.com/publications/noise-characterization-and-error-mitigation-on-ibm-heron-processors-part-1--1
     Reports two-qubit gate error rates approaching 0.1% on Heron.

[13] IBM Quantum Documentation. "Calibration jobs."
     https://quantum.cloud.ibm.com/docs/en/guides/calibration-jobs
     Details daily/hourly calibration procedures for T1/T2, gate fidelities.

[14] The Quantum Insider. "Rigetti Computing Reports 84-Qubit Ankaa-3 System
     Achieves 99.5% Median Two-Qubit Gate Fidelity Milestone." (Dec 2024).
     https://thequantuminsider.com/2024/12/23/rigetti-computing-reports-84-qubit-ankaa-3-system-achieves-99-5-median-two-qubit-gate-fidelity-milestone/

[15] Rigetti QCS. "QPU Specifications."
     https://qcs.rigetti.com/qpus

Noise Model Types
-----------------

1. **Depolarizing Noise**: Models uniform random errors after gates. Each Pauli
   error (X, Y, Z) occurs with equal probability p/3. This is the most common
   generic error model as it captures gate infidelity in a worst-case manner.

2. **Pauli Noise**: Models asymmetric errors where different Pauli operators
   have different probabilities. Useful when hardware exhibits bias (e.g.,
   phase-flip errors dominate over bit-flip in superconducting qubits).

3. **Readout Error**: Models classical bit-flip during measurement. Characterized
   by p01 (0→1 flip probability) and p10 (1→0 flip probability). Often the
   dominant error source in NISQ algorithms.

4. **Combined Noise**: Realistic model combining depolarizing gate errors with
   readout errors. Best approximation of actual NISQ device behavior.

5. **Thermal Noise**: Approximates T1/T2 relaxation using depolarizing channel.
   Error probability scales as gate_time/T1, capturing decoherence effects.

6. **Mixed Noise**: Combines depolarizing (single-qubit) with Pauli (two-qubit)
   errors plus readout. Useful for exploring heterogeneous error scenarios.

Usage Example
-------------

    from qward.algorithms import NoiseModelGenerator, NoiseConfig, PRESET_NOISE_CONFIGS

    # Using preset configurations
    config = PRESET_NOISE_CONFIGS["DEP-MED"]  # Medium depolarizing noise
    noise_model = NoiseModelGenerator.create_from_config(config)

    # Direct creation with custom parameters
    noise_model = NoiseModelGenerator.create_depolarizing(p1=0.005, p2=0.02)

    # Combined realistic noise
    noise_model = NoiseModelGenerator.create_combined(
        p1=0.01,      # 1% single-qubit error
        p2=0.02,      # 2% two-qubit error
        p_readout=0.015  # 1.5% readout error
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Iterable

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error


# =============================================================================
# Default Gate Sets
# =============================================================================
# These gate sets cover the common gates used in quantum circuits.
# Single-qubit gates include standard rotations and Clifford gates.
# Two-qubit gates include controlled operations and SWAP.

DEFAULT_SINGLE_QUBIT_GATES = (
    "u1",
    "u2",
    "u3",  # Universal single-qubit gates
    "h",  # Hadamard
    "x",
    "y",
    "z",  # Pauli gates
    "s",
    "t",  # Phase gates (S = sqrt(Z), T = sqrt(S))
    "rx",
    "ry",
    "rz",  # Rotation gates
    "p",  # Phase gate (generalized)
)

DEFAULT_TWO_QUBIT_GATES = (
    "cx",
    "cy",
    "cz",  # Controlled Pauli gates
    "cp",  # Controlled phase (used in QFT)
    "swap",  # SWAP gate
)


# =============================================================================
# NoiseConfig Dataclass
# =============================================================================


@dataclass
class NoiseConfig:
    """
    Configuration for noise model generation.

    This dataclass encapsulates all parameters needed to create a noise model,
    enabling serialization and systematic experimentation across noise regimes.

    Attributes:
        noise_id: Unique identifier for this noise configuration.
        noise_type: Type of noise model. One of:
            - "none": No noise (ideal simulation)
            - "depolarizing": Uniform depolarizing channel on gates
            - "pauli": Asymmetric Pauli error channel
            - "readout": Measurement errors only
            - "combined": Depolarizing + readout errors
            - "thermal": T1/T2 relaxation approximation
            - "mixed": Depolarizing (1Q) + Pauli (2Q) + readout
        parameters: Dictionary of noise parameters. Keys depend on noise_type:
            - depolarizing: {"p1": float, "p2": float}
            - pauli: {"pX": float, "pY": float, "pZ": float}
            - readout: {"p01": float, "p10": float}
            - combined: {"p1": float, "p2": float, "p_readout": float}
            - thermal: {"T1": float, "T2": float, "gate_time": float}
            - mixed: {"noise_level": float}
        description: Human-readable description of the noise configuration.

    Example:
        >>> config = NoiseConfig(
        ...     noise_id="REALISTIC",
        ...     noise_type="combined",
        ...     parameters={"p1": 0.005, "p2": 0.015, "p_readout": 0.02},
        ...     description="Realistic IBM-like noise"
        ... )
    """

    noise_id: str
    noise_type: str
    parameters: Dict[str, float] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary for JSON serialization."""
        return {
            "noise_id": self.noise_id,
            "noise_type": self.noise_type,
            "parameters": self.parameters,
            "description": self.description,
        }


# =============================================================================
# NoiseModelGenerator Class
# =============================================================================


class NoiseModelGenerator:
    """
    Factory for creating Qiskit Aer noise models with research-justified parameters.

    This class provides static methods for creating various noise models used in
    quantum circuit simulation. All default parameters are based on empirical
    measurements from NISQ hardware (see module docstring for references).

    The noise models support both preset configurations (via NoiseConfig) and
    direct parameter specification for fine-grained control.

    Supported Noise Types:
        - Depolarizing: Uniform random Pauli errors
        - Pauli: Asymmetric X, Y, Z errors (bias-aware)
        - Readout: Measurement bit-flip errors
        - Combined: Gate depolarizing + readout errors
        - Thermal: T1/T2 relaxation approximation
        - Mixed: Heterogeneous error model
    """

    @staticmethod
    def create_from_config(config: NoiseConfig) -> Optional[NoiseModel]:
        """
        Create a noise model from a NoiseConfig object.

        This is the primary entry point for creating noise models from
        configuration objects, enabling systematic experimentation.

        Args:
            config: NoiseConfig specifying the noise type and parameters.

        Returns:
            NoiseModel instance, or None if noise_type is "none".

        Raises:
            ValueError: If the noise_type is not recognized.

        Example:
            >>> from qward.algorithms import NoiseConfig, NoiseModelGenerator
            >>> config = NoiseConfig("TEST", "depolarizing", {"p1": 0.01, "p2": 0.02})
            >>> noise_model = NoiseModelGenerator.create_from_config(config)
        """
        noise_type = (config.noise_type or "none").lower()
        params = config.parameters or {}

        if noise_type == "none":
            return None

        if noise_type == "depolarizing":
            return NoiseModelGenerator.create_depolarizing(
                p1=params.get("p1", 0.01),
                p2=params.get("p2", 0.05),
            )

        if noise_type == "pauli":
            return NoiseModelGenerator.create_pauli(
                px=params.get("pX", params.get("px", 0.01)),
                py=params.get("pY", params.get("py", 0.01)),
                pz=params.get("pZ", params.get("pz", 0.01)),
            )

        if noise_type == "readout":
            return NoiseModelGenerator.create_readout(
                p01=params.get("p01", 0.02),
                p10=params.get("p10", 0.02),
            )

        if noise_type == "combined":
            return NoiseModelGenerator.create_combined(
                p1=params.get("p1", 0.01),
                p2=params.get("p2", 0.05),
                p_readout=params.get("p_readout", 0.02),
            )

        if noise_type == "thermal":
            return NoiseModelGenerator.create_thermal(
                t1=params.get("T1", params.get("t1", 50e-6)),
                t2=params.get("T2", params.get("t2", 70e-6)),
                gate_time=params.get("gate_time", 50e-9),
            )

        if noise_type == "mixed":
            return NoiseModelGenerator.create_mixed(
                noise_level=params.get("noise_level", 0.05),
            )

        raise ValueError(f"Unknown noise type: {config.noise_type}")

    @staticmethod
    def create_by_type(noise_type: str, noise_level: float = 0.05) -> Optional[NoiseModel]:
        """
        Create a noise model by simple type string and uniform noise level.

        This is a convenience method for quick noise model creation when
        detailed parameter control is not needed.

        Args:
            noise_type: Type of noise ("none", "depolarizing", "pauli",
                       "mixed", "readout", "combined").
            noise_level: Base noise probability (default 0.05 = 5%).
                        Two-qubit errors are scaled to 2x this value.

        Returns:
            NoiseModel instance, or None if noise_type is None or "none".

        Raises:
            ValueError: If the noise_type is not recognized.
        """
        if noise_type is None:
            return None
        noise_type = noise_type.lower()

        if noise_type == "none":
            return None
        if noise_type == "depolarizing":
            return NoiseModelGenerator.create_depolarizing(p1=noise_level, p2=noise_level * 2)
        if noise_type == "pauli":
            prob = noise_level / 3
            return NoiseModelGenerator.create_pauli(px=prob, py=prob, pz=prob)
        if noise_type == "mixed":
            return NoiseModelGenerator.create_mixed(noise_level=noise_level)
        if noise_type == "readout":
            return NoiseModelGenerator.create_readout(p01=noise_level, p10=noise_level)
        if noise_type == "combined":
            return NoiseModelGenerator.create_combined(
                p1=noise_level, p2=noise_level * 2, p_readout=noise_level
            )

        raise ValueError(f"Unknown noise model type: {noise_type}")

    @staticmethod
    def create_depolarizing(
        p1: float = 0.01,
        p2: float = 0.05,
        *,
        single_qubit_gates: Iterable[str] = DEFAULT_SINGLE_QUBIT_GATES,
        two_qubit_gates: Iterable[str] = DEFAULT_TWO_QUBIT_GATES,
    ) -> NoiseModel:
        """
        Create a depolarizing noise model.

        The depolarizing channel models uniform random errors where each Pauli
        operator (X, Y, Z) is applied with equal probability p/3. This is the
        most commonly used generic error model in quantum computing research.

        **Parameter Justification:**

        - p1 (single-qubit): Default 0.01 (1%) represents moderate NISQ hardware.
          State-of-the-art superconducting qubits achieve 0.01%-0.1% (1e-4 to 1e-3).
          Reference: IBM reports ~99.9% single-qubit fidelity [1].

        - p2 (two-qubit): Default 0.05 (5%) is conservative for NISQ devices.
          Typical two-qubit gate errors are 0.5%-2% for CX/CZ gates.
          Reference: CZ gates show 0.9%-1.3% error rates [3].

        Args:
            p1: Single-qubit depolarizing error probability.
                Typical range: 1e-4 (best) to 1e-2 (worst).
            p2: Two-qubit depolarizing error probability.
                Typical range: 1e-3 (best) to 1e-1 (worst).
            single_qubit_gates: Gates to apply single-qubit noise to.
            two_qubit_gates: Gates to apply two-qubit noise to.

        Returns:
            NoiseModel with depolarizing errors on specified gates.

        Example:
            >>> # Realistic IBM-like noise
            >>> noise = NoiseModelGenerator.create_depolarizing(p1=0.001, p2=0.01)
            >>> # High-noise stress test
            >>> noise = NoiseModelGenerator.create_depolarizing(p1=0.05, p2=0.10)
        """
        noise_model = NoiseModel()

        depol_1q = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(depol_1q, list(single_qubit_gates))

        depol_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, list(two_qubit_gates))

        return noise_model

    @staticmethod
    def create_pauli(
        px: float = 0.01,
        py: float = 0.01,
        pz: float = 0.01,
        *,
        single_qubit_gates: Iterable[str] = DEFAULT_SINGLE_QUBIT_GATES,
        two_qubit_gates: Iterable[str] = DEFAULT_TWO_QUBIT_GATES,
    ) -> NoiseModel:
        """
        Create a Pauli noise model with asymmetric error probabilities.

        Unlike depolarizing noise, Pauli noise allows different probabilities
        for X (bit-flip), Y (bit+phase flip), and Z (phase-flip) errors. This
        is useful for modeling hardware with known error bias.

        **Parameter Justification:**

        Many superconducting qubit systems exhibit phase-flip (Z) bias, where
        dephasing errors dominate over bit-flip errors. The Pauli Twirling
        Approximation (PTA) shows that Pauli channels closely approximate
        general error channels for threshold analysis [4].

        Typical bias ratios:
        - Symmetric (transmon): pX ≈ pY ≈ pZ
        - Z-biased (flux qubits): pZ >> pX, pY
        - X-biased (cat qubits): pX >> pY, pZ

        Args:
            px: X (bit-flip) error probability.
            py: Y (bit+phase flip) error probability.
            pz: Z (phase-flip) error probability.
            single_qubit_gates: Gates to apply single-qubit noise to.
            two_qubit_gates: Gates to apply two-qubit noise to.

        Returns:
            NoiseModel with Pauli errors on specified gates.

        Example:
            >>> # Symmetric Pauli noise
            >>> noise = NoiseModelGenerator.create_pauli(px=0.01, py=0.01, pz=0.01)
            >>> # Z-biased noise (dephasing dominant)
            >>> noise = NoiseModelGenerator.create_pauli(px=0.005, py=0.005, pz=0.02)
        """
        noise_model = NoiseModel()

        # Single-qubit Pauli channel
        pi = max(0.0, 1 - px - py - pz)
        pauli_1q = pauli_error([("X", px), ("Y", py), ("Z", pz), ("I", pi)])
        noise_model.add_all_qubit_quantum_error(pauli_1q, list(single_qubit_gates))

        # Two-qubit Pauli channel (correlated errors)
        pi2 = max(0.0, 1 - px - py - pz)
        pauli_2q = pauli_error([("XX", px), ("YY", py), ("ZZ", pz), ("II", pi2)])
        noise_model.add_all_qubit_quantum_error(pauli_2q, list(two_qubit_gates))

        return noise_model

    @staticmethod
    def create_readout(p01: float = 0.02, p10: float = 0.02) -> NoiseModel:
        """
        Create a readout (measurement) error model.

        Readout errors model classical bit-flip during measurement, where a
        qubit in state |0⟩ is incorrectly measured as |1⟩ (p01) or vice versa
        (p10). Readout error is often the dominant error source in NISQ devices.

        **Parameter Justification:**

        - p01 (0→1 error): Default 0.02 (2%) is typical for superconducting qubits.
        - p10 (1→0 error): Default 0.02 (2%) is typical for superconducting qubits.

        Empirical measurements:
        - Typical superconducting: 1%-5% readout error [4]
        - Optimized readout: 0.25%-1.5% error [5, 6]
        - Trapped ions: 0.1%-0.5% SPAM error [8]

        Note: In real hardware, p01 and p10 are often asymmetric due to
        T1 decay (excited state decays to ground during measurement).

        Args:
            p01: Probability of measuring |1⟩ when state is |0⟩.
                 Typical range: 0.005 (optimized) to 0.05 (typical).
            p10: Probability of measuring |0⟩ when state is |1⟩.
                 Typical range: 0.005 (optimized) to 0.05 (typical).
                 Often higher than p01 due to T1 relaxation.

        Returns:
            NoiseModel with readout errors on all qubits.

        Example:
            >>> # Typical readout error
            >>> noise = NoiseModelGenerator.create_readout(p01=0.02, p10=0.02)
            >>> # Asymmetric (T1-influenced) readout
            >>> noise = NoiseModelGenerator.create_readout(p01=0.01, p10=0.03)
        """
        noise_model = NoiseModel()

        # Confusion matrix: [[P(0|0), P(1|0)], [P(0|1), P(1|1)]]
        readout_err = ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
        noise_model.add_all_qubit_readout_error(readout_err)

        return noise_model

    @staticmethod
    def create_combined(
        p1: float = 0.01,
        p2: float = 0.05,
        p_readout: float = 0.02,
        *,
        single_qubit_gates: Iterable[str] = DEFAULT_SINGLE_QUBIT_GATES,
        two_qubit_gates: Iterable[str] = DEFAULT_TWO_QUBIT_GATES,
    ) -> NoiseModel:
        """
        Create a combined depolarizing + readout noise model.

        This is the most realistic noise model for NISQ simulations, combining
        gate errors (depolarizing) with measurement errors (readout). This
        captures both coherent evolution errors and classical readout mistakes.

        **Parameter Justification:**

        The combined model uses typical NISQ values:
        - p1 = 0.01 (1%): Moderate single-qubit gate error
        - p2 = 0.05 (5%): Conservative two-qubit gate error
        - p_readout = 0.02 (2%): Typical measurement error

        For more realistic IBM-like behavior, use:
        - p1 = 0.001-0.005, p2 = 0.01-0.02, p_readout = 0.01-0.02

        Args:
            p1: Single-qubit depolarizing error probability.
            p2: Two-qubit depolarizing error probability.
            p_readout: Symmetric readout error probability (p01 = p10).
            single_qubit_gates: Gates to apply single-qubit noise to.
            two_qubit_gates: Gates to apply two-qubit noise to.

        Returns:
            NoiseModel with depolarizing gate errors and readout errors.

        Example:
            >>> # Realistic NISQ noise
            >>> noise = NoiseModelGenerator.create_combined(
            ...     p1=0.005, p2=0.015, p_readout=0.02
            ... )
        """
        noise_model = NoiseModel()

        # Gate errors (depolarizing)
        depol_1q = depolarizing_error(p1, 1)
        noise_model.add_all_qubit_quantum_error(depol_1q, list(single_qubit_gates))

        depol_2q = depolarizing_error(p2, 2)
        noise_model.add_all_qubit_quantum_error(depol_2q, list(two_qubit_gates))

        # Readout errors (symmetric)
        readout_err = ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        noise_model.add_all_qubit_readout_error(readout_err)

        return noise_model

    @staticmethod
    def create_thermal(
        t1: float = 50e-6,
        t2: float = 70e-6,
        gate_time: float = 50e-9,
        *,
        single_qubit_gates: Iterable[str] = DEFAULT_SINGLE_QUBIT_GATES,
    ) -> NoiseModel:
        """
        Create a thermal noise model approximating T1/T2 relaxation.

        This model approximates decoherence effects using a depolarizing channel
        with error probability proportional to gate_time / T1. While simplified
        (true T1/T2 requires amplitude/phase damping channels), this captures
        the scaling behavior of thermal relaxation.

        **Parameter Justification:**

        Typical coherence times for superconducting qubits:
        - T1 (energy relaxation): 50-200 μs
        - T2 (dephasing): 50-100 μs (T2 ≤ 2*T1)
        - Gate time: 20-100 ns for single-qubit gates

        Error rate approximation: p ≈ gate_time / T1
        - For T1=50μs, gate_time=50ns: p ≈ 0.001 (0.1%)

        Note: For accurate T1/T2 modeling, use Qiskit's thermal_relaxation_error
        which implements proper amplitude and phase damping channels.

        Args:
            t1: T1 relaxation time in seconds. Default 50μs.
            t2: T2 dephasing time in seconds. Default 70μs.
                (Currently unused; included for future extension)
            gate_time: Gate duration in seconds. Default 50ns.
            single_qubit_gates: Gates to apply thermal noise to.

        Returns:
            NoiseModel with thermal-approximated depolarizing errors.

        Example:
            >>> # Good coherence (long T1)
            >>> noise = NoiseModelGenerator.create_thermal(t1=100e-6, gate_time=30e-9)
            >>> # Poor coherence (short T1)
            >>> noise = NoiseModelGenerator.create_thermal(t1=20e-6, gate_time=100e-9)
        """
        noise_model = NoiseModel()

        if t1 <= 0:
            p_thermal = 0.0
        else:
            p_thermal = min(1.0, gate_time / t1)

        depol_thermal = depolarizing_error(p_thermal, 1)
        noise_model.add_all_qubit_quantum_error(depol_thermal, list(single_qubit_gates))

        return noise_model

    @staticmethod
    def create_mixed(
        noise_level: float = 0.05,
        *,
        single_qubit_gates: Iterable[str] = DEFAULT_SINGLE_QUBIT_GATES,
        two_qubit_gates: Iterable[str] = DEFAULT_TWO_QUBIT_GATES,
    ) -> NoiseModel:
        """
        Create a mixed noise model combining multiple error types.

        This model uses:
        - Depolarizing errors for single-qubit gates
        - Pauli errors for two-qubit gates
        - Readout errors for measurement

        The mixed model is useful for exploring heterogeneous error scenarios
        where different gate types have different error characteristics.

        Args:
            noise_level: Base noise probability (default 0.05 = 5%).
                        Applied uniformly to 1Q depolarizing and readout.
                        Pauli 2Q errors split as noise_level/3 per Pauli type.
            single_qubit_gates: Gates to apply depolarizing noise to.
            two_qubit_gates: Gates to apply Pauli noise to.

        Returns:
            NoiseModel with mixed error types.

        Example:
            >>> # Moderate mixed noise
            >>> noise = NoiseModelGenerator.create_mixed(noise_level=0.05)
        """
        noise_model = NoiseModel()

        # Single-qubit: depolarizing
        depol_1q = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(depol_1q, list(single_qubit_gates))

        # Two-qubit: Pauli (symmetric)
        prob = noise_level / 3
        pi2 = max(0.0, 1 - noise_level)
        pauli_2q = pauli_error([("XX", prob), ("YY", prob), ("ZZ", prob), ("II", pi2)])
        noise_model.add_all_qubit_quantum_error(pauli_2q, list(two_qubit_gates))

        # Readout
        readout_err = ReadoutError([[1 - noise_level, noise_level], [noise_level, 1 - noise_level]])
        noise_model.add_all_qubit_readout_error(readout_err)

        return noise_model


# =============================================================================
# Preset Noise Configurations
# =============================================================================
# These presets cover a range of noise regimes from near-ideal to worst-case,
# enabling systematic study of algorithm performance under varying noise.
#
# Naming convention:
#   - IDEAL: No noise
#   - *-LOW: Near state-of-the-art hardware (optimistic)
#   - *-MED: Typical NISQ hardware (realistic baseline)
#   - *-HIGH: Challenging conditions (stress test)

PRESET_NOISE_CONFIGS: Dict[str, NoiseConfig] = {
    # =========================================================================
    # Ideal (no noise)
    # =========================================================================
    "IDEAL": NoiseConfig(
        noise_id="IDEAL",
        noise_type="none",
        parameters={},
        description="No noise - ideal quantum simulation",
    ),
    # =========================================================================
    # Depolarizing Noise Presets
    # =========================================================================
    # Based on gate fidelity data from IBM, Google, Rigetti processors.
    # p1: single-qubit error, p2: two-qubit error
    "DEP-LOW": NoiseConfig(
        noise_id="DEP-LOW",
        noise_type="depolarizing",
        parameters={"p1": 0.001, "p2": 0.005},
        description=(
            "Low depolarizing (p1=0.1%, p2=0.5%) - "
            "Near state-of-the-art superconducting qubits. "
            "Ref: IBM Eagle ~99.9% 1Q fidelity"
        ),
    ),
    "DEP-MED": NoiseConfig(
        noise_id="DEP-MED",
        noise_type="depolarizing",
        parameters={"p1": 0.005, "p2": 0.02},
        description=(
            "Medium depolarizing (p1=0.5%, p2=2%) - "
            "Typical NISQ hardware baseline. "
            "Ref: Average IBM/Google device performance"
        ),
    ),
    "DEP-HIGH": NoiseConfig(
        noise_id="DEP-HIGH",
        noise_type="depolarizing",
        parameters={"p1": 0.01, "p2": 0.05},
        description=(
            "High depolarizing (p1=1%, p2=5%) - "
            "Challenging NISQ conditions for stress testing. "
            "Ref: Older/noisy qubits or high crosstalk"
        ),
    ),
    # =========================================================================
    # Readout Error Presets
    # =========================================================================
    # Based on measurement fidelity from literature [5, 6].
    # p01: 0→1 flip, p10: 1→0 flip
    "READ-LOW": NoiseConfig(
        noise_id="READ-LOW",
        noise_type="readout",
        parameters={"p01": 0.005, "p10": 0.005},
        description=(
            "Low readout error (0.5%) - "
            "Optimized readout with ML post-processing. "
            "Ref: arXiv:2307.07765 achieved 0.25%"
        ),
    ),
    "READ-MED": NoiseConfig(
        noise_id="READ-MED",
        noise_type="readout",
        parameters={"p01": 0.015, "p10": 0.02},
        description=(
            "Medium readout error (1.5-2%) - "
            "Typical superconducting qubit readout. "
            "Ref: arXiv:2308.02079 ~1.5% for 17-qubit readout"
        ),
    ),
    "READ-HIGH": NoiseConfig(
        noise_id="READ-HIGH",
        noise_type="readout",
        parameters={"p01": 0.03, "p10": 0.05},
        description=(
            "High readout error (3-5%) - "
            "Asymmetric (T1-influenced) worst-case. "
            "Ref: Fast readout or poor signal-to-noise"
        ),
    ),
    # =========================================================================
    # Combined Noise Presets (Gate + Readout)
    # =========================================================================
    # Most realistic for NISQ simulation studies.
    "COMB-LOW": NoiseConfig(
        noise_id="COMB-LOW",
        noise_type="combined",
        parameters={"p1": 0.001, "p2": 0.005, "p_readout": 0.01},
        description=(
            "Low combined noise - " "Best-case NISQ scenario for near-term advantage studies"
        ),
    ),
    "COMB-MED": NoiseConfig(
        noise_id="COMB-MED",
        noise_type="combined",
        parameters={"p1": 0.005, "p2": 0.015, "p_readout": 0.02},
        description=("Medium combined noise - " "Realistic IBM/Google-like performance baseline"),
    ),
    "COMB-HIGH": NoiseConfig(
        noise_id="COMB-HIGH",
        noise_type="combined",
        parameters={"p1": 0.01, "p2": 0.05, "p_readout": 0.03},
        description=("High combined noise - " "Stress test for algorithm fault tolerance"),
    ),
    # =========================================================================
    # Pauli Noise Presets
    # =========================================================================
    # For studying error bias (Z-dominated dephasing common in superconducting).
    "PAULI-SYM": NoiseConfig(
        noise_id="PAULI-SYM",
        noise_type="pauli",
        parameters={"pX": 0.01, "pY": 0.01, "pZ": 0.01},
        description=("Symmetric Pauli noise (1% each) - " "No error bias, uniform X/Y/Z errors"),
    ),
    "PAULI-ZBIAS": NoiseConfig(
        noise_id="PAULI-ZBIAS",
        noise_type="pauli",
        parameters={"pX": 0.005, "pY": 0.005, "pZ": 0.02},
        description=(
            "Z-biased Pauli noise - "
            "Phase-flip dominant (typical superconducting). "
            "Ref: Nature Sci Rep 10.1038/srep14670"
        ),
    ),
    # =========================================================================
    # Thermal Noise Presets
    # =========================================================================
    # Based on typical T1/T2 coherence times.
    "THERMAL-GOOD": NoiseConfig(
        noise_id="THERMAL-GOOD",
        noise_type="thermal",
        parameters={"T1": 100e-6, "T2": 100e-6, "gate_time": 30e-9},
        description=(
            "Good coherence (T1=100μs, gate=30ns) - " "Modern IBM/Google qubits. Error ~0.03%"
        ),
    ),
    "THERMAL-POOR": NoiseConfig(
        noise_id="THERMAL-POOR",
        noise_type="thermal",
        parameters={"T1": 30e-6, "T2": 40e-6, "gate_time": 100e-9},
        description=(
            "Poor coherence (T1=30μs, gate=100ns) - " "Older or noisy qubits. Error ~0.3%"
        ),
    ),
    # =========================================================================
    # Hardware-Specific Presets (Based on Real Calibration Data)
    # =========================================================================
    # These presets match actual QPU specifications from provider dashboards.
    # Data source: quantum.cloud.ibm.com/computers, qcs.rigetti.com/qpus
    # --- IBM Heron r3 (Best current IBM hardware) ---
    "IBM-HERON-R3": NoiseConfig(
        noise_id="IBM-HERON-R3",
        noise_type="combined",
        parameters={"p1": 0.0005, "p2": 0.00115, "p_readout": 0.0046},
        description=(
            "IBM Heron r3 (ibm_boston) - "
            "2Q error: 0.113%, readout: 0.46%. "
            "Ref: quantum.cloud.ibm.com Jan 2026 [11]"
        ),
    ),
    # --- IBM Heron r2 (Typical current IBM hardware) ---
    "IBM-HERON-R2": NoiseConfig(
        noise_id="IBM-HERON-R2",
        noise_type="combined",
        parameters={"p1": 0.001, "p2": 0.0026, "p_readout": 0.0127},
        description=(
            "IBM Heron r2 (ibm_marrakesh) - "
            "2Q error: 0.26%, readout: 1.27%. "
            "Ref: quantum.cloud.ibm.com Jan 2026 [11]"
        ),
    ),
    # --- IBM Heron r1 (Older Heron generation) ---
    "IBM-HERON-R1": NoiseConfig(
        noise_id="IBM-HERON-R1",
        noise_type="combined",
        parameters={"p1": 0.001, "p2": 0.0025, "p_readout": 0.0293},
        description=(
            "IBM Heron r1 (ibm_torino) - "
            "2Q error: 0.25%, readout: 2.93%. "
            "Ref: quantum.cloud.ibm.com Jan 2026 [11]"
        ),
    ),
    # --- Rigetti Ankaa-3 ---
    "RIGETTI-ANKAA3": NoiseConfig(
        noise_id="RIGETTI-ANKAA3",
        noise_type="combined",
        parameters={"p1": 0.002, "p2": 0.005, "p_readout": 0.02},
        description=(
            "Rigetti Ankaa-3 (84 qubits) - "
            "99.5% median 2Q fidelity (0.5% error). "
            "Ref: Quantum Insider Dec 2024 [14]"
        ),
    ),
}


def get_preset_noise_config(noise_id: str) -> NoiseConfig:
    """
    Get a preset noise configuration by ID.

    Args:
        noise_id: One of the preset IDs (e.g., "IDEAL", "DEP-MED", "COMB-HIGH").

    Returns:
        NoiseConfig for the requested preset.

    Raises:
        ValueError: If the noise_id is not recognized.

    Example:
        >>> config = get_preset_noise_config("COMB-MED")
        >>> print(config.description)
        "Medium combined noise - Realistic IBM/Google-like performance baseline"
    """
    if noise_id not in PRESET_NOISE_CONFIGS:
        raise ValueError(
            f"Unknown preset noise_id: {noise_id}. "
            f"Available: {list(PRESET_NOISE_CONFIGS.keys())}"
        )
    return PRESET_NOISE_CONFIGS[noise_id]


def list_preset_noise_configs() -> None:
    """Print a summary of all available preset noise configurations."""
    print("=" * 80)
    print("PRESET NOISE CONFIGURATIONS")
    print("=" * 80)
    print()
    for noise_id, config in PRESET_NOISE_CONFIGS.items():
        print(f"  {noise_id}:")
        print(f"    Type: {config.noise_type}")
        if config.parameters:
            params_str = ", ".join(f"{k}={v}" for k, v in config.parameters.items())
            print(f"    Parameters: {params_str}")
        print(f"    Description: {config.description}")
        print()


__all__ = [
    "NoiseConfig",
    "NoiseModelGenerator",
    "PRESET_NOISE_CONFIGS",
    "get_preset_noise_config",
    "list_preset_noise_configs",
    "DEFAULT_SINGLE_QUBIT_GATES",
    "DEFAULT_TWO_QUBIT_GATES",
]
