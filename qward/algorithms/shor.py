"""Shor's algorithm (order finding) for qWard.

Classical helpers are module-level functions. Quantum order finding is
encapsulated in ``Shor`` / ``ShorCircuitGenerator``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate, UnitaryGate


# ---------------------------------------------------------------------------
# Classical helpers
# ---------------------------------------------------------------------------


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def is_prime_power(n: int) -> Optional[Tuple[int, int]]:
    """Return (p, k) if n = p^k with k > 1, else None."""
    if n < 4:
        return None
    for k in range(2, int(math.log2(n)) + 1):
        root = round(n ** (1 / k))
        if root >= 2 and root**k == n and is_prime(root):
            return root, k
    return None


@dataclass
class PrecheckResult:
    """Outcome of classical_precheck before any quantum circuit is built."""

    needs_quantum: bool
    reason: str
    factor: Optional[int] = None
    factors: Optional[Tuple[int, int]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "needs_quantum": self.needs_quantum,
            "reason": self.reason,
            "factor": self.factor,
            "factors": list(self.factors) if self.factors else None,
            "details": self.details,
        }


def classical_precheck(N: int, a: int) -> PrecheckResult:
    """Resolve cases that never need a quantum computer."""
    if N < 2:
        return PrecheckResult(False, "invalid_N", details={"N": N})
    if N % 2 == 0:
        return PrecheckResult(
            False, "N_even", factor=2, factors=(2, N // 2), details={"N": N}
        )
    if is_prime(N):
        return PrecheckResult(False, "N_prime", details={"N": N})
    pp = is_prime_power(N)
    if pp is not None:
        p, k = pp
        return PrecheckResult(
            False,
            "N_prime_power",
            factor=p,
            factors=(p, N // p),
            details={"p": p, "k": k},
        )
    g = math.gcd(a, N)
    if g > 1:
        return PrecheckResult(
            False,
            "lucky_gcd",
            factor=g,
            factors=(g, N // g),
            details={"a": a, "gcd": g},
        )
    if not (1 < a < N):
        return PrecheckResult(False, "invalid_a", details={"a": a, "N": N})
    return PrecheckResult(True, "needs_order_finding", details={"a": a, "N": N})


def a_pow_2k_mod_n(a: int, k: int, N: int) -> int:
    """Compute a^(2^k) mod N by repeated squaring."""
    result = a % N
    for _ in range(k):
        result = (result * result) % N
    return result


@dataclass
class StepPlanEntry:
    """One controlled-multiplication slot in the QPE ladder."""

    control_index: int
    k: int
    b: int
    status: str  # "quantum" | "identity" | "reuse"
    reuse_of: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "control_index": self.control_index,
            "k": self.k,
            "b": self.b,
            "status": self.status,
            "reuse_of": self.reuse_of,
        }


def build_step_plan(N: int, a: int, num_control: int) -> List[StepPlanEntry]:
    """Decide per control qubit whether M_{a^(2^k)} is needed."""
    plan: List[StepPlanEntry] = []
    first_seen: Dict[int, int] = {}
    for k in range(num_control):
        b = a_pow_2k_mod_n(a, k, N)
        if b == 1:
            plan.append(StepPlanEntry(k, k, b, "identity"))
        elif b in first_seen:
            plan.append(StepPlanEntry(k, k, b, "reuse", reuse_of=first_seen[b]))
        else:
            first_seen[b] = k
            plan.append(StepPlanEntry(k, k, b, "quantum"))
    return plan


def bitstring_to_phase(bitstring: str, num_control: int) -> float:
    """Convert a measured bitstring to phase in [0, 1).

    Interpret bitstring as big-endian int(bits, 2) / 2^m after stripping spaces.
    """
    bits = bitstring.replace(" ", "").strip()
    if len(bits) < num_control:
        bits = bits.zfill(num_control)
    elif len(bits) > num_control:
        bits = bits[-num_control:]
    decimal = int(bits, 2)
    return decimal / (2**num_control)


def phase_to_order(phase: float, N: int) -> Tuple[int, Fraction]:
    """Continued-fraction expansion: phase ≈ s/r → candidate order r."""
    if phase == 0:
        return 1, Fraction(0, 1)
    frac = Fraction(phase).limit_denominator(N)
    return frac.denominator, frac


def factors_from_order(a: int, r: int, N: int) -> Dict[str, Any]:
    """Classical post-processing: even order → gcd(a^{r/2} ± 1, N)."""
    result: Dict[str, Any] = {
        "a": a,
        "r": r,
        "N": N,
        "even_order": r % 2 == 0 and r > 0,
        "factors": None,
        "nontrivial": False,
        "x": None,
        "reason": None,
    }
    if r <= 0:
        result["reason"] = "invalid_order"
        return result
    if r % 2 == 1:
        result["reason"] = "odd_order"
        return result
    x = pow(a, r // 2, N)
    result["x"] = x
    if x == 1 or x == N - 1:
        result["reason"] = "trivial_x_pm_1"
        return result
    f1 = math.gcd(x - 1, N)
    f2 = math.gcd(x + 1, N)
    factors = sorted({f for f in (f1, f2) if 1 < f < N})
    result["factors"] = factors if factors else None
    result["nontrivial"] = bool(factors)
    result["reason"] = "success" if factors else "no_factor"
    return result


def analyze_counts(
    counts: Dict[str, int],
    a: int,
    N: int,
    num_control: int,
    *,
    true_order: Optional[int] = None,
) -> Dict[str, Any]:
    """Post-process a counts histogram into phases, orders, and factors."""
    shots = sum(counts.values())
    rows: List[Dict[str, Any]] = []
    for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        phase = bitstring_to_phase(bitstring, num_control)
        r, frac = phase_to_order(phase, N)
        factor_info = factors_from_order(a, r, N)
        order_ok = true_order is not None and r == true_order
        rows.append(
            {
                "bitstring": bitstring.replace(" ", ""),
                "count": count,
                "probability": count / shots if shots else 0.0,
                "phase": phase,
                "fraction": f"{frac.numerator}/{frac.denominator}",
                "order_guess": r,
                "order_matches_true": order_ok,
                "factoring": factor_info,
            }
        )
    best = rows[0] if rows else None
    return {
        "shots": shots,
        "num_control": num_control,
        "true_order": true_order,
        "outcomes": rows,
        "best": best,
    }


def classical_order(a: int, N: int) -> int:
    """Brute-force multiplicative order of a modulo N."""
    if math.gcd(a, N) != 1:
        raise ValueError(f"gcd({a}, {N}) != 1")
    val = 1
    for r in range(1, N + 1):
        val = (val * a) % N
        if val == 1:
            return r
    raise RuntimeError(f"No order found for a={a}, N={N}")


# ---------------------------------------------------------------------------
# Modular multiplication gates
# ---------------------------------------------------------------------------


def modular_multiply_permutation_matrix(b: int, N: int, num_qubits: int) -> np.ndarray:
    """Permutation matrix for |y> → |b·y mod N> on a num_qubits register."""
    dim = 2**num_qubits
    mat = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        if y < N:
            dest = (b * y) % N
        else:
            dest = y
        mat[dest, y] = 1.0
    return mat


def modular_multiply_gate(
    b: int,
    N: int,
    num_qubits: int,
    *,
    strategy: str = "permutation",
    a: Optional[int] = None,
) -> QuantumCircuit:
    """Build an (uncontrolled) modular multiplication circuit on ``num_qubits``."""
    if strategy == "swap_network":
        return _swap_network_mb(b, N, num_qubits, a=a)
    qc = QuantumCircuit(num_qubits, name=f"M_{b}")
    mat = modular_multiply_permutation_matrix(b, N, num_qubits)
    qc.append(UnitaryGate(mat, label=f"M_{b}"), range(num_qubits))
    return qc


def _swap_network_mb(
    b: int, N: int, num_qubits: int, *, a: Optional[int] = None
) -> QuantumCircuit:
    """
    Shallow circuits for N=15 hardware demo.

    Hardware uses a=7: surviving powers are M_7 and M_4 (cycle 1→7→4→13→1).
    a=2 IBM-style M_2 / M_4 kept for teaching comparison.
    """
    if N != 15 or num_qubits != 4:
        raise ValueError("swap_network currently supports only N=15 with 4 target qubits")

    qc = QuantumCircuit(4, name=f"M_{b}")

    if b == 1:
        return qc

    if b == 2:
        qc.swap(0, 1)
        qc.swap(1, 2)
        qc.swap(2, 3)
        return qc

    if b == 4:
        qc.swap(1, 3)
        qc.swap(0, 2)
        return qc

    if b == 7:
        # Exact permutation for M_7; labeled swap_network toy strategy.
        mat = modular_multiply_permutation_matrix(7, 15, 4)
        qc.append(UnitaryGate(mat, label="M_7"), range(4))
        return qc

    mat = modular_multiply_permutation_matrix(b, N, num_qubits)
    qc.append(UnitaryGate(mat, label=f"M_{b}"), range(num_qubits))
    return qc


def controlled_modular_multiply(
    b: int,
    N: int,
    num_qubits: int,
    *,
    strategy: str = "permutation",
    a: Optional[int] = None,
):
    """Return a controlled modular-multiplication gate."""
    base = modular_multiply_gate(b, N, num_qubits, strategy=strategy, a=a)
    gate = base.to_gate()
    gate.name = f"cM_{b}"
    return gate.control(1)


# ---------------------------------------------------------------------------
# Shor core class
# ---------------------------------------------------------------------------


class Shor:
    """Order-finding circuit for Shor's algorithm."""

    def __init__(
        self,
        N: int,
        a: int,
        *,
        num_control: int | None = None,
        strategy: str = "permutation",
        use_barriers: bool = True,
    ):
        self.N = N
        self.a = a
        self.strategy = strategy
        self.use_barriers = use_barriers
        self.num_target = math.floor(math.log2(N - 1)) + 1
        self.num_control = (
            num_control if num_control is not None else 2 * self.num_target
        )
        self.num_qubits = self.num_control + self.num_target
        self.precheck = classical_precheck(N, a)
        self.step_plan = build_step_plan(N, a, self.num_control)
        self.true_order: Optional[int] = None
        if self.precheck.needs_quantum:
            try:
                self.true_order = classical_order(a, N)
            except Exception:
                self.true_order = None
        self.expected_period = self.true_order
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        if not self.precheck.needs_quantum:
            qc = QuantumCircuit(1, 1, name="shor_classical_only")
            qc.measure(0, 0)
            return qc

        control = QuantumRegister(self.num_control, name="C")
        target = QuantumRegister(self.num_target, name="T")
        output = ClassicalRegister(self.num_control, name="out")
        qc = QuantumCircuit(control, target, output, name=f"Shor({self.N},{self.a})")

        qc.x(target[0])
        if self.use_barriers:
            qc.barrier()

        gate_cache: Dict[int, Any] = {}

        for entry in self.step_plan:
            qc.h(control[entry.control_index])
            if entry.status == "identity":
                continue
            b = entry.b
            if b not in gate_cache:
                gate_cache[b] = controlled_modular_multiply(
                    b,
                    self.N,
                    self.num_target,
                    strategy=self.strategy,
                    a=self.a,
                )
            qc.append(
                gate_cache[b],
                [control[entry.control_index], *list(target)],
            )

        if self.use_barriers:
            qc.barrier()

        qc.compose(QFTGate(self.num_control).inverse(), qubits=control, inplace=True)
        qc.measure(control, output)
        # Decompose custom controlled unitaries for Aer / hardware compatibility
        return qc.decompose(reps=3)

    def success_criteria(self, outcome: str) -> bool:
        """Primary success: continued fractions recover true order."""
        if self.true_order is None:
            return False
        phase = bitstring_to_phase(outcome, self.num_control)
        r, _ = phase_to_order(phase, self.N)
        return r == self.true_order

    def expected_distribution(self) -> Dict[str, float]:
        """Ideal QPE peak distribution over control bitstrings."""
        if self.true_order is None or not self.precheck.needs_quantum:
            return {}
        r = self.true_order
        m = self.num_control
        dim = 2**m
        probs = {format(i, f"0{m}b"): 0.0 for i in range(dim)}
        weight = 1.0 / r
        for k in range(r):
            phase = k / r
            idx = int(round(phase * dim)) % dim
            probs[format(idx, f"0{m}b")] += weight
        total = sum(probs.values())
        if total > 0:
            probs = {b: p / total for b, p in probs.items() if p > 0}
        return probs

    def draw(self, **kwargs):
        output = kwargs.pop("output", "mpl")
        return self.circuit.draw(output=output, **kwargs)

    def plan_dict(self) -> Dict[str, Any]:
        return {
            "N": self.N,
            "a": self.a,
            "num_control": self.num_control,
            "num_target": self.num_target,
            "strategy": self.strategy,
            "precheck": self.precheck.to_dict(),
            "step_plan": [e.to_dict() for e in self.step_plan],
            "true_order": self.true_order,
            "quantum_gates": sum(1 for e in self.step_plan if e.status != "identity"),
            "identity_skipped": sum(1 for e in self.step_plan if e.status == "identity"),
        }


class ShorCircuitGenerator:
    """Preset Shor circuits for experiments and demos."""

    PRESETS: Dict[str, Dict[str, Any]] = {
        "N15_a2": {"N": 15, "a": 2, "true_order": 4},
        "N15_a7": {"N": 15, "a": 7, "true_order": 4},
        "N21_a2": {"N": 21, "a": 2, "true_order": 6},
        "N21_a5": {"N": 21, "a": 5, "true_order": 6},
    }

    def __init__(
        self,
        test_case: str = "N15_a7",
        *,
        num_control: int | None = None,
        strategy: str = "permutation",
        use_barriers: bool = True,
    ):
        if test_case not in self.PRESETS:
            raise ValueError(
                f"Unknown test_case {test_case!r}. Available: {list(self.PRESETS)}"
            )
        params = self.PRESETS[test_case]
        self.test_case = test_case
        self.N = params["N"]
        self.a = params["a"]
        self.shor = Shor(
            self.N,
            self.a,
            num_control=num_control,
            strategy=strategy,
            use_barriers=use_barriers,
        )
        self.circuit = self.shor.circuit
        self.true_order = params["true_order"]

    def success_criteria(self, outcome: str) -> bool:
        return self.shor.success_criteria(outcome)

    def expected_distribution(self) -> dict:
        return self.shor.expected_distribution()

    def draw(self, **kwargs):
        return self.shor.draw(**kwargs)
