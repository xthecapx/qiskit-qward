"""Tests for qward.algorithms.shor."""

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from qward.algorithms.shor import (
    Shor,
    ShorCircuitGenerator,
    a_pow_2k_mod_n,
    analyze_counts,
    bitstring_to_phase,
    build_step_plan,
    classical_order,
    classical_precheck,
    factors_from_order,
    modular_multiply_gate,
    phase_to_order,
)


def test_classical_precheck_even():
    r = classical_precheck(14, 3)
    assert not r.needs_quantum
    assert r.reason == "N_even"
    assert r.factor == 2


def test_classical_precheck_lucky_gcd():
    r = classical_precheck(15, 6)
    assert not r.needs_quantum
    assert r.reason == "lucky_gcd"
    assert r.factor == 3


def test_classical_precheck_needs_quantum():
    r = classical_precheck(15, 7)
    assert r.needs_quantum


def test_step_plan_n15_a2_collapses():
    plan = build_step_plan(15, 2, 8)
    assert [e.b for e in plan] == [2, 4, 1, 1, 1, 1, 1, 1]
    assert sum(1 for e in plan if e.status == "identity") == 6


def test_step_plan_n15_a7():
    plan = build_step_plan(15, 7, 8)
    assert plan[0].b == 7
    assert plan[1].b == 4
    assert plan[2].status == "identity"


def test_bitstring_phase_order_m8():
    # Known peak: 01000000 → 64/256 = 0.25 → r=4
    phase = bitstring_to_phase("01000000", 8)
    assert abs(phase - 0.25) < 1e-12
    r, frac = phase_to_order(phase, 15)
    assert r == 4
    assert frac.numerator == 1 and frac.denominator == 4


def test_factors_from_order_n15_a7():
    info = factors_from_order(7, 4, 15)
    assert info["nontrivial"]
    assert set(info["factors"]) == {3, 5}


def test_factors_n21_a5_trivial():
    # order 6 but x ≡ -1
    r = classical_order(5, 21)
    assert r == 6
    info = factors_from_order(5, 6, 21)
    assert info["reason"] == "trivial_x_pm_1"


def test_m7_permutation_matches_classical():
    gate = modular_multiply_gate(7, 15, 4, strategy="swap_network")
    op = Operator(gate)
    mat = op.data
    for y in range(15):
        # Apply to |y>
        vec = np.zeros(16, dtype=complex)
        vec[y] = 1.0
        out = mat @ vec
        dest = int(np.argmax(np.abs(out)))
        assert dest == (7 * y) % 15


def test_m4_swap_network():
    gate = modular_multiply_gate(4, 15, 4, strategy="swap_network")
    op = Operator(gate)
    mat = op.data
    for y in range(15):
        vec = np.zeros(16, dtype=complex)
        vec[y] = 1.0
        out = mat @ vec
        dest = int(np.argmax(np.abs(out)))
        assert dest == (4 * y) % 15


def test_shor_aer_n15_a7_finds_factors():
    from qiskit import transpile

    shor = Shor(15, 7, strategy="permutation")
    sim = AerSimulator(method="statevector")
    circuit = transpile(shor.circuit, backend=sim, optimization_level=1)
    result = sim.run(circuit, shots=256).result()
    counts = {k.replace(" ", ""): v for k, v in result.get_counts().items()}
    analysis = analyze_counts(counts, 7, 15, shor.num_control, true_order=4)
    # At least one outcome should recover order 4 or nontrivial factor
    ok = any(
        o["order_matches_true"] or (o["factoring"] and o["factoring"]["nontrivial"])
        for o in analysis["outcomes"]
    )
    assert ok


def test_shor_generator_presets():
    gen = ShorCircuitGenerator("N15_a7")
    assert gen.true_order == 4
    assert gen.circuit.num_qubits >= 8


def test_a_pow_2k():
    assert a_pow_2k_mod_n(7, 0, 15) == 7
    assert a_pow_2k_mod_n(7, 1, 15) == 4
    assert a_pow_2k_mod_n(7, 2, 15) == 1
