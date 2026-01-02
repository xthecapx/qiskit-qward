"""
Variational Quantum Time Evolution (VQTE) experiment runner para QWARD

Genera variaciones automáticas de VQTE sobre Hamiltonianos sencillos (por ejemplo, modelo de Ising o Heisenberg),
ejecuta las métricas LOC, Halstead, Behavioral y Quantum Software Quality, y guarda los resultados en un CSV.
"""

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli
from ..metrics import BehavioralMetrics, ElementMetrics, StructuralMetrics, QuantumSpecificMetrics
import os


def create_vqte_circuit_variational(
    num_qubits, p, evolution_type="imaginary", ansatz_type="ising", delta_t=0.1
):
    """
    Crea un circuito inspirado en la evolución variacional (VarQTE/VarQITE) para analizar métricas estructurales.

    Parámetros:
        num_qubits (int): número de qubits
        p (int): número de pasos (profundidad variacional)
        evolution_type (str): "imaginary" (VarQITE) o "real" (VarQRTE)
        ansatz_type (str): tipo de ansatz ("ising", "heisenberg", "su2")
        delta_t (float): paso de tiempo
    """
    params = ParameterVector("theta", p)
    qc = QuantumCircuit(num_qubits)

    # Inicialización en |+>^n
    qc.h(range(num_qubits))

    for step in range(p):
        theta = params[step]

        if ansatz_type == "ising":
            # Modelo de Ising: ZZ acoplado
            for i in range(num_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(2 * theta * delta_t, i + 1)
                qc.cx(i, i + 1)

        elif ansatz_type == "heisenberg":
            # Modelo de Heisenberg (interacciones XX + YY + ZZ)
            for i in range(num_qubits - 1):
                # Interacción XX
                qc.cx(i, i + 1)
                qc.rx(2 * theta * delta_t, i + 1)
                qc.cx(i, i + 1)

                # Interacción YY
                qc.sdg(i)
                qc.cx(i, i + 1)
                qc.ry(2 * theta * delta_t, i + 1)
                qc.cx(i, i + 1)
                qc.s(i)

                # Interacción ZZ
                qc.cx(i, i + 1)
                qc.rz(2 * theta * delta_t, i + 1)
                qc.cx(i, i + 1)

        elif ansatz_type == "su2":
            # Ansatz genérico tipo EfficientSU2
            qc.ry(theta * delta_t, range(num_qubits))
            for i in range(num_qubits - 1):
                qc.cz(i, i + 1)

        for q in range(num_qubits):
            if evolution_type == "real":
                # Evolución en tiempo real: rotación tipo RX
                qc.rx(2 * theta * delta_t, q)
            elif evolution_type == "imaginary":
                # Evolución en tiempo imaginario: rotación tipo RY
                qc.ry(2 * theta * delta_t, q)

    qc.measure_all()
    return qc


def run_experiments(
    num_instances=3,
    num_qubits_list=None,
    p_list=None,
    seed=123,
    output_csv="vqte_metrics_results.csv",
):
    if num_qubits_list is None:
        num_qubits_list = [5, 6, 7]
    if p_list is None:
        p_list = [1, 2, 3]
    results = []
    np.random.seed(seed)
    for n in num_qubits_list:
        for p in p_list:
            for idx in range(num_instances):

                for evo_type in ["imaginary", "real"]:
                    for ansatz in ["ising", "heisenberg", "su2"]:
                        qc = create_vqte_circuit_variational(
                            n, p, evolution_type=evo_type, ansatz_type=ansatz
                        )

                row = {"num_qubits": n, "p": p, "instance_id": idx}
                # Element
                element_metrics = ElementMetrics(qc).get_metrics()
                row.update({f"element_{k}": v for k, v in element_metrics.dict().items()})
                # Structural
                structural_metrics = StructuralMetrics(qc).get_metrics()
                row.update({f"structural_{k}": v for k, v in structural_metrics.dict().items()})
                # Behavioral
                behavioral_metrics = BehavioralMetrics(qc).get_metrics()
                row.update({f"behavioral_{k}": v for k, v in behavioral_metrics.dict().items()})
                # Quantum Specific
                quantum_specific_metrics = QuantumSpecificMetrics(qc).get_metrics()
                row.update(
                    {f"quantum_specific_{k}": v for k, v in quantum_specific_metrics.dict().items()}
                )
                results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Resultados guardados en {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    run_experiments()
