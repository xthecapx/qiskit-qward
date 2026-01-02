"""
Projected Variational Quantum Dynamics (p-VQD) experiment runner para QWARD

Genera variaciones de ansatz para p-VQD, ejecuta las métricas preruntime
(LOC, Halstead, Behavioral, y Quantum Software Quality), y guarda los resultados en un CSV.
"""

import numpy as np
import pandas as pd
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import ParameterVector

from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, state_fidelity

import os


def create_ghz_state(num_qubits):
    """Crea el Statevector para un estado GHZ de n qubits."""
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return Statevector(qc)


TARGET_STATES = {
    4: create_ghz_state(4),
    6: create_ghz_state(6),
    8: create_ghz_state(8),
    10: create_ghz_state(10),
}


def ghz_state(num_qubits: int) -> Statevector:
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return Statevector.from_instruction(qc)


def ising_evolved_state(num_qubits: int, t: float = 0.5, h: float = 1.0) -> Statevector:
    """Genera un estado |ψ(t)> = exp(-i H t) |000..0> con H tipo Ising (Z_i Z_{i+1})."""
    # Construir Hamiltoniano de Ising simple
    dim = 2**num_qubits
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    # Pauli Z matrices
    pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity = np.eye(2, dtype=complex)

    # Construir H = sum(Z_i ⊗ Z_{i+1})
    for i in range(num_qubits - 1):
        op_list = [identity] * num_qubits
        op_list[i] = pauli_z
        op_list[i + 1] = pauli_z
        term = op_list[0]
        for op in op_list[1:]:
            term = np.kron(term, op)
        hamiltonian += term

    hamiltonian *= h / (num_qubits - 1)
    psi0 = np.zeros((dim,), dtype=complex)
    psi0[0] = 1.0  # |000...0>

    # Evolución temporal: e^{-i H t} |ψ₀⟩
    unitary = expm(-1j * hamiltonian * t)
    psi_t = unitary @ psi0
    return Statevector(psi_t)


def evaluate_ansatz_performance(qc: QuantumCircuit, target_state: Statevector, seed: int):
    """
    Evalúa qué tan satisfactorio es un ansatz para generar un estado objetivo.

    Esta función realiza los siguientes pasos:
    1. Asigna parámetros aleatorios (pero reproducibles gracias a la semilla) al ansatz.
    2. Modifica el circuito para guardar explícitamente el vector de estado final.
    3. Simula el circuito usando el 'statevector_simulator'.
    4. Calcula la fidelidad (state_fidelity) entre el estado generado y el estado objetivo.

    Parámetros:
        qc (QuantumCircuit): El circuito del ansatz con parámetros libres.
        target_state (Statevector): El estado cuántico que queremos alcanzar.
        seed (int): Semilla para el generador de números aleatorios para garantizar la reproducibilidad.

    Retorna:
        dict: Un diccionario que contiene la métrica de rendimiento calculada.
    """
    # Usamos un generador de números aleatorios con semilla para que los resultados sean consistentes
    rng = np.random.default_rng(seed)

    # Asignamos valores aleatorios a los parámetros del circuito
    num_params = qc.num_parameters
    if num_params > 0:
        random_params = rng.uniform(0, 2 * np.pi, num_params)
        bound_qc = qc.assign_parameters(random_params)
    else:
        # Si el circuito no tiene parámetros, no es necesario asignar nada
        bound_qc = qc

    bound_qc.save_statevector()

    # Usamos el simulador de vectores de estado para obtener el resultado ideal (sin ruido)
    simulator = AerSimulator(method="statevector")
    result = simulator.run(bound_qc).result()
    output_state = result.get_statevector()

    # Calculamos la fidelidad.
    fidelity = state_fidelity(target_state, output_state)

    return {"performance_state_fidelity": fidelity}


def create_pvqd_ansatz(num_qubits, reps, ansatz_type="su2"):
    """
    Crea un circuito de ansatz variacional para ser usado en un algoritmo p-VQD.

    Parámetros:
        num_qubits (int): Número de qubits del ansatz.
        reps (int): Número de repeticiones o capas del ansatz (profundidad).
        ansatz_type (str): Tipo de ansatz a generar ("su2", "linear_entanglement", "two_local").

    Retorna:
        QuantumCircuit: El circuito del ansatz variacional listo para el análisis de métricas.
    """
    if ansatz_type == "su2":
        # Ansatz estándar EfficientSU2, usado comúnmente en tutoriales de Qiskit
        ansatz = EfficientSU2(num_qubits, reps=reps)

    elif ansatz_type == "linear_entanglement":
        # Ansatz con capas de rotación y entrelazamiento lineal (CNOTs)
        ansatz = QuantumCircuit(num_qubits)
        params = ParameterVector("θ", (reps + 1) * num_qubits)
        param_idx = 0

        for _ in range(reps):
            # Capa de rotaciones
            for i in range(num_qubits):
                ansatz.ry(params[param_idx], i)
                param_idx += 1

            # Capa de entrelazamiento lineal
            for i in range(num_qubits - 1):
                ansatz.cx(i, i + 1)
            ansatz.barrier()

        # Capa final de rotaciones
        for i in range(num_qubits):
            ansatz.ry(params[param_idx], i)
            param_idx += 1

    elif ansatz_type == "two_local":
        # Otro ansatz común: rotaciones en todos los qubits y entrelazamiento circular
        ansatz = QuantumCircuit(num_qubits)
        params = ParameterVector("θ", (reps + 1) * num_qubits)
        param_idx = 0

        for _ in range(reps):
            # Capa de rotaciones en Y y Z
            for i in range(num_qubits):
                ansatz.ry(params[param_idx], i)
                param_idx += 1

            # Capa de entrelazamiento circular
            for i in range(num_qubits):
                ansatz.cz(i, (i + 1) % num_qubits)  # Conecta el último con el primero
            ansatz.barrier()

        # Capa final de rotaciones
        for i in range(num_qubits):
            ansatz.ry(params[param_idx], i)
            param_idx += 1

    else:
        raise ValueError(f"Tipo de ansatz '{ansatz_type}' no reconocido.")

    # A diferencia del VQTE, en los algoritmos de Qiskit no se suelen añadir las mediciones
    # al ansatz, ya que el framework se encarga de calcular los observables.
    # Para el análisis estructural del ansatz, el circuito está completo así.

    return ansatz.decompose()


def run_experiments(
    num_instances=3,
    num_qubits_list=None,
    reps_list=None,
    seed=42,
    output_csv="pvqd_metrics_results.csv",
):
    """
    Ejecuta el pipeline de experimentos para generar y medir los ansatz de p-VQD
    contra distintos estados objetivo (GHZ e Ising).
    """
    if num_qubits_list is None:
        num_qubits_list = [4, 5, 6]
    if reps_list is None:
        reps_list = [1, 2, 3]
    results = []
    np.random.seed(seed)

    print("Iniciando la generación y análisis de circuitos de ansatz p-VQD...")

    for n in num_qubits_list:
        # Generar ambos estados objetivo para comparar correlaciones
        target_states = {"ghz": ghz_state(n), "ising": ising_evolved_state(n, t=0.5)}

        for target_name, target_state in target_states.items():
            for reps in reps_list:
                for idx in range(num_instances):
                    for ansatz_type in ["su2", "linear_entanglement", "two_local"]:

                        qc = create_pvqd_ansatz(n, reps, ansatz_type=ansatz_type)

                        row = {
                            "num_qubits": n,
                            "reps": reps,
                            "ansatz_type": ansatz_type,
                            "target_type": target_name,
                            "instance_id": idx,
                        }

                        # --- Calcular las métricas preruntime ---
                        element_metrics = ElementMetrics(qc).get_metrics()
                        row.update(
                            {f"element_{k}": v for k, v in element_metrics.model_dump().items()}
                        )

                        structural_metrics = StructuralMetrics(qc).get_metrics()
                        row.update(
                            {
                                f"structural_{k}": v
                                for k, v in structural_metrics.model_dump().items()
                            }
                        )

                        behavioral_metrics = BehavioralMetrics(qc).get_metrics()
                        row.update(
                            {
                                f"behavioral_{k}": v
                                for k, v in behavioral_metrics.model_dump().items()
                            }
                        )

                        quantum_specific_metrics = QuantumSpecificMetrics(qc).get_metrics()
                        row.update(
                            {
                                f"quantum_specific_{k}": v
                                for k, v in quantum_specific_metrics.model_dump().items()
                            }
                        )

                        # --- Calcular métrica de rendimiento (fidelidad) ---
                        instance_seed = seed + idx * 100 + reps * 10 + len(ansatz_type)
                        performance_metrics = evaluate_ansatz_performance(
                            qc, target_state, seed=instance_seed
                        )
                        row.update(performance_metrics)

                        results.append(row)

    print("Análisis completado. Guardando resultados...")

    df = pd.DataFrame(results)

    # Asegurémonos de que la columna de rendimiento esté al final
    if "performance_state_fidelity" in df.columns:
        cols = [c for c in df.columns if c != "performance_state_fidelity"] + [
            "performance_state_fidelity"
        ]
        df = df[cols]

    df.to_csv(output_csv, index=False)

    print(f"Resultados guardados exitosamente en {os.path.abspath(output_csv)}")
    print("\nVista previa de los resultados:")
    print(df.head())


if __name__ == "__main__":
    run_experiments(
        num_instances=5,
        num_qubits_list=[4, 6, 8, 10],  # Ajustado para coincidir con TARGET_STATES
        reps_list=[1, 2, 4, 6],
        seed=123,
        output_csv="pvqd_ansatz_metrics_with_performance.csv",
    )
