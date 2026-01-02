"""
QAOA MaxCut experiment runner for QWARD

Genera variaciones automáticas de QAOA para MaxCut sobre grafos aleatorios y diferentes valores de p,
ejecuta las métricas LOC, Halstead, Behavioral y Quantum Software Quality, y guarda los resultados en un CSV.
"""

import numpy as np
import networkx as nx
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from ..metrics import StructuralMetrics, BehavioralMetrics, ElementMetrics, QuantumSpecificMetrics
import os


def create_qaoa_maxcut_circuit(graph, p, params=None):
    n = graph.number_of_nodes()
    qc = QuantumCircuit(n)
    if params is None:
        params = ParameterVector("theta", 2 * p)
    # Inicialización en superposición
    qc.h(range(n))
    # Alternar operadores de costo y mezcla
    for i in range(p):
        gamma = params[i]
        beta = params[p + i]
        # Operador de costo (MaxCut)
        for u, v in graph.edges():
            qc.cx(u, v)
            qc.rz(-gamma, v)
            qc.cx(u, v)
        # Operador de mezcla
        for q in range(n):
            qc.rx(2 * beta, q)
    qc.measure_all()
    return qc


def run_experiments(
    num_graphs=5,
    num_nodes_list=None,
    p_list=None,
    seed=42,
    output_csv="qaoa_metrics_results.csv",
):
    if num_nodes_list is None:
        num_nodes_list = [1, 2, 3]
    if p_list is None:
        p_list = [1, 2, 3]
    results = []
    np.random.seed(seed)
    for n in num_nodes_list:
        for p in p_list:
            for gidx in range(num_graphs):
                # Grafo aleatorio
                graph = nx.erdos_renyi_graph(n, 0.5, seed=seed + gidx)
                qc = create_qaoa_maxcut_circuit(graph, p)
                # Ejecutar métricas
                row = {"num_nodes": n, "p": p, "graph_id": gidx, "edges": list(graph.edges())}
                # Element
                element_metrics = ElementMetrics(qc).get_metrics()
                row.update({f"element_{k}": v for k, v in element_metrics.dict().items()})
                # Structural
                loc_metrics = StructuralMetrics(qc).get_metrics()
                row.update({f"structural_{k}": v for k, v in loc_metrics.dict().items()})
                # Behavioral
                behavioral_metrics = BehavioralMetrics(qc).get_metrics()
                row.update({f"behavioral_{k}": v for k, v in behavioral_metrics.dict().items()})
                # Quantum Specific
                quantum_specific_metrics = QuantumSpecificMetrics(qc).get_metrics()
                row.update(
                    {f"quantum_specific_{k}": v for k, v in quantum_specific_metrics.dict().items()}
                )
                # Guardar resultados
                results.append(row)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Resultados guardados en {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    run_experiments(3, [5, 10, 15], [5, 5, 5], seed=123, output_csv="qaoa_metrics_test_results.csv")
