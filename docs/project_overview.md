# Qiskit Qward Project Overview

Qiskit Qward is a Python library designed to assist in the analysis of quantum circuits and their execution quality, particularly when targeting quantum processing units (QPUs) or simulators. It offers a structured way to apply various metrics to quantum circuits and their results.

## Key Components

The library revolves around the following key components:

1.  **`Scanner` (`qward.Scanner`)**: 
    -   The main orchestrator for analysis.
    -   Takes a Qiskit `QuantumCircuit` and optionally execution `Job` or `Result` objects.
    -   Users add various `Metric` objects to the `Scanner` to perform specific analyses.

2.  **`Metric` System (`qward.metrics`)**:
    -   A collection of classes for performing specific analyses. Each metric class typically focuses on a particular aspect of the circuit or its execution results.
    -   Key built-in metrics include:
        -   `QiskitMetrics`: For basic circuit properties (depth, width, gate counts).
        -   `ComplexityMetrics`: For advanced circuit complexity analysis (based on D. Shami's work) and Quantum Volume estimation.
        -   `SuccessRate`: For calculating success/error rates from execution counts against user-defined criteria.
    -   Extensible: Users can create custom metrics by inheriting from `qward.metrics.base_metric.Metric`.

3.  **`Result` (`qward.Result`)**:
    -   A helper class to encapsulate Qiskit job execution results, particularly counts and metadata, for use with the `Scanner` and certain metrics.

4.  **`QiskitRuntimeService` (`qward.runtime.qiskit_runtime.QiskitRuntimeService`)**:
    -   Extends Qiskit's own runtime service client, offering a streamlined way to run circuits, particularly with a `run_and_watch` method for easier job monitoring on IBM Quantum hardware (though it can be used with simulators too via Qiskit's backend system).

5.  **Example Implementations (`qward/examples/`)**:
    -   The `qward/examples/` directory contains scripts and notebooks (e.g., `aer.py`, `run_on_aer.ipynb`) demonstrating how to use the `Scanner` with various metrics and Qiskit's Aer simulator.

## Getting Started

To get started with Qiskit Qward, refer to the [Beginner's Guide](beginners_guide.md) and [Quickstart Guide](quickstart_guide.md) for installation instructions and basic usage patterns.

## Documentation Structure

-   **Beginner's Guide**: Introduction to Qward concepts and basic examples.
-   **Quickstart Guide**: Installation and core usage patterns.
-   **Technical Documentation**: More detailed component descriptions (see `docs/architecture.md`).
-   **API Documentation**: Full API reference (see `docs/apidocs/index.rst`).
-   **Examples**: Practical code examples in the `qward/examples/` directory.

## Project Goals

The primary goals of the Qiskit Qward project are to:

1.  Provide a flexible toolkit for evaluating quantum circuit properties and execution performance.
2.  Enable comparison between simulated and real hardware execution through consistent metric application.
3.  Support quantum algorithm analysis through a standardized and extensible metrics framework.
4.  Offer robust circuit complexity and Quantum Volume estimation capabilities.

## Background

Qward aims to simplify the process of analyzing quantum circuits and understanding their behavior. As quantum computing continues to evolve, developers and researchers need tools to assess how their quantum algorithms perform, both in simulation and on actual hardware.

## Solution Explanation

Qward helps bridge the gap between theoretical quantum algorithms and their practical implementation by providing:

-   A `Scanner` to orchestrate the application of various analytical `Metric` objects.
-   Comprehensive metrics for circuit structure (`QiskitMetrics`), advanced complexity and QV estimation (`ComplexityMetrics`), and execution outcome analysis (`SuccessRate`).
-   An extensible system for users to define their own custom metrics.
-   Utilities like `Result` for easier data handling and `QiskitRuntimeService` for streamlined execution.

This framework enables researchers and developers to gain deeper insights into their circuits, identify performance characteristics, understand error sources when results are available, and potentially optimize their quantum algorithms.

## Current Implementation Status

The project is under active development. Key implemented features include:

### Implemented Features
-   Core `Scanner` class for metric-based analysis.
-   `Result` class for handling execution data.
-   `QiskitRuntimeService` for enhanced job execution.
-   Metrics System:
    -   `QiskitMetrics` for circuit basics.
    -   `ComplexityMetrics` including detailed complexity categories and Quantum Volume estimation.
    -   `SuccessRate` metric for execution outcome analysis.
    -   Abstract `Metric` base class for custom metric development.
-   Integration with Qiskit's `QuantumCircuit`, `Job`, and `Result` objects.
-   Example usage scripts in `qward/examples/`.

### In Progress / Potential Future Enhancements
-   Expansion of built-in metrics (e.g., specific noise analysis, advanced error mitigation metrics).
-   More sophisticated data management and visualization tools integrated with the `Scanner` or as separate utilities.
-   Tighter integration with different Qiskit providers or other quantum SDKs.
-   Development of more complex example use-cases and tutorials.
-   Formalized parameter sweeping and experiment management utilities.

