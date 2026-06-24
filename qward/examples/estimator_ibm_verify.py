"""Verify QWARD Estimator metrics from a completed IBM QPU job.

Reads the job_id saved by estimator_ibm_experiment.py, retrieves results
via scan_job, and prints the EstimatorSchema metrics.

Usage:
    uv run qward/examples/estimator_ibm_verify.py
"""

import json

import numpy as np

from qward.scan import scan_job


def main():
    with open("qward/examples/estimator_ibm_result.json") as f:
        data = json.load(f)

    job_id = data["job_id"]
    ideal = np.array(data["ideal_values"])
    labels = data["observables"]

    print(f"Retrieving job: {job_id}")
    print(f"Backend: {data['backend']}")
    print(f"Observables: {labels}")
    print(f"Ideal values: {ideal.tolist()}")
    print("-" * 60)

    results = scan_job(
        job_id,
        ideal_expectation_values=ideal,
        observable_labels=labels,
    )

    print("\n=== Meta ===")
    print(results["_meta"].to_string(index=False))

    print("\n=== FidelityMetrics (EstimatorSchema) ===")
    df = results["FidelityMetrics"]
    print(df.to_string(index=False))

    print("\n=== Key Metrics ===")
    for col in [
        "mean_observable_fidelity",
        "mean_success_probability",
        "mean_snr",
        "depolarization_factor",
    ]:
        if col in df.columns:
            print(f"  {col}: {df[col].iloc[0]:.4f}")


if __name__ == "__main__":
    main()
