"""
QiskitRuntimeService class for QWARD.
"""

import time
from typing import Any, Dict, Optional, Union

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_aer import AerJob
from qiskit.providers.job import Job as QiskitJob
from qiskit_ibm_runtime import (
    SamplerV2 as Sampler,
    RuntimeJobV2 as RuntimeJob,
    QiskitRuntimeService as QiskitRuntimeServiceBase,
)

from qward.result import Result


class QiskitRuntimeService(QiskitRuntimeServiceBase):
    """
    Extended QiskitRuntimeService for QWARD.

    This class extends the QiskitRuntimeService class to provide enhanced functionality
    for quantum circuit execution.
    """

    def __init__(self, circuit: QuantumCircuit, backend: Union[Backend, str], **kwargs):
        """
        Initialize a QiskitRuntimeService object.

        Args:
            circuit: The quantum circuit to execute
            backend: The backend to execute the circuit on
            **kwargs: Additional arguments to pass to the QiskitRuntimeService constructor
        """
        super().__init__(**kwargs)
        self._circuit = circuit
        self._backend = backend
        self._job: Optional[Union[AerJob, QiskitJob]] = None
        self._result: Optional[Result] = None

    @property
    def circuit(self) -> QuantumCircuit:
        """
        Get the quantum circuit.

        Returns:
            QuantumCircuit: The quantum circuit
        """
        return self._circuit

    @property
    def backend(self) -> Union[Backend, str]:
        """
        Get the backend.

        Returns:
            Union[Backend, str]: The backend
        """
        return self._backend

    @property
    def job(self) -> Optional[Union[AerJob, QiskitJob]]:
        """
        Get the job.

        Returns:
            Optional[Union[AerJob, QiskitJob]]: The job
        """
        return self._job

    @property
    def result(self) -> Optional[Result]:
        """
        Get the result.

        Returns:
            Optional[Result]: The result
        """
        return self._result

    def run(self) -> None:
        """
        Run the circuit on the backend.
        """
        # Create a sampler
        sampler = Sampler(backend=self._backend)

        # Run the circuit
        self._job = sampler.run(self._circuit)

    def check_status(self) -> str:
        """
        Check the status of the job.

        Returns:
            str: The status of the job
        """
        if self._job is None:
            return "No job"

        return self._job.status().name

    def get_results(self) -> Result:
        """
        Get the results of the job.

        Returns:
            Result: The results of the job
        """
        if self._job is None:
            return Result()

        # Get the job result
        job_result = self._job.result()

        # Create a Result object
        self._result = Result(
            job=self._job,
            quasi_dists=job_result.quasi_dists if hasattr(job_result, "quasi_dists") else [],
            metadata=job_result.metadata if hasattr(job_result, "metadata") else {},
        )

        return self._result

    def run_and_watch(
        self, polling_interval: float = 1.0, timeout: Optional[float] = None
    ) -> Result:
        """
        Run the circuit and watch the job status.

        Args:
            polling_interval: The interval between status checks in seconds
            timeout: The timeout in seconds, or None for no timeout

        Returns:
            Result: The results of the job
        """
        # Run the circuit
        self.run()

        # Watch the job status
        start_time = time.time()

        while True:
            # Check if the job is done
            status = self.check_status()
            if status == "DONE":
                break

            # Check if the job has failed
            if status in ["ERROR", "CANCELLED"]:
                raise RuntimeError(f"Job failed with status: {status}")

            # Check if we've timed out
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Job timed out after {timeout} seconds")

            # Wait for the next polling interval
            time.sleep(polling_interval)

        # Get the results
        return self.get_results()
