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

    def backend(self, name=None, instance=None, use_fractional_gates=False) -> Union[Backend, str]:
        """
        Get the backend.

        This method overrides the parent class method but returns the stored backend
        instead of looking up a backend by name.

        Args:
            name: Ignored, kept for compatibility with parent class
            instance: Ignored, kept for compatibility with parent class
            use_fractional_gates: Ignored, kept for compatibility with parent class

        Returns:
            Union[Backend, str]: The backend
        """
        return self._backend

    def job(self, job_id=None) -> Optional[Union[AerJob, QiskitJob]]:
        """
        Get the job.

        This method overrides the parent class method but returns the stored job
        instead of looking up a job by ID.

        Args:
            job_id: Ignored, kept for compatibility with parent class

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
            counts=job_result.get_counts() if hasattr(job_result, "get_counts") else {},
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

    def _create_result(self, job: QiskitJob) -> Result:
        """
        Create a Result object from a Qiskit job.

        Args:
            job: The Qiskit job

        Returns:
            Result: The created Result object
        """
        result = job.result()
        counts = result.get_counts()
        return Result(job=job, counts=counts)
