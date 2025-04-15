"""
Result class for QWARD.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from qiskit_aer import AerJob
from qiskit.providers.job import Job as QiskitJob


class Result:
    """
    Class for storing and managing quantum circuit execution results.

    This class provides methods for storing, loading, and analyzing the results
    of quantum circuit executions.
    """

    def __init__(
        self,
        job: Optional[Union[AerJob, QiskitJob]] = None,
        counts: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a Result object.

        Args:
            job: The job that executed the circuit
            counts: The measurement counts from the job result
            metadata: Additional metadata about the result
        """
        self._job = job
        self._counts = counts or {}
        self._metadata = metadata or {}

    @property
    def job(self) -> Optional[Union[AerJob, QiskitJob]]:
        """
        Get the job that executed the circuit.

        Returns:
            Optional[Union[AerJob, QiskitJob]]: The job that executed the circuit
        """
        return self._job

    @property
    def counts(self) -> Dict[str, int]:
        """
        Get the measurement counts from the job result.

        Returns:
            Dict[str, int]: The measurement counts
        """
        return self._counts

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata about the result.

        Returns:
            Dict[str, Any]: The metadata about the result
        """
        return self._metadata

    def save(self, path: str) -> None:
        """
        Save the result to a file.

        Args:
            path: The path to save the result to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Prepare data for saving
        data = {
            "counts": self._counts,
            "metadata": self._metadata,
        }

        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Result":
        """
        Load a result from a file.

        Args:
            path: The path to load the result from

        Returns:
            Result: The loaded result
        """
        # Load from file
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create result object
        return cls(
            counts=data.get("counts", {}),
            metadata=data.get("metadata", {}),
        )

    def update_from_job(self) -> None:
        """
        Update the result from the job.
        """
        if self._job is None:
            return

        # Get the result from the job
        result = self._job.result()

        # Update counts
        self._counts = result.get_counts()

        # Update metadata
        self._metadata.update(
            {
                "shots": result.shots,
                "status": result.status,
                "success": result.success,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the result
        """
        return {
            "counts": self._counts,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Result":
        """
        Create a Result object from a dictionary.

        Args:
            data: Dictionary containing the result data

        Returns:
            Result: The created Result object
        """
        return cls(
            counts=data.get("counts", {}),
            metadata=data.get("metadata", {}),
        )

    def save_to_file(self, filename: str) -> None:
        """Save the result to a file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> "Result":
        """Load a result from a file."""
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
