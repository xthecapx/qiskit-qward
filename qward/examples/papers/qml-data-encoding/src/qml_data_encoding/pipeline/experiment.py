"""Experiment pipeline: preprocessing + encoding + circuit building."""

from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

from qml_data_encoding.encodings import (
    AngleEncoding,
    IQPEncoding,
    ReuploadingEncoding,
    AmplitudeEncoding,
    BasisEncoding,
)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class ExperimentPipeline:
    """End-to-end pipeline: preprocessing, encoding, circuit building.

    Args:
        encoding: Encoding method name.
        preprocessing: Preprocessing level name.
        n_qubits: Number of qubits for the encoding.
        ansatz_reps: Number of ansatz repetitions.
        pca_components: Number of PCA components (optional).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        encoding: str,
        preprocessing: str,
        n_qubits: int,
        ansatz_reps: int = 2,
        pca_components: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.encoding_name = encoding
        self.preprocessing_name = preprocessing
        self.n_qubits = n_qubits
        self.ansatz_reps = ansatz_reps
        self.pca_components = pca_components
        self.seed = seed

        # Build preprocessing pipeline
        self._scaler: Optional[object] = None
        self._pca: Optional[PCA] = None
        self._fitted = False
        self._init_preprocessing()

        # Build encoding
        self._encoder = self._make_encoder()

    def _init_preprocessing(self) -> None:
        """Initialize preprocessing objects."""
        if self.preprocessing_name == "minmax_pi":
            self._scaler = MinMaxScaler(feature_range=(0, np.pi))
        elif self.preprocessing_name == "zscore_sigmoid":
            self._scaler = StandardScaler()
        elif self.preprocessing_name == "pca_minmax":
            if self.pca_components is not None:
                self._pca = PCA(
                    n_components=self.pca_components,
                    random_state=self.seed,
                )
            self._scaler = MinMaxScaler(feature_range=(0, np.pi))
        # "none" -> no scaler

    def _make_encoder(self):
        """Instantiate the encoding object."""
        if self.encoding_name == "angle_ry":
            return AngleEncoding(n_features=self.n_qubits, rotation_axis="y")
        elif self.encoding_name == "iqp_full":
            return IQPEncoding(n_features=self.n_qubits, interaction="full")
        elif self.encoding_name == "reuploading":
            return ReuploadingEncoding(n_features=self.n_qubits, n_layers=2)
        elif self.encoding_name == "amplitude":
            return AmplitudeEncoding(n_features=self.n_qubits)
        elif self.encoding_name == "basis":
            return BasisEncoding(n_features=self.n_qubits)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding_name}")

    def fit_preprocessing(self, X: np.ndarray) -> None:
        """Fit the preprocessing pipeline on training data.

        Args:
            X: Training features of shape (n_samples, n_features).
        """
        data = X.copy()
        if self._pca is not None:
            data = self._pca.fit_transform(data)
        if self._scaler is not None:
            if self.preprocessing_name == "zscore_sigmoid":
                self._scaler.fit(data)
            else:
                self._scaler.fit(data)
        self._fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted preprocessing to data.

        Args:
            X: Feature array of shape (n_samples, n_features).

        Returns:
            Transformed data.
        """
        data = X.copy()
        if self._pca is not None:
            data = self._pca.transform(data)
        if self._scaler is not None:
            if self.preprocessing_name == "zscore_sigmoid":
                data = self._scaler.transform(data)
                data = np.pi * _sigmoid(data)
            else:
                data = self._scaler.transform(data)
        return data

    def build_circuit(self, x: np.ndarray) -> QuantumCircuit:
        """Build the encoding circuit for a single (preprocessed) data point.

        Args:
            x: Preprocessed feature vector of shape (n_qubits,).

        Returns:
            QuantumCircuit for the encoded data point.
        """
        return self._encoder.encode(x)
