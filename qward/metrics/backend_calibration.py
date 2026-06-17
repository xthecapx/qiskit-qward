"""
Backend calibration metrics collector.

Extracts median gate errors, readout errors, T1/T2 times, and operational
qubit count from IBM BackendV2 or AWS Braket devices.
"""

import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from qward.metrics.backend_metric_base import BackendMetricCollector
from qward.schemas.backend_calibration_schema import BackendCalibrationSchema


class BackendCalibrationCollector(BackendMetricCollector):
    """
    Collects backend calibration data at execution time.

    Supports IBM Qiskit BackendV2 (via backend.target) and AWS Braket devices.
    """

    def is_available(self) -> bool:
        """Check if backend exposes calibration properties."""
        return self._detect_provider() is not None

    def get_metrics(self) -> BackendCalibrationSchema:
        """Extract calibration metrics from backend."""
        provider = self._detect_provider()
        if provider == "ibm":
            return self._extract_ibm()
        elif provider == "aws":
            return self._extract_aws()
        else:
            return BackendCalibrationSchema(
                num_operational_qubits=0,
                backend_name=str(self._backend),
                provider="unknown",
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return metrics as plain dictionary for JSON serialization."""
        return self.get_metrics().model_dump()

    def _detect_provider(self) -> Optional[str]:
        # Raw Braket AwsDevice (has properties + arn, no target)
        if hasattr(self._backend, "properties") and hasattr(self._backend, "arn"):
            return "aws"
        # qiskit-braket-provider wraps AwsDevice in _device
        if hasattr(self._backend, "_device") and hasattr(self._backend._device, "properties"):
            return "aws"
        # IBM BackendV2 (has target with qubit_properties containing T1/T2)
        if hasattr(self._backend, "target") and hasattr(self._backend, "num_qubits"):
            return "ibm"
        return None

    def _extract_ibm(self) -> BackendCalibrationSchema:
        """Extract calibration from IBM BackendV2 via backend.target."""
        target = self._backend.target
        num_qubits = self._backend.num_qubits

        single_qubit_errors: List[float] = []
        two_qubit_errors: List[float] = []
        readout_errors: List[float] = []
        t1_values: List[float] = []
        t2_values: List[float] = []
        operational_count = 0

        for qubit in range(num_qubits):
            qprops = self._get_qubit_properties(target, qubit)
            if qprops is None:
                continue
            operational_count += 1

            if hasattr(qprops, "t1") and qprops.t1 is not None:
                t1_values.append(qprops.t1 * 1e6)  # seconds → microseconds
            if hasattr(qprops, "t2") and qprops.t2 is not None:
                t2_values.append(qprops.t2 * 1e6)

        for op_name in target.operation_names:
            props_map = target[op_name]
            if props_map is None:
                continue
            for qargs, props in props_map.items():
                if props is None or props.error is None:
                    continue
                if op_name == "measure":
                    readout_errors.append(props.error)
                elif len(qargs) == 1:
                    single_qubit_errors.append(props.error)
                elif len(qargs) == 2:
                    two_qubit_errors.append(props.error)

        backend_name = getattr(self._backend, "name", str(self._backend))
        timestamp = datetime.now(timezone.utc).isoformat()

        return BackendCalibrationSchema(
            median_single_qubit_gate_error=(
                statistics.median(single_qubit_errors) if single_qubit_errors else None
            ),
            median_two_qubit_gate_error=(
                statistics.median(two_qubit_errors) if two_qubit_errors else None
            ),
            median_readout_error=(statistics.median(readout_errors) if readout_errors else None),
            median_t1_us=statistics.median(t1_values) if t1_values else None,
            median_t2_us=statistics.median(t2_values) if t2_values else None,
            num_operational_qubits=operational_count,
            backend_name=backend_name,
            calibration_timestamp=timestamp,
            provider="ibm",
        )

    def _get_qubit_properties(self, target, qubit: int):
        """Safely get qubit properties from target."""
        try:
            if hasattr(target, "qubit_properties") and target.qubit_properties is not None:
                props = target.qubit_properties
                if callable(props):
                    return props(qubit)
                elif isinstance(props, (list, tuple)) and qubit < len(props):
                    return props[qubit]
                elif hasattr(props, "__getitem__"):
                    return props[qubit]
        except (IndexError, KeyError, TypeError):
            pass
        return None

    def _extract_aws(self) -> BackendCalibrationSchema:
        """Extract calibration from AWS Braket device properties.

        Tries standardized Braket format first (works across providers),
        falls back to Rigetti-specific provider.specs format.
        """
        # Unwrap qiskit-braket-provider wrapper if present
        if hasattr(self._backend, "_device") and hasattr(self._backend._device, "properties"):
            device = self._backend._device
        else:
            device = self._backend
        backend_name = getattr(device, "name", getattr(device, "arn", str(device)))

        single_qubit_errors: List[float] = []
        two_qubit_errors: List[float] = []
        readout_errors: List[float] = []
        t1_values: List[float] = []
        t2_values: List[float] = []
        operational_count = 0

        try:
            props = device.properties
            if hasattr(props, "dict"):
                props_dict = props.dict()
            elif isinstance(props, dict):
                props_dict = props
            else:
                props_dict = {}

            # Try standardized Braket format first (cross-provider)
            standardized = props_dict.get("standardized", {})
            one_q = standardized.get("oneQubitProperties", {})
            two_q = standardized.get("twoQubitProperties", {})

            if one_q:
                for _qubit_id, qdata in one_q.items():
                    operational_count += 1
                    t1_entry = qdata.get("T1", {})
                    t2_entry = qdata.get("T2", {})
                    if t1_entry and t1_entry.get("value") is not None:
                        val = t1_entry["value"]
                        if t1_entry.get("unit") == "us":
                            t1_values.append(val)
                        else:
                            t1_values.append(val * 1e6)
                    if t2_entry and t2_entry.get("value") is not None:
                        val = t2_entry["value"]
                        if t2_entry.get("unit") == "us":
                            t2_values.append(val)
                        else:
                            t2_values.append(val * 1e6)

                    for fid_entry in qdata.get("oneQubitFidelity", []):
                        fid_type = fid_entry.get("fidelityType", {})
                        fid_name = fid_type.get("name", "")
                        fid_val = fid_entry.get("fidelity")
                        if fid_val is None:
                            continue
                        if fid_name == "READOUT":
                            readout_errors.append(1.0 - fid_val)
                        elif fid_name == "RANDOMIZED_BENCHMARKING":
                            single_qubit_errors.append(1.0 - fid_val)

                for _pair_id, pdata in two_q.items():
                    for fid_entry in pdata.get("twoQubitGateFidelity", []):
                        fid_val = fid_entry.get("fidelity")
                        if fid_val is not None:
                            two_qubit_errors.append(1.0 - fid_val)

            # Fallback: Rigetti-specific provider.specs format
            if not one_q:
                provider_data = props_dict.get("provider", {})
                specs = provider_data.get("specs", {})
                qubit_specs = specs.get("1Q", {})
                gate_specs = specs.get("2Q", {})

                for _qubit_id, qdata in qubit_specs.items():
                    operational_count += 1
                    if "T1" in qdata and qdata["T1"] is not None:
                        t1_values.append(qdata["T1"] * 1e6)
                    if "T2" in qdata and qdata["T2"] is not None:
                        t2_values.append(qdata["T2"] * 1e6)
                    if "f1QRB" in qdata and qdata["f1QRB"] is not None:
                        single_qubit_errors.append(1.0 - qdata["f1QRB"])
                    if "fRO" in qdata and qdata["fRO"] is not None:
                        readout_errors.append(1.0 - qdata["fRO"])

                for _gate_id, gdata in gate_specs.items():
                    if "fCZ" in gdata and gdata["fCZ"] is not None:
                        two_qubit_errors.append(1.0 - gdata["fCZ"])
                    elif "fISWAP" in gdata and gdata["fISWAP"] is not None:
                        two_qubit_errors.append(1.0 - gdata["fISWAP"])

        except Exception:
            pass

        timestamp = datetime.now(timezone.utc).isoformat()

        return BackendCalibrationSchema(
            median_single_qubit_gate_error=(
                statistics.median(single_qubit_errors) if single_qubit_errors else None
            ),
            median_two_qubit_gate_error=(
                statistics.median(two_qubit_errors) if two_qubit_errors else None
            ),
            median_readout_error=(statistics.median(readout_errors) if readout_errors else None),
            median_t1_us=statistics.median(t1_values) if t1_values else None,
            median_t2_us=statistics.median(t2_values) if t2_values else None,
            num_operational_qubits=operational_count,
            backend_name=backend_name,
            calibration_timestamp=timestamp,
            provider="aws",
        )
