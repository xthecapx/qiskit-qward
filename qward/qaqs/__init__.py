"""QAQS: predict execution quality and cost of a quantum circuit before running it.

    from qward.qaqs import predict
    result = predict(circuit, backend)       # backend: a real IBM backend or FakeFez()
    result.footprint                         # the pre-runtime metrics
    result.quality["ibm_fez/success"]        # a prediction with its validation score
    result.cost                              # billed QPU time and resource units

The models are frozen results of the thesis "Estimating the quality of quantum circuits before
execution": quality models for IBM Fez and IBM Kingston (circuit footprint only) and one
execution-cost model for every chip. They ship as plain JSON, run on NumPy alone, and never
load pickled code.
"""

from qward.qaqs.footprint import compute_footprint
from qward.qaqs.models import CHIPS, TARGETS, available_models, model_card
from qward.qaqs.predict import (
    VALIDATED_SHOTS,
    CostPrediction,
    QAQSResult,
    QualityPrediction,
    predict,
    predict_cost,
)

__all__ = [
    "CHIPS",
    "TARGETS",
    "VALIDATED_SHOTS",
    "CostPrediction",
    "QAQSResult",
    "QualityPrediction",
    "available_models",
    "compute_footprint",
    "model_card",
    "predict",
    "predict_cost",
]
