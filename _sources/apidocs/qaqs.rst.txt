=================================
QAQS: Quality and Cost Prediction
=================================

Predict the execution quality and billed cost of a circuit before running it. See the
:doc:`../qaqs` guide for how to use it and how far to trust it.

.. code-block:: python

    from qward.qaqs import predict

    result = predict(circuit, backend)
    result.quality["ibm_fez/success"].value
    result.cost.resource_units

Prediction
----------

.. autofunction:: qward.qaqs.predict.predict

.. autofunction:: qward.qaqs.predict.predict_cost

.. autoclass:: qward.qaqs.predict.QAQSResult
   :members:

.. autoclass:: qward.qaqs.predict.QualityPrediction
   :members:

.. autoclass:: qward.qaqs.predict.CostPrediction
   :members:

Footprint
---------

.. automodule:: qward.qaqs.footprint
   :members: compute_footprint, logical_footprint, compiled_footprint, quantum_specific, ensure_barrier, compile_circuit, scheduled_duration_seconds
   :undoc-members:

Models
------

.. autofunction:: qward.qaqs.models.available_models

.. autofunction:: qward.qaqs.models.model_card

.. autoclass:: qward.qaqs.models.ModelCard
   :members:
