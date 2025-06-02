##############################
QWARD Documentation
##############################

.. toctree::
  :hidden:

  Home <self>


QWARD is a Python library designed to assist in analyzing quantum circuits and their execution quality on quantum processing units (QPUs) or simulators. It helps developers and researchers understand how their quantum algorithms perform by providing tools to collect and interpret various metrics related to circuit structure and execution outcomes.

QWARD enables users to apply a range of metrics to quantum circuits, gather execution data, and analyze performance characteristics. This aids in understanding circuit behavior, complexity, and potential areas for optimization, whether working with simulated or real hardware backends.

**Key Features:**

- **Simple Unified API**: All metric classes use `get_metrics()` for consistent, type-safe access
- **Schema Validation**: Automatic data validation with IDE support and error prevention
- **Comprehensive Metrics**: Circuit structure, complexity analysis, and performance evaluation
- **Rich Visualization**: Beautiful plots and dashboards for analysis results
- **Extensible Design**: Easy to add custom metrics and success criteria

.. toctree::
  :maxdepth: 1
  :caption: Documentation

  quickstart_guide
  beginners_guide
  technical_docs
  project_overview
  architecture
  visualization_guide
  Tutorials <tutorials/index>
  User Guide <how_tos/index>
  API References <apidocs/index>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
