Visualization Base Classes
==========================

The visualization system in QWARD provides a structured approach to creating beautiful, informative plots of quantum circuit analysis results. The base classes define the common interface and utilities used by all visualization strategies.

**Key Features:**
- **Strategy Pattern**: Extensible design for different visualization types
- **Consistent Styling**: Unified plot configuration and appearance
- **Automatic Integration**: Seamless integration with Scanner and metric results
- **Flexible Output**: Support for saving, showing, and customizing plots

Base Visualization Strategy
----------------------------

The `VisualizationStrategy` class is the abstract base class for all visualizers in QWARD. It provides common functionality and defines the interface that all concrete visualizers must implement.

**Core Responsibilities:**
- **Output Management**: Handles directory creation and file path management
- **Plot Configuration**: Integrates with `PlotConfig` for consistent styling
- **Common Utilities**: Provides reusable methods for data validation and plot creation
- **Abstract Interface**: Defines `create_dashboard()` and `plot_all()` methods

**Usage Pattern:**

.. code-block:: python

    from qward.visualization.base import VisualizationStrategy, PlotConfig
    import pandas as pd
    
    class MyCustomVisualizer(VisualizationStrategy):
        def __init__(self, metrics_dict, output_dir="plots", config=None):
            super().__init__(output_dir, config)
            self.metrics_dict = metrics_dict
        
        def create_dashboard(self, save=True, show=False):
            # Create comprehensive dashboard
            dashboard_plots = {}
            # Implementation here...
            return dashboard_plots
        
        def plot_all(self, save=True, show=False):
            # Create all individual plots
            all_plots = {}
            # Implementation here...
            return all_plots

API Reference
-------------

.. automodule:: qward.visualization.base
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: qward.visualization.base.VisualizationStrategy
   :members:
   :undoc-members:
   :show-inheritance:

Plot Configuration
------------------

The `PlotConfig` class provides comprehensive configuration options for plot appearance and behavior.

**Configuration Options:**
- **Appearance**: Figure size, DPI, color palettes, transparency
- **Styling**: Plot styles (default, quantum, minimal)
- **Output**: Save format (PNG, SVG), grid settings
- **Extensible**: Easy to customize for different visualization needs

**Usage Example:**

.. code-block:: python

    from qward.visualization.base import PlotConfig
    
    # Create custom plot configuration
    config = PlotConfig(
        figsize=(12, 8),           # Larger figures
        dpi=150,                   # High quality
        style="quantum",           # Quantum-themed styling
        color_palette=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        save_format="svg",         # Vector graphics
        grid=True,                 # Show grid lines
        alpha=0.8                  # Transparency level
    )
    
    # Use with any visualizer
    from qward.visualization import QiskitVisualizer
    visualizer = QiskitVisualizer(metrics_dict, config=config)

.. autoclass:: qward.visualization.base.PlotConfig
   :members:
   :undoc-members:
   :show-inheritance: 