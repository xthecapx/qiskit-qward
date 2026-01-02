"""
QWARD Visualization Module

This module provides visualization tools for QWARD metrics, focusing on
dynamic metrics that benefit from graphical representation using a strategy pattern.
"""

from .base import VisualizationStrategy, PlotConfig, PlotType, PlotMetadata
from .circuit_performance_visualizer import CircuitPerformanceVisualizer
from .qiskit_metrics_visualizer import QiskitVisualizer
from .complexity_metrics_visualizer import ComplexityVisualizer
from .element_metrics_visualizer import ElementMetricsVisualizer
from .structural_metrics_visualizer import StructuralMetricsVisualizer
from .behavioral_metrics_visualizer import BehavioralMetricsVisualizer
from .quantum_specific_metrics_visualizer import QuantumSpecificMetricsVisualizer
from .visualizer import Visualizer
from .constants import Metrics, Plots
from .ieee_config import (
    IEEEPlotConfig,
    IEEELargePlotConfig,
    IEEEPosterConfig,
    IEEE_CONFIG,
    IEEE_LARGE_CONFIG,
    IEEE_POSTER_CONFIG,
)
from .ieee_styling import (
    apply_ieee_rcparams_styling,
    apply_ieee_styling_to_axes,
    IEEE_FONT_SIZES,
    IEEE_STYLING,
)

__all__ = [
    "VisualizationStrategy",
    "PlotConfig",
    "PlotType",
    "PlotMetadata",
    "CircuitPerformanceVisualizer",
    "QiskitVisualizer",
    "ComplexityVisualizer",
    "ElementMetricsVisualizer",
    "StructuralMetricsVisualizer",
    "BehavioralMetricsVisualizer",
    "QuantumSpecificMetricsVisualizer",
    "Visualizer",
    "Metrics",
    "Plots",
    # IEEE configurations
    "IEEEPlotConfig",
    "IEEELargePlotConfig",
    "IEEEPosterConfig",
    "IEEE_CONFIG",
    "IEEE_LARGE_CONFIG",
    "IEEE_POSTER_CONFIG",
    "apply_ieee_rcparams_styling",
    "apply_ieee_styling_to_axes",
    "IEEE_FONT_SIZES",
    "IEEE_STYLING",
]
