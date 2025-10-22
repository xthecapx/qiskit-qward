"""
IEEE-optimized plot configuration for QWARD visualizations.

This module provides IEEE-compliant PlotConfig classes optimized for IEEE
publication standards, extracted from the styling used in analy.py.
"""

from typing import List, Tuple
from .base import PlotConfig
from .ieee_styling import IEEE_FONT_SIZES, IEEE_STYLING


class IEEEPlotConfig(PlotConfig):
    """
    IEEE-optimized PlotConfig class for publication-quality visualizations.

    This configuration is based on the IEEE formatting styles used in analy.py,
    featuring:
    - Compact figure size suitable for IEEE column widths
    - High DPI (300) for publication quality
    - Professional ColorBrewer palette
    - Bold font weights and appropriate font sizes
    - Clean grid styling
    - Reduced alpha for better print quality
    """

    def __init__(self, figsize: Tuple[int, int] = (8, 6), **kwargs):
        """
        Initialize IEEE PlotConfig.

        Args:
            figsize: Figure size in inches (default: (8, 6) for IEEE standard)
            **kwargs: Additional PlotConfig parameters to override
        """
        # IEEE ColorBrewer palette from analy.py
        ieee_color_palette = [
            "#1b9e77",  # Teal
            "#d95f02",  # Orange
            "#7570b3",  # Purple
            "#e7298a",  # Pink
            "#66a61e",  # Green
            "#e6ab02",  # Yellow
            "#e377c2",  # Pink (light)
            "#7f7f7f",  # Gray
            "#bcbd22",  # Olive
            "#17becf",  # Cyan
        ]

        # IEEE default settings
        ieee_defaults = {
            "dpi": 300,  # High resolution for publication
            "style": "ieee",  # Custom IEEE style identifier
            "color_palette": ieee_color_palette,
            "save_format": "png",  # Standard format for IEEE
            "grid": True,  # Grid enabled with custom styling
            "alpha": 0.7,  # Reduced alpha for better print quality
        }

        # Merge user kwargs with IEEE defaults (user kwargs take precedence)
        final_kwargs = {**ieee_defaults, **kwargs}

        # Initialize parent class with IEEE defaults
        super().__init__(figsize=figsize, **final_kwargs)


class IEEELargePlotConfig(IEEEPlotConfig):
    """IEEE PlotConfig for double-column figures."""

    def __init__(self, figsize: Tuple[int, int] = (10, 8), **kwargs):
        """
        Initialize IEEE Large PlotConfig.

        Args:
            figsize: Figure size in inches (default: (10, 8) for double-column)
            **kwargs: Additional PlotConfig parameters to override
        """
        super().__init__(figsize=figsize, **kwargs)


class IEEEPosterConfig(IEEEPlotConfig):
    """IEEE PlotConfig for poster presentations."""

    def __init__(self, figsize: Tuple[int, int] = (12, 9), **kwargs):
        """
        Initialize IEEE Poster PlotConfig.

        Args:
            figsize: Figure size in inches (default: (12, 9) for poster visibility)
            **kwargs: Additional PlotConfig parameters to override
        """
        # Poster-specific defaults
        poster_defaults = {"save_format": "pdf"}  # Vector format for scaling

        # Merge with user kwargs
        final_kwargs = {**poster_defaults, **kwargs}

        super().__init__(figsize=figsize, **final_kwargs)


# Pre-configured instances for easy access
IEEE_CONFIG = IEEEPlotConfig()
IEEE_LARGE_CONFIG = IEEELargePlotConfig()
IEEE_POSTER_CONFIG = IEEEPosterConfig()
