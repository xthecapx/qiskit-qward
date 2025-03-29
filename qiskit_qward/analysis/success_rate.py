import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Callable, Optional, Union
import numpy as np
from .analysis import Analysis


class SuccessRate(Analysis):
    """
    Class for analyzing success rates of quantum circuits.
    """

    def __init__(self, results_df: pd.DataFrame = None):
        """
        Initialize the SuccessRate analysis.

        Args:
            results_df (pd.DataFrame, optional): DataFrame containing experiment results
        """
        super().__init__(results_df)
        self.success_criteria = self._default_success_criteria()

    def _default_success_criteria(self) -> Callable[[str], bool]:
        """
        Define the default success criteria for the circuit.
        By default, considers all zeros as success.

        Returns:
            Callable[[str], bool]: Function that takes a measurement result and returns True if successful
        """
        return lambda result: result == "0"

    def set_success_criteria(self, criteria: Callable[[str], bool]):
        """
        Set custom success criteria for the circuit.

        Args:
            criteria (Callable[[str], bool]): Function that takes a measurement result and returns True if successful
        """
        self.success_criteria = criteria

    def add_results(self, results: Dict[str, Any]):
        """
        Add new results to the analysis.

        Args:
            results (dict): Dictionary containing experiment results
        """
        self.results_df = pd.concat([self.results_df, pd.DataFrame([results])], ignore_index=True)

    def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze the success rates from the results.

        Args:
            **kwargs: Keyword arguments including:
                target_value (str): The value we consider as success (e.g., '0' for heads)

        Returns:
            dict: Success rate analysis results
        """
        target_value = kwargs.get("target_value", "0")

        if self.results_df.empty:
            raise ValueError("No results available to analyze")

        # Calculate success rate for each job using the custom success criteria
        success_rates = []
        for counts in self.results_df["counts"]:
            total_shots = sum(counts.values())
            successful_shots = 0
            for state, count in counts.items():
                if self.success_criteria(state):
                    successful_shots += count
            success_rates.append(successful_shots / total_shots if total_shots > 0 else 0)

        # Calculate average counts
        avg_heads = self.results_df["counts"].apply(lambda x: x.get("0", 0)).mean()
        avg_tails = self.results_df["counts"].apply(lambda x: x.get("1", 0)).mean()

        analysis = {
            "mean_success_rate": float(np.mean(success_rates)),
            "std_success_rate": float(np.std(success_rates)),
            "min_success_rate": float(np.min(success_rates)),
            "max_success_rate": float(np.max(success_rates)),
            "total_trials": len(self.results_df),
            "target_value": target_value,
            "average_counts": {"heads": float(avg_heads), "tails": float(avg_tails)},
        }

        return analysis

    def plot(self, **kwargs) -> Any:
        """
        Plot the distribution of success rates.

        Args:
            **kwargs: Keyword arguments including:
                target_value (str): The value we consider as success (e.g., '0' for heads)
                ideal_rate (float): The ideal success rate to mark on the plot

        Returns:
            Any: The plot figure or display object
        """
        target_value = kwargs.get("target_value", "0")
        ideal_rate = kwargs.get("ideal_rate", 0.5)

        if self.results_df.empty:
            raise ValueError("No results available to plot")

        # Calculate success rates using the custom success criteria
        success_rates = []
        for counts in self.results_df["counts"]:
            total_shots = sum(counts.values())
            successful_shots = 0
            for state, count in counts.items():
                if self.success_criteria(state):
                    successful_shots += count
            success_rates.append(successful_shots / total_shots if total_shots > 0 else 0)

        plt.figure(figsize=(10, 6))
        plt.hist(success_rates, bins=20, alpha=0.7)
        plt.axvline(x=ideal_rate, color="r", linestyle="--", label=f"Ideal Rate ({ideal_rate:.0%})")
        plt.title("Distribution of Success Rates")
        plt.xlabel("Success Rate")
        plt.ylabel("Count")
        plt.legend()
        return plt.gcf()
