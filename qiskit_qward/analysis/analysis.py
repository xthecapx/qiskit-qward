import pandas as pd
from typing import Dict, Any
from abc import ABC, abstractmethod


class Analysis(ABC):
    """
    Base class for analyzing quantum circuit results and experiments.
    """

    def __init__(self, results_df: pd.DataFrame = None):
        """
        Initialize the Analysis class.

        Args:
            results_df (pd.DataFrame, optional): DataFrame containing experiment results
        """
        self.results_df = results_df if results_df is not None else pd.DataFrame()

    @abstractmethod
    def add_results(self, results: Dict[str, Any]):
        """
        Add new results to the analysis.

        Args:
            results (dict): Dictionary containing experiment results
        """
        pass

    @abstractmethod
    def analyze(self, **kwargs) -> Dict[str, Any]:
        """
        Analyze the results.

        Args:
            **kwargs: Additional analysis parameters

        Returns:
            dict: Analysis results
        """
        pass

    @abstractmethod
    def plot(self, **kwargs):
        """
        Plot the analysis results.

        Args:
            **kwargs: Additional plotting parameters
        """
        pass

    def export_results(self, filename: str):
        """
        Export results to a CSV file.

        Args:
            filename: Name of the file to save results to
        """
        if self.results_df.empty:
            raise ValueError("No results available to export")

        self.results_df.to_csv(filename, index=False)

    def get_results(self) -> pd.DataFrame:
        """
        Get the current results DataFrame.

        Returns:
            pd.DataFrame: The results DataFrame
        """
        return self.results_df

    def clear_results(self):
        """
        Clear all stored results.
        """
        self.results_df = pd.DataFrame()
