"""
Modele uczenia maszynowego dla UCI Drug Consumption Analyzer
"""

from .clustering import ClusterAnalyzer
from .classification import ClassificationManager

__all__ = ['ClusterAnalyzer', 'ClassificationManager']