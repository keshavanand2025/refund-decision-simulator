"""
Refund Decision Simulator
=========================

A simulation framework comparing rule-based and machine learning approaches
for refund decision systems with economic cost evaluation.

Modules:
    - config: Centralized project configuration and constants
    - data_generator: Synthetic dataset generation
    - rule_engine: Rule-based decision strategies
    - model: ML model training, tuning, and evaluation
    - metrics: Economic cost and classification metrics
    - visualization: Publication-quality plotting utilities
"""

from src.config import Config
from src.data_generator import generate_dataset
from src.rule_engine import RuleEngine
from src.model import ModelPipeline
from src.metrics import EconomicMetrics

__version__ = "1.0.0"
__author__ = "Keshav Anand"

__all__ = [
    "Config",
    "generate_dataset",
    "RuleEngine",
    "ModelPipeline",
    "EconomicMetrics",
]
