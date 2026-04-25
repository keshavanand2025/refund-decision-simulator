"""
Refund Decision Simulator (CODA)
=================================

A simulation framework comparing rule-based and machine learning approaches
for refund decision systems with economic cost evaluation.

Implements the Cost-Optimal Decision Algorithm (CODA) with:
    - Per-instance cost-weighted training
    - Cost-optimal threshold search
    - Three-tier decision output (approve / review / deny)
    - Bootstrap statistical validation

Modules:
    - config: Centralized project configuration and constants
    - data_generator: Synthetic dataset generation
    - rule_engine: Rule-based decision strategies
    - model: ML model training (6 models incl. LightGBM, MetaCost, CS-SVM)
    - metrics: Economic cost and classification metrics
    - visualization: Publication-quality plotting utilities
    - coda: Cost-Optimal Decision Algorithm (Algorithm 1)
    - bootstrap_validation: Statistical bootstrap resampling (B=1000)
    - dataset_loader: IEEE-CIS and PaySim dataset loaders with PCA alignment
"""

from src.config import Config
from src.data_generator import generate_dataset
from src.rule_engine import RuleEngine
from src.model import ModelPipeline
from src.metrics import EconomicMetrics
from src.coda import CODA, ThreeTierDecision, run_ablation_study
from src.bootstrap_validation import bootstrap_cost_comparison, pairwise_p_values

__version__ = "2.0.0"
__author__ = "Keshav Anand"

__all__ = [
    "Config",
    "generate_dataset",
    "RuleEngine",
    "ModelPipeline",
    "EconomicMetrics",
    "CODA",
    "ThreeTierDecision",
    "run_ablation_study",
    "bootstrap_cost_comparison",
    "pairwise_p_values",
]
