"""
Refund Decision Simulator
=========================

A simulation framework comparing rule-based and machine learning approaches
for refund decision systems with economic cost evaluation.

Includes the Cost-Optimal Decision Algorithm (CODA) with:
    - Per-instance cost-weighted training
    - Cost-optimal threshold search
    - Three-tier decision output (approve / review / deny)
    - Bootstrap resampling validation
    - Ablation study framework

Modules:
    - config: Centralized project configuration and constants
    - data_generator: Synthetic dataset generation
    - dataset_loader: IEEE-CIS (full-scale) and PaySim dataset loading
    - rule_engine: Rule-based decision strategies
    - model: ML model training, tuning, and evaluation
    - metrics: Economic cost and classification metrics
    - visualization: Publication-quality plotting utilities
    - cost_sensitive_model: Per-instance cost-weighted training
    - threshold_optimizer: Cost-optimal threshold search
    - sensitivity_analysis: Dynamic cost sensitivity analysis
    - pareto_analysis: Multi-objective Pareto front analysis
    - coda: CODA and CODA+ algorithms, three-tier decisions, bootstrap, ablation
"""

from src.config import Config
from src.data_generator import generate_dataset
from src.rule_engine import RuleEngine
from src.model import ModelPipeline
from src.metrics import EconomicMetrics
from src.coda import CODA, CODAPlus, ThreeTierDecision, bootstrap_validation, ablation_study
from src.dataset_loader import load_ieee_cis, load_paysim

__version__ = "2.0.0"
__author__ = "Keshav Anand"

__all__ = [
    "Config",
    "generate_dataset",
    "RuleEngine",
    "ModelPipeline",
    "EconomicMetrics",
    "CODA",
    "CODAPlus",
    "ThreeTierDecision",
    "bootstrap_validation",
    "ablation_study",
    "load_ieee_cis",
    "load_paysim",
]
