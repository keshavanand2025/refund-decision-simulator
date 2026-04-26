"""
Configuration Module
====================

Centralized configuration for the Refund Decision Simulator.
All constants, hyperparameters, and tunable settings are defined here
to ensure reproducibility and easy experimentation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Config:
    """
    Project-wide configuration for the refund decision simulator.

    Attributes:
        random_seed: Global random seed for reproducibility.
        n_samples: Number of synthetic samples to generate.
        test_size: Fraction of data reserved for testing.
        retention_value: Estimated revenue lost when a legitimate
            customer is denied a refund and churns (in ₹).
        fraud_penalty_multiplier: Multiplier applied to order amount
            when a fraudulent refund is approved (e.g., 2.0 = double loss).
        fraud_threshold: Fraud score above which an order is considered
            high-risk for fraud.
        feature_ranges: Min/max bounds for synthetic feature generation.
        model_hyperparams: Hyperparameter grids for model tuning.
    """

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Data Generation ---
    n_samples: int = 1000
    test_size: float = 0.3

    # --- Economic Cost Parameters ---
    retention_value: float = 500.0
    fraud_penalty_multiplier: float = 2.0
    fraud_threshold: float = 0.7

    # --- Feature Ranges ---
    feature_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "order_amount": (100, 2000),
        "delay_minutes": (0, 90),
        "previous_refunds": (0, 5),
        "fraud_score": (0.0, 1.0),
        "complaint_severity": (1, 6),
    })

    # --- Label Generation Weights ---
    label_weights: Dict[str, float] = field(default_factory=lambda: {
        "delay_weight": 0.3,
        "severity_weight": 0.4,
        "fraud_weight": -0.5,
        "delay_threshold": 30,
        "severity_threshold": 3,
        "fraud_threshold": 0.7,
    })

    # --- Model Hyperparameters ---
    model_hyperparams: Dict[str, Dict[str, List[Any]]] = field(
        default_factory=lambda: {
            "LogisticRegression": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "max_iter": [1000],
                "solver": ["lbfgs"],
            },
            "RandomForest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5],
            },
            "GradientBoosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5],
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "use_label_encoder": [False],
                "eval_metric": ["logloss"],
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "verbose": [-1],
            },
        }
    )

    # --- Cross-Validation ---
    cv_folds: int = 5

    # --- Visualization ---
    figure_dpi: int = 150
    color_palette: str = "viridis"

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        if self.n_samples < 100:
            raise ValueError("n_samples must be at least 100.")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")
        if self.retention_value < 0:
            raise ValueError("retention_value cannot be negative.")
        if self.fraud_penalty_multiplier < 1.0:
            raise ValueError("fraud_penalty_multiplier must be >= 1.0.")
