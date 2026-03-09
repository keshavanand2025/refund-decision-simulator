"""
Data Generator Module
=====================

Generates synthetic datasets for refund decision simulation.
Each record represents an order with features relevant to
refund decision-making.

Features generated:
    - order_amount: Value of the order in ₹
    - delay_minutes: Delivery delay in minutes
    - previous_refunds: Number of past refund requests by the customer
    - fraud_score: Probability estimate of the order being fraudulent (0–1)
    - complaint_severity: Customer complaint severity rating (1–5)
    - refunded: Binary target label (1 = refunded, 0 = rejected)
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.config import Config


def generate_dataset(
    config: Optional[Config] = None,
    n_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic refund decision dataset.

    The target variable ``refunded`` is generated probabilistically
    using a sigmoid function over a weighted combination of features,
    ensuring realistic non-linear decision boundaries.

    Args:
        config: Configuration object. Uses default Config() if not provided.
        n_samples: Override the number of samples from config.
        random_seed: Override the random seed from config.

    Returns:
        pd.DataFrame with columns:
            ``order_amount``, ``delay_minutes``, ``previous_refunds``,
            ``fraud_score``, ``complaint_severity``, ``refunded``

    Raises:
        ValueError: If n_samples < 100.

    Example:
        >>> from src.data_generator import generate_dataset
        >>> df = generate_dataset(n_samples=500, random_seed=42)
        >>> df.shape
        (500, 6)
        >>> list(df.columns)
        ['order_amount', 'delay_minutes', 'previous_refunds',
         'fraud_score', 'complaint_severity', 'refunded']
    """
    if config is None:
        config = Config()

    n = n_samples if n_samples is not None else config.n_samples
    seed = random_seed if random_seed is not None else config.random_seed

    if n < 100:
        raise ValueError(f"n_samples must be at least 100, got {n}.")

    rng = np.random.RandomState(seed)
    ranges = config.feature_ranges
    weights = config.label_weights

    # --- Generate features ---
    order_amount = rng.uniform(
        ranges["order_amount"][0], ranges["order_amount"][1], n
    )
    delay_minutes = rng.randint(
        ranges["delay_minutes"][0], ranges["delay_minutes"][1], n
    )
    previous_refunds = rng.randint(
        ranges["previous_refunds"][0], ranges["previous_refunds"][1], n
    )
    fraud_score = rng.uniform(
        ranges["fraud_score"][0], ranges["fraud_score"][1], n
    )
    complaint_severity = rng.randint(
        ranges["complaint_severity"][0], ranges["complaint_severity"][1], n
    )

    # --- Generate probabilistic target label ---
    refund_logit = (
        weights["delay_weight"] * (delay_minutes > weights["delay_threshold"])
        + weights["severity_weight"]
        * (complaint_severity > weights["severity_threshold"])
        + weights["fraud_weight"]
        * (fraud_score > weights["fraud_threshold"])
    )

    refund_prob = _sigmoid(refund_logit)
    refunded = rng.binomial(1, refund_prob)

    # --- Assemble DataFrame ---
    data = pd.DataFrame(
        {
            "order_amount": order_amount,
            "delay_minutes": delay_minutes,
            "previous_refunds": previous_refunds,
            "fraud_score": fraud_score,
            "complaint_severity": complaint_severity,
            "refunded": refunded,
        }
    )

    return data


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid function element-wise.

    Args:
        x: Input array.

    Returns:
        Array of values in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-x))
