"""
Rule Engine Module
==================

Implements multiple rule-based heuristic strategies for refund decisions.
These serve as baselines to compare against machine learning models.

Strategies:
    - **simple**: Original balanced heuristic
    - **conservative**: Bias toward rejection (reduces fraud risk)
    - **lenient**: Bias toward approval (maximizes customer satisfaction)
"""

import numpy as np
import pandas as pd
from typing import List, Literal


# ---------- Individual Decision Functions ----------


def rule_simple(
    order_value: float, delay_minutes: int, past_refunds: int
) -> int:
    """
    Simple balanced rule-based refund decision.

    Logic:
        1. If delivery delay > 30 min → approve refund
        2. If customer has > 3 past refunds → reject (abuse risk)
        3. If order value < 200 → approve (low-cost, retain customer)
        4. Otherwise → reject

    Args:
        order_value: The monetary value of the order.
        delay_minutes: Delivery delay in minutes.
        past_refunds: Number of previous refund requests.

    Returns:
        1 (approve refund) or 0 (reject refund).

    Example:
        >>> rule_simple(150, 10, 0)
        1
        >>> rule_simple(500, 10, 4)
        0
    """
    if delay_minutes > 30:
        return 1
    if past_refunds > 3:
        return 0
    if order_value < 200:
        return 1
    return 0


def rule_conservative(
    order_value: float,
    delay_minutes: int,
    past_refunds: int,
    fraud_score: float,
) -> int:
    """
    Conservative rule — biased toward rejection to minimize fraud losses.

    Logic:
        1. If fraud score > 0.5 → reject immediately
        2. If past refunds > 2 → reject
        3. If delay > 45 min AND order value < 500 → approve
        4. Otherwise → reject

    Args:
        order_value: The monetary value of the order.
        delay_minutes: Delivery delay in minutes.
        past_refunds: Number of previous refund requests.
        fraud_score: Probability of fraud (0–1).

    Returns:
        1 (approve refund) or 0 (reject refund).
    """
    if fraud_score > 0.5:
        return 0
    if past_refunds > 2:
        return 0
    if delay_minutes > 45 and order_value < 500:
        return 1
    return 0


def rule_lenient(
    order_value: float,
    delay_minutes: int,
    complaint_severity: int,
    fraud_score: float,
) -> int:
    """
    Lenient rule — biased toward approval to maximize customer retention.

    Logic:
        1. If fraud score > 0.8 → reject (only reject very high fraud)
        2. If delay > 15 min → approve
        3. If complaint severity >= 3 → approve
        4. If order value < 500 → approve
        5. Otherwise → reject

    Args:
        order_value: The monetary value of the order.
        delay_minutes: Delivery delay in minutes.
        complaint_severity: Customer complaint severity (1–5).
        fraud_score: Probability of fraud (0–1).

    Returns:
        1 (approve refund) or 0 (reject refund).
    """
    if fraud_score > 0.8:
        return 0
    if delay_minutes > 15:
        return 1
    if complaint_severity >= 3:
        return 1
    if order_value < 500:
        return 1
    return 0


# ---------- Batch Application ----------


class RuleEngine:
    """
    Applies rule-based strategies to an entire DataFrame.

    Supports three strategies: ``'simple'``, ``'conservative'``, ``'lenient'``.

    Example:
        >>> engine = RuleEngine()
        >>> predictions = engine.predict(df, strategy='simple')
        >>> predictions_all = engine.predict_all_strategies(df)
    """

    STRATEGIES = ("simple", "conservative", "lenient")

    def predict(
        self,
        data: pd.DataFrame,
        strategy: Literal["simple", "conservative", "lenient"] = "simple",
    ) -> np.ndarray:
        """
        Apply a rule-based strategy to the dataset.

        Args:
            data: DataFrame with required feature columns.
            strategy: One of 'simple', 'conservative', 'lenient'.

        Returns:
            NumPy array of binary predictions (0 or 1).

        Raises:
            ValueError: If strategy is not recognized.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from {self.STRATEGIES}."
            )

        if strategy == "simple":
            return np.array([
                rule_simple(row.order_amount, row.delay_minutes,
                            row.previous_refunds)
                for _, row in data.iterrows()
            ])

        elif strategy == "conservative":
            return np.array([
                rule_conservative(row.order_amount, row.delay_minutes,
                                  row.previous_refunds, row.fraud_score)
                for _, row in data.iterrows()
            ])

        elif strategy == "lenient":
            return np.array([
                rule_lenient(row.order_amount, row.delay_minutes,
                             row.complaint_severity, row.fraud_score)
                for _, row in data.iterrows()
            ])

    def predict_all_strategies(
        self, data: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """
        Apply all three strategies and return a dict of predictions.

        Args:
            data: DataFrame with required feature columns.

        Returns:
            Dictionary mapping strategy name → prediction array.
        """
        return {
            strategy: self.predict(data, strategy)
            for strategy in self.STRATEGIES
        }
