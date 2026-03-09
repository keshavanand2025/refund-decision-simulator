"""
Threshold Optimizer Module
===========================

Novel Contribution #2: Economic cost-optimal decision threshold search.

Standard classifiers use a fixed threshold of 0.5 to convert predicted
probabilities into binary decisions. This module demonstrates that the
**cost-optimal threshold** can differ significantly from 0.5 and depends
on the specific economic cost structure.

Key insight: A threshold that maximizes accuracy may NOT minimize
economic cost, because the business impact of false positives (approving
fraudulent refunds) differs from false negatives (denying legitimate ones).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.metrics import EconomicMetrics, classification_metrics


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    data: pd.DataFrame,
    config: Optional[Config] = None,
    n_steps: int = 100,
) -> pd.DataFrame:
    """
    Evaluate economic cost and accuracy at every threshold in [0, 1].

    For each threshold t, predictions are:
        y_pred = 1 if y_proba >= t, else 0

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for the positive class.
        data: DataFrame with ``order_amount``, ``fraud_score``,
              ``refunded`` columns (aligned with y_true).
        config: Configuration with cost parameters.
        n_steps: Number of threshold steps (default 100 → step size 0.01).

    Returns:
        DataFrame with columns:
            ``threshold``, ``accuracy``, ``precision``, ``recall``,
            ``f1_score``, ``economic_cost``, ``approval_rate``
    """
    config = config or Config()
    econ = EconomicMetrics(config)

    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    rows = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        metrics = classification_metrics(y_true, y_pred)
        cost = econ.calculate_total_cost(data, y_pred)
        approval_rate = y_pred.mean()

        rows.append({
            "threshold": round(t, 4),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "economic_cost": cost,
            "approval_rate": approval_rate,
        })

    return pd.DataFrame(rows)


def find_optimal_thresholds(
    sweep_results: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Find thresholds that optimize different objectives.

    Returns:
        Dictionary with keys:
            - ``cost_optimal``: threshold minimizing economic cost
            - ``accuracy_optimal``: threshold maximizing accuracy
            - ``f1_optimal``: threshold maximizing F1 score

        Each value is a dict with ``threshold`` and the metric value.
    """
    cost_idx = sweep_results["economic_cost"].idxmin()
    acc_idx = sweep_results["accuracy"].idxmax()
    f1_idx = sweep_results["f1_score"].idxmax()

    return {
        "cost_optimal": {
            "threshold": sweep_results.loc[cost_idx, "threshold"],
            "economic_cost": sweep_results.loc[cost_idx, "economic_cost"],
            "accuracy": sweep_results.loc[cost_idx, "accuracy"],
        },
        "accuracy_optimal": {
            "threshold": sweep_results.loc[acc_idx, "threshold"],
            "economic_cost": sweep_results.loc[acc_idx, "economic_cost"],
            "accuracy": sweep_results.loc[acc_idx, "accuracy"],
        },
        "f1_optimal": {
            "threshold": sweep_results.loc[f1_idx, "threshold"],
            "economic_cost": sweep_results.loc[f1_idx, "economic_cost"],
            "accuracy": sweep_results.loc[f1_idx, "accuracy"],
        },
    }


def compare_thresholds_across_models(
    y_true: np.ndarray,
    probas_dict: Dict[str, np.ndarray],
    data: pd.DataFrame,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Compare optimal thresholds across multiple models.

    Args:
        y_true: True labels.
        probas_dict: model_name → predicted probabilities.
        data: Test set DataFrame.
        config: Configuration.

    Returns:
        DataFrame with model, optimal thresholds for cost/accuracy/F1,
        and the gap between cost-optimal and accuracy-optimal thresholds.
    """
    config = config or Config()
    rows = []

    for name, proba in probas_dict.items():
        sweep = sweep_thresholds(y_true, proba, data, config)
        optimal = find_optimal_thresholds(sweep)

        rows.append({
            "Model": name,
            "Cost_Optimal_Threshold": optimal["cost_optimal"]["threshold"],
            "Cost_at_Optimal": optimal["cost_optimal"]["economic_cost"],
            "Accuracy_Optimal_Threshold": optimal["accuracy_optimal"]["threshold"],
            "Accuracy_at_Optimal": optimal["accuracy_optimal"]["accuracy"],
            "Threshold_Gap": abs(
                optimal["cost_optimal"]["threshold"]
                - optimal["accuracy_optimal"]["threshold"]
            ),
            "Cost_at_Default_0.5": sweep[
                sweep["threshold"] == 0.5
            ]["economic_cost"].values[0] if 0.5 in sweep["threshold"].values else None,
            "Cost_Savings_vs_Default": (
                sweep[sweep["threshold"] == 0.5]["economic_cost"].values[0]
                - optimal["cost_optimal"]["economic_cost"]
            ) if 0.5 in sweep["threshold"].values else None,
        })

    return pd.DataFrame(rows)
