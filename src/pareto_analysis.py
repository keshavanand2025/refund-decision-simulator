"""
Pareto Analysis Module
=======================

Novel Contribution #4: Multi-objective Pareto front analysis.

Frames refund decision strategy selection as a multi-objective
optimization problem with two competing objectives:

    1. Maximize accuracy (or equivalently, minimize error rate)
    2. Minimize economic cost

A strategy is **Pareto-optimal** if no other strategy is simultaneously
better on BOTH objectives. The set of all Pareto-optimal strategies
forms the **Pareto front** — the boundary of achievable tradeoffs.

Strategies below the Pareto front are **dominated** (there exists a
better alternative that beats them on both metrics).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.metrics import EconomicMetrics, classification_metrics


def compute_strategy_objectives(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    data: pd.DataFrame,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Compute accuracy and economic cost for all strategies.

    Args:
        y_true: True labels.
        predictions_dict: strategy_name → predictions.
        data: Test set DataFrame.
        config: Configuration.

    Returns:
        DataFrame with columns: Strategy, Accuracy, Economic_Cost,
        Error_Rate, Approval_Rate
    """
    config = config or Config()
    econ = EconomicMetrics(config)
    rows = []

    for name, preds in predictions_dict.items():
        metrics = classification_metrics(y_true, preds)
        cost = econ.calculate_total_cost(data, preds)

        rows.append({
            "Strategy": name,
            "Accuracy": metrics["accuracy"],
            "Error_Rate": 1.0 - metrics["accuracy"],
            "Economic_Cost": cost,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1_score"],
            "Approval_Rate": preds.mean(),
        })

    return pd.DataFrame(rows)


def find_pareto_front(
    objectives_df: pd.DataFrame,
    minimize_cols: List[str] = None,
    maximize_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Identify Pareto-optimal strategies.

    A strategy is Pareto-optimal if no other strategy is strictly better
    on ALL specified objectives simultaneously.

    Default objectives:
        - Minimize: Economic_Cost
        - Maximize: Accuracy

    Args:
        objectives_df: DataFrame with strategy metrics.
        minimize_cols: Columns to minimize (default: ["Economic_Cost"]).
        maximize_cols: Columns to maximize (default: ["Accuracy"]).

    Returns:
        DataFrame of only Pareto-optimal strategies, with an added
        ``is_pareto_optimal`` column.
    """
    if minimize_cols is None:
        minimize_cols = ["Economic_Cost"]
    if maximize_cols is None:
        maximize_cols = ["Accuracy"]

    n = len(objectives_df)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            dominates = True

            for col in minimize_cols:
                if objectives_df.iloc[j][col] > objectives_df.iloc[i][col]:
                    dominates = False
                    break

            if dominates:
                for col in maximize_cols:
                    if objectives_df.iloc[j][col] < objectives_df.iloc[i][col]:
                        dominates = False
                        break

            if dominates:
                is_pareto[i] = False
                break

    result = objectives_df.copy()
    result["Is_Pareto_Optimal"] = is_pareto
    return result


def pareto_summary(pareto_df: pd.DataFrame) -> str:
    """
    Generate a human-readable summary of the Pareto analysis.

    Args:
        pareto_df: Output of find_pareto_front.

    Returns:
        Multi-line string summarizing findings.
    """
    optimal = pareto_df[pareto_df["Is_Pareto_Optimal"]]
    dominated = pareto_df[~pareto_df["Is_Pareto_Optimal"]]

    lines = [
        f"Total strategies evaluated: {len(pareto_df)}",
        f"Pareto-optimal strategies:  {len(optimal)}",
        f"Dominated strategies:       {len(dominated)}",
        "",
        "🏆 Pareto-Optimal Strategies:",
    ]

    for _, row in optimal.iterrows():
        lines.append(
            f"  • {row['Strategy']}: "
            f"Accuracy={row['Accuracy']:.3f}, "
            f"Cost=₹{row['Economic_Cost']:,.0f}"
        )

    if len(dominated) > 0:
        lines.append("")
        lines.append("❌ Dominated Strategies:")
        for _, row in dominated.iterrows():
            lines.append(
                f"  • {row['Strategy']}: "
                f"Accuracy={row['Accuracy']:.3f}, "
                f"Cost=₹{row['Economic_Cost']:,.0f}"
            )

    return "\n".join(lines)


def extended_pareto_with_thresholds(
    y_true: np.ndarray,
    probas_dict: Dict[str, np.ndarray],
    rule_predictions: Dict[str, np.ndarray],
    data: pd.DataFrame,
    config: Optional[Config] = None,
    threshold_steps: int = 20,
) -> pd.DataFrame:
    """
    Build an extended Pareto analysis by including multiple thresholds
    per ML model as separate "strategies".

    This creates a richer set of strategies to analyze:
    - Each rule-based strategy (fixed)
    - Each ML model × multiple thresholds (e.g., 0.3, 0.4, 0.5, 0.6, 0.7)

    Args:
        y_true: True labels.
        probas_dict: model_name → predicted probabilities.
        rule_predictions: rule_strategy_name → predictions.
        data: Test set DataFrame.
        config: Configuration.
        threshold_steps: Number of thresholds to try per model.

    Returns:
        Full Pareto analysis DataFrame with Is_Pareto_Optimal column.
    """
    config = config or Config()

    # Start with rule-based predictions
    all_predictions: Dict[str, np.ndarray] = {}
    for name, preds in rule_predictions.items():
        all_predictions[name] = preds

    # Add ML predictions at multiple thresholds
    thresholds = np.linspace(0.1, 0.9, threshold_steps)
    for model_name, proba in probas_dict.items():
        for t in thresholds:
            strategy_name = f"{model_name}@{t:.2f}"
            all_predictions[strategy_name] = (proba >= t).astype(int)

    # Compute objectives and find Pareto front
    objectives = compute_strategy_objectives(
        y_true, all_predictions, data, config
    )
    return find_pareto_front(objectives)
