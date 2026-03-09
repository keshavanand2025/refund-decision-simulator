"""
Sensitivity Analysis Module
=============================

Novel Contribution #3: Dynamic cost environment analysis.

Demonstrates that the answer to "which strategy is best?" is NOT fixed —
it depends on the cost environment. By varying cost parameters
(retention_value, fraud_penalty_multiplier), we show:

1. Under which cost regimes rule-based systems outperform ML (and vice versa)
2. The crossover points where optimal strategy switches
3. Heatmaps of total cost as a function of two cost parameters
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import Config
from src.metrics import EconomicMetrics


def single_parameter_sweep(
    data: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    parameter_name: str,
    parameter_range: np.ndarray,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Sweep a single cost parameter and compute costs for all strategies.

    Args:
        data: Test set DataFrame.
        predictions_dict: strategy_name → predictions array.
        parameter_name: One of 'retention_value' or 'fraud_penalty_multiplier'.
        parameter_range: Array of values to sweep.
        config: Base configuration (modified parameter overrides).

    Returns:
        DataFrame with columns: parameter_value, strategy_name, total_cost

    Raises:
        ValueError: If parameter_name not supported.
    """
    if parameter_name not in ("retention_value", "fraud_penalty_multiplier"):
        raise ValueError(
            f"Unsupported parameter: {parameter_name}. "
            "Use 'retention_value' or 'fraud_penalty_multiplier'."
        )

    base_config = config or Config()
    rows = []

    for val in parameter_range:
        # Create modified config
        cfg = Config(
            retention_value=(
                val if parameter_name == "retention_value"
                else base_config.retention_value
            ),
            fraud_penalty_multiplier=(
                val if parameter_name == "fraud_penalty_multiplier"
                else base_config.fraud_penalty_multiplier
            ),
            fraud_threshold=base_config.fraud_threshold,
        )
        econ = EconomicMetrics(cfg)

        for strategy_name, preds in predictions_dict.items():
            cost = econ.calculate_total_cost(data, preds)
            rows.append({
                "parameter_value": val,
                "strategy": strategy_name,
                "total_cost": cost,
            })

    return pd.DataFrame(rows)


def find_crossover_points(
    sweep_df: pd.DataFrame,
    strategy_a: str,
    strategy_b: str,
) -> List[float]:
    """
    Find parameter values where strategy A becomes cheaper than B (or vice versa).

    Args:
        sweep_df: Output of single_parameter_sweep.
        strategy_a: Name of first strategy.
        strategy_b: Name of second strategy.

    Returns:
        List of approximate crossover parameter values.
    """
    a_costs = sweep_df[sweep_df["strategy"] == strategy_a].sort_values(
        "parameter_value"
    )
    b_costs = sweep_df[sweep_df["strategy"] == strategy_b].sort_values(
        "parameter_value"
    )

    if len(a_costs) == 0 or len(b_costs) == 0:
        return []

    # Merge on parameter value
    merged = a_costs.merge(
        b_costs, on="parameter_value", suffixes=("_a", "_b")
    )

    diff = merged["total_cost_a"].values - merged["total_cost_b"].values
    crossovers = []

    for i in range(1, len(diff)):
        if diff[i - 1] * diff[i] < 0:  # Sign change
            # Linear interpolation for crossover point
            x0 = merged["parameter_value"].values[i - 1]
            x1 = merged["parameter_value"].values[i]
            y0 = diff[i - 1]
            y1 = diff[i]
            crossover = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossovers.append(round(crossover, 2))

    return crossovers


def dual_parameter_heatmap(
    data: pd.DataFrame,
    predictions_dict: Dict[str, np.ndarray],
    retention_range: np.ndarray,
    fraud_multiplier_range: np.ndarray,
    config: Optional[Config] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute costs on a 2D grid of (retention_value × fraud_multiplier).

    For each cell, determines which strategy achieves the lowest cost.

    Args:
        data: Test set DataFrame.
        predictions_dict: strategy_name → predictions.
        retention_range: Array of retention values to sweep.
        fraud_multiplier_range: Array of fraud multipliers to sweep.
        config: Base configuration.

    Returns:
        Dictionary with:
            - ``cost_grids``: strategy_name → 2D cost array
            - ``best_strategy_grid``: 2D array of indices of the cheapest strategy
            - ``strategy_names``: ordered list of strategy names
            - ``retention_range``: retention values used
            - ``fraud_multiplier_range``: fraud multiplier values used
    """
    base_config = config or Config()
    strategy_names = list(predictions_dict.keys())
    n_ret = len(retention_range)
    n_fraud = len(fraud_multiplier_range)

    cost_grids = {name: np.zeros((n_ret, n_fraud)) for name in strategy_names}
    best_strategy_grid = np.zeros((n_ret, n_fraud), dtype=int)

    for i, ret_val in enumerate(retention_range):
        for j, fraud_val in enumerate(fraud_multiplier_range):
            cfg = Config(
                retention_value=ret_val,
                fraud_penalty_multiplier=fraud_val,
                fraud_threshold=base_config.fraud_threshold,
            )
            econ = EconomicMetrics(cfg)

            best_cost = float("inf")
            best_idx = 0

            for k, (name, preds) in enumerate(predictions_dict.items()):
                cost = econ.calculate_total_cost(data, preds)
                cost_grids[name][i, j] = cost

                if cost < best_cost:
                    best_cost = cost
                    best_idx = k

            best_strategy_grid[i, j] = best_idx

    return {
        "cost_grids": cost_grids,
        "best_strategy_grid": best_strategy_grid,
        "strategy_names": strategy_names,
        "retention_range": retention_range,
        "fraud_multiplier_range": fraud_multiplier_range,
    }


def winner_summary(
    heatmap_result: Dict,
) -> pd.DataFrame:
    """
    Summarize what percentage of the cost-parameter space each strategy wins.

    Args:
        heatmap_result: Output of dual_parameter_heatmap.

    Returns:
        DataFrame with strategy names and their win percentages.
    """
    grid = heatmap_result["best_strategy_grid"]
    names = heatmap_result["strategy_names"]
    total_cells = grid.size

    rows = []
    for i, name in enumerate(names):
        wins = (grid == i).sum()
        rows.append({
            "Strategy": name,
            "Wins": int(wins),
            "Win_Pct": round(wins / total_cells * 100, 1),
        })

    return pd.DataFrame(rows).sort_values("Win_Pct", ascending=False)
