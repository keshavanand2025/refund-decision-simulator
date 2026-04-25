"""
Bootstrap Validation Module
============================

Statistical validation via bootstrap resampling (B = 1,000).
Tests null hypothesis H₀: mean cost difference between strategies = 0.

Produces:
    - Bootstrap cost distributions per strategy
    - Pairwise p-values for cost differences
    - 95% confidence intervals
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import Config


def bootstrap_cost_comparison(
    X: np.ndarray,
    y: np.ndarray,
    order_values: np.ndarray,
    model_pipelines: Dict[str, Pipeline],
    config: Optional[Config] = None,
    B: int = 1000,
    sample_size: Optional[int] = None,
    random_seed: int = 42,
    use_cost_weights: bool = True,
) -> pd.DataFrame:
    """Run bootstrap resampling to validate cost differences.

    At each iteration b = 1..B:
        1. Draw stratified bootstrap sample (with replacement)
        2. Split into train/validation
        3. Train each model and compute C_total
        4. Record costs for all strategies

    Args:
        X: Feature matrix.
        y: Binary labels.
        order_values: Per-instance order amounts.
        model_pipelines: Dict of name -> sklearn Pipeline.
        config: Configuration with cost parameters.
        B: Number of bootstrap iterations (default 1000).
        sample_size: Bootstrap sample size (default = len(X)).
        random_seed: Base random seed.
        use_cost_weights: Whether to apply cost-weighted training.

    Returns:
        DataFrame with columns: Strategy, Bootstrap_Costs (list),
        Mean_Cost, Std_Cost, CI_Lower, CI_Upper.
    """
    config = config or Config()
    rng = np.random.RandomState(random_seed)
    n = len(X)
    sample_size = sample_size or n

    alpha = config.fraud_penalty_multiplier
    beta = config.retention_value

    # Storage for per-strategy costs across bootstrap iterations
    cost_distributions: Dict[str, List[float]] = {
        name: [] for name in model_pipelines
    }

    for b in range(B):
        # Stratified bootstrap sample
        idx = rng.choice(n, size=sample_size, replace=True)
        X_boot = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values
        y_boot = y[idx] if isinstance(y, np.ndarray) else y.iloc[idx].values
        ov_boot = order_values[idx] if isinstance(order_values, np.ndarray) else order_values.iloc[idx].values

        # 70/30 split within bootstrap sample
        split = int(0.7 * sample_size)
        X_tr, X_val = X_boot[:split], X_boot[split:]
        y_tr, y_val = y_boot[:split], y_boot[split:]
        ov_tr, ov_val = ov_boot[:split], ov_boot[split:]

        for name, pipe in model_pipelines.items():
            try:
                pipe_clone = clone(pipe)

                # Compute weights if needed
                if use_cost_weights:
                    v_bar = np.mean(ov_tr)
                    weights = np.where(
                        y_tr == 1,
                        alpha * ov_tr / v_bar,
                        beta / v_bar,
                    )
                    pipe_clone.fit(X_tr, y_tr, classifier__sample_weight=weights)
                else:
                    pipe_clone.fit(X_tr, y_tr)

                # Predict and compute cost
                preds = pipe_clone.predict(X_val)
                fn = (preds == 0) & (y_val == 1)
                fp = (preds == 1) & (y_val == 0)
                tp = (preds == 1) & (y_val == 1)
                cost = (
                    alpha * np.sum(ov_val[fp])
                    + beta * np.sum(fn)
                    + np.sum(ov_val[tp])
                )
                cost_distributions[name].append(float(cost))
            except Exception:
                # Skip failed iterations (e.g., single-class bootstrap)
                cost_distributions[name].append(np.nan)

    # Compute statistics
    results = []
    for name, costs in cost_distributions.items():
        costs_clean = [c for c in costs if not np.isnan(c)]
        if costs_clean:
            arr = np.array(costs_clean)
            results.append({
                "Strategy": name,
                "Mean_Cost": np.mean(arr),
                "Std_Cost": np.std(arr),
                "CI_Lower": np.percentile(arr, 2.5),
                "CI_Upper": np.percentile(arr, 97.5),
                "N_Valid": len(costs_clean),
            })

    return pd.DataFrame(results)


def pairwise_p_values(
    X: np.ndarray,
    y: np.ndarray,
    order_values: np.ndarray,
    model_pipelines: Dict[str, Pipeline],
    config: Optional[Config] = None,
    B: int = 1000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Compute pairwise bootstrap p-values for cost differences.

    p-value = (1/B) * sum_b 1[C_s1(b) >= C_s2(b)]

    Tests H₀: mean cost difference = 0 for each pair.

    Args:
        X, y, order_values: Dataset.
        model_pipelines: Dict of strategy name -> pipeline.
        config: Configuration.
        B: Bootstrap iterations.
        random_seed: Random seed.

    Returns:
        DataFrame with columns: Strategy_1, Strategy_2, p_value, Significant.
    """
    config = config or Config()
    rng = np.random.RandomState(random_seed)
    n = len(X)
    alpha = config.fraud_penalty_multiplier
    beta = config.retention_value

    names = list(model_pipelines.keys())
    costs_per_iter: Dict[str, List[float]] = {name: [] for name in names}

    for b in range(B):
        idx = rng.choice(n, size=n, replace=True)
        X_boot = X[idx] if isinstance(X, np.ndarray) else X.iloc[idx].values
        y_boot = y[idx] if isinstance(y, np.ndarray) else y.iloc[idx].values
        ov_boot = order_values[idx] if isinstance(order_values, np.ndarray) else order_values.iloc[idx].values

        split = int(0.7 * n)
        X_tr, X_val = X_boot[:split], X_boot[split:]
        y_tr, y_val = y_boot[:split], y_boot[split:]
        ov_val = ov_boot[split:]

        for name, pipe in model_pipelines.items():
            try:
                p = clone(pipe)
                p.fit(X_tr, y_tr)
                preds = p.predict(X_val)
                fn = (preds == 0) & (y_val == 1)
                fp = (preds == 1) & (y_val == 0)
                tp = (preds == 1) & (y_val == 1)
                cost = alpha * np.sum(ov_val[fp]) + beta * np.sum(fn) + np.sum(ov_val[tp])
                costs_per_iter[name].append(float(cost))
            except Exception:
                costs_per_iter[name].append(np.nan)

    # Pairwise comparisons
    results = []
    for i, s1 in enumerate(names):
        for s2 in names[i + 1:]:
            c1 = np.array(costs_per_iter[s1])
            c2 = np.array(costs_per_iter[s2])
            valid = ~(np.isnan(c1) | np.isnan(c2))
            c1_v, c2_v = c1[valid], c2[valid]

            if len(c1_v) > 0:
                p_val = np.mean(c1_v >= c2_v)
                results.append({
                    "Strategy_1": s1,
                    "Strategy_2": s2,
                    "p_value": round(p_val, 4),
                    "Significant": "Yes" if p_val < 0.01 or p_val > 0.99 else "No",
                })

    return pd.DataFrame(results)
