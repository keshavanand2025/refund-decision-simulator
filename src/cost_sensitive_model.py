"""
Cost-Sensitive Model Module
============================

Novel Contribution #1: Per-instance cost-weighted training.

Instead of treating all misclassifications equally (standard accuracy
optimization), this module derives sample-specific weights from the
economic cost model:

    - False Positive (approve fraudulent refund):
        weight ∝ order_amount × fraud_penalty_multiplier
    - False Negative (deny legitimate refund):
        weight ∝ retention_value

Models trained with these weights directly learn to minimize total
economic cost rather than classification error rate.

This approach differs from standard class-weight balancing (which
assigns uniform weights per class) by incorporating **instance-level**
economic information.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from src.config import Config


def compute_sample_weights(
    data: pd.DataFrame,
    config: Optional[Config] = None,
) -> np.ndarray:
    """
    Derive per-instance training weights from economic cost parameters.

    The weight of each sample reflects the economic impact of
    misclassifying that particular instance:

    - For refundable orders (y=1): weight = order_amount + fraud bonus
      (cost of failing to approve → retention loss, but if approved
      wrongly for fraud → large cost)
    - For non-refundable orders (y=0): weight = order_amount × multiplier
      if high fraud risk (cost of wrongly approving → direct loss)

    The final weights are normalized to have mean = 1.0 to preserve
    gradient magnitudes during training.

    Args:
        data: DataFrame with ``order_amount``, ``fraud_score``,
              ``refunded`` columns.
        config: Configuration with cost parameters.

    Returns:
        NumPy array of per-sample weights (same length as data).

    Example:
        >>> weights = compute_sample_weights(data)
        >>> model.fit(X_train, y_train, sample_weight=weights[train_idx])
    """
    config = config or Config()

    weights = np.ones(len(data), dtype=np.float64)

    for i in range(len(data)):
        row = data.iloc[i]
        if row["refunded"] == 1:
            # Cost of wrongly denying this refund = retention loss
            # Weight higher for high-value customers
            weights[i] = config.retention_value + row["order_amount"] * 0.1
        else:
            # Cost of wrongly approving this = refund amount + fraud penalty
            base_cost = row["order_amount"]
            if row["fraud_score"] > config.fraud_threshold:
                base_cost *= config.fraud_penalty_multiplier
            weights[i] = base_cost

    # Normalize: mean weight = 1.0
    weights = weights / weights.mean()

    return weights


class CostSensitivePipeline:
    """
    ML pipeline that trains models with cost-derived sample weights.

    Compared to the standard ``ModelPipeline``, this pipeline:
    1. Computes per-instance weights from the economic cost model
    2. Passes these weights to ``fit(sample_weight=...)``
    3. Results in models that minimize cost, not just accuracy

    Attributes:
        config: Project configuration.
        models_standard: Models trained without cost weights (baseline).
        models_cost_weighted: Models trained with cost weights (novel).
        comparison_df: Side-by-side comparison of both approaches.

    Example:
        >>> pipeline = CostSensitivePipeline()
        >>> pipeline.prepare_data(data)
        >>> pipeline.train_both()
        >>> comparison = pipeline.compare()
    """

    MODEL_CLASSES = {
        "LogisticRegression": LogisticRegression,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
    }

    if _XGBOOST_AVAILABLE:
        MODEL_CLASSES["XGBoost"] = XGBClassifier

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.models_standard: Dict[str, Pipeline] = {}
        self.models_cost_weighted: Dict[str, Pipeline] = {}

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.sample_weights: Optional[np.ndarray] = None

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "refunded",
    ) -> None:
        """
        Split data and compute sample weights for the training set.

        Args:
            data: Full dataset.
            target_column: Name of target column.
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=y,
            )
        )

        # Store full rows for cost calculation
        self.train_data = data.loc[self.X_train.index].reset_index(drop=True)
        self.test_data = data.loc[self.X_test.index].reset_index(drop=True)

        # Compute cost-based sample weights for training set
        self.sample_weights = compute_sample_weights(
            self.train_data, self.config
        )

    def _build_pipeline(self, model_name: str) -> Pipeline:
        """Build a StandardScaler + Classifier pipeline."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", self.MODEL_CLASSES[model_name](
                random_state=self.config.random_seed
            )),
        ])

    def train_both(self) -> None:
        """
        Train each model twice: once standard, once with cost weights.

        This enables direct comparison of accuracy-optimized vs
        cost-optimized models on the same data split.
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() first.")

        for name in self.MODEL_CLASSES:
            # Standard training (no sample weights)
            pipe_std = self._build_pipeline(name)
            pipe_std.fit(self.X_train, self.y_train)
            self.models_standard[name] = pipe_std

            # Cost-weighted training
            pipe_cw = self._build_pipeline(name)
            pipe_cw.fit(
                self.X_train, self.y_train,
                classifier__sample_weight=self.sample_weights,
            )
            self.models_cost_weighted[name] = pipe_cw

    def compare(self) -> pd.DataFrame:
        """
        Compare standard vs cost-weighted models on test set.

        Returns:
            DataFrame with columns: Model, Training_Type,
            Accuracy, Economic_Cost
        """
        from src.metrics import EconomicMetrics

        econ = EconomicMetrics(self.config)
        rows = []

        for name in self.MODEL_CLASSES:
            for label, models_dict in [
                ("Standard", self.models_standard),
                ("Cost-Weighted", self.models_cost_weighted),
            ]:
                if name not in models_dict:
                    continue

                preds = models_dict[name].predict(self.X_test)
                acc = accuracy_score(self.y_test, preds)
                cost = econ.calculate_total_cost(self.test_data, preds)

                rows.append({
                    "Model": name,
                    "Training_Type": label,
                    "Accuracy": round(acc, 4),
                    "Economic_Cost": round(cost, 2),
                })

        return pd.DataFrame(rows)

    def get_cost_reduction(self) -> pd.DataFrame:
        """
        Calculate the cost reduction achieved by cost-weighted training.

        Returns:
            DataFrame showing per-model cost reduction (absolute and %).
        """
        comparison = self.compare()
        rows = []

        for name in self.MODEL_CLASSES:
            std = comparison[
                (comparison["Model"] == name)
                & (comparison["Training_Type"] == "Standard")
            ]
            cw = comparison[
                (comparison["Model"] == name)
                & (comparison["Training_Type"] == "Cost-Weighted")
            ]

            if len(std) > 0 and len(cw) > 0:
                std_cost = std["Economic_Cost"].values[0]
                cw_cost = cw["Economic_Cost"].values[0]
                reduction = std_cost - cw_cost
                pct = (reduction / std_cost * 100) if std_cost > 0 else 0

                rows.append({
                    "Model": name,
                    "Standard_Cost": std_cost,
                    "CostWeighted_Cost": cw_cost,
                    "Reduction": round(reduction, 2),
                    "Reduction_Pct": round(pct, 2),
                })

        return pd.DataFrame(rows)
