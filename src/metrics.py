"""
Metrics Module
==============

Comprehensive evaluation metrics for refund decision systems,
including both classification accuracy and economic cost analysis.

The economic cost model considers three components:
    1. **Refund cost**: Direct cost of approving a refund.
    2. **Fraud penalty**: Additional loss when a fraudulent order is refunded.
    3. **Retention loss**: Revenue lost when a legitimate refund is wrongly denied.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Config


class EconomicMetrics:
    """
    Economic cost evaluation for refund decision systems.

    Cost Model:
        - **Approved refund (pred=1)**:
            cost = order_amount + (fraud_penalty if fraud_score > threshold)
        - **Denied refund (pred=0) for a legitimately refundable order**:
            cost = retention_value (customer churn risk)

    Example:
        >>> metrics = EconomicMetrics()
        >>> cost = metrics.calculate_total_cost(data, predictions)
        >>> breakdown = metrics.cost_breakdown(data, predictions)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize with cost parameters from config.

        Args:
            config: Configuration object. Uses defaults if not provided.
        """
        self.config = config or Config()

    def calculate_total_cost(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
    ) -> float:
        """
        Calculate total economic cost of a refund decision strategy.

        Args:
            data: DataFrame containing ``order_amount``, ``fraud_score``,
                  and ``refunded`` columns.
            predictions: Array of binary predictions (0 or 1).

        Returns:
            Total economic cost as a float.

        Raises:
            ValueError: If predictions length doesn't match data length.
        """
        if len(predictions) != len(data):
            raise ValueError(
                f"Predictions length ({len(predictions)}) does not match "
                f"data length ({len(data)})."
            )

        total_cost = 0.0
        fraud_threshold = self.config.fraud_threshold
        retention_value = self.config.retention_value
        fraud_multiplier = self.config.fraud_penalty_multiplier

        for i, pred in enumerate(predictions):
            row = data.iloc[i]
            if pred == 1:
                # Cost of issuing refund
                total_cost += row["order_amount"]
                # Additional penalty if likely fraudulent
                if row["fraud_score"] > fraud_threshold:
                    total_cost += row["order_amount"] * (fraud_multiplier - 1)
            else:
                # Cost of denying a legitimate refund (customer churn)
                if row["refunded"] == 1:
                    total_cost += retention_value

        return total_cost

    def cost_breakdown(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
    ) -> Dict[str, float]:
        """
        Break down total cost into its three components.

        Args:
            data: DataFrame with order data.
            predictions: Array of binary predictions.

        Returns:
            Dictionary with keys:
                ``refund_cost``, ``fraud_penalty``, ``retention_loss``,
                ``total_cost``
        """
        refund_cost = 0.0
        fraud_penalty = 0.0
        retention_loss = 0.0

        for i, pred in enumerate(predictions):
            row = data.iloc[i]
            if pred == 1:
                refund_cost += row["order_amount"]
                if row["fraud_score"] > self.config.fraud_threshold:
                    fraud_penalty += row["order_amount"] * (
                        self.config.fraud_penalty_multiplier - 1
                    )
            else:
                if row["refunded"] == 1:
                    retention_loss += self.config.retention_value

        return {
            "refund_cost": round(refund_cost, 2),
            "fraud_penalty": round(fraud_penalty, 2),
            "retention_loss": round(retention_loss, 2),
            "total_cost": round(
                refund_cost + fraud_penalty + retention_loss, 2
            ),
        }

    def compare_strategies(
        self,
        data: pd.DataFrame,
        predictions_dict: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """
        Compare economic costs across multiple strategies.

        Args:
            data: DataFrame with order data.
            predictions_dict: Dictionary of strategy_name → predictions.

        Returns:
            DataFrame with cost breakdown per strategy.
        """
        rows = []
        for name, preds in predictions_dict.items():
            breakdown = self.cost_breakdown(data, preds)
            breakdown["strategy"] = name
            rows.append(breakdown)

        return pd.DataFrame(rows).set_index("strategy")


# ---------- Classification Metrics ----------


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class (optional).

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally AUC.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["auc_roc"] = float("nan")

    return metrics


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        2×2 confusion matrix as numpy array.
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    Generate a classification report as a DataFrame.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        target_names: Optional display names for classes.

    Returns:
        DataFrame with precision, recall, f1, and support per class.
    """
    if target_names is None:
        target_names = ["Rejected (0)", "Refunded (1)"]

    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    return pd.DataFrame(report).T
