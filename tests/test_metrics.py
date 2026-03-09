"""Tests for the metrics module."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.metrics import (
    EconomicMetrics,
    classification_metrics,
    get_confusion_matrix,
    get_classification_report,
)


@pytest.fixture
def sample_data():
    """Sample dataset for cost calculations."""
    return pd.DataFrame({
        "order_amount": [100.0, 500.0, 200.0, 800.0, 300.0],
        "fraud_score": [0.1, 0.8, 0.3, 0.9, 0.2],
        "refunded": [1, 0, 1, 0, 1],
    })


class TestEconomicMetrics:
    """Test suite for EconomicMetrics."""

    def test_all_approved_cost(self, sample_data):
        """All approvals should sum to refund costs + fraud penalties."""
        metrics = EconomicMetrics()
        preds = np.array([1, 1, 1, 1, 1])
        cost = metrics.calculate_total_cost(sample_data, preds)
        assert cost > 0

    def test_all_rejected_cost(self, sample_data):
        """All rejections should only incur retention losses."""
        config = Config(retention_value=500)
        metrics = EconomicMetrics(config)
        preds = np.array([0, 0, 0, 0, 0])
        cost = metrics.calculate_total_cost(sample_data, preds)
        # 3 refundable orders × 500 = 1500
        assert cost == 1500.0

    def test_zero_cost_perfect(self):
        """Approving exactly the refundable orders with no fraud = minimal cost."""
        data = pd.DataFrame({
            "order_amount": [100.0, 200.0],
            "fraud_score": [0.1, 0.1],
            "refunded": [1, 0],
        })
        metrics = EconomicMetrics()
        preds = np.array([1, 0])
        cost = metrics.calculate_total_cost(data, preds)
        # Only refund cost for order 1: 100
        assert cost == 100.0

    def test_cost_breakdown_keys(self, sample_data):
        """Breakdown should contain all expected keys."""
        metrics = EconomicMetrics()
        preds = np.array([1, 0, 1, 0, 1])
        breakdown = metrics.cost_breakdown(sample_data, preds)
        assert "refund_cost" in breakdown
        assert "fraud_penalty" in breakdown
        assert "retention_loss" in breakdown
        assert "total_cost" in breakdown

    def test_cost_breakdown_adds_up(self, sample_data):
        """Component costs should sum to total cost."""
        metrics = EconomicMetrics()
        preds = np.array([1, 1, 0, 0, 1])
        breakdown = metrics.cost_breakdown(sample_data, preds)
        assert breakdown["total_cost"] == pytest.approx(
            breakdown["refund_cost"]
            + breakdown["fraud_penalty"]
            + breakdown["retention_loss"]
        )

    def test_length_mismatch_raises(self, sample_data):
        """Mismatched prediction length should raise ValueError."""
        metrics = EconomicMetrics()
        with pytest.raises(ValueError, match="does not match"):
            metrics.calculate_total_cost(sample_data, np.array([1, 0]))

    def test_compare_strategies(self, sample_data):
        """compare_strategies should return a DataFrame."""
        metrics = EconomicMetrics()
        strategies = {
            "all_approve": np.ones(5, dtype=int),
            "all_reject": np.zeros(5, dtype=int),
        }
        result = metrics.compare_strategies(sample_data, strategies)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "total_cost" in result.columns


class TestClassificationMetrics:
    """Test suite for classification metric functions."""

    def test_perfect_predictions(self):
        """Perfect predictions should give accuracy = 1.0."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_worst_predictions(self):
        """Fully inverted predictions should give accuracy = 0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        metrics = classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_auc_with_proba(self):
        """AUC should be returned when probabilities are provided."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        metrics = classification_metrics(y_true, y_pred, y_proba)
        assert "auc_roc" in metrics
        assert metrics["auc_roc"] == 1.0

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2x2."""
        cm = get_confusion_matrix(
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
        )
        assert cm.shape == (2, 2)

    def test_classification_report_type(self):
        """Classification report should return a DataFrame."""
        report = get_classification_report(
            np.array([0, 1, 0, 1]),
            np.array([0, 1, 1, 0]),
        )
        assert isinstance(report, pd.DataFrame)
        assert "precision" in report.columns
