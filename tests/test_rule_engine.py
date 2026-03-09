"""Tests for the rule engine module."""

import numpy as np
import pandas as pd
import pytest

from src.rule_engine import (
    rule_simple,
    rule_conservative,
    rule_lenient,
    RuleEngine,
)


class TestRuleSimple:
    """Test suite for the simple rule-based decision function."""

    def test_high_delay_approves(self):
        """Orders with delay > 30 should be approved."""
        assert rule_simple(1000, 45, 0) == 1

    def test_many_past_refunds_rejects(self):
        """Customers with > 3 past refunds should be rejected."""
        assert rule_simple(1000, 10, 4) == 0

    def test_low_order_approves(self):
        """Low-value orders (< 200) should be approved."""
        assert rule_simple(150, 10, 0) == 1

    def test_default_rejects(self):
        """High-value, no-delay, few-refund orders should be rejected."""
        assert rule_simple(500, 10, 2) == 0

    def test_delay_overrides_past_refunds(self):
        """Delay > 30 should approve even with many past refunds."""
        assert rule_simple(500, 40, 4) == 1

    def test_returns_binary(self):
        """Output should always be 0 or 1."""
        for _ in range(20):
            result = rule_simple(
                np.random.uniform(100, 2000),
                np.random.randint(0, 90),
                np.random.randint(0, 5),
            )
            assert result in (0, 1)


class TestRuleConservative:
    """Test suite for the conservative rule function."""

    def test_high_fraud_rejects(self):
        """High fraud score (> 0.5) should always reject."""
        assert rule_conservative(100, 60, 0, 0.6) == 0

    def test_many_refunds_rejects(self):
        """Past refunds > 2 should reject."""
        assert rule_conservative(100, 60, 3, 0.3) == 0

    def test_delay_and_low_value_approves(self):
        """Long delay + low order + low fraud should approve."""
        assert rule_conservative(300, 50, 0, 0.2) == 1

    def test_default_rejects(self):
        """Borderline cases should default to rejection."""
        assert rule_conservative(800, 20, 1, 0.3) == 0


class TestRuleLenient:
    """Test suite for the lenient rule function."""

    def test_very_high_fraud_rejects(self):
        """Only very high fraud (> 0.8) rejects."""
        assert rule_lenient(100, 60, 5, 0.9) == 0

    def test_moderate_delay_approves(self):
        """Delay > 15 should approve."""
        assert rule_lenient(1000, 20, 1, 0.5) == 1

    def test_high_severity_approves(self):
        """Complaint severity >= 3 should approve."""
        assert rule_lenient(1000, 5, 3, 0.5) == 1

    def test_low_value_approves(self):
        """Low order value should approve."""
        assert rule_lenient(300, 5, 1, 0.3) == 1


class TestRuleEngine:
    """Test suite for the RuleEngine batch processor."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create a small sample DataFrame for testing."""
        return pd.DataFrame({
            "order_amount": [150, 800, 300, 1500, 100],
            "delay_minutes": [10, 45, 60, 5, 20],
            "previous_refunds": [0, 4, 1, 0, 3],
            "fraud_score": [0.1, 0.6, 0.3, 0.9, 0.2],
            "complaint_severity": [2, 4, 5, 1, 3],
        })

    def test_predict_simple(self, sample_data):
        """Simple predictions should be correct length and binary."""
        engine = RuleEngine()
        preds = engine.predict(sample_data, strategy="simple")
        assert len(preds) == len(sample_data)
        assert all(p in (0, 1) for p in preds)

    def test_predict_conservative(self, sample_data):
        """Conservative predictions should work."""
        engine = RuleEngine()
        preds = engine.predict(sample_data, strategy="conservative")
        assert len(preds) == len(sample_data)

    def test_predict_lenient(self, sample_data):
        """Lenient predictions should work."""
        engine = RuleEngine()
        preds = engine.predict(sample_data, strategy="lenient")
        assert len(preds) == len(sample_data)

    def test_invalid_strategy_raises(self, sample_data):
        """Unknown strategy should raise ValueError."""
        engine = RuleEngine()
        with pytest.raises(ValueError, match="Unknown strategy"):
            engine.predict(sample_data, strategy="unknown")

    def test_predict_all_strategies(self, sample_data):
        """predict_all_strategies should return all three."""
        engine = RuleEngine()
        result = engine.predict_all_strategies(sample_data)
        assert set(result.keys()) == {"simple", "conservative", "lenient"}
        for preds in result.values():
            assert len(preds) == len(sample_data)

    def test_lenient_approves_more_than_conservative(self, sample_data):
        """Lenient should generally approve more than conservative."""
        engine = RuleEngine()
        lenient = engine.predict(sample_data, strategy="lenient")
        conservative = engine.predict(sample_data, strategy="conservative")
        assert sum(lenient) >= sum(conservative)
