"""Tests for the data generator module."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_generator import generate_dataset


class TestGenerateDataset:
    """Test suite for the generate_dataset function."""

    def test_default_shape(self):
        """Dataset should have 1000 rows and 6 columns by default."""
        df = generate_dataset()
        assert df.shape == (1000, 6)

    def test_custom_sample_size(self):
        """Should support custom sample sizes."""
        df = generate_dataset(n_samples=500)
        assert len(df) == 500

    def test_minimum_sample_size(self):
        """Should accept exactly 100 samples."""
        df = generate_dataset(n_samples=100)
        assert len(df) == 100

    def test_too_few_samples_raises(self):
        """Should raise ValueError for n_samples < 100."""
        with pytest.raises(ValueError, match="at least 100"):
            generate_dataset(n_samples=50)

    def test_expected_columns(self):
        """Dataset should contain all expected feature columns."""
        df = generate_dataset()
        expected = [
            "order_amount",
            "delay_minutes",
            "previous_refunds",
            "fraud_score",
            "complaint_severity",
            "refunded",
        ]
        assert list(df.columns) == expected

    def test_column_types(self):
        """Numeric columns should have correct dtypes."""
        df = generate_dataset()
        assert df["order_amount"].dtype == np.float64
        assert df["fraud_score"].dtype == np.float64
        assert df["refunded"].dtype in [np.int32, np.int64, np.intp]

    def test_order_amount_range(self):
        """Order amounts should be within configured range."""
        df = generate_dataset()
        assert df["order_amount"].min() >= 100
        assert df["order_amount"].max() <= 2000

    def test_fraud_score_range(self):
        """Fraud scores should be between 0 and 1."""
        df = generate_dataset()
        assert df["fraud_score"].min() >= 0.0
        assert df["fraud_score"].max() <= 1.0

    def test_complaint_severity_range(self):
        """Complaint severity should be within 1-5."""
        df = generate_dataset()
        assert df["complaint_severity"].min() >= 1
        assert df["complaint_severity"].max() <= 5

    def test_target_is_binary(self):
        """Refunded column should only contain 0 and 1."""
        df = generate_dataset()
        assert set(df["refunded"].unique()).issubset({0, 1})

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical datasets."""
        df1 = generate_dataset(random_seed=123)
        df2 = generate_dataset(random_seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different datasets."""
        df1 = generate_dataset(random_seed=1)
        df2 = generate_dataset(random_seed=2)
        assert not df1.equals(df2)

    def test_custom_config(self):
        """Should respect custom Config object."""
        config = Config(n_samples=200, random_seed=99)
        df = generate_dataset(config=config)
        assert len(df) == 200

    def test_override_beats_config(self):
        """Explicit n_samples should override config value."""
        config = Config(n_samples=1000)
        df = generate_dataset(config=config, n_samples=300)
        assert len(df) == 300
