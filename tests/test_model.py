"""Tests for the model module."""

import numpy as np
import pytest

from src.config import Config
from src.data_generator import generate_dataset
from src.model import ModelPipeline, get_available_models


@pytest.fixture(scope="module")
def trained_pipeline():
    """Create a pipeline with data prepared and one model trained."""
    config = Config(n_samples=300, random_seed=42)
    data = generate_dataset(config=config)
    pipeline = ModelPipeline(config=config)
    pipeline.prepare_data(data)
    pipeline.train_model("LogisticRegression", tune_hyperparams=False)
    return pipeline


class TestModelPipeline:
    """Test suite for the ModelPipeline class."""

    def test_available_models(self):
        """Should have at least 3 models available."""
        models = get_available_models()
        assert len(models) >= 3
        assert "LogisticRegression" in models
        assert "RandomForest" in models

    def test_prepare_data_splits(self, trained_pipeline):
        """prepare_data should create train/test splits."""
        assert trained_pipeline.X_train is not None
        assert trained_pipeline.X_test is not None
        assert len(trained_pipeline.X_train) > len(trained_pipeline.X_test)

    def test_train_produces_model(self, trained_pipeline):
        """Training should store a model in the dict."""
        assert "LogisticRegression" in trained_pipeline.models

    def test_predict_shape(self, trained_pipeline):
        """Predictions should match test set length."""
        preds = trained_pipeline.predict("LogisticRegression")
        assert len(preds) == len(trained_pipeline.X_test)

    def test_predict_binary(self, trained_pipeline):
        """Predictions should be 0 or 1."""
        preds = trained_pipeline.predict("LogisticRegression")
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, trained_pipeline):
        """Probability predictions should match test set length."""
        proba = trained_pipeline.predict_proba("LogisticRegression")
        assert len(proba) == len(trained_pipeline.X_test)

    def test_predict_proba_range(self, trained_pipeline):
        """Probabilities should be between 0 and 1."""
        proba = trained_pipeline.predict_proba("LogisticRegression")
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_feature_importance(self, trained_pipeline):
        """Feature importance should have correct number of features."""
        importance = trained_pipeline.get_feature_importance(
            "LogisticRegression"
        )
        assert len(importance) == trained_pipeline.X_train.shape[1]

    def test_results_summary(self, trained_pipeline):
        """Summary should produce a DataFrame with expected columns."""
        summary = trained_pipeline.get_results_summary()
        assert "Model" in summary.columns
        assert "Test_Accuracy" in summary.columns
        assert len(summary) >= 1

    def test_untrained_model_raises(self, trained_pipeline):
        """Predicting with untrained model should raise ValueError."""
        with pytest.raises(ValueError, match="not trained"):
            trained_pipeline.predict("NonExistent")

    def test_unprepared_data_raises(self):
        """Training before prepare_data should raise ValueError."""
        pipeline = ModelPipeline()
        with pytest.raises(ValueError, match="prepare_data"):
            pipeline.train_model("LogisticRegression")

    def test_train_random_forest(self):
        """RandomForest should train successfully."""
        config = Config(n_samples=200, random_seed=42)
        data = generate_dataset(config=config)
        pipeline = ModelPipeline(config=config)
        pipeline.prepare_data(data)
        pipeline.train_model("RandomForest", tune_hyperparams=False)
        preds = pipeline.predict("RandomForest")
        assert len(preds) == len(pipeline.X_test)

    def test_stratified_split(self):
        """Train/test split should be stratified."""
        config = Config(n_samples=500, random_seed=42)
        data = generate_dataset(config=config)
        pipeline = ModelPipeline(config=config)
        pipeline.prepare_data(data)

        train_ratio = pipeline.y_train.mean()
        test_ratio = pipeline.y_test.mean()
        assert abs(train_ratio - test_ratio) < 0.05
