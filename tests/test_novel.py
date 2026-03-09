"""Tests for the novel research contribution modules."""

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.data_generator import generate_dataset
from src.rule_engine import RuleEngine
from src.model import ModelPipeline


@pytest.fixture(scope="module")
def research_setup():
    """Set up data, rule predictions, and ML model for all tests."""
    config = Config(n_samples=300, random_seed=42)
    data = generate_dataset(config=config)

    # Rule predictions
    engine = RuleEngine()
    rule_preds = engine.predict_all_strategies(data)

    # ML pipeline (fast, no tuning)
    pipeline = ModelPipeline(config=config)
    pipeline.prepare_data(data)
    pipeline.train_model("LogisticRegression", tune_hyperparams=False)
    pipeline.train_model("RandomForest", tune_hyperparams=False)

    test_data = data.loc[pipeline.X_test.index].reset_index(drop=True)

    return {
        "config": config,
        "data": data,
        "pipeline": pipeline,
        "rule_preds": rule_preds,
        "test_data": test_data,
        "y_test": pipeline.y_test,
    }


# ========== Cost-Sensitive Model Tests ==========


class TestCostSensitiveModel:
    """Tests for Novel Contribution #1."""

    def test_compute_sample_weights_shape(self, research_setup):
        from src.cost_sensitive_model import compute_sample_weights
        weights = compute_sample_weights(research_setup["data"])
        assert len(weights) == len(research_setup["data"])

    def test_weights_are_positive(self, research_setup):
        from src.cost_sensitive_model import compute_sample_weights
        weights = compute_sample_weights(research_setup["data"])
        assert np.all(weights > 0)

    def test_weights_normalized(self, research_setup):
        from src.cost_sensitive_model import compute_sample_weights
        weights = compute_sample_weights(research_setup["data"])
        assert abs(weights.mean() - 1.0) < 1e-6

    def test_cost_sensitive_pipeline_trains(self, research_setup):
        from src.cost_sensitive_model import CostSensitivePipeline
        cs = CostSensitivePipeline(research_setup["config"])
        cs.prepare_data(research_setup["data"])
        cs.train_both()
        assert len(cs.models_standard) > 0
        assert len(cs.models_cost_weighted) > 0

    def test_comparison_dataframe(self, research_setup):
        from src.cost_sensitive_model import CostSensitivePipeline
        cs = CostSensitivePipeline(research_setup["config"])
        cs.prepare_data(research_setup["data"])
        cs.train_both()
        df = cs.compare()
        assert "Model" in df.columns
        assert "Training_Type" in df.columns
        assert "Economic_Cost" in df.columns
        assert len(df) > 0

    def test_cost_reduction_report(self, research_setup):
        from src.cost_sensitive_model import CostSensitivePipeline
        cs = CostSensitivePipeline(research_setup["config"])
        cs.prepare_data(research_setup["data"])
        cs.train_both()
        reduction = cs.get_cost_reduction()
        assert "Reduction_Pct" in reduction.columns


# ========== Threshold Optimizer Tests ==========


class TestThresholdOptimizer:
    """Tests for Novel Contribution #2."""

    def test_sweep_returns_correct_shape(self, research_setup):
        from src.threshold_optimizer import sweep_thresholds
        proba = research_setup["pipeline"].predict_proba("LogisticRegression")
        y_test = research_setup["y_test"].values
        result = sweep_thresholds(
            y_test, proba, research_setup["test_data"],
            n_steps=10,
        )
        assert len(result) == 11  # 0.0, 0.1, ..., 1.0

    def test_sweep_has_required_columns(self, research_setup):
        from src.threshold_optimizer import sweep_thresholds
        proba = research_setup["pipeline"].predict_proba("LogisticRegression")
        result = sweep_thresholds(
            research_setup["y_test"].values, proba,
            research_setup["test_data"], n_steps=10,
        )
        assert "threshold" in result.columns
        assert "economic_cost" in result.columns
        assert "accuracy" in result.columns

    def test_find_optimal_thresholds(self, research_setup):
        from src.threshold_optimizer import sweep_thresholds, find_optimal_thresholds
        proba = research_setup["pipeline"].predict_proba("LogisticRegression")
        sweep = sweep_thresholds(
            research_setup["y_test"].values, proba,
            research_setup["test_data"], n_steps=50,
        )
        optimal = find_optimal_thresholds(sweep)
        assert "cost_optimal" in optimal
        assert "accuracy_optimal" in optimal
        assert 0.0 <= optimal["cost_optimal"]["threshold"] <= 1.0

    def test_compare_across_models(self, research_setup):
        from src.threshold_optimizer import compare_thresholds_across_models
        probas = {
            "LR": research_setup["pipeline"].predict_proba("LogisticRegression"),
            "RF": research_setup["pipeline"].predict_proba("RandomForest"),
        }
        result = compare_thresholds_across_models(
            research_setup["y_test"].values, probas,
            research_setup["test_data"],
        )
        assert len(result) == 2
        assert "Threshold_Gap" in result.columns


# ========== Sensitivity Analysis Tests ==========


class TestSensitivityAnalysis:
    """Tests for Novel Contribution #3."""

    def test_single_parameter_sweep(self, research_setup):
        from src.sensitivity_analysis import single_parameter_sweep
        test_preds = {"simple": RuleEngine().predict(
            research_setup["test_data"], "simple"
        )}
        result = single_parameter_sweep(
            research_setup["test_data"], test_preds,
            "retention_value", np.array([100, 500, 1000]),
        )
        assert len(result) == 3  # 3 param values × 1 strategy
        assert "total_cost" in result.columns

    def test_invalid_parameter_raises(self, research_setup):
        from src.sensitivity_analysis import single_parameter_sweep
        with pytest.raises(ValueError, match="Unsupported parameter"):
            single_parameter_sweep(
                research_setup["test_data"], {},
                "invalid_param", np.array([1, 2]),
            )

    def test_find_crossover_points(self, research_setup):
        from src.sensitivity_analysis import single_parameter_sweep, find_crossover_points
        test_preds = {
            "simple": RuleEngine().predict(research_setup["test_data"], "simple"),
            "conservative": RuleEngine().predict(research_setup["test_data"], "conservative"),
        }
        sweep = single_parameter_sweep(
            research_setup["test_data"], test_preds,
            "retention_value", np.linspace(100, 2000, 20),
        )
        crossovers = find_crossover_points(sweep, "simple", "conservative")
        assert isinstance(crossovers, list)

    def test_dual_parameter_heatmap(self, research_setup):
        from src.sensitivity_analysis import dual_parameter_heatmap
        test_preds = {
            "simple": RuleEngine().predict(research_setup["test_data"], "simple"),
        }
        result = dual_parameter_heatmap(
            research_setup["test_data"], test_preds,
            np.array([100, 500]), np.array([1.0, 2.0]),
        )
        assert "cost_grids" in result
        assert "best_strategy_grid" in result
        assert result["best_strategy_grid"].shape == (2, 2)

    def test_winner_summary(self, research_setup):
        from src.sensitivity_analysis import dual_parameter_heatmap, winner_summary
        test_preds = {
            "A": RuleEngine().predict(research_setup["test_data"], "simple"),
            "B": RuleEngine().predict(research_setup["test_data"], "conservative"),
        }
        heatmap = dual_parameter_heatmap(
            research_setup["test_data"], test_preds,
            np.array([100, 500, 1000]), np.array([1.0, 2.0, 3.0]),
        )
        summary = winner_summary(heatmap)
        assert "Win_Pct" in summary.columns
        assert summary["Win_Pct"].sum() == pytest.approx(100.0)


# ========== Pareto Analysis Tests ==========


class TestParetoAnalysis:
    """Tests for Novel Contribution #4."""

    def test_compute_objectives(self, research_setup):
        from src.pareto_analysis import compute_strategy_objectives
        test_preds = {
            "simple": RuleEngine().predict(research_setup["test_data"], "simple"),
            "LR": research_setup["pipeline"].predict("LogisticRegression"),
        }
        result = compute_strategy_objectives(
            research_setup["y_test"].values, test_preds,
            research_setup["test_data"],
        )
        assert "Accuracy" in result.columns
        assert "Economic_Cost" in result.columns
        assert len(result) == 2

    def test_find_pareto_front(self, research_setup):
        from src.pareto_analysis import compute_strategy_objectives, find_pareto_front
        test_preds = {
            "simple": RuleEngine().predict(research_setup["test_data"], "simple"),
            "conservative": RuleEngine().predict(research_setup["test_data"], "conservative"),
            "LR": research_setup["pipeline"].predict("LogisticRegression"),
        }
        objectives = compute_strategy_objectives(
            research_setup["y_test"].values, test_preds,
            research_setup["test_data"],
        )
        pareto = find_pareto_front(objectives)
        assert "Is_Pareto_Optimal" in pareto.columns
        assert pareto["Is_Pareto_Optimal"].any()

    def test_pareto_summary_string(self, research_setup):
        from src.pareto_analysis import (
            compute_strategy_objectives, find_pareto_front, pareto_summary,
        )
        test_preds = {
            "A": RuleEngine().predict(research_setup["test_data"], "simple"),
            "B": research_setup["pipeline"].predict("LogisticRegression"),
        }
        objectives = compute_strategy_objectives(
            research_setup["y_test"].values, test_preds,
            research_setup["test_data"],
        )
        pareto = find_pareto_front(objectives)
        summary = pareto_summary(pareto)
        assert "Pareto-Optimal" in summary
        assert isinstance(summary, str)
