"""
CODA: Cost-Optimal Decision Algorithm
======================================

Formal implementation of Algorithm 1 from the paper.
Integrates: per-instance cost weighting, cost-optimal threshold search,
and three-tier decision output (approve / review / deny).

This module also provides ablation study functionality to isolate
the contribution of each CODA component.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from src.config import Config


@dataclass
class CODAResult:
    """Container for CODA algorithm outputs.

    Attributes:
        model: Trained model pipeline.
        threshold: Cost-optimal threshold t*.
        delta: Confidence margin for three-tier decisions.
        cost_at_threshold: C_total at the optimal threshold.
        weights: Per-instance training weights used.
    """
    model: Any
    threshold: float
    delta: float
    cost_at_threshold: float
    weights: np.ndarray


class ThreeTierDecision:
    """Three-tier decision output from CODA.

    Classifies predictions into:
        - approve: f(x) <= t* - delta  (low fraud risk)
        - review:  t* - delta < f(x) < t* + delta  (uncertain)
        - deny:    f(x) >= t* + delta  (high fraud risk)
    """

    def __init__(self, threshold: float, delta: float = 0.10):
        self.threshold = threshold
        self.delta = delta

    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply three-tier decision rule.

        Args:
            probabilities: Model predicted probabilities for fraud class.

        Returns:
            Array of strings: 'approve', 'review', or 'deny'.
        """
        decisions = np.full(len(probabilities), 'review', dtype=object)
        decisions[probabilities <= self.threshold - self.delta] = 'approve'
        decisions[probabilities >= self.threshold + self.delta] = 'deny'
        return decisions

    def get_tier_distribution(self, probabilities: np.ndarray) -> Dict[str, float]:
        """Compute the proportion of claims in each tier.

        Args:
            probabilities: Model predicted probabilities.

        Returns:
            Dictionary with tier names and their proportions.
        """
        decisions = self.decide(probabilities)
        n = len(decisions)
        return {
            'approve': np.sum(decisions == 'approve') / n,
            'review': np.sum(decisions == 'review') / n,
            'deny': np.sum(decisions == 'deny') / n,
        }


class CODA:
    """
    Cost-Optimal Decision Algorithm (Algorithm 1 from paper).

    Steps:
        1. Compute per-instance weights from cost parameters.
        2. Train model M with sample_weight = w.
        3. Search cost-optimal threshold t* over candidate set.
        4. Set confidence margin delta.
        5. Construct three-tier decision rule R(x).
        6. Return R(x), t*, C(t*).

    Args:
        config: Project configuration with cost parameters.
        delta: Confidence margin for three-tier output. Default 0.10.
        threshold_candidates: Candidate thresholds to search.
            Default: {0.05, 0.10, ..., 0.95}.

    Example:
        >>> coda = CODA()
        >>> result = coda.fit(model_class, X_train, y_train, X_val, y_val,
        ...                   order_values_train, order_values_val)
        >>> decisions = coda.predict(X_test, prob_test)
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        delta: float = 0.10,
        threshold_candidates: Optional[np.ndarray] = None,
    ):
        self.config = config or Config()
        self.delta = delta
        self.threshold_candidates = (
            threshold_candidates
            if threshold_candidates is not None
            else np.arange(0.05, 1.00, 0.05)
        )
        self.result_: Optional[CODAResult] = None
        self.decision_rule_: Optional[ThreeTierDecision] = None

    def _compute_weights(
        self,
        y: np.ndarray,
        order_values: np.ndarray,
    ) -> np.ndarray:
        """Step 1: Compute per-instance weights.

        w_i = alpha * v_i / v_bar  if y_i = fraud
        w_i = beta / v_bar         if y_i = legit

        Args:
            y: Binary labels (1 = fraud/refund, 0 = legit).
            order_values: Per-instance order amounts.

        Returns:
            Per-instance weight array.
        """
        alpha = self.config.fraud_penalty_multiplier
        beta = self.config.retention_value
        v_bar = np.mean(order_values)

        weights = np.where(
            y == 1,
            alpha * order_values / v_bar,
            beta / v_bar,
        )
        return weights

    def _compute_cost(
        self,
        predictions: np.ndarray,
        y_true: np.ndarray,
        order_values: np.ndarray,
    ) -> float:
        """Compute total economic cost C_total.

        C_total = sum_i [alpha * v_i * 1(FN_i) + beta * 1(FP_i) + v_i * 1(TP_i)]

        Args:
            predictions: Binary predictions (0 or 1).
            y_true: True labels.
            order_values: Per-instance order values.

        Returns:
            Total economic cost.
        """
        alpha = self.config.fraud_penalty_multiplier
        beta = self.config.retention_value

        fn = (predictions == 0) & (y_true == 1)  # Missed fraud / denied legit
        fp = (predictions == 1) & (y_true == 0)  # Approved fraud
        tp = (predictions == 1) & (y_true == 1)  # Approved legit

        cost = (
            alpha * np.sum(order_values[fp])
            + beta * np.sum(fn)
            + np.sum(order_values[tp])
        )
        return float(cost)

    def _search_threshold(
        self,
        probabilities: np.ndarray,
        y_true: np.ndarray,
        order_values: np.ndarray,
    ) -> Tuple[float, float]:
        """Step 3: Search cost-optimal threshold t*.

        Args:
            probabilities: Predicted probabilities.
            y_true: True labels.
            order_values: Per-instance order values.

        Returns:
            Tuple of (optimal_threshold, cost_at_threshold).
        """
        best_t = 0.5
        best_cost = float('inf')

        for t in self.threshold_candidates:
            preds = (probabilities >= t).astype(int)
            cost = self._compute_cost(preds, y_true, order_values)
            if cost < best_cost:
                best_cost = cost
                best_t = t

        return best_t, best_cost

    def fit(
        self,
        model_pipeline: Pipeline,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        order_values_train: np.ndarray,
        order_values_val: np.ndarray,
        use_weights: bool = True,
        use_threshold: bool = True,
    ) -> CODAResult:
        """Run the full CODA algorithm.

        Args:
            model_pipeline: Sklearn pipeline (scaler + classifier).
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features.
            y_val: Validation labels.
            order_values_train: Training set order amounts.
            order_values_val: Validation set order amounts.
            use_weights: Whether to apply cost weights (for ablation).
            use_threshold: Whether to optimise threshold (for ablation).

        Returns:
            CODAResult with trained model, threshold, and cost.
        """
        # Step 1: Compute weights
        weights = self._compute_weights(y_train, order_values_train)

        # Step 2: Train with sample weights
        classifier_step = model_pipeline.named_steps.get('classifier')
        if use_weights and classifier_step is not None:
            model_pipeline.fit(
                X_train, y_train,
                classifier__sample_weight=weights,
            )
        else:
            model_pipeline.fit(X_train, y_train)

        # Step 3: Search optimal threshold
        if use_threshold:
            probas = model_pipeline.predict_proba(X_val)[:, 1]
            t_star, cost_star = self._search_threshold(
                probas, y_val, order_values_val
            )
        else:
            t_star = 0.5
            preds = model_pipeline.predict(X_val)
            cost_star = self._compute_cost(preds, y_val, order_values_val)

        # Steps 4-5: Set delta and build decision rule
        self.decision_rule_ = ThreeTierDecision(t_star, self.delta)

        self.result_ = CODAResult(
            model=model_pipeline,
            threshold=t_star,
            delta=self.delta,
            cost_at_threshold=cost_star,
            weights=weights,
        )
        return self.result_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply three-tier decision rule.

        Args:
            X: Feature matrix.

        Returns:
            Array of decisions: 'approve', 'review', or 'deny'.
        """
        if self.result_ is None:
            raise ValueError("Call fit() before predict().")
        probas = self.result_.model.predict_proba(X)[:, 1]
        return self.decision_rule_.decide(probas)

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Apply binary decision at optimal threshold.

        Args:
            X: Feature matrix.

        Returns:
            Binary prediction array (0 or 1).
        """
        if self.result_ is None:
            raise ValueError("Call fit() before predict_binary().")
        probas = self.result_.model.predict_proba(X)[:, 1]
        return (probas >= self.result_.threshold).astype(int)


def run_ablation_study(
    model_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    order_values_train: np.ndarray,
    order_values_val: np.ndarray,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """Run ablation study isolating each CODA component.

    Tests four configurations:
        1. Baseline (no weighting, no threshold)
        2. Weighting only
        3. Threshold only
        4. Full CODA (weighting + threshold)

    Args:
        model_class: Sklearn classifier class (e.g., GradientBoostingClassifier).
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        order_values_train, order_values_val: Order amounts.
        config: Configuration with cost parameters.

    Returns:
        DataFrame with columns: Configuration, Weighting, Threshold,
        C_total, Delta_vs_Baseline.
    """
    config = config or Config()

    configurations = [
        ("Baseline (no CODA)", False, False),
        ("Weighting only", True, False),
        ("Threshold only", False, True),
        ("Full CODA", True, True),
    ]

    results = []
    baseline_cost = None

    for name, use_w, use_t in configurations:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model_class(random_state=config.random_seed)),
        ])

        coda = CODA(config=config)
        result = coda.fit(
            pipeline, X_train, y_train, X_val, y_val,
            order_values_train, order_values_val,
            use_weights=use_w, use_threshold=use_t,
        )

        if baseline_cost is None:
            baseline_cost = result.cost_at_threshold

        delta_pct = (
            (result.cost_at_threshold - baseline_cost) / baseline_cost * 100
            if baseline_cost > 0 else 0
        )

        results.append({
            "Configuration": name,
            "Weighting": "✓" if use_w else "✗",
            "Threshold": "✓" if use_t else "✗",
            "C_total": round(result.cost_at_threshold, 2),
            "Delta_vs_Baseline": f"{delta_pct:+.1f}%" if name != configurations[0][0] else "—",
            "Optimal_t": result.threshold,
        })

    return pd.DataFrame(results)
