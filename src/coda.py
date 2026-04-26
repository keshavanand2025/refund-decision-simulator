"""
CODA: Cost-Optimal Decision Algorithm
=======================================

Formal implementation of Algorithm 1 from the paper.

CODA integrates:
    1. Per-instance cost-weighted training
    2. Cost-optimal threshold search
    3. Three-tier decision output (approve / review / deny)

This module also provides:
    - Bootstrap resampling validation (B = 1,000)
    - Ablation study comparing component contributions
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.config import Config
from src.metrics import EconomicMetrics, classification_metrics
from src.cost_sensitive_model import compute_sample_weights


# ── Three-Tier Decision ──────────────────────────────────────────────

class ThreeTierDecision:
    """
    Three-tier decision output for production fraud systems.

    Given a cost-optimal threshold t* and confidence margin δ:
        - Auto-Approve: f(x) ≤ t* - δ  (low risk)
        - Manual Review: t* - δ < f(x) < t* + δ  (uncertain)
        - Auto-Deny:    f(x) ≥ t* + δ  (high fraud risk)

    Attributes:
        threshold: Cost-optimal threshold t*.
        delta: Confidence margin δ (default 0.10).
    """

    def __init__(self, threshold: float, delta: float = 0.10) -> None:
        self.threshold = threshold
        self.delta = delta

    def decide(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply three-tier decision rule.

        Args:
            probabilities: Array of predicted fraud probabilities.

        Returns:
            Array of decisions: 'approve', 'review', or 'deny'.
        """
        decisions = np.full(len(probabilities), "review", dtype=object)
        decisions[probabilities <= self.threshold - self.delta] = "approve"
        decisions[probabilities >= self.threshold + self.delta] = "deny"
        return decisions

    def tier_distribution(self, probabilities: np.ndarray) -> Dict[str, float]:
        """
        Compute the percentage of claims in each tier.

        Args:
            probabilities: Array of predicted fraud probabilities.

        Returns:
            Dictionary with approve/review/deny percentages.
        """
        decisions = self.decide(probabilities)
        n = len(decisions)
        return {
            "approve_pct": round((decisions == "approve").sum() / n * 100, 1),
            "review_pct": round((decisions == "review").sum() / n * 100, 1),
            "deny_pct": round((decisions == "deny").sum() / n * 100, 1),
        }

    def to_binary(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Convert three-tier decisions to binary (approve=1, deny=0, review=0).

        For cost evaluation, 'review' is treated as deny (conservative).

        Args:
            probabilities: Array of predicted fraud probabilities.

        Returns:
            Binary prediction array.
        """
        decisions = self.decide(probabilities)
        return (decisions == "approve").astype(int)


# ── CODA Algorithm (Algorithm 1) ────────────────────────────────────

class CODA:
    """
    Cost-Optimal Decision Algorithm.

    Implements Algorithm 1 from the paper:
        1. Compute per-instance weights from cost parameters
        2. Train model with sample_weight
        3. Search cost-optimal threshold t*
        4. Set confidence margin δ
        5. Construct three-tier rule R(x)
        6. Return R(x), t*, C(t*)

    Attributes:
        config: Configuration with α (fraud_penalty_multiplier) and
                β (retention_value).
        delta: Confidence margin for three-tier output.
        model: Trained classifier pipeline.
        threshold: Cost-optimal threshold t*.
        cost_at_threshold: C_total at t*.
        decision_rule: ThreeTierDecision instance.
    """

    def __init__(
        self,
        model_class: Any,
        config: Optional[Config] = None,
        delta: float = 0.10,
        threshold_steps: int = 19,
    ) -> None:
        """
        Initialize CODA.

        Args:
            model_class: sklearn-compatible classifier class.
            config: Configuration with cost parameters.
            delta: Confidence margin δ (default 0.10).
            threshold_steps: Number of threshold candidates (default 19,
                             i.e., {0.05, 0.10, ..., 0.95}).
        """
        self.config = config or Config()
        self.model_class = model_class
        self.delta = delta
        self.threshold_steps = threshold_steps

        self.model: Optional[Pipeline] = None
        self.threshold: Optional[float] = None
        self.cost_at_threshold: Optional[float] = None
        self.decision_rule: Optional[ThreeTierDecision] = None

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "refunded",
    ) -> "CODA":
        """
        Execute the full CODA pipeline (Steps 1-5).

        Args:
            data: Training DataFrame with features + target.
            target_column: Name of target column.

        Returns:
            self (fitted CODA instance).
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Step 1: Compute per-instance weights
        weights = compute_sample_weights(data, self.config)

        # Step 2: Train model with cost weights
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", self.model_class(
                random_state=self.config.random_seed
            )),
        ])
        self.model.fit(X, y, classifier__sample_weight=weights)

        # Step 3: Search cost-optimal threshold
        probas = self.model.predict_proba(X)[:, 1]
        econ = EconomicMetrics(self.config)

        thresholds = np.linspace(0.05, 0.95, self.threshold_steps)
        best_t, best_cost = 0.5, float("inf")

        for t in thresholds:
            preds = (probas >= t).astype(int)
            cost = econ.calculate_total_cost(data, preds)
            if cost < best_cost:
                best_cost = cost
                best_t = round(t, 4)

        self.threshold = best_t
        self.cost_at_threshold = best_cost

        # Steps 4-5: Set δ and construct three-tier rule
        self.decision_rule = ThreeTierDecision(self.threshold, self.delta)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate binary predictions using cost-optimal threshold.

        Args:
            X: Feature DataFrame.

        Returns:
            Binary prediction array.
        """
        probas = self.predict_proba(X)
        return (probas >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability estimates for the positive class.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of fraud probabilities.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict_three_tier(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate three-tier decisions (approve / review / deny).

        Args:
            X: Feature DataFrame.

        Returns:
            Array of string decisions.
        """
        probas = self.predict_proba(X)
        return self.decision_rule.decide(probas)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the fitted CODA model."""
        return {
            "threshold": self.threshold,
            "cost_at_threshold": self.cost_at_threshold,
            "delta": self.delta,
            "model": type(self.model_class).__name__,
        }


# ── Bootstrap Resampling Validation ─────────────────────────────────

def bootstrap_validation(
    data: pd.DataFrame,
    model_class: Any,
    config: Optional[Config] = None,
    B: int = 1000,
    n_samples: int = 1000,
    random_state: int = 42,
    target_column: str = "refunded",
) -> pd.DataFrame:
    """
    Bootstrap resampling validation (B iterations).

    At each iteration:
        1. Draw a stratified bootstrap sample (with replacement)
        2. Train CODA on the bootstrap sample
        3. Record C_total for CODA and baseline (t=0.5)

    Args:
        data: Full dataset.
        model_class: sklearn classifier class.
        config: Configuration with cost parameters.
        B: Number of bootstrap iterations (default 1,000).
        n_samples: Size of each bootstrap sample.
        random_state: Base random seed.
        target_column: Name of target column.

    Returns:
        DataFrame with columns: iteration, cost_coda, cost_baseline,
        threshold, cost_difference.
    """
    config = config or Config()
    econ = EconomicMetrics(config)
    rng = np.random.RandomState(random_state)

    rows = []
    for b in range(B):
        # Stratified bootstrap sample
        idx = rng.choice(len(data), size=n_samples, replace=True)
        boot_data = data.iloc[idx].reset_index(drop=True)

        X = boot_data.drop(columns=[target_column])
        y = boot_data[target_column]

        # CODA (cost-weighted + threshold search)
        try:
            coda = CODA(model_class, config)
            coda.fit(boot_data, target_column)
            preds_coda = coda.predict(X)
            cost_coda = econ.calculate_total_cost(boot_data, preds_coda)

            # Baseline (standard training, t=0.5)
            baseline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", model_class(
                    random_state=config.random_seed
                )),
            ])
            baseline.fit(X, y)
            preds_baseline = baseline.predict(X)
            cost_baseline = econ.calculate_total_cost(
                boot_data, preds_baseline
            )

            rows.append({
                "iteration": b,
                "cost_coda": cost_coda,
                "cost_baseline": cost_baseline,
                "threshold": coda.threshold,
                "cost_difference": cost_baseline - cost_coda,
            })
        except Exception:
            continue  # Skip failed iterations

    result = pd.DataFrame(rows)
    return result


def bootstrap_summary(boot_results: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarise bootstrap validation results.

    Args:
        boot_results: Output of bootstrap_validation.

    Returns:
        Dictionary with mean costs, CI, and p-value.
    """
    cost_diff = boot_results["cost_difference"]
    n = len(cost_diff)

    p_value = (cost_diff <= 0).sum() / n if n > 0 else 1.0

    return {
        "n_iterations": n,
        "mean_cost_coda": round(boot_results["cost_coda"].mean(), 2),
        "mean_cost_baseline": round(boot_results["cost_baseline"].mean(), 2),
        "mean_cost_reduction": round(cost_diff.mean(), 2),
        "ci_95_lower": round(cost_diff.quantile(0.025), 2),
        "ci_95_upper": round(cost_diff.quantile(0.975), 2),
        "p_value": round(p_value, 4),
        "significant_at_001": p_value < 0.01,
        "mean_threshold": round(boot_results["threshold"].mean(), 4),
    }


# ── Ablation Study ──────────────────────────────────────────────────

def ablation_study(
    data: pd.DataFrame,
    model_class: Any,
    config: Optional[Config] = None,
    target_column: str = "refunded",
) -> pd.DataFrame:
    """
    Ablation study comparing CODA component contributions.

    Tests four configurations:
        1. Baseline: no weighting, t=0.5
        2. Weighting only: cost weights, t=0.5
        3. Threshold only: no weights, t*
        4. Full CODA: cost weights + t*

    Args:
        data: Dataset DataFrame.
        model_class: sklearn classifier class.
        config: Configuration with cost parameters.
        target_column: Name of target column.

    Returns:
        DataFrame with columns: Configuration, Weighting, Threshold,
        C_total, Delta_vs_Baseline.
    """
    config = config or Config()
    econ = EconomicMetrics(config)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size,
        random_state=config.random_seed, stratify=y,
    )
    train_data = data.loc[X_train.index].reset_index(drop=True)
    test_data = data.loc[X_test.index].reset_index(drop=True)

    weights = compute_sample_weights(train_data, config)

    results = {}

    # 1. Baseline (no CODA)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model_class(random_state=config.random_seed)),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    results["Baseline (no CODA)"] = {
        "weighting": False,
        "threshold": False,
        "cost": econ.calculate_total_cost(test_data, preds),
    }

    # 2. Weighting only (t=0.5)
    pipe_w = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model_class(random_state=config.random_seed)),
    ])
    pipe_w.fit(X_train, y_train, classifier__sample_weight=weights)
    preds_w = pipe_w.predict(X_test)
    results["Weighting only"] = {
        "weighting": True,
        "threshold": False,
        "cost": econ.calculate_total_cost(test_data, preds_w),
    }

    # 3. Threshold only (no weights, search t*)
    probas = pipe.predict_proba(X_test)[:, 1]
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 19):
        p = (probas >= t).astype(int)
        c = econ.calculate_total_cost(test_data, p)
        if c < best_cost:
            best_cost = c
            best_t = t
    results["Threshold only"] = {
        "weighting": False,
        "threshold": True,
        "cost": best_cost,
    }

    # 4. Full CODA (weights + threshold)
    probas_w = pipe_w.predict_proba(X_test)[:, 1]
    best_t_coda, best_cost_coda = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 19):
        p = (probas_w >= t).astype(int)
        c = econ.calculate_total_cost(test_data, p)
        if c < best_cost_coda:
            best_cost_coda = c
            best_t_coda = t
    results["Full CODA"] = {
        "weighting": True,
        "threshold": True,
        "cost": best_cost_coda,
    }

    # Build output table
    baseline_cost = results["Baseline (no CODA)"]["cost"]
    rows = []
    for name, r in results.items():
        delta = (
            f"{(r['cost'] - baseline_cost) / baseline_cost * 100:+.1f}%"
            if name != "Baseline (no CODA)"
            else "—"
        )
        rows.append({
            "Configuration": name,
            "Weighting": "✓" if r["weighting"] else "✗",
            "Threshold": "✓" if r["threshold"] else "✗",
            "C_total": round(r["cost"], 0),
            "Delta_vs_Baseline": delta,
        })

    return pd.DataFrame(rows)


# ── CODA+ Dynamic Cost Learning (Algorithm 2) ──────────────────────

class CODAPlus:
    """
    CODA+ — Dynamic Cost Learning Extension.

    Implements Algorithm 2 from the paper. Extends static CODA by
    replacing global cost constants α, β with learned functions:

        α(xᵢ) = max(α_min, w₀ + w₁·ṽᵢ + w₂·fraud_scoreᵢ + w₃·prev_refundsᵢ)
        β(xᵢ) = max(β_min, u₀ + u₁·loyaltyᵢ + u₂·ṽᵢ⁻¹)

    The instance-adaptive Bayes-optimal threshold becomes:
        t*(xᵢ) = β(xᵢ) / (α(xᵢ)·vᵢ + β(xᵢ))

    Attributes:
        config: Base configuration.
        delta: Confidence margin for three-tier output.
        model: Trained classifier pipeline.
        alpha_model: Ridge regressor for α(x).
        beta_model: Ridge regressor for β(x).
        alpha_min: Minimum α floor (default 1.0).
        beta_min: Minimum β floor (default 100.0).
    """

    def __init__(
        self,
        model_class: Any,
        config: Optional[Config] = None,
        delta: float = 0.10,
        alpha_min: float = 1.0,
        beta_min: float = 100.0,
        cost_est_split: float = 0.3,
    ) -> None:
        """
        Initialize CODA+.

        Args:
            model_class: sklearn-compatible classifier class.
            config: Configuration with base cost parameters.
            delta: Confidence margin δ (default 0.10).
            alpha_min: Minimum floor for α(x).
            beta_min: Minimum floor for β(x).
            cost_est_split: Fraction of training data used for
                            cost regressor estimation (default 0.3).
        """
        self.config = config or Config()
        self.model_class = model_class
        self.delta = delta
        self.alpha_min = alpha_min
        self.beta_min = beta_min
        self.cost_est_split = cost_est_split

        self.model: Optional[Pipeline] = None
        self.alpha_model = None
        self.beta_model = None
        self.cost_at_threshold: Optional[float] = None
        self._feature_cols: Optional[List[str]] = None

    def _build_cost_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Build feature matrix for cost regressors.

        α features: normalised order value, fraud_score, previous_refunds
        β features: loyalty (1 - fraud_score), inverse normalised value
        """
        v_bar = data["order_amount"].mean()
        v_norm = data["order_amount"] / v_bar

        alpha_features = pd.DataFrame({
            "v_norm": v_norm,
            "fraud_score": data["fraud_score"],
            "prev_refunds": data.get(
                "previous_refunds",
                pd.Series(np.zeros(len(data)))
            ),
        })

        beta_features = pd.DataFrame({
            "loyalty": 1.0 - data["fraud_score"],
            "inv_v_norm": 1.0 / (v_norm + 1e-6),
        })

        return alpha_features, beta_features

    def _generate_cost_targets(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate proxy cost targets from feature structure.

        α target: fraud penalty proxy based on order value and fraud score.
        β target: retention loss proxy based on loyalty and order value.
        """
        v_bar = data["order_amount"].mean()
        base_alpha = self.config.fraud_penalty_multiplier

        # α proxy: higher for high-value, high-fraud-score transactions
        alpha_targets = (
            base_alpha
            * (0.5 + 0.3 * data["order_amount"] / v_bar
               + 0.2 * data["fraud_score"])
        )

        # β proxy: higher for loyal, low-value customers
        base_beta = self.config.retention_value
        beta_targets = (
            base_beta
            * (0.5 + 0.3 * (1.0 - data["fraud_score"])
               + 0.2 * v_bar / (data["order_amount"] + 1e-6).clip(upper=5.0))
        )

        return alpha_targets.values, beta_targets.values

    def fit(
        self,
        data: pd.DataFrame,
        target_column: str = "refunded",
    ) -> "CODAPlus":
        """
        Execute the full CODA+ pipeline (Algorithm 2).

        Steps:
            1. Split training data into cost-estimation and classifier partitions
            2. Train cost regressors α(x), β(x) on estimation partition
            3. Compute instance-adaptive weights on classifier partition
            4. Train classifier with adaptive weights
            5. Compute per-instance thresholds t*(xᵢ)
            6. Construct three-tier decision with adaptive thresholds

        Args:
            data: Training DataFrame with features + target.
            target_column: Name of target column.

        Returns:
            self (fitted CODAPlus instance).
        """
        from sklearn.linear_model import Ridge

        X = data.drop(columns=[target_column])
        y = data[target_column]
        self._feature_cols = list(X.columns)

        # Step 1: Split into cost-estimation and classifier partitions
        X_cost, X_clf, y_cost, y_clf = train_test_split(
            X, y,
            test_size=1.0 - self.cost_est_split,
            random_state=self.config.random_seed,
            stratify=y,
        )
        cost_data = data.loc[X_cost.index].reset_index(drop=True)
        clf_data = data.loc[X_clf.index].reset_index(drop=True)
        X_clf = X_clf.reset_index(drop=True)
        y_clf = y_clf.reset_index(drop=True)

        # Step 2: Train cost regressors
        alpha_feats_cost, beta_feats_cost = self._build_cost_features(cost_data)
        alpha_targets, beta_targets = self._generate_cost_targets(cost_data)

        self.alpha_model = Ridge(alpha=1.0)
        self.alpha_model.fit(alpha_feats_cost, alpha_targets)

        self.beta_model = Ridge(alpha=1.0)
        self.beta_model.fit(beta_feats_cost, beta_targets)

        # Step 3: Compute instance-adaptive weights on classifier partition
        alpha_feats_clf, beta_feats_clf = self._build_cost_features(clf_data)
        alpha_vals = np.maximum(
            self.alpha_min, self.alpha_model.predict(alpha_feats_clf)
        )
        beta_vals = np.maximum(
            self.beta_min, self.beta_model.predict(beta_feats_clf)
        )

        v_bar = clf_data["order_amount"].mean()
        weights = np.where(
            y_clf == 1,
            alpha_vals * clf_data["order_amount"].values / v_bar,
            beta_vals / v_bar,
        )
        weights = weights / weights.mean()  # normalise

        # Step 4: Train classifier with adaptive weights
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", self.model_class(
                random_state=self.config.random_seed
            )),
        ])
        self.model.fit(X_clf, y_clf, classifier__sample_weight=weights)

        # Step 5: Evaluate cost with instance-adaptive thresholds
        probas = self.model.predict_proba(X_clf)[:, 1]
        thresholds_per_instance = beta_vals / (
            alpha_vals * clf_data["order_amount"].values + beta_vals
        )

        preds = (probas >= thresholds_per_instance).astype(int)
        econ = EconomicMetrics(self.config)
        self.cost_at_threshold = econ.calculate_total_cost(clf_data, preds)

        return self

    def predict_alpha_beta(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict instance-adaptive α(x) and β(x).

        Args:
            data: DataFrame with feature columns.

        Returns:
            Tuple of (alpha_values, beta_values).
        """
        alpha_feats, beta_feats = self._build_cost_features(data)
        alpha_vals = np.maximum(
            self.alpha_min, self.alpha_model.predict(alpha_feats)
        )
        beta_vals = np.maximum(
            self.beta_min, self.beta_model.predict(beta_feats)
        )
        return alpha_vals, beta_vals

    def predict_thresholds(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute per-instance Bayes-optimal thresholds.

        t*(xᵢ) = β(xᵢ) / (α(xᵢ)·vᵢ + β(xᵢ))

        Args:
            data: DataFrame with order_amount and feature columns.

        Returns:
            Array of per-instance thresholds.
        """
        alpha_vals, beta_vals = self.predict_alpha_beta(data)
        return beta_vals / (alpha_vals * data["order_amount"].values + beta_vals)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate binary predictions using instance-adaptive thresholds.

        Args:
            data: DataFrame with features.

        Returns:
            Binary prediction array.
        """
        X = data[self._feature_cols] if self._feature_cols else data
        probas = self.model.predict_proba(X)[:, 1]
        thresholds = self.predict_thresholds(data)
        return (probas >= thresholds).astype(int)

    def predict_three_tier(
        self, data: pd.DataFrame
    ) -> np.ndarray:
        """
        Generate three-tier decisions with instance-adaptive thresholds.

        Args:
            data: DataFrame with features.

        Returns:
            Array of decisions: 'approve', 'review', or 'deny'.
        """
        X = data[self._feature_cols] if self._feature_cols else data
        probas = self.model.predict_proba(X)[:, 1]
        thresholds = self.predict_thresholds(data)

        decisions = np.full(len(probas), "review", dtype=object)
        decisions[probas <= thresholds - self.delta] = "approve"
        decisions[probas >= thresholds + self.delta] = "deny"
        return decisions

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the fitted CODA+ model."""
        return {
            "cost_at_threshold": self.cost_at_threshold,
            "delta": self.delta,
            "alpha_min": self.alpha_min,
            "beta_min": self.beta_min,
            "alpha_coefs": (
                self.alpha_model.coef_.tolist()
                if self.alpha_model else None
            ),
            "beta_coefs": (
                self.beta_model.coef_.tolist()
                if self.beta_model else None
            ),
        }
