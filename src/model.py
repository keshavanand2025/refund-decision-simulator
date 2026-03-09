"""
Model Module
============

Provides a unified pipeline for training, tuning, and evaluating
multiple machine learning classifiers for refund decision prediction.

Supported Models:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

from src.config import Config

warnings.filterwarnings("ignore", category=UserWarning)


# ---------- Model Registry ----------

_MODEL_REGISTRY: Dict[str, Any] = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

if _XGBOOST_AVAILABLE:
    _MODEL_REGISTRY["XGBoost"] = XGBClassifier


def get_available_models() -> List[str]:
    """Return the list of available model names."""
    return list(_MODEL_REGISTRY.keys())


# ---------- Model Pipeline ----------


class ModelPipeline:
    """
    End-to-end ML pipeline: split → scale → train → tune → evaluate.

    Attributes:
        config: Project configuration object.
        models: Dictionary of trained (name → pipeline) pairs.
        best_params: Dictionary of best hyperparameters per model.
        cv_scores: Dictionary of cross-validation scores per model.
        X_train, X_test, y_train, y_test: Train/test splits.

    Example:
        >>> pipeline = ModelPipeline()
        >>> pipeline.prepare_data(df)
        >>> pipeline.train_all()
        >>> results = pipeline.get_results_summary()
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the model pipeline.

        Args:
            config: Project configuration. Uses default if not provided.
        """
        self.config = config or Config()
        self.models: Dict[str, Pipeline] = {}
        self.best_params: Dict[str, Dict] = {}
        self.cv_scores: Dict[str, np.ndarray] = {}

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self._feature_columns: Optional[List[str]] = None

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str = "refunded",
        exclude_columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train/test sets.

        Args:
            data: Full dataset DataFrame.
            target_column: Name of the target column.
            exclude_columns: Additional columns to drop (e.g., rule predictions).

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).

        Raises:
            ValueError: If target_column is not found in data.
        """
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not in data.")

        drop_cols = [target_column]
        if exclude_columns:
            drop_cols.extend(exclude_columns)

        X = data.drop(columns=[c for c in drop_cols if c in data.columns])
        y = data[target_column]

        self._feature_columns = list(X.columns)

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=y,
            )
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_model(
        self,
        model_name: str,
        tune_hyperparams: bool = True,
    ) -> Pipeline:
        """
        Train a single model with optional hyperparameter tuning.

        The pipeline includes StandardScaler → Classifier.

        Args:
            model_name: Name of the model (must be in registry).
            tune_hyperparams: Whether to run GridSearchCV.

        Returns:
            Trained sklearn Pipeline.

        Raises:
            ValueError: If model_name is not available or data not prepared.
        """
        if self.X_train is None:
            raise ValueError("Call prepare_data() before training.")

        if model_name not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {get_available_models()}"
            )

        classifier_class = _MODEL_REGISTRY[model_name]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier_class(
                random_state=self.config.random_seed
            )),
        ])

        if tune_hyperparams and model_name in self.config.model_hyperparams:
            param_grid = {
                f"classifier__{k}": v
                for k, v in self.config.model_hyperparams[model_name].items()
            }

            grid_search = GridSearchCV(
                pipe,
                param_grid,
                cv=self.config.cv_folds,
                scoring="accuracy",
                n_jobs=-1,
                error_score="raise",
            )
            grid_search.fit(self.X_train, self.y_train)

            self.models[model_name] = grid_search.best_estimator_
            self.best_params[model_name] = grid_search.best_params_
        else:
            pipe.fit(self.X_train, self.y_train)
            self.models[model_name] = pipe

        # Cross-validation score
        self.cv_scores[model_name] = cross_val_score(
            self.models[model_name],
            self.X_train,
            self.y_train,
            cv=self.config.cv_folds,
            scoring="accuracy",
        )

        return self.models[model_name]

    def train_all(self, tune_hyperparams: bool = True) -> Dict[str, Pipeline]:
        """
        Train all available models.

        Args:
            tune_hyperparams: Whether to tune hyperparameters.

        Returns:
            Dictionary of model_name → trained Pipeline.
        """
        for name in get_available_models():
            self.train_model(name, tune_hyperparams=tune_hyperparams)
        return self.models

    def predict(self, model_name: str) -> np.ndarray:
        """
        Generate predictions on the test set.

        Args:
            model_name: Name of the trained model.

        Returns:
            Array of binary predictions.

        Raises:
            ValueError: If the model hasn't been trained.
        """
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not trained. "
                f"Call train_model('{model_name}') first."
            )
        return self.models[model_name].predict(self.X_test)

    def predict_proba(self, model_name: str) -> np.ndarray:
        """
        Generate probability estimates for the positive class.

        Args:
            model_name: Name of the trained model.

        Returns:
            Array of probabilities for class 1 (refund approved).
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained.")
        return self.models[model_name].predict_proba(self.X_test)[:, 1]

    def get_feature_importance(self, model_name: str) -> pd.Series:
        """
        Extract feature importance from a trained model.

        Uses ``feature_importances_`` for tree models and
        ``coef_`` for linear models.

        Args:
            model_name: Name of the trained model.

        Returns:
            pd.Series with feature names as index and importances as values,
            sorted descending.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not trained.")

        classifier = self.models[model_name].named_steps["classifier"]

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            raise AttributeError(
                f"Model '{model_name}' does not support feature importance."
            )

        return pd.Series(
            importances,
            index=self._feature_columns,
            name=model_name,
        ).sort_values(ascending=False)

    def get_results_summary(self) -> pd.DataFrame:
        """
        Generate a summary table of all trained models.

        Returns:
            DataFrame with columns:
                Model, CV_Mean_Accuracy, CV_Std, Test_Accuracy
        """
        from sklearn.metrics import accuracy_score

        rows = []
        for name in self.models:
            preds = self.predict(name)
            rows.append({
                "Model": name,
                "CV_Mean_Accuracy": self.cv_scores[name].mean(),
                "CV_Std": self.cv_scores[name].std(),
                "Test_Accuracy": accuracy_score(self.y_test, preds),
                "Best_Params": self.best_params.get(name, {}),
            })

        return pd.DataFrame(rows).sort_values(
            "Test_Accuracy", ascending=False
        ).reset_index(drop=True)
