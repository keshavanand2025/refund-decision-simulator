"""
Visualization Module
====================

Publication-quality plotting utilities for the Refund Decision Simulator.
All plots use seaborn + matplotlib with consistent professional styling.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_curve, auc

# Use non-interactive backend for environments without display
matplotlib.use("Agg")

# ---------- Global Style ----------

STYLE_CONFIG = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 11,
}


def apply_style() -> None:
    """Apply the dark professional theme globally."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("viridis")


# ---------- Confusion Matrix ----------


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a styled confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        title: Plot title.
        labels: Display names for classes.
        save_path: Optional file path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    apply_style()

    if labels is None:
        labels = ["Rejected", "Refunded"]

    from src.metrics import get_confusion_matrix

    cm = get_confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="#30363d",
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------- Cost Comparison ----------


def plot_cost_comparison(
    cost_data: pd.DataFrame,
    title: str = "Economic Cost Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot stacked bar chart comparing cost breakdown across strategies.

    Args:
        cost_data: DataFrame with columns ``refund_cost``, ``fraud_penalty``,
                   ``retention_loss`` and strategy names as index.
        title: Plot title.
        save_path: Optional file path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    cost_columns = ["refund_cost", "fraud_penalty", "retention_loss"]
    colors = ["#58a6ff", "#f78166", "#3fb950"]
    labels_display = ["Refund Cost", "Fraud Penalty", "Retention Loss"]

    x = np.arange(len(cost_data))
    width = 0.5
    bottom = np.zeros(len(cost_data))

    for col, color, label in zip(cost_columns, colors, labels_display):
        if col in cost_data.columns:
            values = cost_data[col].values
            ax.bar(x, values, width, bottom=bottom, label=label, color=color,
                   edgecolor="#30363d", linewidth=0.5)
            bottom += values

    ax.set_xlabel("Strategy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Cost (₹)", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(cost_data.index, rotation=30, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------- Feature Importance ----------


def plot_feature_importance(
    importances: pd.Series,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importances.

    Args:
        importances: pd.Series with feature names as index.
        title: Plot title.
        save_path: Optional file path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    sorted_imp = importances.sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_imp)))

    ax.barh(sorted_imp.index, sorted_imp.values, color=colors,
            edgecolor="#30363d", linewidth=0.5)
    ax.set_xlabel("Importance", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------- Model Comparison ----------


def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot grouped bar chart comparing model accuracies (CV mean vs test).

    Args:
        results_df: DataFrame from ModelPipeline.get_results_summary().
        save_path: Optional file path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_df))
    width = 0.3

    bars1 = ax.bar(x - width / 2, results_df["CV_Mean_Accuracy"], width,
                   label="CV Mean Accuracy", color="#58a6ff",
                   edgecolor="#30363d", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, results_df["Test_Accuracy"], width,
                   label="Test Accuracy", color="#3fb950",
                   edgecolor="#30363d", linewidth=0.5)

    # Error bars for CV std
    ax.errorbar(
        x - width / 2,
        results_df["CV_Mean_Accuracy"],
        yerr=results_df["CV_Std"],
        fmt="none",
        ecolor="#f0f6fc",
        capsize=3,
    )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_title(
        "Model Performance Comparison",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=8, color="#c9d1d9")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom',
                fontsize=8, color="#c9d1d9")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------- ROC Curves ----------


def plot_roc_curves(
    y_true: np.ndarray,
    probas_dict: Dict[str, np.ndarray],
    title: str = "ROC Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot overlaid ROC curves for multiple models.

    Args:
        y_true: True labels.
        probas_dict: Dictionary of model_name → predicted probabilities.
        title: Plot title.
        save_path: Optional file path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#f0e68c"]

    for (name, probas), color in zip(probas_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, probas)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------- Data Distribution ----------


def plot_data_distribution(
    data: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot feature distributions with refund class overlay.

    Args:
        data: DataFrame with features and ``refunded`` column.
        save_path: Optional file path to save the figure.

    Returns:
        matplotlib Figure object.
    """
    apply_style()

    features = [c for c in data.columns if c != "refunded"]
    n_features = len(features)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        if i >= len(axes):
            break
        ax = axes[i]
        for label, color in [(0, "#f78166"), (1, "#58a6ff")]:
            subset = data[data["refunded"] == label][feat]
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label=f"{'Refunded' if label == 1 else 'Rejected'}",
                    edgecolor="#30363d", linewidth=0.3)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # Hide unused axes
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature Distributions by Refund Outcome",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
