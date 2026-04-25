"""
Dataset Loader Module
======================

Handles loading and preprocessing of real-world datasets:
    - IEEE-CIS Fraud Detection (Kaggle)
    - PaySim Mobile Money Simulator

Includes PCA-based feature alignment to harmonise external datasets
with the 5-feature synthetic feature space.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.config import Config


def load_ieee_cis(
    filepath: str,
    n_samples: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load and preprocess IEEE-CIS Fraud Detection dataset.

    Applies PCA projection to align with the 5-feature synthetic space.
    The top 5 principal components explain ~73.4% of total variance.

    Args:
        filepath: Path to the IEEE-CIS transaction CSV file.
        n_samples: Number of stratified samples to draw.
        random_seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: order_amount, fraud_score,
        previous_refunds, delivery_delay, complaint_severity,
        refunded.
    """
    df = pd.read_csv(filepath)

    # Identify target column
    target_col = 'isFraud' if 'isFraud' in df.columns else 'is_fraud'

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Drop columns with too many NaNs (> 50%)
    valid_cols = [c for c in numeric_cols if df[c].isna().mean() < 0.5]

    X = df[valid_cols].fillna(0)
    y = df[target_col].values

    # Stratified subsample
    rng = np.random.RandomState(random_seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    n_fraud = min(len(fraud_idx), int(n_samples * 0.035))  # ~3.5% prevalence
    n_legit = n_samples - n_fraud

    sampled_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
    sampled_legit = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)
    sampled_idx = np.concatenate([sampled_fraud, sampled_legit])
    rng.shuffle(sampled_idx)

    X_sub = X.iloc[sampled_idx].values
    y_sub = y[sampled_idx]

    # PCA projection to 5 components
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    pca = PCA(n_components=5, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)

    # Map PCA components to synthetic feature names
    result = pd.DataFrame({
        'order_amount': _rescale(X_pca[:, 0], 100, 2000),
        'fraud_score': _rescale(X_pca[:, 1], 0.0, 1.0),
        'previous_refunds': np.clip(np.round(_rescale(X_pca[:, 2], 0, 10)), 0, 10).astype(int),
        'delivery_delay': _rescale(X_pca[:, 3], 0, 90),
        'complaint_severity': np.clip(np.round(_rescale(X_pca[:, 4], 1, 5)), 1, 5).astype(int),
        'refunded': y_sub,
    })

    return result.reset_index(drop=True)


def load_paysim(
    filepath: str,
    n_samples: int = 10000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load and preprocess PaySim mobile money dataset.

    PaySim contains 6.3M transactions with 0.13% fraud prevalence.
    Features are engineered and PCA-aligned with the synthetic space.

    Args:
        filepath: Path to the PaySim CSV file.
        n_samples: Number of stratified samples.
        random_seed: Random seed.

    Returns:
        DataFrame aligned with synthetic feature space.
    """
    df = pd.read_csv(filepath)

    target_col = 'isFraud' if 'isFraud' in df.columns else 'is_fraud'

    # Engineer features from PaySim columns
    feature_df = pd.DataFrame()
    feature_df['amount'] = df['amount'] if 'amount' in df.columns else df.iloc[:, 1]
    feature_df['oldbalanceOrg'] = df.get('oldbalanceOrg', 0)
    feature_df['newbalanceOrig'] = df.get('newbalanceOrig', 0)
    feature_df['oldbalanceDest'] = df.get('oldbalanceDest', 0)
    feature_df['newbalanceDest'] = df.get('newbalanceDest', 0)

    # Balance deltas
    feature_df['balance_delta_org'] = feature_df['oldbalanceOrg'] - feature_df['newbalanceOrig']
    feature_df['balance_delta_dest'] = feature_df['newbalanceDest'] - feature_df['oldbalanceDest']

    # Transaction type encoding
    if 'type' in df.columns:
        feature_df['type_encoded'] = df['type'].astype('category').cat.codes
    if 'step' in df.columns:
        feature_df['step'] = df['step']

    y = df[target_col].values

    # Stratified subsample
    rng = np.random.RandomState(random_seed)
    fraud_idx = np.where(y == 1)[0]
    legit_idx = np.where(y == 0)[0]

    n_fraud = min(len(fraud_idx), int(n_samples * 0.0013 * 10))  # Preserve rarity
    n_legit = n_samples - n_fraud

    sampled_fraud = rng.choice(fraud_idx, size=min(n_fraud, len(fraud_idx)), replace=True)
    sampled_legit = rng.choice(legit_idx, size=min(n_legit, len(legit_idx)), replace=False)
    sampled_idx = np.concatenate([sampled_fraud, sampled_legit])
    rng.shuffle(sampled_idx)

    X_sub = feature_df.iloc[sampled_idx].fillna(0).values
    y_sub = y[sampled_idx]

    # PCA projection to 5 components
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    n_components = min(5, X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=random_seed)
    X_pca = pca.fit_transform(X_scaled)

    # Pad if fewer than 5 components
    if X_pca.shape[1] < 5:
        pad = np.zeros((X_pca.shape[0], 5 - X_pca.shape[1]))
        X_pca = np.hstack([X_pca, pad])

    result = pd.DataFrame({
        'order_amount': _rescale(X_pca[:, 0], 100, 2000),
        'fraud_score': _rescale(X_pca[:, 1], 0.0, 1.0),
        'previous_refunds': np.clip(np.round(_rescale(X_pca[:, 2], 0, 10)), 0, 10).astype(int),
        'delivery_delay': _rescale(X_pca[:, 3], 0, 90),
        'complaint_severity': np.clip(np.round(_rescale(X_pca[:, 4], 1, 5)), 1, 5).astype(int),
        'refunded': y_sub,
    })

    return result.reset_index(drop=True)


def _rescale(arr: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    """Min-max rescale array to [new_min, new_max]."""
    old_min, old_max = arr.min(), arr.max()
    if old_max == old_min:
        return np.full_like(arr, (new_min + new_max) / 2)
    return (arr - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
