"""
Dataset Loader Module
======================

Loads and preprocesses real-world datasets for cross-domain validation:
    - IEEE-CIS Fraud Detection (Kaggle)
    - PaySim Mobile Money Simulator

Both datasets are harmonised to a 5-feature PCA-projected space
aligned with the synthetic feature axes, following the methodology
described in Dal Pozzolo et al. [9].
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
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and preprocess the IEEE-CIS Fraud Detection dataset.

    Applies stratified subsampling and PCA projection to align
    with the 5-feature synthetic space.

    Args:
        filepath: Path to the IEEE-CIS transaction CSV file.
        n_samples: Number of samples to subsample (stratified).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns: order_amount, fraud_score,
        previous_refunds, delivery_delay, complaint_severity,
        refunded (target).
    """
    df = pd.read_csv(filepath)

    # Identify target column
    target_col = "isFraud" if "isFraud" in df.columns else "is_fraud"

    # Select numeric features only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # Drop columns with >50% missing
    valid_cols = [
        c for c in numeric_cols
        if df[c].isna().mean() < 0.5
    ]

    X = df[valid_cols].fillna(0)
    y = df[target_col]

    # Stratified subsample
    np.random.seed(random_state)
    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].index

    n_fraud = min(len(fraud_idx), int(n_samples * 0.035))  # ~3.5% prevalence
    n_legit = n_samples - n_fraud

    sampled_fraud = np.random.choice(fraud_idx, n_fraud, replace=False)
    sampled_legit = np.random.choice(legit_idx, n_legit, replace=False)
    sampled_idx = np.concatenate([sampled_fraud, sampled_legit])
    np.random.shuffle(sampled_idx)

    X_sub = X.loc[sampled_idx].reset_index(drop=True)
    y_sub = y.loc[sampled_idx].reset_index(drop=True)

    # PCA projection to 5 components
    return _pca_project(X_sub, y_sub, n_components=5, random_state=random_state)


def load_paysim(
    filepath: str,
    n_samples: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load and preprocess the PaySim mobile money dataset.

    PaySim has extreme class imbalance (0.13% fraud). Features are
    engineered and PCA-projected to align with the synthetic space.

    Args:
        filepath: Path to the PaySim CSV file.
        n_samples: Number of samples to subsample (stratified).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns: order_amount, fraud_score,
        previous_refunds, delivery_delay, complaint_severity,
        refunded (target).
    """
    df = pd.read_csv(filepath)

    # PaySim column mapping
    target_col = "isFraud"

    # Engineer features
    features = pd.DataFrame()
    features["amount"] = df["amount"]
    features["balance_delta"] = (
        df["newbalanceOrig"] - df["oldbalanceOrg"]
    ).abs()
    features["dest_balance_delta"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    ).abs()
    features["step"] = df["step"]

    # Encode transaction type
    if "type" in df.columns:
        type_dummies = pd.get_dummies(df["type"], prefix="type")
        features = pd.concat([features, type_dummies], axis=1)

    y = df[target_col]

    # Stratified subsample preserving fraud ratio
    np.random.seed(random_state)
    fraud_idx = y[y == 1].index
    legit_idx = y[y == 0].index

    n_fraud = min(len(fraud_idx), max(13, int(n_samples * 0.0013)))
    n_legit = n_samples - n_fraud

    sampled_fraud = np.random.choice(
        fraud_idx, n_fraud, replace=len(fraud_idx) < n_fraud
    )
    sampled_legit = np.random.choice(legit_idx, n_legit, replace=False)
    sampled_idx = np.concatenate([sampled_fraud, sampled_legit])
    np.random.shuffle(sampled_idx)

    X_sub = features.loc[sampled_idx].fillna(0).reset_index(drop=True)
    y_sub = y.loc[sampled_idx].reset_index(drop=True)

    return _pca_project(X_sub, y_sub, n_components=5, random_state=random_state)


def _pca_project(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Project features into a 5-dimensional PCA space aligned with
    synthetic feature semantics.

    The top 5 principal components are mapped to:
        PC1 → order_amount (transaction value)
        PC2 → fraud_score (fraud propensity)
        PC3 → previous_refunds (behavioural history)
        PC4 → delivery_delay (temporal delay)
        PC5 → complaint_severity (complaint intensity)

    Args:
        X: Feature matrix.
        y: Target labels.
        n_components: Number of PCA components.
        random_state: Random seed.

    Returns:
        DataFrame with synthetic-aligned feature names + target.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, X_scaled.shape[1]),
              random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # Pad if fewer components than requested
    if X_pca.shape[1] < n_components:
        pad = np.zeros((X_pca.shape[0], n_components - X_pca.shape[1]))
        X_pca = np.hstack([X_pca, pad])

    feature_names = [
        "order_amount",
        "fraud_score",
        "previous_refunds",
        "delivery_delay",
        "complaint_severity",
    ]

    result = pd.DataFrame(X_pca[:, :5], columns=feature_names)

    # Rescale to match synthetic feature ranges
    from src.config import Config
    config = Config()
    for col in feature_names:
        col_min, col_max = config.feature_ranges.get(col, (0, 1))
        col_data = result[col]
        if col_data.std() > 0:
            result[col] = (
                (col_data - col_data.min())
                / (col_data.max() - col_data.min())
                * (col_max - col_min)
                + col_min
            )

    result["refunded"] = y.values

    variance_explained = pca.explained_variance_ratio_.sum()
    print(
        f"PCA variance explained ({pca.n_components_} components): "
        f"{variance_explained:.1%}"
    )

    return result
