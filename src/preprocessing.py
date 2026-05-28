"""Data preprocessing utilities: scaling, SMOTE, PCA."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE


def apply_scaling(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def apply_smote(X_train, y_train, random_state=42):
    """Apply SMOTE to handle class imbalance."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def apply_pca(X_train, X_test, n_components=0.95):
    """Apply PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA: Reduced from {X_train.shape[1]} to {X_train_pca.shape[1]} components")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return X_train_pca, X_test_pca, pca, X_train_pca.shape[1]

