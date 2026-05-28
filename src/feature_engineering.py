"""Feature engineering utilities for telecom churn prediction."""

import numpy as np
import pandas as pd


def create_phase_drift_features(df, feature_groups):
    """
    Create phase drift features: deviation from good-phase baseline.
    Phase drift = month_8 - avg(month_6, month_7)
    
    Args:
        df (pd.DataFrame): Input dataframe with monthly columns
        feature_groups (list): List of feature prefixes (e.g., ['arpu', 'loc_mou'])
    
    Returns:
        pd.DataFrame: DataFrame with phase drift features added
    """
    df_engineered = df.copy()
    
    for feature in feature_groups:
        col_6 = f"{feature}_6"
        col_7 = f"{feature}_7"
        col_8 = f"{feature}_8"
        
        if col_6 in df.columns and col_7 in df.columns and col_8 in df.columns:
            good_phase_avg = (df[col_6] + df[col_7]) / 2
            df_engineered[f"{feature}_phase_drift"] = df[col_8] - good_phase_avg
    
    return df_engineered


def create_mom_diff_features(df, feature_groups):
    """
    Create month-over-month difference features.
    
    Args:
        df (pd.DataFrame): Input dataframe with monthly columns
        feature_groups (list): List of feature prefixes (e.g., ['arpu', 'loc_mou'])
    
    Returns:
        pd.DataFrame: DataFrame with MoM difference features added
    """
    df_engineered = df.copy()
    
    for feature in feature_groups:
        col_6 = f"{feature}_6"
        col_7 = f"{feature}_7"
        col_8 = f"{feature}_8"
        
        if col_6 in df.columns and col_7 in df.columns:
            df_engineered[f"{feature}_diff_7_6"] = df[col_7] - df[col_6]
        
        if col_7 in df.columns and col_8 in df.columns:
            df_engineered[f"{feature}_diff_8_7"] = df[col_8] - df[col_7]
    
    return df_engineered


def create_monthly_avg_features(df, feature_groups):
    """
    Create monthly average features across June-August.
    
    Args:
        df (pd.DataFrame): Input dataframe with monthly columns
        feature_groups (list): List of feature prefixes
    
    Returns:
        pd.DataFrame: DataFrame with monthly average features added
    """
    df_engineered = df.copy()
    
    for feature in feature_groups:
        col_6 = f"{feature}_6"
        col_7 = f"{feature}_7"
        col_8 = f"{feature}_8"
        
        cols_to_avg = [col for col in [col_6, col_7, col_8] if col in df.columns]
        if len(cols_to_avg) > 0:
            df_engineered[f"{feature}_monthly"] = df[cols_to_avg].mean(axis=1)
    
    return df_engineered


def create_high_value_customer_flag(df, arpu_threshold=500, recharge_threshold=1000):
    """
    Derive high-value customer flag based on ARPU and recharge thresholds.
    
    Args:
        df (pd.DataFrame): Input dataframe
        arpu_threshold (float): ARPU threshold for high-value classification
        recharge_threshold (float): Recharge amount threshold
    
    Returns:
        pd.DataFrame: DataFrame with high_value_customer flag added
    """
    df_engineered = df.copy()
    
    # Check if required columns exist
    if 'arpu_6' in df.columns and 'total_rech_amt_6' in df.columns:
        avg_arpu = df['arpu_6']  # Use June as baseline
        df_engineered['high_value_customer'] = (
            (avg_arpu >= arpu_threshold) | (df['total_rech_amt_6'] >= recharge_threshold)
        ).astype(int)
    else:
        print("Warning: Required columns for high_value_customer not found")
    
    return df_engineered


def engineer_features(df, feature_groups=None, apply_hvc_filter=False):
    """
    Apply all feature engineering transformations.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_groups (list): Feature prefixes to engineer. If None, uses default.
        apply_hvc_filter (bool): If True, filter to high-value customers only
    
    Returns:
        pd.DataFrame: Feature-engineered dataframe
    """
    if feature_groups is None:
        feature_groups = [
            'arpu', 'onnet_mou', 'offnet_mou', 'roam_og_mou', 
            'loc_og_mou', 'std_og_mou', 'isd_og_mou', 'spl_og_mou',
            'total_rech_amt', 'vol_2g', 'vol_3g', 'monthly_2g', 'sachet_2g'
        ]
    
    print("Starting feature engineering...")
    
    # Create derived features
    df_engineered = create_high_value_customer_flag(df)
    df_engineered = create_phase_drift_features(df_engineered, feature_groups)
    df_engineered = create_mom_diff_features(df_engineered, feature_groups)
    df_engineered = create_monthly_avg_features(df_engineered, feature_groups)
    
    print(f"Features created. Shape: {df_engineered.shape}")
    
    if apply_hvc_filter:
        df_engineered = df_engineered[df_engineered['high_value_customer'] == 1].reset_index(drop=True)
        print(f"Filtered to high-value customers. Shape: {df_engineered.shape}")
    
    return df_engineered
