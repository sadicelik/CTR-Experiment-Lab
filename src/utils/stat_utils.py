from typing import List, Literal, Optional

import numpy as np
import pandas as pd


def zscore_elimination(
    input_df: pd.DataFrame, feature_columns: List[str], threshold: int = 3
) -> pd.DataFrame:
    """
    Eliminate the outliers based on Z-score elimination.
    Suitable for normal distributed data.
    """
    df = input_df.copy()

    for feature in feature_columns:
        if pd.api.types.is_numeric_dtype(df[feature]):  # Type check
            mean, std = df[feature].mean(), df[feature].std()
            z_scores = (df[feature] - mean) / std

            df = df[np.abs(z_scores) <= threshold]

    return df


def iqr_elimination(
    input_df: pd.DataFrame, feature_columns: List[str], k: float = 1.5
) -> pd.DataFrame:
    """
    Eliminate the outliers based on IQR elimination.
    Suitable for skewed distributed data.
    """
    df = input_df.copy()

    for feature in feature_columns:
        if pd.api.types.is_numeric_dtype(df[feature]):  # Type check
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - k * IQR
            upper = Q3 + k * IQR

            df = df[(df[feature] >= lower) & (df[feature] <= upper)]

    return df


def percentile_capping(
    input_df: pd.DataFrame,
    feature_columns: List[str],
    q: float = 0.98,
    side: Literal["right", "left", "both"] = "both",
    gap_factor: Optional[float] = 0.5,
) -> pd.DataFrame:
    """
    Cap outliers at percentile thresholds for both right and left tail.
    Replace outlier values with threshold values at bounds.
    """
    df = input_df.copy()

    for feature in feature_columns:
        if pd.api.types.is_numeric_dtype(df[feature]):  # Type check
            feature_max = df[feature].max()
            feature_min = df[feature].min()

            upper_percentile = df[feature].quantile(q)
            lower_percentile = df[feature].quantile(1 - q)

            # Right tail
            if side in ("right", "both"):
                should_cap_right = (
                    True
                    if gap_factor is None
                    else (upper_percentile < (gap_factor * feature_max))
                )
                if should_cap_right:
                    df.loc[df[feature] >= upper_percentile, feature] = upper_percentile

            # Left tail
            if side in ("left", "both"):
                should_cap_left = (
                    True
                    if gap_factor is None
                    else (lower_percentile > (gap_factor * feature_min))
                )
                if should_cap_left:
                    df.loc[df[feature] <= lower_percentile, feature] = lower_percentile

    return df
