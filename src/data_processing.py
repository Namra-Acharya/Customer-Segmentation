from typing import Iterable

import numpy as np
import pandas as pd


EXCLUDED_FEATURES = {"customer_id", "city_tier", "preferred_channel", "membership"}


def load_csv(uploaded_file) -> pd.DataFrame:
    """Load a CSV uploaded to Streamlit or a local file-like object."""
    df = pd.read_csv(uploaded_file)
    if df.empty:
        raise ValueError("The CSV contains no customer records.")
    return df


def numeric_feature_candidates(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are suitable for the initial model."""
    numeric = df.select_dtypes(include=["number"]).columns
    return [column for column in numeric if column not in EXCLUDED_FEATURES]


def validate_clustering_features(
    df: pd.DataFrame, feature_names: Iterable[str]
) -> list[str]:
    features = list(feature_names)
    if not features:
        raise ValueError("Select at least one numerical feature.")

    missing_columns = [name for name in features if name not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    non_numeric = [
        name for name in features if not pd.api.types.is_numeric_dtype(df[name])
    ]
    if non_numeric:
        raise ValueError(
            "Only numerical columns can be used: " + ", ".join(non_numeric)
        )

    all_missing = [name for name in features if df[name].isna().all()]
    if all_missing:
        raise ValueError(
            "These selected features contain only missing values: "
            + ", ".join(all_missing)
        )
    return features


def prepare_feature_matrix(
    df: pd.DataFrame, feature_names: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve the original selected values and provide a median-imputed copy."""
    features = validate_clustering_features(df, feature_names)
    original = df[features].copy()
    prepared = original.copy()
    prepared = prepared.apply(
        lambda column: column.fillna(column.median())
        if column.isna().any()
        else column
    )

    if prepared.isna().any().any():
        raise ValueError("Missing values remain after imputation.")
    if not np.isfinite(prepared.to_numpy(dtype=float)).all():
        raise ValueError("Selected features contain infinite values.")
    return original, prepared
