"""Feature engineering and target construction for both modelling tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive chemistry ratios that are more informative than raw concentrations."""
    out = df.copy()

    out["total_acidity"] = out["fixed acidity"] + out["volatile acidity"] + out["citric acid"]
    out["bound_so2"] = (out["total sulfur dioxide"] - out["free sulfur dioxide"]).clip(lower=0)

    safe_total_so2 = out["total sulfur dioxide"].replace(0, np.nan)
    out["free_so2_ratio"] = (out["free sulfur dioxide"] / safe_total_so2).fillna(0.0)

    # Residual sugar relative to alcohol separates sweet low-alcohol wines from dry,
    # high-alcohol ones far better than either raw column.
    out["sugar_alcohol_ratio"] = out["residual sugar"] / out["alcohol"].replace(0, np.nan)
    out["sulphates_alcohol"] = out["sulphates"] / out["alcohol"].replace(0, np.nan)
    out["acidity_ph"] = out["total_acidity"] / out["pH"].replace(0, np.nan)

    out["is_red"] = (out[config.WINE_TYPE].astype(str) == "red").astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns() -> list[str]:
    return [*config.BASE_FEATURES, *config.ENGINEERED_NUMERIC, *config.FLAG_FEATURES]


def build_regression_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and the ordinal quality score (3-9) for regression."""
    engineered = add_features(df)
    return engineered[feature_columns()].copy(), engineered[config.TARGET].astype(int)


def build_classification_matrix(
    df: pd.DataFrame, threshold: int = config.GOOD_THRESHOLD
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and a binary 'good wine' label for classification."""
    engineered = add_features(df)
    target = (engineered[config.TARGET] >= threshold).astype(int)
    return engineered[feature_columns()].copy(), target


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the derived binary label for EDA and reporting."""
    out = add_features(df)
    out["is_good"] = (out[config.TARGET] >= config.GOOD_THRESHOLD).astype(int)
    return out
