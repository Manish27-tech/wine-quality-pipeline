"""Cleaning: dtype coercion, duplicate removal and range clipping."""

from __future__ import annotations

import pandas as pd

from . import config


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in config.BASE_FEATURES + [config.TARGET]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out[config.TARGET] = out[config.TARGET].astype("Int64")
    out[config.WINE_TYPE] = out[config.WINE_TYPE].astype(str)
    return out


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove duplicate measurements and clip physically implausible values.

    Duplicate rows in this dataset are genuine repeat lab measurements rather than
    data-entry errors, but keeping them would let the same wine appear in both the
    training and validation folds and inflate cross-validation scores.
    """
    typed = coerce_types(df)
    log: dict = {"rows_in": int(len(df))}

    duplicates_by_type = (
        typed.groupby(config.WINE_TYPE, observed=True)
        .apply(lambda frame: int(frame.duplicated().sum()), include_groups=False)
        .to_dict()
    )
    log["duplicates_by_type"] = {k: int(v) for k, v in duplicates_by_type.items()}

    before = len(typed)
    typed = typed.drop_duplicates().reset_index(drop=True)
    log["duplicates_removed"] = int(before - len(typed))

    # Clip instead of drop: a single implausible reading should not delete a wine.
    clipped: dict[str, int] = {}
    for column, (low, high) in config.VALID_RANGES.items():
        series = typed[column]
        out_of_range = int(((series < low) | (series > high)).sum())
        if out_of_range:
            typed[column] = series.clip(lower=low, upper=high)
            clipped[column] = out_of_range
    log["values_clipped"] = clipped

    log["rows_out"] = int(len(typed))
    log["quality_counts"] = {
        str(int(level)): int(count)
        for level, count in typed[config.TARGET].value_counts().sort_index().items()
    }
    log["good_wine_rate"] = round(
        float((typed[config.TARGET] >= config.GOOD_THRESHOLD).mean()), 4
    )
    typed.attrs["cleaning_log"] = log
    return typed, log


def save_processed(df: pd.DataFrame) -> str:
    destination = config.PROCESSED_DIR / config.PROCESSED_FILENAME
    df.to_csv(destination, index=False)
    return str(destination)
