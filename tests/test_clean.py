"""Tests for cleaning, de-duplication and clipping."""

from __future__ import annotations

import pandas as pd

from wine import config
from wine.clean import clean, coerce_types


def test_coerce_types_sets_quality_dtype(raw_frame: pd.DataFrame) -> None:
    typed = coerce_types(raw_frame)
    assert typed[config.TARGET].dtype == "Int64"
    assert pd.api.types.is_numeric_dtype(typed["alcohol"])


def test_clean_removes_exact_duplicates(raw_frame: pd.DataFrame) -> None:
    frame = pd.concat([raw_frame, raw_frame.iloc[[0, 1]]], ignore_index=True)
    cleaned, log = clean(frame)

    assert log["rows_in"] == len(frame)
    assert log["duplicates_removed"] == 2
    assert log["rows_out"] == len(raw_frame)
    assert not cleaned.duplicated().any()


def test_clean_reports_duplicates_per_type(raw_frame: pd.DataFrame) -> None:
    frame = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    _, log = clean(frame)
    assert sum(log["duplicates_by_type"].values()) == 1
    assert set(log["duplicates_by_type"]) == {"red", "white"}


def test_clean_clips_implausible_measurements(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "alcohol"] = 99.0
    frame.loc[1, "pH"] = 1.0
    cleaned, log = clean(frame)

    low, high = config.VALID_RANGES["alcohol"]
    assert cleaned.loc[0, "alcohol"] == high
    low_ph, high_ph = config.VALID_RANGES["pH"]
    assert cleaned.loc[1, "pH"] == low_ph
    assert log["values_clipped"]["alcohol"] >= 1
    assert log["values_clipped"]["pH"] >= 1


def test_clean_records_class_balance(clean_frame: pd.DataFrame) -> None:
    _, log = clean(clean_frame)
    assert 0 < log["good_wine_rate"] < 1
    assert sum(log["quality_counts"].values()) == len(clean_frame)
