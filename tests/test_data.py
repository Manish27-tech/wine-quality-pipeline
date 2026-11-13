"""Tests for schema validation and loading behaviour."""

from __future__ import annotations

import pandas as pd
import pytest

from wine import config, data


def test_validate_returns_checks(raw_frame: pd.DataFrame) -> None:
    report = data.validate(raw_frame)
    assert report["rows"] == len(raw_frame)
    assert len(report["checks"]) >= 6
    assert all(check["passed"] for check in report["checks"] if check["critical"])


def test_validate_raises_on_missing_feature(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(data.DataValidationError, match="expected_columns"):
        data.validate(raw_frame.drop(columns=["residual sugar"]))


def test_validate_raises_when_a_wine_type_is_missing(raw_frame: pd.DataFrame) -> None:
    only_red = raw_frame[raw_frame[config.WINE_TYPE] == "red"].copy()
    with pytest.raises(data.DataValidationError, match="wine_types"):
        data.validate(only_red)


def test_validate_raises_on_missing_measurements(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "alcohol"] = None
    with pytest.raises(data.DataValidationError, match="no_missing_measurements"):
        data.validate(frame)


def test_validate_reports_out_of_range_values(raw_frame: pd.DataFrame) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "alcohol"] = 40.0
    report = data.validate(frame)
    check = next(c for c in report["checks"] if c["check"] == "physical_ranges")
    assert not check["passed"]


def test_load_raw_tags_wine_type(tmp_path, raw_frame: pd.DataFrame) -> None:
    paths = {}
    for wine_type in ("red", "white"):
        subset = raw_frame[raw_frame[config.WINE_TYPE] == wine_type].copy()
        path = tmp_path / f"winequality-{wine_type}.csv"
        subset.drop(columns=[config.WINE_TYPE]).to_csv(path, sep=";", index=False)
        paths[wine_type] = path

    loaded = data.load_raw(paths)
    assert set(loaded[config.WINE_TYPE]) == {"red", "white"}
    assert len(loaded) == len(raw_frame)
