"""Tests for engineered chemistry features and target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wine import config
from wine.features import (
    add_features,
    add_targets,
    build_classification_matrix,
    build_regression_matrix,
    feature_columns,
)


def test_total_acidity_is_the_sum_of_acid_components(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)
    expected = (
        engineered["fixed acidity"]
        + engineered["volatile acidity"]
        + engineered["citric acid"]
    )
    assert np.allclose(engineered["total_acidity"], expected)


def test_free_so2_ratio_is_bounded(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)
    ratio = engineered["free_so2_ratio"]
    assert ratio.notna().all()
    assert ratio.between(0, 1).all()


def test_bound_so2_is_non_negative(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)
    assert (engineered["bound_so2"] >= 0).all()


def test_is_red_flag_matches_wine_type(clean_frame: pd.DataFrame) -> None:
    engineered = add_features(clean_frame)
    expected = (engineered[config.WINE_TYPE].astype(str) == "red").astype(int)
    assert (engineered["is_red"] == expected).all()


def test_regression_matrix_shape_and_target(clean_frame: pd.DataFrame) -> None:
    X, y = build_regression_matrix(clean_frame)
    assert list(X.columns) == feature_columns()
    assert len(X) == len(y)
    assert y.between(0, 10).all()
    assert not X.isna().any().any()


def test_classification_target_uses_threshold(clean_frame: pd.DataFrame) -> None:
    X, y = build_classification_matrix(clean_frame)
    expected = (clean_frame[config.TARGET] >= config.GOOD_THRESHOLD).astype(int)
    assert (y.reset_index(drop=True) == expected.reset_index(drop=True)).all()
    assert set(y.unique()) <= {0, 1}


def test_add_targets_attaches_binary_label(clean_frame: pd.DataFrame) -> None:
    enriched = add_targets(clean_frame)
    assert "is_good" in enriched.columns
    assert enriched["is_good"].isin([0, 1]).all()
