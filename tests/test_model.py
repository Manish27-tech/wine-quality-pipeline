"""Tests for preprocessing, regression metrics and classification helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from wine import config
from wine.features import build_classification_matrix, build_regression_matrix
from wine.model import (
    _select_threshold,
    build_preprocessor,
    classification_candidates,
    classification_metrics,
    feature_importance,
    regression_candidates,
    regression_metrics,
)


@pytest.fixture()
def regression_data(clean_frame: pd.DataFrame):
    return build_regression_matrix(clean_frame)


@pytest.fixture()
def classification_data(clean_frame: pd.DataFrame):
    return build_classification_matrix(clean_frame)


def test_preprocessor_output_is_finite(regression_data) -> None:
    X, y = regression_data
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X, y)

    assert transformed.shape == (len(X), len(config.BASE_FEATURES)
                                 + len(config.ENGINEERED_NUMERIC) + 1)
    assert np.isfinite(transformed).all()


def test_ridge_pipeline_fits_and_predicts(regression_data) -> None:
    X, y = regression_data
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", Ridge(alpha=1.0))])
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_threshold_selection_stays_in_range(classification_data) -> None:
    X, y = classification_data
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced")),
    ])
    threshold, curve = _select_threshold(pipeline, X, y)

    assert 0.1 <= threshold <= 0.9
    assert len(curve) > 50
    assert max(curve, key=lambda row: row["f1"])["threshold"] == pytest.approx(threshold)


def test_classification_metrics_on_perfect_predictions(classification_data) -> None:
    _, y = classification_data
    probabilities = y.astype(float).to_numpy()
    metrics = classification_metrics(y, probabilities, 0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_regression_metrics_match_manual_computation() -> None:
    y_true = pd.Series([5.0, 6.0, 7.0, 8.0])
    predictions = np.array([4.5, 5.5, 7.5, 8.5])
    metrics = regression_metrics(y_true, predictions)

    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["mean_error"] == pytest.approx(0.0)
    assert metrics["within_one_point"] == 1.0


def test_feature_importance_is_sorted_and_clean(regression_data) -> None:
    X, y = regression_data
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", Ridge())]).fit(X, y)
    importances = feature_importance(pipeline, top_n=4)

    assert len(importances) == 4
    values = [item["importance"] for item in importances]
    assert values == sorted(values, reverse=True)
    assert all("__" not in item["feature"] for item in importances)


def test_candidates_cover_three_families() -> None:
    assert set(regression_candidates()) == {"ridge", "random_forest", "gradient_boosting"}
    assert set(classification_candidates()) == {
        "logistic_regression", "random_forest", "gradient_boosting"
    }
