"""Regression and classification modelling for wine quality."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import loguniform, randint  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config  # noqa: E402
from .features import (  # noqa: E402
    build_classification_matrix,
    build_regression_matrix,
    feature_columns,
)

plt.rcParams["figure.autolayout"] = True


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """All wine features are numeric, so a single impute + scale branch suffices."""
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer([("num", numeric, feature_columns())], remainder="drop")


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(config.FIGURE_DIR / filename, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def feature_importance(pipeline: Pipeline, top_n: int = 15) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    names = list(preprocessor.get_feature_names_out())

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:  # pragma: no cover - every candidate exposes one of the two
        return []

    order = np.argsort(values)[::-1][:top_n]
    return [
        {"feature": names[index].split("__", 1)[-1], "importance": round(float(values[index]), 5)}
        for index in order
    ]


def _select_threshold(pipeline: Pipeline, X_train, y_train) -> tuple[float, list[dict]]:
    folds = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    out_of_fold = np.zeros(len(y_train))
    for train_index, valid_index in folds.split(X_train, y_train):
        fold_model = clone(pipeline)
        fold_model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        out_of_fold[valid_index] = fold_model.predict_proba(X_train.iloc[valid_index])[:, 1]

    curve = [
        {
            "threshold": float(t),
            "f1": float(f1_score(y_train, out_of_fold >= t)),
            "precision": float(precision_score(y_train, out_of_fold >= t, zero_division=0)),
            "recall": float(recall_score(y_train, out_of_fold >= t, zero_division=0)),
        }
        for t in np.round(np.arange(0.10, 0.91, 0.01), 2)
    ]
    best = max(curve, key=lambda row: row["f1"])
    return best["threshold"], curve


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def regression_candidates() -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=config.RANDOM_STATE,
        ),
    }


REGRESSION_SPACES: dict[str, dict] = {
    "ridge": {"model__alpha": loguniform(1e-2, 1e3)},
    "random_forest": {
        "model__n_estimators": randint(250, 600),
        "model__max_depth": [None, 8, 14, 20],
        "model__min_samples_leaf": randint(1, 6),
        "model__max_features": [0.5, 0.8, 1.0],
    },
    "gradient_boosting": {
        "model__n_estimators": randint(200, 500),
        "model__learning_rate": loguniform(0.02, 0.15),
        "model__max_depth": randint(2, 5),
        "model__subsample": [0.7, 0.85, 1.0],
    },
}


def compare_regressors(X_train, y_train) -> tuple[str, list[dict]]:
    folds = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows: list[dict] = []
    print("[regression] cross-validating candidates")
    for name, estimator in regression_candidates().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                n_jobs=-1, error_score="raise")
        row = {
            "model": name,
            "rmse_mean": round(float(-scores["test_rmse"].mean()), 4),
            "rmse_std": round(float(scores["test_rmse"].std()), 4),
            "mae_mean": round(float(-scores["test_mae"].mean()), 4),
            "r2_mean": round(float(scores["test_r2"].mean()), 4),
            "fit_seconds": round(float(scores["fit_time"].mean()), 2),
        }
        rows.append(row)
        print(f"[regression]   {name:<18} rmse={row['rmse_mean']:.4f} "
              f"mae={row['mae_mean']:.4f} r2={row['r2_mean']:.4f}")

    rows.sort(key=lambda item: item["rmse_mean"])
    return rows[0]["model"], rows


def tune_regressor(name: str, X_train, y_train) -> tuple[Pipeline, dict, float]:
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", regression_candidates()[name]),
    ])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=REGRESSION_SPACES[name],
        n_iter=config.SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=config.SEARCH_FOLDS, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    print(f"[regression] tuned {name}: cv rmse={-search.best_score_:.4f} "
          f"({elapsed:.1f}s, {config.SEARCH_ITER} configs)")
    best_params = {
        key.replace("model__", ""): (value.item() if hasattr(value, "item") else value)
        for key, value in search.best_params_.items()
    }
    return search.best_estimator_, best_params, elapsed


def regression_metrics(y_true, predictions) -> dict:
    errors = np.asarray(predictions) - np.asarray(y_true)
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, predictions))), 4),
        "mae": round(float(mean_absolute_error(y_true, predictions)), 4),
        "r2": round(float(r2_score(y_true, predictions)), 4),
        "median_abs_error": round(float(np.median(np.abs(errors))), 4),
        "within_one_point": round(float((np.abs(errors) <= 1).mean()), 4),
        "within_half_point": round(float((np.abs(errors) <= 0.5).mean()), 4),
        "mean_error": round(float(errors.mean()), 4),
    }


def plot_regression_diagnostics(y_test, predictions, importances: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].scatter(y_test, predictions, s=12, alpha=0.35, color="#4C72B0",
                    edgecolors="none")
    limits = [float(y_test.min()) - 0.5, float(y_test.max()) + 0.5]
    axes[0].plot(limits, limits, linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("Actual quality")
    axes[0].set_ylabel("Predicted quality")
    axes[0].set_title("Actual vs predicted")

    residuals = np.asarray(predictions) - np.asarray(y_test)
    sns.histplot(residuals, bins=30, ax=axes[1], color="#C44E52", kde=True)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual (predicted − actual)")
    axes[1].set_title("Residual distribution")
    fig.suptitle("Regression diagnostics", fontsize=13)
    _save(fig, "08_regression_diagnostics.png")

    frame = pd.DataFrame(importances).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.barplot(data=frame, x="importance", y="feature", hue="feature",
                legend=False, palette="crest", ax=ax)
    ax.set_title("What drives predicted quality")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save(fig, "09_regression_importance.png")


def train_regression(df: pd.DataFrame) -> dict:
    started = time.perf_counter()
    X, y = build_regression_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    print(f"[regression] train={len(X_train):,} | test={len(X_test):,} | "
          f"target mean={y.mean():.2f}")

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline.predict(X_test))))

    best_name, comparison = compare_regressors(X_train, y_train)
    pipeline, best_params, search_seconds = tune_regressor(best_name, X_train, y_train)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, predictions)
    importances = feature_importance(pipeline)

    metrics.update({
        "selected_model": best_name,
        "best_params": best_params,
        "baseline_rmse": round(baseline_rmse, 4),
        "improvement_over_baseline_pct": round(
            (baseline_rmse - metrics["rmse"]) / baseline_rmse * 100, 2
        ),
        "cv_comparison": comparison,
        "search_seconds": round(search_seconds, 1),
        "feature_importance": importances,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    })

    plot_regression_diagnostics(y_test, predictions, importances)
    joblib.dump({"pipeline": pipeline, "metrics": metrics, "features": feature_columns()},
                config.ARTIFACT_DIR / "regression_model.joblib")

    print(f"[regression] test rmse={metrics['rmse']:.4f} mae={metrics['mae']:.4f} "
          f"r2={metrics['r2']:.4f} (±1 point: {metrics['within_one_point']:.1%})")
    return metrics


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classification_candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=config.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=config.RANDOM_STATE,
        ),
    }


CLASSIFICATION_SPACES: dict[str, dict] = {
    "logistic_regression": {
        "model__C": loguniform(1e-2, 1e2),
        "model__solver": ["lbfgs", "liblinear"],
    },
    "random_forest": {
        "model__n_estimators": randint(250, 600),
        "model__max_depth": [None, 8, 14, 20],
        "model__min_samples_leaf": randint(1, 6),
        "model__max_features": ["sqrt", 0.5, 0.8],
    },
    "gradient_boosting": {
        "model__n_estimators": randint(200, 500),
        "model__learning_rate": loguniform(0.02, 0.15),
        "model__max_depth": randint(2, 5),
        "model__subsample": [0.7, 0.85, 1.0],
    },
}


def compare_classifiers(X_train, y_train) -> tuple[str, list[dict]]:
    folds = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    rows: list[dict] = []
    print("[classification] cross-validating candidates")
    for name, estimator in classification_candidates().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        scores = cross_validate(pipeline, X_train, y_train, cv=folds, scoring=scoring,
                                n_jobs=-1, error_score="raise")
        row = {"model": name}
        for metric in scoring:
            row[f"{metric}_mean"] = round(float(scores[f"test_{metric}"].mean()), 4)
            row[f"{metric}_std"] = round(float(scores[f"test_{metric}"].std()), 4)
        row["fit_seconds"] = round(float(scores["fit_time"].mean()), 2)
        rows.append(row)
        print(f"[classification]   {name:<20} auc={row['roc_auc_mean']:.4f} "
              f"f1={row['f1_mean']:.4f} recall={row['recall_mean']:.4f}")

    rows.sort(key=lambda item: item["roc_auc_mean"], reverse=True)
    return rows[0]["model"], rows


def tune_classifier(name: str, X_train, y_train) -> tuple[Pipeline, dict, float]:
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", classification_candidates()[name]),
    ])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=CLASSIFICATION_SPACES[name],
        n_iter=config.SEARCH_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(
            n_splits=config.SEARCH_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
        ),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    print(f"[classification] tuned {name}: cv auc={search.best_score_:.4f} "
          f"({elapsed:.1f}s, {config.SEARCH_ITER} configs)")
    best_params = {
        key.replace("model__", ""): (value.item() if hasattr(value, "item") else value)
        for key, value in search.best_params_.items()
    }
    return search.best_estimator_, best_params, elapsed


def classification_metrics(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y_true, probabilities)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_classification_diagnostics(y_test, probabilities, metrics: dict,
                                    importances: list[dict]) -> None:
    matrix = metrics["confusion_matrix"]
    array = np.array([[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]])

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sns.heatmap(array, annot=True, fmt=",d", cmap="Blues", cbar=False,
                xticklabels=["Not good", "Good"], yticklabels=["Not good", "Good"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix @ {metrics['decision_threshold']:.2f}")
    _save(fig, "10_classification_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(fpr, tpr, linewidth=2, color=config.PALETTE["red"],
                 label=f"AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, linewidth=2, color=config.PALETTE["red"],
                 label=f"AP = {metrics['average_precision']:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(loc="lower left")
    _save(fig, "11_classification_curves.png")

    frame = pd.DataFrame(importances).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.barplot(data=frame, x="importance", y="feature", hue="feature",
                legend=False, palette="flare", ax=ax)
    ax.set_title("Drivers of the 'good wine' probability")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    _save(fig, "12_classification_importance.png")


def train_classification(df: pd.DataFrame) -> dict:
    started = time.perf_counter()
    X, y = build_classification_matrix(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    print(f"[classification] train={len(X_train):,} | test={len(X_test):,} | "
          f"positive rate={y.mean():.3f}")

    best_name, comparison = compare_classifiers(X_train, y_train)
    pipeline, best_params, search_seconds = tune_classifier(best_name, X_train, y_train)
    threshold, curve = _select_threshold(pipeline, X_train, y_train)

    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, probabilities, threshold)
    importances = feature_importance(pipeline)

    operating_points = []
    for candidate in sorted({0.30, 0.40, round(threshold, 2), 0.50, 0.60}):
        row = classification_metrics(y_test, probabilities, candidate)
        operating_points.append({
            "threshold": candidate,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        })

    metrics.update({
        "selected_model": best_name,
        "best_params": best_params,
        "decision_threshold": round(float(threshold), 2),
        "threshold_selection": "maximise F1 on 5-fold out-of-fold probabilities",
        "threshold_curve": curve[::5],
        "operating_points": operating_points,
        "cv_comparison": comparison,
        "search_seconds": round(search_seconds, 1),
        "feature_importance": importances,
        "runtime_seconds": round(time.perf_counter() - started, 1),
    })

    plot_classification_diagnostics(y_test, probabilities, metrics, importances)
    joblib.dump({
        "pipeline": pipeline,
        "threshold": float(threshold),
        "metrics": metrics,
        "features": feature_columns(),
    }, config.ARTIFACT_DIR / "classification_model.joblib")

    print(f"[classification] test auc={metrics['roc_auc']:.4f} "
          f"f1={metrics['f1']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f}")
    return metrics


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train(df: pd.DataFrame) -> dict:
    started = time.perf_counter()
    regression = train_regression(df)
    classification = train_classification(df)

    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "rows": int(len(df)),
            "rows_by_type": {
                str(k): int(v) for k, v in df[config.WINE_TYPE].value_counts().items()
            },
            "features": len(feature_columns()),
            "good_threshold": config.GOOD_THRESHOLD,
        },
        "regression": regression,
        "classification": classification,
        "total_runtime_seconds": round(time.perf_counter() - started, 1),
    }

    path = config.REPORT_DIR / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[model] metrics -> {path.relative_to(config.ROOT)}")
    print(f"[model] total runtime {metrics['total_runtime_seconds']}s")
    return metrics


def load_model(task: str = "regression") -> dict:
    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")
    return joblib.load(config.ARTIFACT_DIR / f"{task}_model.joblib")


def predict(df: pd.DataFrame, task: str = "regression", bundle: dict | None = None):
    from .features import add_features

    bundle = bundle or load_model(task)
    engineered = add_features(df)
    matrix = engineered[bundle["features"]]
    if task == "regression":
        return bundle["pipeline"].predict(matrix)
    return bundle["pipeline"].predict_proba(matrix)[:, 1]
