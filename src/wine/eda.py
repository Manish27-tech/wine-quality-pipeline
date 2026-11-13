"""Exploratory data analysis for the wine quality dataset."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .features import add_features, add_targets  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.autolayout"] = True


def _save(fig: plt.Figure, filename: str) -> None:
    path = config.FIGURE_DIR / filename
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] figure -> {path.relative_to(config.ROOT)}")


def plot_quality_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    overall = df[config.TARGET].value_counts().sort_index()
    sns.barplot(x=overall.index.astype(str), y=overall.values, ax=axes[0],
                hue=overall.index.astype(str), legend=False, palette="flare")
    axes[0].set_title("Overall quality distribution")
    axes[0].set_xlabel("Quality score")
    axes[0].set_ylabel("Bottles")
    for index, value in enumerate(overall.values):
        axes[0].text(index, value + 30, f"{value:,}", ha="center", fontsize=9)

    for wine_type, color in config.PALETTE.items():
        subset = df.loc[df[config.WINE_TYPE].astype(str) == wine_type]
        counts = subset[config.TARGET].value_counts(normalize=True).sort_index()
        axes[1].plot(counts.index, counts.values, marker="o", linewidth=2,
                     color=color, label=f"{wine_type} (n={len(subset):,})")
    axes[1].set_title("Quality share by wine type")
    axes[1].set_xlabel("Quality score")
    axes[1].set_ylabel("Share of bottles")
    axes[1].legend()

    fig.suptitle("Quality is concentrated at 5–6; 7+ is rare", fontsize=13)
    _save(fig, "01_quality_distribution.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    columns = config.BASE_FEATURES
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    for ax, column in zip(axes.ravel(), columns, strict=False):
        sns.histplot(data=df, x=column, hue=config.WINE_TYPE, bins=35,
                     element="step", stat="density", common_norm=False, ax=ax,
                     palette=config.PALETTE, legend=(column == columns[0]))
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes.ravel()[len(columns):]:
        ax.axis("off")
    fig.suptitle("Physicochemical distributions: red vs white", fontsize=13)
    _save(fig, "02_feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    engineered = add_features(df)
    columns = [*config.BASE_FEATURES, *config.ENGINEERED_NUMERIC, config.TARGET]
    correlation = engineered[columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"shrink": 0.75},
                annot_kws={"size": 6}, ax=ax)
    ax.set_title("Correlation matrix (engineered features included)")
    _save(fig, "03_correlation_heatmap.png")


def plot_correlations_with_quality(df: pd.DataFrame) -> None:
    engineered = add_features(df)
    columns = [*config.BASE_FEATURES, *config.ENGINEERED_NUMERIC]
    correlations = (
        engineered[columns + [config.TARGET]].corr(numeric_only=True)[config.TARGET]
        .drop(config.TARGET).sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = [config.PALETTE["white"] if value < 0 else config.PALETTE["red"]
              for value in correlations]
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Linear correlation with quality score")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("")
    for index, value in enumerate(correlations.values):
        offset = 0.005 if value >= 0 else -0.005
        ax.text(value + offset, index, f"{value:+.2f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=8)
    ax.set_xlim(correlations.min() - 0.1, correlations.max() + 0.1)
    _save(fig, "04_correlation_with_quality.png")


def plot_quality_by_type(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    mix = (
        df.groupby([config.TARGET, config.WINE_TYPE], observed=True)
        .size().unstack(fill_value=0)
        .div(df.groupby(config.WINE_TYPE, observed=True).size(), axis=1)
        .fillna(0)
    )
    mix.plot(kind="bar", stacked=True, ax=axes[0], color=list(config.PALETTE.values()))
    axes[0].set_title("Quality composition within each wine type")
    axes[0].set_xlabel("Quality score")
    axes[0].set_ylabel("Share")
    axes[0].legend(title="")

    good = (
        df.assign(is_good=(df[config.TARGET] >= config.GOOD_THRESHOLD).astype(int))
        .groupby(config.WINE_TYPE, observed=True)["is_good"].mean()
    )
    sns.barplot(x=good.index.astype(str), y=good.values, ax=axes[1],
                hue=good.index.astype(str), legend=False,
                palette=list(config.PALETTE.values()))
    axes[1].set_title(f"Share rated {config.GOOD_THRESHOLD}+")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Share of bottles")
    for index, value in enumerate(good.values):
        axes[1].text(index, value + 0.005, f"{value:.1%}", ha="center", fontsize=10)

    _save(fig, "05_quality_by_wine_type.png")


def plot_alcohol_and_acidity(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    sns.boxplot(data=df, x=config.TARGET, y="alcohol", hue=config.WINE_TYPE,
                palette=config.PALETTE, ax=axes[0], fliersize=1.5)
    axes[0].set_title("Alcohol rises with quality")
    axes[0].set_xlabel("Quality score")
    axes[0].legend(title="")

    sns.scatterplot(
        data=df.sample(min(len(df), 1200), random_state=config.RANDOM_STATE),
        x="volatile acidity", y="alcohol", hue=config.TARGET,
        palette="flare", size=config.TARGET, sizes=(8, 30), alpha=0.6, ax=axes[1],
        legend=False,
    )
    axes[1].set_title("High alcohol + low volatile acidity = high quality")

    _save(fig, "06_alcohol_acidity_landscape.png")


def plot_wine_type_profile(df: pd.DataFrame) -> None:
    engineered = add_features(df)
    columns = [*config.BASE_FEATURES, *config.ENGINEERED_NUMERIC]
    standardised = (
        engineered[columns] - engineered[columns].mean()
    ) / engineered[columns].std(ddof=0)
    standardised[config.WINE_TYPE] = engineered[config.WINE_TYPE].astype(str)
    profile = standardised.groupby(config.WINE_TYPE, observed=True).mean().T

    fig, ax = plt.subplots(figsize=(8, 6.5))
    profile.plot(kind="barh", ax=ax, color=list(config.PALETTE.values()))
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Standardised chemistry profile: red vs white")
    ax.set_xlabel("Standard deviations from the mean")
    ax.set_ylabel("")
    ax.legend(title="")
    _save(fig, "07_wine_type_profile.png")


def compute_summary(df: pd.DataFrame) -> dict:
    engineered = add_targets(df)

    quality_counts = {
        str(int(level)): int(count)
        for level, count in engineered[config.TARGET].value_counts().sort_index().items()
    }
    quality_by_type = {
        wine_type: {
            str(int(level)): int(count)
            for level, count in subset[config.TARGET].value_counts().sort_index().items()
        }
        for wine_type, subset in engineered.groupby(config.WINE_TYPE, observed=True)
    }

    numeric_summary = {}
    for column in [*config.BASE_FEATURES, *config.ENGINEERED_NUMERIC]:
        series = engineered[column].astype(float)
        numeric_summary[column] = {
            "mean": round(float(series.mean()), 3),
            "std": round(float(series.std()), 3),
            "min": round(float(series.min()), 3),
            "median": round(float(series.median()), 3),
            "max": round(float(series.max()), 3),
        }

    correlations = (
        engineered[[*config.BASE_FEATURES, *config.ENGINEERED_NUMERIC, config.TARGET]]
        .corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    good_rate_by_type = (
        engineered.groupby(config.WINE_TYPE, observed=True)["is_good"].mean().round(4).to_dict()
    )

    return {
        "rows": int(len(engineered)),
        "columns": int(engineered.shape[1]),
        "rows_by_type": {
            str(k): int(v)
            for k, v in engineered[config.WINE_TYPE].value_counts().items()
        },
        "quality_counts": quality_counts,
        "quality_by_type": quality_by_type,
        "good_wine_rate": round(float(engineered["is_good"].mean()), 4),
        "good_wine_rate_by_type": {str(k): float(v) for k, v in good_rate_by_type.items()},
        "numeric_summary": numeric_summary,
        "top_correlations_with_quality": [
            {"feature": name, "correlation": round(float(value), 4)}
            for name, value in correlations.head(12).items()
        ],
        "median_alcohol_by_quality": {
            str(int(level)): round(float(value), 2)
            for level, value in engineered.groupby(config.TARGET)["alcohol"].median().items()
        },
    }


def run_eda(df: pd.DataFrame) -> dict:
    """Generate all EDA artefacts and persist the summary JSON."""
    summary = compute_summary(df)

    plot_quality_distribution(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_correlations_with_quality(df)
    plot_quality_by_type(df)
    plot_alcohol_and_acidity(df)
    plot_wine_type_profile(df)

    summary_path = config.REPORT_DIR / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eda] summary -> {summary_path.relative_to(config.ROOT)}")
    return summary
