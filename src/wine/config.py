"""Central configuration: paths, dataset locations and modelling constants."""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
ARTIFACT_DIR = ROOT / "artifacts"
NOTEBOOK_DIR = ROOT / "notebooks"

for _directory in (RAW_DIR, PROCESSED_DIR, REPORT_DIR, FIGURE_DIR, ARTIFACT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# UCI Machine Learning Repository, Wine Quality (Cortez et al., 2009).
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality"
RAW_FILES = {
    "red": "winequality-red.csv",
    "white": "winequality-white.csv",
}
PROCESSED_FILENAME = "wine_quality_clean.csv"

TARGET = "quality"
WINE_TYPE = "wine_type"

# 5-6 dominate the distribution; >= 7 is the conventional "good wine" cut.
GOOD_THRESHOLD = 7

BASE_FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

ENGINEERED_NUMERIC = [
    "total_acidity",
    "bound_so2",
    "free_so2_ratio",
    "sugar_alcohol_ratio",
    "sulphates_alcohol",
    "acidity_ph",
]

FLAG_FEATURES = ["is_red"]

# Plausibility bounds used by the validation layer. Values outside these ranges
# indicate a corrupted or mis-parsed file, not an unusual wine.
VALID_RANGES = {
    "fixed acidity": (3.5, 16.0),
    "volatile acidity": (0.05, 1.60),
    "citric acid": (0.0, 1.70),
    "residual sugar": (0.5, 70.0),
    "chlorides": (0.005, 0.65),
    "free sulfur dioxide": (0.0, 300.0),
    "total sulfur dioxide": (5.0, 450.0),
    "density": (0.985, 1.045),
    "pH": (2.7, 4.1),
    "sulphates": (0.2, 2.1),
    "alcohol": (7.5, 15.5),
    "quality": (0, 10),
}

# --------------------------------------------------------------------------- #
# Modelling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITER = 10
SEARCH_FOLDS = 3

FIG_DPI = 140
PALETTE = {"red": "#C44E52", "white": "#DDAA33"}
