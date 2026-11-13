"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wine import config


@pytest.fixture()
def raw_frame() -> pd.DataFrame:
    """A small synthetic batch that respects the real dataset's plausibility ranges."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    size = 320
    frame = pd.DataFrame({
        "fixed acidity": np.round(rng.uniform(4.0, 14.0, size), 2),
        "volatile acidity": np.round(rng.uniform(0.12, 1.2, size), 3),
        "citric acid": np.round(rng.uniform(0.0, 1.0, size), 3),
        "residual sugar": np.round(rng.uniform(0.9, 40.0, size), 2),
        "chlorides": np.round(rng.uniform(0.02, 0.4, size), 4),
        "free sulfur dioxide": np.round(rng.uniform(1.0, 100.0, size), 1),
        "total sulfur dioxide": np.round(rng.uniform(10.0, 300.0, size), 1),
        "density": np.round(rng.uniform(0.987, 1.005, size), 5),
        "pH": np.round(rng.uniform(2.9, 3.9, size), 2),
        "sulphates": np.round(rng.uniform(0.3, 1.6, size), 2),
        "alcohol": np.round(rng.uniform(8.0, 14.5, size), 2),
        "quality": rng.integers(3, 10, size),
        config.WINE_TYPE: rng.choice(["red", "white"], size, p=[0.3, 0.7]),
    })
    # Free SO2 must never exceed total SO2 in a real measurement.
    frame["free sulfur dioxide"] = np.minimum(
        frame["free sulfur dioxide"], frame["total sulfur dioxide"] * 0.5
    )
    return frame


@pytest.fixture()
def clean_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    from wine.clean import clean

    cleaned, _ = clean(raw_frame)
    return cleaned
