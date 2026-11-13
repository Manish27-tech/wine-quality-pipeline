# Model Card — Wine Quality

## Overview
| Field | Regression | Classification |
| --- | --- | --- |
| Task | Predict quality score | Predict quality ≥ 7 |
| Algorithm | `random_forest` | `random_forest` |
| Rows | 5,320 | 5,320 |
| Features | 18 | 18 |
| Trained | 2026-11-13T12:46:25+00:00 | 2026-11-13T12:46:25+00:00 |

## Performance (hold-out, 20%)
| Metric | Value |
| --- | --- |
| RMSE (quality points) | 0.6599 |
| MAE (quality points) | 0.5085 |
| R² | 0.4208 |
| Within ±1 point | 88.0% |
| ROC AUC (good wine) | 0.8735 |
| F1 (good wine) | 0.6089 |
| Decision threshold | 0.33 |

## Intended use
Decision support for winemaking QC and batch screening. **Not** a substitute for
sensory panels, and not for labelling claims about health or taste.

## Limitations
- Dataset covers Portuguese *Vinho Verde* only; transfer to other regions is unproven.
- Scores are discretised and noisy, capping achievable R².
- No vintage or variety features, so year-to-year drift is unmodelled.

## Ethical considerations
- `wine_type` is a legitimate physicochemical grouping, but models can use it as a
  shortcut; always report per-type metrics.
- Alcohol is a protected public-health concern — do not use this model to market
  higher-alcohol wines as "better".
