# Wine Quality — End-to-End Regression & Classification

[![CI](https://github.com/Manish27-tech/wine-quality-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Manish27-tech/wine-quality-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Two modelling tasks on one dataset: predict the **ordinal quality score** of a wine
(regression) and decide whether it clears the **"good wine"** bar of 7+ (classification).

Built on the [UCI Wine Quality dataset](https://archive.ics.uci.edu/dataset/186/wine+quality)
(Cortez et al., 2009): 6,497 wines, 11 physicochemical measurements, red and white
*Vinho Verde* samples. After removing 1,177 duplicate lab measurements, 5,320 wines remain.

---

## Results (hold-out, 1,064 wines)

### Regression — predicted quality (3–9)

| Metric | Value |
| --- | --- |
| RMSE | **0.660** quality points |
| MAE | **0.509** quality points |
| R² | **0.421** |
| Within ±1 quality point | **88.0%** |
| Baseline RMSE (predict the mean) | 0.868 |
| Improvement over baseline | **23.9%** |

### Classification — quality ≥ 7

| Metric | Value |
| --- | --- |
| ROC AUC | **0.874** |
| Average precision | **0.624** |
| F1 | **0.609** |
| Precision | 0.531 |
| Recall | 0.713 |
| Accuracy | 0.826 |
| Brier score | 0.106 |
| Decision threshold | 0.33 |

Candidate comparison (5-fold CV on the training split):

| Task | Model | Headline CV metric |
| --- | --- | --- |
| Regression | **Random forest** | RMSE **0.699** |
| Regression | Gradient boosting | RMSE 0.700 |
| Regression | Ridge | RMSE 0.733 |
| Classification | **Random forest** | ROC AUC **0.850** |
| Classification | Gradient boosting | ROC AUC 0.840 |
| Classification | Logistic regression | ROC AUC 0.832 |

The full analysis with figures is in **[reports/REPORT.md](reports/REPORT.md)**.

---

## Pipeline

```
download (red + white)  →  validate  →  de-duplicate  →  engineer chemistry ratios
                                                                      ↓
      reports  ←  regression + classification  ←  EDA  ←────────────┘
```

| Module | Responsibility |
| --- | --- |
| `src/wine/data.py` | Download, `;`-aware loading, 8 validation checks |
| `src/wine/clean.py` | Type coercion, duplicate removal, range clipping |
| `src/wine/features.py` | Six chemistry ratios + wine-type flag; both targets |
| `src/wine/eda.py` | Summary statistics + 7 figures |
| `src/wine/model.py` | Regression **and** classification workflows |
| `src/wine/report.py` | Report + model card generation |
| `src/wine/cli.py` | `download` / `eda` / `train` / `all` / `score` |

---

## Quickstart

```bash
git clone https://github.com/Manish27-tech/wine-quality-pipeline.git
cd wine-quality-pipeline
pip install -r requirements.txt

python run.py all       # download → EDA → both models → reports (~2.5 min)
python run.py eda
python run.py train
python run.py score --input data/raw/winequality-red.csv --task classification
```

Notebook (executed, outputs committed):

```bash
pip install -r requirements-dev.txt
python scripts/build_notebook.py
```

---

## Data quality decisions

- **1,177 exact duplicate rows.** These are repeat measurements of the same wine, not
  data-entry errors. Keeping them lets the same bottle appear in both training and
  validation folds and inflates every metric — so they are removed once and the count is
  reported per wine type (red 240, white 937).
- **No missing values.** The validation layer enforces that.
- **Plausibility ranges** are defined per measurement and enforced, so a mis-parsed file
  (wrong separator, truncated download) fails loudly instead of producing a plausible
  report.
- **`wine_type` is a feature, not a shortcut to hide.** Per-type metrics are reported
  because colour is confounded with chemistry.

---

## What the analysis finds

1. **Alcohol is the strongest single signal** (r ≈ +0.44 with quality), followed by
   **volatile acidity** (negative) and **sulphates**.
2. **Quality is highly discretised.** 5s and 6s account for 77% of bottles; 3s and 9s
   are rare enough that no model can predict them reliably.
3. **Red and white are different problems.** White wines skew sweeter and denser, reds
   carry more sulphates and volatile acidity; the model has to separate style from
   quality.
4. **The regression ceiling is real.** Physicochemical measurements explain ~42% of the
   variance in expert scores — the rest is sensory-panel noise and unmeasured factors.

---

## Design decisions

- **De-duplicate before splitting.** Otherwise near-identical wines leak across folds.
- **Two targets, one feature set.** Keeps the comparison honest: the classification
  model is the regression problem with a business threshold applied.
- **Threshold tuned on out-of-fold probabilities**, then reported with a precision/recall
  table so a winery can move it.
- **Metrics that mean something to a winemaker:** RMSE in quality points and
  "within ±1 point" are more actionable than raw R².
- **Deterministic.** All splits, searches and models seeded with `RANDOM_STATE = 42`.

---

## Project structure

```
wine-quality-pipeline/
├── src/wine/             # importable pipeline package
├── tests/                # 25 unit tests (pytest)
├── notebooks/            # executed EDA + modelling notebook
├── scripts/              # notebook builder
├── data/raw/             # committed UCI CSVs for offline reproducibility
├── reports/              # REPORT.md, metrics.json, model_card.md, figures/
├── .github/workflows/    # CI: lint + tests + end-to-end pipeline
├── run.py
└── pyproject.toml
```

---

## Testing

```bash
pip install -r requirements-dev.txt
ruff check src tests scripts
pytest -q
```

CI lints, tests on Python 3.11 and 3.12, then runs the whole pipeline and uploads the
generated reports as build artefacts.

---

## Data & licence

Dataset: UCI Wine Quality, Cortez et al. (2009), public. Code: MIT — see
[LICENSE](LICENSE).
