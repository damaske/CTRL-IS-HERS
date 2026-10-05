# Mashable Article Popularity Prediction

Kaggle competition: binary classification — will an article get **≥ 1400** shares?  
Metric: **ROC-AUC**. Test is newer than train (temporal shift).

| | |
|---|---|
| Train | 31,715 articles |
| Test | 7,929 articles |
| Positive rate (train) | ≈ 54.7% |
| Result | **1st place** (best of 4 submissions) |

---

## Repository

```
├── README.md           # short overview
├── requirements.txt
└── solution.ipynb      # full experimental notebook (run this)
```

All hypotheses, ablations, tables, and plots live in **`solution.ipynb`**.

---

## Method (final)

1. **Features:** clip broken ratios, treat `-1` keyword mins as missing, `log1p` on heavy tails, density ratios (images/links per 100 words), weekday/channel one-hot, PCA on keyword & sentiment blocks.
2. **Validation:** chronological sort; single 80/20 time split for fast checks; **TimeSeriesSplit (5 folds)** for final decisions.
3. **Models:** LightGBM (Optuna-tuned) + CatBoost, **rank blend** (weight ≈ 0.3 / 0.7), multi-seed average, fit on full train.

---

## What was tried and rejected

| Idea | Local signal | Verdict |
|------|----------------|---------|
| Drop `month` / `kw_min_min` / `kw_avg_min` (adversarial top features) | +0.003 on one 80/20 split | **Rejected** — TimeSeriesSplit gain within noise; public LB worse |
| Heavy stacking / 3rd model | extra complexity | Not needed for 1st place |
| Aggressive tree complexity | no gain | Kept moderate trees |

---

## Reproduce

```bash
pip install -r requirements.txt
# place train.csv, test.csv, sample_submission.csv (or use Kaggle Input)
```

Open `solution.ipynb` and **Run All**. Last section writes `submission.csv`.

---

## Dependencies

`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `catboost`, `optuna`, `scipy`
