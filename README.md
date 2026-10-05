# Mashable Article Popularity Prediction

Kaggle competition: binary classification — will an article get **≥ 1400** shares?  
Metric: **ROC-AUC**. Test set is newer than train (temporal shift).

| | |
|---|---|
| Train | 31,715 articles |
| Test | 7,929 articles |
| Popular share (train) | ≈ 54.7% |
| Result | **1st place** (best of 4 submissions) |

---

## Repository layout

```
├── README.md                 # this file
├── requirements.txt          # dependencies
└── solution.ipynb            # full pipeline + all experiments
```

Competition data is expected as Kaggle Input (`train.csv`, `test.csv`, `sample_submission.csv`).

---

## Approach summary

1. **Features:** clip ratios, `log1p` for skewed counts, images/links per 100 words, weekday/channel one-hot, PCA on keyword and sentiment blocks.
2. **Validation:** data sorted by `publish_date`; first a single 80/20 time split, then **TimeSeriesSplit** (5 folds) so gains of ±0.002 are not treated as signal.
3. **Models:** LightGBM + CatBoost, **rank blend**.
4. **Train/test shift:** adversarial validation (AUC ≈ 0.998). Dropping time-leaking features (`month`, `kw_min_min`, `kw_avg_min`) improved a **single** 80/20 split, but TimeSeriesSplit showed the gain was noise → final model keeps the **full** feature set.
5. **Hyperparameters:** Optuna gave ~+0.001 on one split and ~0 on time-CV; params kept (no harm).
6. **Final:** LGBM (Optuna params, 5 seeds) + CatBoost (3 seeds), rank blend, fit on full train.

---

## Experiment timeline

### 1. Baseline pipeline
**Hypothesis:** careful preprocessing + a conservative LightGBM is a solid start.

- Clip `n_unique_tokens` and related ratios to [0, 1]
- `-1` in `kw_min_min` / `kw_avg_min` → NaN
- `log1p` for length, links, keyword and self-reference features
- New: `imgs_per_100w`, `links_per_100w`, `self_link_share`, `kw_best_vs_avg`
- One-hot: `weekday`, `channel`; `month`, `is_weekend`
- PCA: 3 components on keywords, 4 on sentiment

**Result (single 80/20 time split):** LGBM ≈ **0.7518**

**Takeaway:** baseline works; next add models and check stability.

---

### 2. CatBoost + rank blend
**Hypothesis:** a second model + rank blend yields +0.002–0.005.

| LGBM weight | Val AUC |
|-------------|---------|
| 0 (CB only) | 0.7542 |
| 0.3 | **0.7545** |
| 0.5 | 0.7542 |
| 0.7 | 0.7535 |
| 1 (LGBM only) | 0.7518 |

**Takeaway:** CatBoost is slightly stronger; 0.3/0.7 blend is a good default. First strong submission used this setup (5 LGBM seeds + 3 CB seeds, full features).

---

### 3. Adversarial validation and feature dropping
**Hypothesis:** strong train↔test shift (AUC **0.9985**) breaks validation; removing top shift features should improve generalization.

Adv model top features: `month`, `kw_max_avg`, `kw_min_min`, `kw_avg_min`, …

| Feature set | AUC (single 80/20) |
|-------------|--------------------|
| all | 0.7518 |
| without `month` | 0.7543 |
| without `month`, `kw_min_min`, `kw_avg_min` (**FEATS**) | **0.7549** |
| without `month` and all `kw_*` | 0.7375 (bad) |

On FEATS, LGBM+CB blend reached **0.7557** (weight ~0.5).

**Problem:** single split only. After **TimeSeriesSplit (5 folds)**:

| Feature set | mean AUC | std |
|-------------|----------|-----|
| all features | 0.7228 | 0.0227 |
| FEATS (no month / kw_min_min / kw_avg_min) | 0.7253 | 0.0233 |

Difference **+0.0025** is within noise (±0.023). Fold-wise results mixed.

**Takeaway:** dropping “temporal” features was **not confirmed** by honest CV. Final solution **keeps** them. Lesson: adversarial detects shift but does not mean a feature is useless for the target.

---

### 4. TimeSeriesSplit as the main criterion
**Why:** gains of 0.001–0.002 on one block are often noise; 5 time folds filter that out.

Used for:
- decide whether to drop features;
- validate Optuna;
- (in experiments) score oof blends.

---

### 5. Optuna (LightGBM)
**Hypothesis:** tuning `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda` gives a small boost.

- Best trial on one 80/20: **0.7548**
- Same params on TimeSeriesSplit: mean **0.7235** ± 0.0224  
  (baseline params: 0.7228 ± 0.0227)

**Takeaway:** honest CV gain ≈ 0. Best params are no worse than baseline → used in the final LGBM.

Example params found:
```text
learning_rate ≈ 0.012
num_leaves = 34
min_child_samples = 132
subsample ≈ 0.78
colsample_bytree ≈ 0.30
reg_lambda ≈ 0.71
```

---

### 6. Three-model ensemble (experiment)
Tried LGBM + CatBoost + XGBoost with oof and weight search.  
Because of TimeSeriesSplit (early rows never appear in val), raw oof AUC looked artificially low (~0.62) until NaNs were masked.  
For the winning place, **two** models (LGBM+CB) without stacking were enough — simpler and more stable.

---

### 7. Final submission (1st place)
- Features: **all** (no drop)
- LGBM: Optuna params, **5** random seeds, average probabilities
- CatBoost: depth=6, lr=0.03, **3** seeds, average
- Mix: **rank blend** (LGBM weight in the 0.3–0.5 range — val spread ~0.0003)
- Fit on **full** train

Four submissions uploaded; the **last** one got the highest public score and **1st place**.

---

## Why these choices

| Decision | Why | Outcome |
|----------|-----|---------|
| Sort by date + time split | Test is newer than train | Validation closer to leaderboard |
| TimeSeriesSplit (5) | Single 80/20 is noisy at ±0.002 | Rejected feature drop |
| Keep `month`/kw | CV did not support drop | More stable generalization |
| Rank blend LGBM+CB | Different model errors | ~+0.003 vs LGBM alone |
| Multiple seeds | Lower variance | Smoother submission |
| Optuna | Cheap chance of a gain | Params no worse than baseline |
| No heavy stacking | Complexity / oof artifacts | 1st place with a simple blend |

---

## How to reproduce

```bash
pip install -r requirements.txt
# data: train.csv, test.csv, sample_submission.csv
# on Kaggle — Add Competition Data
```

Open `solution.ipynb` and run top to bottom.  
The last section writes `submission.csv`.

---

## Dependencies

- Python ≥ 3.10  
- numpy, pandas, scikit-learn  
- lightgbm, catboost  
- optuna (optional, tuning section)  
- xgboost (optional, third-model experiment)

---

## `solution.ipynb` structure

1. Load and overview  
2. Feature prep + PCA  
3. Baseline LightGBM (single time split)  
4. CatBoost and rank blend  
5. Adversarial validation  
6. Feature-set comparison (one split vs TimeSeriesSplit)  
7. Optuna  
8. Final training and `submission.csv`  

Each section: **hypothesis → code → result → takeaway**.

---

## Author note

Solution evolved across several Kaggle notebook versions.  
Main lesson: **temporal shift exists, but “obvious” time-leaking features should not always be dropped** — only after multi-fold time CV.
