# Mashable Article Popularity Prediction

**1st place** solution for a Kaggle binary classification competition:  
predict whether a Mashable article will receive **≥ 1,400 shares**.

| Item | Value |
|------|--------|
| Metric | ROC-AUC |
| Train / Test | 31,715 / 7,929 articles |
| Positive rate (train) | ≈ 54.7% |
| Key challenge | **Temporal shift** — test period is strictly after train |
| Public result | **1st place** (best of 4 submitted versions) |


> **Educational project.** This work was done as a coursework-style Kaggle assignment  
> within the **HSE** programme *«IT Ecosystem Control»* (*«Бир экосистем контрол»*, НИУ ВШЭ).  
> The goal is to practice a full ML cycle: EDA, time-aware validation, ablation, and a reproducible final model — not production deployment.

---

## Why this problem is non-trivial

The label is driven by social dynamics that change over months.  
A model that memorizes “what worked in early 2013” fails on later articles.

We measured the shift with **adversarial validation** (train vs test classifier):

```text
Adversarial AUC ≈ 0.998
Top separating features: month, kw_min_min, kw_avg_min, keyword averages
```

Almost perfect separation means standard random CV would be misleading.  
All serious decisions in this project use **time-ordered validation**.

---

## Final pipeline (what scored 1st)

```text
raw CSV
  → clean ratios / log1p / density features / one-hot day & channel
  → PCA on keyword block (3 PCs) and sentiment block (4 PCs)
  → LightGBM (Optuna params, 5 random seeds, average)
  → CatBoost (depth 6, 3 seeds, average)
  → rank blend (weight ≈ 0.3 LGBM + 0.7 CatBoost)
  → submission.csv
```

**Design choices that mattered:**

1. **Chronological splits only** — last 20% of train by date for fast checks; TimeSeriesSplit (5 folds) for final calls.
2. **Keep “time-leaking” features** — adversarial flags them, but multi-fold time CV showed dropping them was noise (see below).
3. **Rank blend, not probability average** — AUC cares about order; ranks align two libraries better than raw scores.
4. **Seed averaging** — cheap variance reduction on a medium-sized table.

---

## Experiment highlights

Full tables, plots, and code live in [`solution.ipynb`](solution.ipynb).  
Summary of the path:

| Stage | What we tried | Validation signal | Decision |
|-------|----------------|-------------------|----------|
| Baseline features + PCA | clip, log1p, ratios, one-hot, PCA | LGBM ≈ **0.7518** (80/20 time) | keep |
| + CatBoost rank blend | weights 0…1 | best ≈ **0.7545** at w=0.3 | use blend |
| Adversarial validation | train vs test classifier | AUC **0.998** | investigate drop |
| Drop `month` + kw mins | single 80/20 | local **0.7549** | *candidate* |
| Same drop, TimeSeriesSplit×5 | mean AUC | **+0.0025** inside ±0.023 noise | **reject drop** |
| Optuna (LGBM) | 40 trials | +noise on time CV | keep params |
| Complexity sweep | leaves 7 / 15 / 31 | leaves=15 competitive | aligned |
| Final multi-seed blend | full train | best public LB | **submit** |

### The important negative result

Dropping features that separate train from test **looked good on one split** and **hurt the leaderboard**.  
TimeSeriesSplit made the gain disappear into fold noise.  

Lesson: under temporal shift, **never trust a single validation block** for ±0.001–0.003 claims.

---

## Repository layout

```text
.
├── README.md              # this file
├── requirements.txt       # Python dependencies
└── solution.ipynb         # end-to-end experiments + plots + final train
```

`solution.ipynb` sections:

0. Setup  
1. Data load & EDA plots  
2. Feature engineering  
3. PCA (variance plots)  
4. Baseline LightGBM (ROC, importances)  
5. CatBoost & rank-blend curves  
6. Adversarial validation  
7. Feature-drop ablation (single split)  
8. TimeSeriesSplit confirmation  
9. Optuna search  
10. Model-complexity sweep  
11. **Final training → `submission.csv`**  
12. Decision log  

---

## How to reproduce

```bash
pip install -r requirements.txt
```

Place competition files (`train.csv`, `test.csv`, `sample_submission.csv`) where the notebook can find them (Kaggle Input or local folder).

Open `solution.ipynb` → **Run All**.  
The last code section writes `submission.csv`.

Expected runtime on CPU: on the order of **10–20 minutes** (Optuna + multi-seed CatBoost dominate).

---

## Tech stack

| Library | Role |
|---------|------|
| pandas / numpy | data & features |
| matplotlib / seaborn | all figures in the notebook |
| scikit-learn | PCA, scaling, TimeSeriesSplit, metrics |
| LightGBM | primary gradient boosting model |
| CatBoost | second model for the blend |
| Optuna | short hyperparameter search |
| scipy | rankdata for blending |

---

## Context

Educational assignment for the **HSE** course *«IT Ecosystem Control»* (*«Бир экосистем контрол»*, НИУ ВШЭ):  
a closed/course Kaggle competition on Mashable article share prediction.

The winning submission was the last of four public uploads. Earlier versions either used a weaker blend or incorrectly dropped time-related features after adversarial validation alone.
