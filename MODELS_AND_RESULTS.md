# BCI Performance Prediction: Regression & Classifier Models — How, Why & Results

This document explains the regression and classifier models built for predicting BCI decoder performance from early motor imagery trials, including the most recent changes (March 2026).

---

## 1. Overview: What We Built

**Goal:** Predict a subject's final BCI decoder accuracy from only the first 15 trials, enabling faster calibration and early user screening.

**Approach:**
- **Input:** Neurophysiological features extracted from the first 15 trials only (no data leakage).
- **Target:** Final accuracy from a full CSP+LDA decoder trained on all trials.
- **Output:** Continuous accuracy prediction (regression) and HIGH/LOW performer tier (classifier).

---

## 2. Data Pipeline

```
PhysioNet MI (109 subjects)
         │
         ├── Phase 1: generate_ground_truth_labels.py
         │   • CSP+LDA on all trials → per-subject accuracy (ground truth)
         │   • Output: src/results/ground_truth_labels.json
         │
         ├── Phase 2: extract_early_trial_features.py
         │   • First 15 trials → 23 base neurophysiological features
         │   • Output: src/results/early_trial_features.json
         │
         ├── merge_features.py (MOST RECENT ADDITION)
         │   • Merges Phase 2 + Daniel + Andrew + resting-state → 38 features
         │   • Output: src/results/early_trial_features_merged.json
         │
         └── Phase 3: train_performance_predictor.py
             • X = early features, y = ground truth accuracy
             • Feature selection → train 4 regressors + 1 RF classifier
             • Output: src/results/models/*.pkl, model_evaluation.json
```

---

## 3. Feature Sets

### 3.1 Base Features (23 — Phase 2)

| Category      | Count | Examples |
|---------------|-------|----------|
| Band Power    | 6     | mu/beta for C3, Cz, C4 |
| CSP Patterns  | 4     | eigenvalue ratio, class separability |
| ERD/ERS       | 6     | mu/beta desynchronization (C3, Cz, C4) |
| Variability   | 4     | CV, inter-trial correlation, stability |
| SNR           | 3     | mean, std, max |

### 3.2 Merged Features (38 — Most Recent)

The **merge_features.py** step adds:

- **Daniel (5):** `rpl`, `smr_strength`, `mu_erd_real_C3`, `beta_erd_real_C3`, `mu_erd_imagined_C3`
- **Andrew (7):** entropy, LZC, TAR for LRF/FF beta imagined
- **Resting-state (3):** `resting_rpl_alpha`, `resting_tar`, `resting_pse_avg` — validated as BCI predictors via Spearman + FDR

### 3.3 Selected Features (12 — After Feature Selection)

After Spearman correlation + permutation tests + Benjamini–Hochberg FDR (α=0.05) and |ρ| ≥ 0.1:

1. `band_power_beta_C3`
2. `band_power_beta_C4`
3. `csp_class_separability`
4. `erdrs_mu_C3`
5. `erdrs_mu_Cz`
6. `snr_mean`
7. `rpl` (Daniel)
8. `smr_strength` (Daniel)
9. `mu_erd_imagined_C3` (Daniel)
10. `resting_rpl_alpha` (resting-state)
11. `resting_tar` (resting-state)
12. `resting_pse_avg` (resting-state)

**Why:** Merged features improve prediction by adding validated resting-state and task-based predictors from EDA.

---

## 4. Regression Models

### 4.1 Models Trained

| Model | Algorithm | Hyperparameters | Purpose |
|-------|-----------|-----------------|---------|
| **Random Forest** | `RandomForestRegressor` | n_estimators=200, max_depth=5, min_samples_split=10, min_samples_leaf=5, max_features='sqrt' | Best-performing |
| Gradient Boosting | `GradientBoostingRegressor` | n_estimators=200, lr=0.05, max_depth=3, subsample=0.8 | Ensemble baseline |
| SVM | `SVR` | kernel='rbf', C=0.1, gamma='scale', epsilon=0.1 | Non-linear baseline |
| Ridge Regression | `Ridge` | alpha=10.0 | Linear baseline |

### 4.2 Training Setup

- **Preprocessing:** `StandardScaler` on all features.
- **Feature selection:** Spearman correlation + 5000 permutation tests + Benjamini–Hochberg FDR (α=0.05), plus minimum |ρ| ≥ 0.1.
- **Evaluation:** 5-fold KFold CV.
- **Metrics:** RMSE, MAE, R², Pearson r, Spearman r.
- **Best model:** Chosen by R².

### 4.3 Regression Results (99 subjects, 5-fold CV)

| Model | R² | Pearson r | RMSE | MAE |
|-------|-----|-----------|------|-----|
| **Random Forest** | **0.302** | **0.580** | **0.126** | **0.099** |
| Ridge Regression | 0.269 | 0.530 | 0.130 | 0.101 |
| SVM | 0.246 | 0.492 | 0.134 | 0.105 |
| Gradient Boosting | 0.162 | 0.521 | 0.136 | 0.107 |

**Interpretation:**
- R² ≈ 0.30 is realistic when predicting from only 15 early trials.
- Earlier R² ≈ 0.95 was inflated by circular features (fold accuracies from full data).
- Random Forest performs best and is used in production.

---

## 5. Classifier Model (HIGH vs LOW Performer)

### 5.1 Purpose

Binary classification for pre-screening: HIGH (≥65% accuracy) vs LOW (<65%).

### 5.2 Model

- **Algorithm:** `RandomForestClassifier`
- **Hyperparameters:** n_estimators=200, max_depth=4, max_features=10, class_weight={0: 2, 1: 1}
- **Threshold:** 0.65 (HIGH ≥ 0.65, LOW < 0.65)
- **Evaluation:** Leave-One-Out CV (appropriate for small sample)

### 5.3 Classifier Results (99 subjects, LOOCV)

| Metric | Value |
|--------|-------|
| **LOOCV Accuracy** | **78.8%** |
| Class distribution | HIGH=26, LOW=73 |

**Confusion Matrix (rows=actual, cols=predicted):**

|  | Predicted LOW | Predicted HIGH |
|--|---------------|----------------|
| **Actual LOW** | 65 | 8 |
| **Actual HIGH** | 13 | 13 |

**Per-class metrics:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| LOW | 0.83 | 0.89 | 0.86 |
| HIGH | 0.62 | 0.50 | 0.55 |

**Why class_weight={0: 2, 1: 1}:** LOW is majority (73 vs 26). Higher weight on LOW reduces false negatives for LOW and improves overall calibration.

---

## 6. Ground Truth: CSP+LDA Decoder

- **Pipeline:** MetaBCI CSP (8 components) → `StandardScaler` → `LinearDiscriminantAnalysis`
- **LDA:** solver='lsqr', shrinkage=0.5
- **Task:** Per-trial motor imagery classification (T1 vs T2)
- **Use:** Produces per-subject accuracy used as ground truth for regression and classifier.

---

## 7. Most Recent Changes (March 2026)

### 7.1 Merged Feature Pipeline

- **merge_features.py** expanded to combine:
  - Phase 2 early-trial features (23)
  - Daniel’s EDA features (5)
  - Andrew’s features (7)
  - Resting-state features (3) — validated predictors
- **Result:** 38 features → 12 selected after feature selection.

### 7.2 Feature Selection

- Switched to **Spearman correlation + permutation tests + Benjamini–Hochberg FDR** instead of f_regression.
- **Why:** More robust to non-linearity and controls multiple testing.

### 7.3 Ridge Regression Added

- Added Ridge as a linear baseline for comparison with non-linear models.

### 7.4 RF Classifier

- RF classifier for HIGH/LOW tiering with class weighting for imbalance.
- LOOCV accuracy 78.8%; better at identifying LOW performers (recall 0.89).

### 7.5 Prediction Server Updates

- Loads best regression model and RF classifier.
- Tiering: EXPERT (≥0.80), PROFICIENT (≥0.65), DEVELOPING (≥0.50), EARLY (<0.50).
- Optional blending: `blended_acc = 0.7 * actual + 0.3 * predicted` when actual is available.

---

## 8. Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| First 15 trials only | Avoid data leakage; reflect real early calibration |
| Spearman + permutation + FDR | Robust feature selection and multiple-testing control |
| class_weight for classifier | Handle LOW/HIGH imbalance |
| LOOCV for classifier | Appropriate for small sample (99 subjects) |
| Merged features | Combine validated predictors from EDA |
| CSP+LDA for ground truth | Standard MI-BCI approach; MetaBCI for consistency |
| R² ≈ 0.30 as “good” | Honest when predicting from 15 trials; avoids circularity |

---

## 9. File Reference

| File | Purpose |
|------|---------|
| `src/train_performance_predictor.py` | Trains all regression models and RF classifier |
| `src/generate_ground_truth_labels.py` | CSP+LDA ground truth generation |
| `src/extract_early_trial_features.py` | Early-trial feature extraction |
| `src/merge_features.py` | Merges feature sources (Phase 2 + Daniel + Andrew + resting-state) |
| `src/prediction_server.py` | Flask API and model loading |
| `src/results/model_evaluation.json` | Regression metrics |
| `src/results/models/rf_classifier_metadata.json` | Classifier metrics |

---

**Generated:** March 3, 2026  
**Last model training:** March 2, 2026
