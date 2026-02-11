# Phase 3: Performance Prediction Models

This phase trains Machine Learning models to predict a subject's final BCI accuracy using only features extracted from their first 15 trials.

## Overview

- **Input (X)**: 23 Early Trial Features (from Phase 2)
- **Target (y)**: Final Ground Truth Accuracy (from Phase 1)
- **Goal**: Regress `y` from `X` to estimate BCI illiteracy/performance early.
- **Validation**: Leave-One-Subject-Out (LOSO) / 5-Fold Cross-Validation.

## Models Trained

We evaluated four regression models:
1.  **Random Forest Regressor** (Best Performance)
2.  **Gradient Boosting Regressor**
3.  **Support Vector Regressor (SVR)**
4.  **Ridge Regression** (Linear baseline)

## Results

Evaluation based on 99 valid subjects (overlapping dataset).

| Model | R² Score | Pearson Correlation (r) | RMSE |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **0.240** | 0.524 | 0.133 |
| Gradient Boosting | 0.187 | **0.539** | 0.132 |
| Ridge Regression | 0.187 | 0.460 | 0.137 |
| SVM | 0.168 | 0.412 | 0.140 |

### Interpretation
- **Correlation (r ~ 0.52)**: Indicates a moderate positive relationship. Subjects with "better" early features (stronger ERD, cleaner CSP patterns) tend to have higher final accuracy.
- **R² (0.24)**: The low R² suggests that while we can predict the *trend*, precise accuracy prediction is difficult with only 15 trials. This is a realistic result for non-leaky early prediction.

## Usage

### Train and Evaluate Models

```bash
python src/train_performance_predictor.py
```

This script:
1. Loads features from `results/early_trial_features.json`.
2. Loads targets from `results/ground_truth_labels.json`.
3. Trains all models using cross-validation.
4. Saves the best model to `results/models/random_forest_model.pkl`.
5. Outputs a comparison report.

## Files
- `src/train_performance_predictor.py`: Main training script.
- `results/models/`: Directory containing saved `.pkl` models.
- `results/model_evaluation.json`: Detailed metrics for all models.
