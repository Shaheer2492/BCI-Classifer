# Phase 3: Performance Prediction Model Training

This phase trains machine learning models to predict BCI decoder performance from early-trial features.

## Overview

- **Input**: Ground truth labels from Phase 1
- **Models**: Random Forest, Gradient Boosting, SVM
- **Task**: Regression (predict accuracy scores)
- **Cross-validation**: 5-fold
- **Output**: Trained models and evaluation metrics

## Models Implemented

### 1. Random Forest Regressor
- **n_estimators**: 100 trees
- **max_depth**: 10
- **Advantages**: Robust, handles non-linear relationships
- **Performance**: R² ≈ 0.94

### 2. Gradient Boosting Regressor
- **n_estimators**: 100
- **learning_rate**: 0.1
- **max_depth**: 5
- **Advantages**: High accuracy, sequential learning
- **Performance**: R² ≈ 0.95 (**Best Model**)

### 3. Support Vector Machine (SVR)
- **kernel**: RBF
- **C**: 1.0
- **Advantages**: Effective in high dimensions
- **Performance**: R² ≈ 0.82

## Features Extracted

From ground truth data (simulating early trial features):

1. **Trial Information**
   - Number of trials
   - Number of channels

2. **Class Distribution**
   - Class 0 ratio
   - Class 1 ratio

3. **Early Performance Indicators**
   - Mean accuracy (first 2 folds)
   - Std accuracy (first 2 folds)
   - Max accuracy (first 2 folds)
   - Min accuracy (first 2 folds)

4. **Class-specific Performance**
   - Class 0 accuracy
   - Class 1 accuracy

**Total Features**: 10

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python src/train_performance_predictor.py
```

This will:
1. Load ground truth labels from Phase 1
2. Extract features for each subject
3. Train all three models with 5-fold cross-validation
4. Evaluate and compare models
5. Save trained models to `src/results/models/`
6. Save evaluation metrics to `src/results/model_evaluation.json`

### Expected Runtime

- **Total**: ~10-30 seconds (depends on number of subjects)

## Output Files

### Trained Models
```
src/results/models/
├── random_forest_model.pkl
├── gradient_boosting_model.pkl
├── svm_model.pkl
└── feature_scaler.pkl
```

### Evaluation Results
```json
{
  "metadata": {
    "generated_at": "2026-01-28T...",
    "n_subjects": 99,
    "n_folds": 5,
    "best_model": "Gradient Boosting"
  },
  "models": {
    "Random Forest": {
      "rmse_mean": 0.0370,
      "mae_mean": 0.0258,
      "r2_mean": 0.9439,
      "pearson_r": 0.9957
    },
    "Gradient Boosting": {
      "rmse_mean": 0.0332,
      "mae_mean": 0.0246,
      "r2_mean": 0.9504,
      "pearson_r": 0.9999
    },
    "SVM": {
      "rmse_mean": 0.0653,
      "mae_mean": 0.0473,
      "r2_mean": 0.8203,
      "pearson_r": 0.9618
    }
  }
}
```

## Evaluation Metrics

### Root Mean Squared Error (RMSE)
- Measures average prediction error
- Lower is better
- **Best**: Gradient Boosting (0.0332)

### Mean Absolute Error (MAE)
- Average absolute difference between predicted and actual
- Lower is better
- **Best**: Gradient Boosting (0.0246)

### R² Score (Coefficient of Determination)
- Proportion of variance explained
- Range: [0, 1], higher is better
- **Best**: Gradient Boosting (0.9504)

### Pearson Correlation
- Linear relationship between predicted and actual
- Range: [-1, 1], closer to 1 is better
- **Best**: Gradient Boosting (0.9999)

## Model Selection

**Winner: Gradient Boosting**

Selected based on:
- Highest R² score (0.9504)
- Lowest RMSE (0.0332)
- Lowest MAE (0.0246)
- Highest Pearson correlation (0.9999)

This model will be used in the real-time BCI demo for performance prediction.

## Using Trained Models

### Load and Predict

```python
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('src/results/models/gradient_boosting_model.pkl')
scaler = joblib.load('src/results/models/feature_scaler.pkl')

# Prepare features (10 features as described above)
features = np.array([[45, 64, 0.51, 0.49, 0.68, 0.05, 0.72, 0.64, 0.67, 0.69]])

# Scale and predict
features_scaled = scaler.transform(features)
predicted_accuracy = model.predict(features_scaled)

print(f"Predicted BCI Accuracy: {predicted_accuracy[0]:.2%}")
```

## Next Steps

1. **Phase 4**: Integrate model into real-time BCI demo
2. Use model to predict performance from simulated PhysioNet data
3. Display real-time predictions in the web interface

## Future Improvements

- Extract richer features from actual early trial EEG data
- Implement deep learning models (CNN, LSTM)
- Add feature importance analysis
- Hyperparameter optimization with grid search
- Ensemble methods combining multiple models

## References

- **Random Forest**: Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- **Gradient Boosting**: Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine.
- **SVM**: Vapnik, V. (1995). The nature of statistical learning theory.
