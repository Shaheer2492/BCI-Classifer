# BCI-Classifier
Brain Computer Interface performance prediction using MetaBCI decoder and machine learning.

## Overview

This project predicts BCI decoder performance from early-trial motor imagery data using machine learning models trained on PhysioNet MI dataset.

### Key Features

- **MetaBCI CSP+LDA Decoder**: Ground truth generation using MetaBCI's CSP implementation
- **Early Trial Feature Extraction**: 27 neurophysiological features from first 15 trials
- **ML Performance Prediction**: Random Forest, Gradient Boosting, and SVM models for early prediction
- **Real-time Web Demo**: Interactive visualization with live performance predictions
- **PhysioNet MI Dataset**: 109 subjects with motor imagery tasks

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Ground Truth Labels (Phase 1)

```bash
python src/generate_ground_truth_labels.py
```

This processes all 109 subjects from PhysioNet MI dataset (~1-2 hours).

### 3. Extract Early Trial Features (Phase 2)

```bash
python src/extract_early_trial_features.py
```

Extracts 27 neurophysiological features from first 15 trials per subject (~5-10 minutes).

### 4. Train ML Models (Phase 3)

```bash
python src/train_performance_predictor.py
```

Trains Random Forest, Gradient Boosting, and SVM models on early trial features (~10-30 seconds).

### 5. Start Prediction Server

```bash
python src/prediction_server.py
```

Runs Flask server on `http://localhost:5000` for ML predictions.

### 6. View Web Demo

Open `docs/ml-demo.html` in your browser or run:

```bash
cd docs
python -m http.server 8000
# Visit http://localhost:8000/ml-demo.html
```

## Project Structure

```
BCI-Classifer/
├── src/
│   ├── generate_ground_truth_labels.py  # Phase 1: Ground truth generation
│   ├── extract_early_trial_features.py  # Phase 2: Early trial features
│   ├── train_performance_predictor.py   # Phase 3: ML model training
│   ├── prediction_server.py             # Flask API server
│   └── results/
│       ├── ground_truth_labels.json     # Subject accuracies (Phase 1)
│       ├── early_trial_features.json    # Early features (Phase 2)
│       ├── model_evaluation.json        # Model performance metrics
│       └── models/                      # Trained ML models
├── docs/
│   ├── index.html                       # Main landing page
│   ├── ml-demo.html                     # ML-enhanced BCI demo
│   ├── results.html                     # Ground truth results
│   └── js/
│       ├── bci-demo.js                  # Base demo
│       └── bci-demo-ml.js               # ML-enhanced demo
├── PHASE1_README.md                     # Ground truth generation docs
├── PHASE2_README.md                     # Early trial feature extraction docs
├── PHASE3_README.md                     # ML model training docs
└── requirements.txt                     # Python dependencies
```

## Pipeline Architecture

### Phase 1: Ground Truth Generation
- Load PhysioNet MI dataset (109 subjects)
- Apply MetaBCI CSP+LDA decoder
- 5-fold cross-validation per subject
- Extract accuracy metrics
- Output: `ground_truth_labels.json`

### Phase 2: Early Trial Feature Extraction
- Load first 15 trials per subject
- Extract 27 neurophysiological features:
  - Band Power (mu/beta, C3/Cz/C4): 6 features
  - CSP Patterns (spatial filter quality): 4 features
  - ERD/ERS (desynchronization): 6 features
  - Trial Variability (consistency): 4 features
  - Signal-to-Noise Ratio: 3 features
- Output: `early_trial_features.json`

### Phase 3: ML Model Training
- Load early trial features (Phase 2) as input
- Load ground truth accuracies (Phase 1) as targets
- Train 3 models: Random Forest, Gradient Boosting, SVM
- Evaluate with RMSE, MAE, R², Pearson correlation
- Select best model based on cross-validation
- Output: Trained models + evaluation metrics

### Phase 4: Real-time Prediction
- Flask API serves ML predictions
- Web demo visualizes brain activity
- Real-time performance prediction
- PhysioNet subject simulation

## Model Performance

**Note:** With Phase 2 implementation, models now predict from **early trials only** (first 15 trials), not from full dataset statistics. Expected R² range: 0.5-0.8 (realistic for early prediction).

Run the pipeline to see updated performance metrics:

```bash
# After running Phase 1, 2, and 3:
cat src/results/model_evaluation.json
```

Previous performance (before Phase 2, using circular features):
- Gradient Boosting: R²=0.95 (misleadingly high)
- Random Forest: R²=0.94
- SVM: R²=0.82

**Current approach is more honest:** Predicting final accuracy from only 15 early trials is the correct research goal.

## API Endpoints

### GET /api/health
Health check for prediction server.

### POST /api/predict
Predict BCI performance from features.

```json
{
  "features": [45, 64, 0.51, 0.49, 0.68, 0.05, 0.72, 0.64, 0.67, 0.69]
}
```

### GET /api/simulate_subject
Get random PhysioNet subject with predictions.

### GET /api/subjects
Get all subjects with actual vs predicted accuracies.

## Technologies

- **MetaBCI**: BCI decoder framework
- **MNE**: EEG data processing
- **scikit-learn**: Machine learning models
- **Flask**: REST API server
- **Vanilla JS**: Interactive web demo

## References

- PhysioNet Motor Imagery Database
- MetaBCI Framework
- Common Spatial Patterns (CSP)
- Gradient Boosting Machines

## License

MIT License

## Contributors

- Shaheer Khan (shk021@ucsd.edu)
