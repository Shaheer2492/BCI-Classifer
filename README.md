# BCI-Classifier
Brain Computer Interface performance prediction using MetaBCI decoder and machine learning.

## Overview

This project predicts BCI decoder performance from early-trial motor imagery data using machine learning models trained on PhysioNet MI dataset.

### Key Features

- **MetaBCI CSP+LDA Decoder**: Ground truth generation using MetaBCI's CSP implementation
- **ML Performance Prediction**: Random Forest, Gradient Boosting (best: R²=0.95), and SVM models
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

### 3. Train ML Models (Phase 3)

```bash
python src/train_performance_predictor.py
```

Trains Random Forest, Gradient Boosting, and SVM models (~10-30 seconds).

### 4. Start Prediction Server

```bash
python src/prediction_server.py
```

Runs Flask server on `http://localhost:5000` for ML predictions.

### 5. View Web Demo

Open `website/ml-demo.html` in your browser or run:

```bash
cd website
python -m http.server 8000
# Visit http://localhost:8000/ml-demo.html
```

## Project Structure

```
BCI-Classifer/
├── src/
│   ├── generate_ground_truth_labels.py  # Phase 1: Ground truth generation
│   ├── train_performance_predictor.py   # Phase 3: ML model training
│   ├── prediction_server.py             # Flask API server
│   └── results/
│       ├── ground_truth_labels.json     # Subject accuracies
│       ├── model_evaluation.json        # Model performance metrics
│       └── models/                      # Trained ML models
├── website/
│   ├── index.html                       # Main landing page
│   ├── ml-demo.html                     # ML-enhanced BCI demo
│   ├── results.html                     # Ground truth results
│   └── js/
│       ├── bci-demo.js                  # Base demo
│       └── bci-demo-ml.js               # ML-enhanced demo
├── PHASE1_README.md                     # Ground truth generation docs
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

### Phase 3: ML Model Training
- Extract features from ground truth data
- Train 3 models: Random Forest, Gradient Boosting, SVM
- Evaluate with RMSE, MAE, R², Pearson correlation
- Select best model (Gradient Boosting: R²=0.95)
- Output: Trained models + evaluation metrics

### Phase 4: Real-time Prediction
- Flask API serves ML predictions
- Web demo visualizes brain activity
- Real-time performance prediction
- PhysioNet subject simulation

## Model Performance

| Model | RMSE | MAE | R² | Pearson r |
|-------|------|-----|----|-----------| 
| **Gradient Boosting** | **0.0332** | **0.0246** | **0.9504** | **0.9999** |
| Random Forest | 0.0370 | 0.0258 | 0.9439 | 0.9957 |
| SVM | 0.0653 | 0.0473 | 0.8203 | 0.9618 |

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
