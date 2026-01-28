# Quick Start Guide

Get the BCI Classifier system up and running in minutes!

## Prerequisites

- Python 3.9+
- pip package manager
- ~3GB disk space (for PhysioNet dataset)

## Installation

```bash
# Clone the repository
git clone https://github.com/Shaheer2492/BCI-Classifer.git
cd BCI-Classifer

# Install dependencies
pip install -r requirements.txt
```

## Step-by-Step Setup

### Step 1: Generate Ground Truth Labels (Optional - Already Done!)

The ground truth labels are already included in `src/results/ground_truth_labels.json`.

If you want to regenerate them:

```bash
python src/generate_ground_truth_labels.py
```

⏱️ **Time**: ~1-2 hours for all 109 subjects

### Step 2: Train ML Models (Optional - Already Done!)

Pre-trained models are included in `src/results/models/`.

To retrain:

```bash
python src/train_performance_predictor.py
```

⏱️ **Time**: ~10-30 seconds

**Output:**
```
Model Comparison
            Model            RMSE             MAE              R²
    Random Forest 0.0370 ± 0.0336 0.0258 ± 0.0099 0.9439 ± 0.0323
Gradient Boosting 0.0332 ± 0.0214 0.0246 ± 0.0055 0.9504 ± 0.0153
              SVM 0.0653 ± 0.0506 0.0473 ± 0.0120 0.8203 ± 0.0723

Best model: Gradient Boosting (R² = 0.9504)
```

### Step 3: Start the Prediction Server

```bash
python src/prediction_server.py
```

You should see:

```
============================================================
BCI Performance Prediction Server
============================================================
✅ Loaded Gradient Boosting model
✅ Loaded feature scaler
✅ Loaded ground truth data with 99 subjects

🚀 Starting server on http://localhost:5000

Available endpoints:
  GET  /api/health           - Health check
  POST /api/predict          - Predict from features
  GET  /api/simulate_subject - Get random subject simulation
  GET  /api/subjects         - Get all subjects with predictions
============================================================
```

### Step 4: View the Web Demo

**Option A: Simple File Open**
```bash
open website/ml-demo.html
```

**Option B: Local Web Server (Recommended)**
```bash
cd website
python -m http.server 8000
```

Then visit: http://localhost:8000/ml-demo.html

## Using the ML Demo

1. **Load a Subject**: Click "Load New Subject" to get a random PhysioNet subject
2. **Enable ML Predictions**: Toggle "Use ML Predictions" switch
3. **Start Session**: Click "Start BCI Session" to begin simulation
4. **Watch Real-time Predictions**: See brain activity and ML performance predictions

## Testing the API

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Get Random Subject
```bash
curl http://localhost:5000/api/simulate_subject
```

### Manual Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [45, 64, 0.51, 0.49, 0.68, 0.05, 0.72, 0.64, 0.67, 0.69]}'
```

## What's Included

✅ **Ground Truth Labels**: 99 successful subjects from PhysioNet MI  
✅ **Trained Models**: Random Forest, Gradient Boosting, SVM  
✅ **Prediction Server**: Flask API for real-time predictions  
✅ **Web Demo**: Interactive BCI visualization with ML  
✅ **Documentation**: Complete guides for all phases  

## Troubleshooting

### Server Won't Start
- Check if port 5000 is available
- Ensure models are trained: `python src/train_performance_predictor.py`

### Web Demo Shows "Server Offline"
- Make sure prediction server is running: `python src/prediction_server.py`
- Check console for CORS errors

### Import Errors
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`
- Check Python version: `python --version` (need 3.9+)

## Next Steps

- Explore `PHASE1_README.md` for ground truth generation details
- Read `PHASE3_README.md` for ML model information
- Check `README.md` for complete project overview
- View results at `website/results.html`

## Support

For issues or questions:
- GitHub Issues: https://github.com/Shaheer2492/BCI-Classifer/issues
- Email: shk021@ucsd.edu

---

**Enjoy exploring BCI performance prediction! 🧠🤖**
