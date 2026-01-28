"""
BCI Performance Prediction Server
----------------------------------
Flask server that provides real-time BCI performance predictions
for the web demo using the trained Gradient Boosting model.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for web demo

# Load trained model and scaler
MODEL_PATH = Path('src/results/models/gradient_boosting_model.pkl')
SCALER_PATH = Path('src/results/models/feature_scaler.pkl')
GROUND_TRUTH_PATH = Path('src/results/ground_truth_labels.json')

model = None
scaler = None
ground_truth_data = None


def load_models():
    """Load the trained model and scaler."""
    global model, scaler, ground_truth_data

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"✅ Loaded Gradient Boosting model from {MODEL_PATH}")
        print(f"✅ Loaded feature scaler from {SCALER_PATH}")

        # Load ground truth data for simulation
        with open(GROUND_TRUTH_PATH, 'r') as f:
            ground_truth_data = json.load(f)
        print(f"✅ Loaded ground truth data with {len(ground_truth_data['subjects'])} subjects")

        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })


@app.route('/api/predict', methods=['POST'])
def predict_performance():
    """
    Predict BCI performance from features.

    Expected JSON body:
    {
        "features": [n_trials, n_channels, class0_ratio, class1_ratio,
                     mean_acc_early, std_acc_early, max_acc_early, min_acc_early,
                     class0_acc, class1_acc]
    }
    """
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            'success': True,
            'predicted_accuracy': float(prediction),
            'confidence': 0.95  # Based on model R² score
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/simulate_subject', methods=['GET'])
def simulate_subject():
    """
    Get simulated data for a random subject from PhysioNet.

    Returns subject data and predicted performance.
    """
    try:
        # Get a random successful subject
        successful_subjects = [s for s in ground_truth_data['subjects'] if s['success']]
        subject = np.random.choice(successful_subjects)

        # Extract features (same as in training)
        features = extract_features_from_subject(subject)

        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        predicted_accuracy = model.predict(features_scaled)[0]

        return jsonify({
            'success': True,
            'subject_id': subject['subject_id'],
            'actual_accuracy': subject['accuracy'],
            'predicted_accuracy': float(predicted_accuracy),
            'n_trials': subject['n_trials'],
            'n_channels': subject['n_channels'],
            'class_distribution': subject['class_distribution'],
            'features': features.tolist()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/subjects', methods=['GET'])
def get_all_subjects():
    """Get all subjects with their actual and predicted accuracies."""
    try:
        subjects_data = []

        for subject in ground_truth_data['subjects']:
            if subject['success']:
                # Extract features
                features = extract_features_from_subject(subject)

                # Scale and predict
                features_scaled = scaler.transform(features.reshape(1, -1))
                predicted_accuracy = model.predict(features_scaled)[0]

                subjects_data.append({
                    'subject_id': subject['subject_id'],
                    'actual_accuracy': subject['accuracy'],
                    'predicted_accuracy': float(predicted_accuracy),
                    'error': abs(subject['accuracy'] - predicted_accuracy)
                })

        return jsonify({
            'success': True,
            'subjects': subjects_data,
            'count': len(subjects_data)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def extract_features_from_subject(subject_data):
    """Extract features from subject data (same as training)."""
    features = []

    # Basic features
    features.append(subject_data['n_trials'])
    features.append(subject_data['n_channels'])

    # Class distribution
    class_dist = subject_data['class_distribution']
    total_trials = sum(class_dist.values())
    features.append(class_dist.get('0', 0) / total_trials)
    features.append(class_dist.get('1', 0) / total_trials)

    # Fold accuracy statistics
    fold_accs = subject_data['fold_accuracies']
    features.append(np.mean(fold_accs[:2]))
    features.append(np.std(fold_accs[:2]))
    features.append(np.max(fold_accs[:2]))
    features.append(np.min(fold_accs[:2]))

    # Class accuracies
    class_accs = subject_data['class_accuracies']
    features.append(class_accs.get('0', 0.5))
    features.append(class_accs.get('1', 0.5))

    return np.array(features)


if __name__ == '__main__':
    print("="*60)
    print("BCI Performance Prediction Server")
    print("="*60)

    if load_models():
        print("\n🚀 Starting server on http://localhost:5000")
        print("\nAvailable endpoints:")
        print("  GET  /api/health           - Health check")
        print("  POST /api/predict          - Predict from features")
        print("  GET  /api/simulate_subject - Get random subject simulation")
        print("  GET  /api/subjects         - Get all subjects with predictions")
        print("\n" + "="*60)

        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n❌ Failed to load models. Please run train_performance_predictor.py first.")
