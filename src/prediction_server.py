"""
BCI Performance Prediction Server
----------------------------------
Flask server that provides real-time BCI performance predictions
for the web demo using pre-computed merged features and the best trained model.
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import json
import random
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for web demo

# Paths
MODEL_DIR = Path('src/results/models')
GROUND_TRUTH_PATH = Path('src/results/ground_truth_labels.json')
FEATURES_PATH = Path('src/results/early_trial_features.json')
EVAL_PATH = Path('src/results/model_evaluation.json')

model = None
scaler = None
selector = None
ground_truth_data = None
features_data = None
features_lookup = None  # subject_id -> feature vector


def load_models():
    """Load the trained model, scaler, feature selector, and data."""
    global model, scaler, selector, ground_truth_data, features_data, features_lookup

    try:
        # Determine best model from evaluation results
        best_model_name = 'random_forest'
        if EVAL_PATH.exists():
            with open(EVAL_PATH, 'r') as f:
                eval_data = json.load(f)
            best_model_name = eval_data['metadata']['best_model'].lower().replace(' ', '_')
        model_path = MODEL_DIR / f'{best_model_name}_model.pkl'

        scaler_path = MODEL_DIR / 'feature_scaler.pkl'
        selector_path = MODEL_DIR / 'feature_selector.pkl'

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print(f"Loaded {best_model_name} model from {model_path}")
        print(f"Loaded feature scaler from {scaler_path}")

        if selector_path.exists():
            selector = joblib.load(selector_path)
            print(f"Loaded feature selector from {selector_path}")

        # Load ground truth data
        with open(GROUND_TRUTH_PATH, 'r') as f:
            raw_gt = json.load(f)
            
        # Convert subjects list to dict for fast lookup
        ground_truth_data = {
            'metadata': raw_gt.get('metadata', {}),
            'summary': raw_gt.get('summary', {}),
            'subjects': {}
        }
        
        for subj in raw_gt['subjects']:
            ground_truth_data['subjects'][str(subj['subject_id'])] = subj
            
        print(f"Loaded ground truth data for {len(ground_truth_data['subjects'])} subjects")

        # Load pre-computed merged features
        with open(FEATURES_PATH, 'r') as f:
            features_data = json.load(f)

        # Build lookup: subject_id -> feature vector
        features_lookup = {}
        for subject in features_data['subjects']:
            if subject['success']:
                features_lookup[subject['subject_id']] = np.array(subject['features'])

        print(f"Loaded {len(features_lookup)} subject feature vectors "
              f"({features_data['metadata']['n_features']} features each)")

        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False


def predict_for_subject(subject_id):
    """Look up pre-computed features and predict for a subject."""
    if subject_id not in features_lookup:
        return None

    features = features_lookup[subject_id].reshape(1, -1)
    features_scaled = scaler.transform(features)
    if selector is not None:
        features_scaled = selector.transform(features_scaled)
    return float(model.predict(features_scaled)[0])


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'n_subjects': len(features_lookup) if features_lookup else 0,
        'n_features': features_data['metadata']['n_features'] if features_data else 0
    })


@app.route('/api/predict', methods=['POST'])
def predict_performance():
    """
    Predict BCI performance from a raw feature vector.

    Expected JSON body:
    {
        "features": [val1, val2, ...]  // merged feature vector
    }
    """
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)

        features_scaled = scaler.transform(features)
        if selector is not None:
            features_scaled = selector.transform(features_scaled)
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            'success': True,
            'predicted_accuracy': float(prediction),
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/simulate_subject', methods=['GET'])
def simulate_subject():
    """Returns data for a random subject (simulate new session)"""
    if not ground_truth_data:
        return jsonify({'error': 'Ground truth data not loaded'}), 503
        
    # Pick random subject
    subject_ids = list(ground_truth_data['subjects'].keys())
    random_id = random.choice(subject_ids) # random.choice on keys list returns a key
    
    # keys are strings, get_subject_data handles int or str conversion but let's pass int
    return get_subject_data(int(random_id))


@app.route('/api/subject/<int:subject_id>', methods=['GET'])
def get_subject_data(subject_id):
    """Returns data for a specific subject"""
    if not ground_truth_data:
        return jsonify({'error': 'Ground truth data not loaded'}), 503
        
    str_id = str(subject_id)
    if str_id not in ground_truth_data['subjects']:
        return jsonify({'error': f'Subject {subject_id} not found'}), 404
        
    # Get ground truth
    subject_gt = ground_truth_data['subjects'][str_id]
    actual_acc = subject_gt['accuracy']
    
    # Get features and predict
    features = features_lookup.get(int(str_id), [])
    
    if len(features) == 0:
         # Try looking up by string if int failed
         features = features_lookup.get(str_id, [])

    predicted_acc = 0.0
    
    # Convert to list for JSON serialization if it's numpy array
    features_list = []
    if hasattr(features, 'tolist'):
        features_list = features.tolist()
    else:
        features_list = list(features)
    
    if len(features) > 0 and model is not None:
        try:
            # Reshape for prediction
            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)
            X_selected = X_scaled
            if selector is not None:
                X_selected = selector.transform(X_scaled)
            predicted_acc = float(model.predict(X_selected)[0])
        except Exception as e:
            print(f"Prediction error for subject {subject_id}: {e}")
            predicted_acc = 0.5 # Fallback
            
    return jsonify({
        'success': True,
        'subject_id': int(subject_id),
        'actual_accuracy': actual_acc,
        'predicted_accuracy': predicted_acc,
        'fold_accuracies': subject_gt.get('fold_accuracies', []),
        'accuracy_std': subject_gt.get('accuracy_std', 0.0),
        'n_trials': subject_gt.get('n_trials', 0),
        'n_channels': 64,
        'class_distribution': subject_gt.get('class_distribution', {}),
        'features': features_list
    })


@app.route('/api/subjects', methods=['GET'])
def get_all_subjects():
    """Get all subjects with their actual and predicted accuracies."""
    try:
        subjects_data = []

        for subject in ground_truth_data['subjects'].values():
            sid = subject['subject_id']
            if subject['success'] and sid in features_lookup:
                predicted_accuracy = predict_for_subject(sid)
                subjects_data.append({
                    'subject_id': sid,
                    'actual_accuracy': subject['accuracy'],
                    'predicted_accuracy': predicted_accuracy,
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


@app.route('/api/population_stats', methods=['GET'])
def get_population_stats():
    """Return population accuracy distribution for benchmarking."""
    if not ground_truth_data:
        return jsonify({'error': 'Ground truth data not loaded'}), 503
    accuracies = sorted([
        s['accuracy'] for s in ground_truth_data['subjects'].values()
        if s.get('success', False)
    ])
    return jsonify({
        'success': True,
        'accuracies': accuracies,
        'n_subjects': len(accuracies),
        'mean': float(np.mean(accuracies)),
        'std': float(np.std(accuracies)),
        'min': float(np.min(accuracies)),
        'max': float(np.max(accuracies))
    })


if __name__ == '__main__':
    print("=" * 60)
    print("BCI Performance Prediction Server")
    print("=" * 60)

    if load_models():
        print("\nStarting server on http://localhost:5001")
        print("\nAvailable endpoints:")
        print("  GET  /api/health           - Health check")
        print("  POST /api/predict          - Predict from features")
        print("  GET  /api/simulate_subject - Get random subject simulation")
        print("  GET  /api/subjects         - Get all subjects with predictions")
        print("\n" + "=" * 60)

        app.run(host='0.0.0.0', port=5001, debug=True)
    else:
        print("\nFailed to load models. Please run train_performance_predictor.py first.")
