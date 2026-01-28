"""
Phase 3: Train Performance Prediction Models
---------------------------------------------
Train machine learning models to predict BCI decoder performance from early-trial features.

Implements multiple models:
- Random Forest
- Gradient Boosting (XGBoost)
- Support Vector Machine (SVM)

Uses features extracted from early trials to predict final decoder accuracy.
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import joblib
from tqdm import tqdm

warnings.filterwarnings('ignore')


class BCIPerformancePredictor:
    """Train ML models to predict BCI decoder performance from early-trial features."""

    def __init__(self, n_folds=5, random_state=42):
        """
        Initialize the performance predictor.

        Parameters
        ----------
        n_folds : int
            Number of folds for cross-validation (default: 5)
        random_state : int
            Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.results = {}

    def extract_early_trial_features(self, subject_data: Dict) -> np.ndarray:
        """
        Extract features from subject data for prediction.

        For now, uses basic statistical features from the ground truth data.
        In a full implementation, this would extract features from early trials.

        Parameters
        ----------
        subject_data : dict
            Subject data from ground truth labels

        Returns
        -------
        features : ndarray
            Feature vector for the subject
        """
        features = []

        # Basic features from available data
        features.append(subject_data['n_trials'])
        features.append(subject_data['n_channels'])

        # Class distribution features
        class_dist = subject_data['class_distribution']
        total_trials = sum(class_dist.values())
        features.append(class_dist.get('0', 0) / total_trials)  # Class 0 ratio
        features.append(class_dist.get('1', 0) / total_trials)  # Class 1 ratio

        # Fold accuracy statistics (simulating early trial variability)
        fold_accs = subject_data['fold_accuracies']
        features.append(np.mean(fold_accs[:2]))  # Mean of first 2 folds (early trials)
        features.append(np.std(fold_accs[:2]))   # Std of first 2 folds
        features.append(np.max(fold_accs[:2]))   # Max of first 2 folds
        features.append(np.min(fold_accs[:2]))   # Min of first 2 folds

        # Class accuracy features
        class_accs = subject_data['class_accuracies']
        features.append(class_accs.get('0', 0.5))
        features.append(class_accs.get('1', 0.5))

        return np.array(features)

    def load_ground_truth_data(self, json_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Load ground truth data and extract features.

        Parameters
        ----------
        json_path : str
            Path to ground truth labels JSON file

        Returns
        -------
        X : ndarray
            Feature matrix (n_subjects, n_features)
        y : ndarray
            Target accuracies (n_subjects,)
        subject_ids : list
            List of subject IDs
        """
        print(f"Loading ground truth data from {json_path}...")

        with open(json_path, 'r') as f:
            data = json.load(f)

        X_list = []
        y_list = []
        subject_ids = []

        for subject in data['subjects']:
            if subject['success']:
                features = self.extract_early_trial_features(subject)
                X_list.append(features)
                y_list.append(subject['accuracy'])
                subject_ids.append(subject['subject_id'])

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"Loaded {len(subject_ids)} successful subjects")
        print(f"Feature shape: {X.shape}")
        print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")

        return X, y, subject_ids

    def build_models(self) -> Dict:
        """
        Build machine learning models for performance prediction.

        Returns
        -------
        models : dict
            Dictionary of model name -> model instance
        """
        models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'SVM': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1
            )
        }

        return models

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray,
                      model_name: str) -> Dict:
        """
        Evaluate a model using cross-validation.

        Parameters
        ----------
        model : sklearn estimator
            Model to evaluate
        X : ndarray
            Feature matrix
        y : ndarray
            Target values
        model_name : str
            Name of the model

        Returns
        -------
        results : dict
            Evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")

        # Cross-validation
        cv = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Evaluate with multiple metrics
        mse_scores = -cross_val_score(model, X, y, cv=cv,
                                      scoring='neg_mean_squared_error', n_jobs=-1)
        mae_scores = -cross_val_score(model, X, y, cv=cv,
                                      scoring='neg_mean_absolute_error', n_jobs=-1)
        r2_scores = cross_val_score(model, X, y, cv=cv,
                                    scoring='r2', n_jobs=-1)

        # Train on full data for predictions
        model.fit(X, y)
        y_pred = model.predict(X)

        # Calculate correlation
        pearson_r, pearson_p = pearsonr(y, y_pred)
        spearman_r, spearman_p = spearmanr(y, y_pred)

        results = {
            'model_name': model_name,
            'mse_mean': float(np.mean(mse_scores)),
            'mse_std': float(np.std(mse_scores)),
            'mae_mean': float(np.mean(mae_scores)),
            'mae_std': float(np.std(mae_scores)),
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'rmse_mean': float(np.sqrt(np.mean(mse_scores))),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'predictions': y_pred.tolist(),
            'actual': y.tolist()
        }

        print(f"  RMSE: {results['rmse_mean']:.4f} (±{np.sqrt(results['mse_std']):.4f})")
        print(f"  MAE:  {results['mae_mean']:.4f} (±{results['mae_std']:.4f})")
        print(f"  R²:   {results['r2_mean']:.4f} (±{results['r2_std']:.4f})")
        print(f"  Pearson r: {results['pearson_r']:.4f} (p={results['pearson_p']:.4f})")

        return results

    def train_all_models(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train and evaluate all models.

        Parameters
        ----------
        X : ndarray
            Feature matrix
        y : ndarray
            Target values

        Returns
        -------
        results : dict
            Results for all models
        """
        print("\n" + "="*60)
        print("Training Performance Prediction Models")
        print("="*60)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['feature_scaler'] = scaler

        # Build models
        models = self.build_models()

        # Train and evaluate each model
        results = {}
        for model_name, model in models.items():
            model_results = self.evaluate_model(model, X_scaled, y, model_name)
            results[model_name] = model_results
            self.models[model_name] = model

        return results

    def select_best_model(self, results: Dict) -> str:
        """
        Select the best model based on R² score.

        Parameters
        ----------
        results : dict
            Results from all models

        Returns
        -------
        best_model_name : str
            Name of the best performing model
        """
        best_model = max(results.items(), key=lambda x: x[1]['r2_mean'])
        return best_model[0]

    def save_models(self, output_dir: str):
        """
        Save trained models and scalers.

        Parameters
        ----------
        output_dir : str
            Directory to save models
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_filename = model_name.lower().replace(' ', '_') + '_model.pkl'
            model_path = output_path / model_filename
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")

        # Save scaler
        scaler_path = output_path / 'feature_scaler.pkl'
        joblib.dump(self.scalers['feature_scaler'], scaler_path)
        print(f"Saved feature scaler to {scaler_path}")

    def save_results(self, results: Dict, output_path: str, subject_ids: List[int]):
        """
        Save evaluation results to JSON.

        Parameters
        ----------
        results : dict
            Evaluation results
        output_path : str
            Path to save results
        subject_ids : list
            List of subject IDs
        """
        best_model = self.select_best_model(results)

        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'n_subjects': len(subject_ids),
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'best_model': best_model
            },
            'models': results,
            'subject_ids': subject_ids
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to {output_path}")
        print(f"Best model: {best_model} (R² = {results[best_model]['r2_mean']:.4f})")

    def generate_comparison_report(self, results: Dict) -> pd.DataFrame:
        """
        Generate a comparison report of all models.

        Parameters
        ----------
        results : dict
            Results from all models

        Returns
        -------
        df : DataFrame
            Comparison table
        """
        comparison_data = []

        for model_name, model_results in results.items():
            comparison_data.append({
                'Model': model_name,
                'RMSE': f"{model_results['rmse_mean']:.4f} ± {np.sqrt(model_results['mse_std']):.4f}",
                'MAE': f"{model_results['mae_mean']:.4f} ± {model_results['mae_std']:.4f}",
                'R²': f"{model_results['r2_mean']:.4f} ± {model_results['r2_std']:.4f}",
                'Pearson r': f"{model_results['pearson_r']:.4f}",
                'Spearman r': f"{model_results['spearman_r']:.4f}"
            })

        df = pd.DataFrame(comparison_data)
        return df


def main():
    """Main execution function."""
    # Paths
    ground_truth_path = 'src/results/ground_truth_labels.json'
    output_dir = 'src/results/models'
    results_path = 'src/results/model_evaluation.json'

    # Check if ground truth exists
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        print("Please run generate_ground_truth_labels.py first.")
        return

    # Initialize predictor
    predictor = BCIPerformancePredictor(n_folds=5, random_state=42)

    # Load data
    X, y, subject_ids = predictor.load_ground_truth_data(ground_truth_path)

    # Train all models
    results = predictor.train_all_models(X, y)

    # Generate comparison report
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    comparison_df = predictor.generate_comparison_report(results)
    print(comparison_df.to_string(index=False))

    # Save models and results
    predictor.save_models(output_dir)
    predictor.save_results(results, results_path, subject_ids)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
