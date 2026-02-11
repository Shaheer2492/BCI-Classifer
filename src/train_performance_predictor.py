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
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
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

    def load_early_trial_features(self, features_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Load early trial features and ground truth labels.
        
        This method loads features extracted from ONLY the first N trials
        (Phase 2) and matches them with final accuracies (Phase 1) to enable
        true early prediction of BCI performance.

        Parameters
        ----------
        features_path : str
            Path to early trial features JSON file
        labels_path : str
            Path to ground truth labels JSON file

        Returns
        -------
        X : ndarray
            Feature matrix from early trials (n_subjects, n_features)
        y : ndarray
            Target accuracies from full dataset (n_subjects,)
        subject_ids : list
            List of subject IDs
        """
        print(f"Loading early trial features from {features_path}...")
        print(f"Loading ground truth labels from {labels_path}...")

        # Load early trial features
        with open(features_path, 'r') as f:
            features_data = json.load(f)

        # Load ground truth labels (targets)
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)

        # Create lookup for ground truth accuracies
        gt_lookup = {}
        for subject in labels_data['subjects']:
            if subject['success']:
                gt_lookup[subject['subject_id']] = subject['accuracy']

        # Match features with ground truth labels
        X_list = []
        y_list = []
        subject_ids = []

        for subject in features_data['subjects']:
            if subject['success'] and subject['subject_id'] in gt_lookup:
                # Features from early trials (Phase 2)
                features = np.array(subject['features'])
                X_list.append(features)
                
                # Target accuracy from full dataset (Phase 1)
                y_list.append(gt_lookup[subject['subject_id']])
                subject_ids.append(subject['subject_id'])

        X = np.array(X_list)
        y = np.array(y_list)

        print(f"\n{'='*60}")
        print(f"Data Loading Summary")
        print(f"{'='*60}")
        print(f"Matched subjects: {len(subject_ids)}")
        print(f"Feature shape: {X.shape}")
        print(f"Features per subject: {X.shape[1]}")
        print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"Target mean: {y.mean():.3f} ± {y.std():.3f}")
        
        # Display feature names if available
        if 'feature_names' in features_data['metadata']:
            feature_names = features_data['metadata']['feature_names']
            print(f"\nFeature categories:")
            print(f"  - Band Power: {sum(1 for f in feature_names if 'band_power' in f)} features")
            print(f"  - CSP Patterns: {sum(1 for f in feature_names if 'csp' in f)} features")
            print(f"  - ERD/ERS (Phase 2): {sum(1 for f in feature_names if 'erdrs' in f)} features")
            print(f"  - Variability: {sum(1 for f in feature_names if 'variability' in f)} features")
            print(f"  - SNR: {sum(1 for f in feature_names if 'snr' in f)} features")
            print(f"  - Daniel's features: {sum(1 for f in feature_names if f in ['rpl', 'smr_strength'] or 'erd_' in f or 'ers_' in f)} features")
            print(f"  - Andrew's features: {sum(1 for f in feature_names if 'entropy' in f or 'lzc' in f or 'tar' in f)} features")
        print(f"{'='*60}\n")

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
                n_estimators=200,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=self.random_state
            ),
            'SVM': SVR(
                kernel='rbf',
                C=0.1,
                gamma='scale',
                epsilon=0.1
            ),
            'Ridge Regression': Ridge(
                alpha=10.0
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

        # Get out-of-fold predictions for honest correlation estimates
        y_pred_cv = cross_val_predict(model, X, y, cv=cv)
        pearson_r, pearson_p = pearsonr(y, y_pred_cv)
        spearman_r, spearman_p = spearmanr(y, y_pred_cv)

        # Train on full data for final model and in-sample predictions
        model.fit(X, y)
        y_pred = model.predict(X)

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

        # Feature selection to avoid overfitting with many features and few samples
        n_features_to_select = min(20, X_scaled.shape[1])
        selector = SelectKBest(score_func=f_regression, k=n_features_to_select)
        X_scaled = selector.fit_transform(X_scaled, y)
        self.scalers['feature_selector'] = selector

        selected_mask = selector.get_support()
        print(f"Feature selection: {n_features_to_select}/{X.shape[1]} features retained")

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

        # Save feature selector
        selector_path = output_path / 'feature_selector.pkl'
        joblib.dump(self.scalers['feature_selector'], selector_path)
        print(f"Saved feature selector to {selector_path}")

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
    features_path = 'src/results/early_trial_features.json'
    ground_truth_path = 'src/results/ground_truth_labels.json'
    output_dir = 'src/results/models'
    results_path = 'src/results/model_evaluation.json'

    # Check if required files exist
    if not os.path.exists(features_path):
        print(f"Error: Early trial features not found at {features_path}")
        print("Please run extract_early_trial_features.py first (Phase 2).")
        return
    
    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        print("Please run generate_ground_truth_labels.py first (Phase 1).")
        return

    # Initialize predictor
    predictor = BCIPerformancePredictor(n_folds=5, random_state=42)

    # Load data (features from early trials, targets from full dataset)
    X, y, subject_ids = predictor.load_early_trial_features(features_path, ground_truth_path)

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
    print("\nNote: This uses merged features from Phase 2 early trials + EEG-Project")
    print("features. Pearson r is now computed out-of-fold (cross-validated) for")
    print("honest evaluation. Feature selection retains the top features to avoid")
    print("overfitting with the expanded feature set.")


if __name__ == '__main__':
    main()
