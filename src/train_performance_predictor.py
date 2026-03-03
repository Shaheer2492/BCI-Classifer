"""
Train Performance Prediction Models
---------------------------------------------
Train machine learning models to predict BCI decoder performance from early-trial features.

Implements multiple models:
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Ridge Regression

Feature selection uses Spearman correlation with permutation tests and
Benjamini-Hochberg FDR correction (methodology from EDA notebook analysis).
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, confusion_matrix, classification_report,
)
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
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

    def select_features_spearman_fdr(
        self, X: np.ndarray, y: np.ndarray,
        feature_names: List[str] = None, alpha: float = 0.05,
        n_permutations: int = 5000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select features using Spearman correlation with permutation-based
        p-values and Benjamini-Hochberg FDR correction.

        Methodology adapted from the EDA notebook analysis which demonstrated
        that this approach produces more statistically rigorous feature sets
        than univariate f_regression.

        Parameters
        ----------
        X : ndarray
            Feature matrix (n_subjects, n_features)
        y : ndarray
            Target values (n_subjects,)
        feature_names : list, optional
            Names of features for reporting
        alpha : float
            FDR significance threshold (default: 0.05)
        n_permutations : int
            Number of permutations for p-value estimation

        Returns
        -------
        selected_mask : ndarray of bool
            Boolean mask of selected features
        selection_results : ndarray
            Array of (spearman_r, p_value, fdr_reject) per feature
        """
        n_features = X.shape[1]
        correlations = np.zeros(n_features)
        p_values = np.zeros(n_features)

        rng = np.random.RandomState(self.random_state)

        for i in range(n_features):
            rho, _ = spearmanr(X[:, i], y)
            correlations[i] = rho

            null_distribution = np.zeros(n_permutations)
            for p in range(n_permutations):
                y_perm = rng.permutation(y)
                null_distribution[p], _ = spearmanr(X[:, i], y_perm)

            p_values[i] = np.mean(np.abs(null_distribution) >= np.abs(rho))

        p_values = np.clip(p_values, 1.0 / n_permutations, 1.0)

        reject, pvals_corrected, _, _ = multipletests(
            p_values, alpha=alpha, method='fdr_bh'
        )

        min_abs_corr = 0.1
        selected_mask = reject & (np.abs(correlations) >= min_abs_corr)

        if not np.any(selected_mask):
            top_k = min(10, n_features)
            top_indices = np.argsort(np.abs(correlations))[::-1][:top_k]
            selected_mask[top_indices] = True
            print(f"  WARNING: No features passed FDR at alpha={alpha}. "
                  f"Falling back to top {top_k} by |Spearman r|.")

        print(f"\n  Spearman + FDR Feature Selection (alpha={alpha}):")
        print(f"  {'Feature':<40} {'rho':>8} {'p-raw':>10} {'p-FDR':>10} {'Keep':>6}")
        print(f"  {'-'*74}")

        for i in range(n_features):
            name = feature_names[i] if feature_names else f"feature_{i}"
            keep = "YES" if selected_mask[i] else ""
            print(f"  {name:<40} {correlations[i]:>8.4f} "
                  f"{p_values[i]:>10.4f} {pvals_corrected[i]:>10.4f} {keep:>6}")

        n_selected = int(np.sum(selected_mask))
        print(f"\n  Selected {n_selected}/{n_features} features")

        return selected_mask, np.column_stack([
            correlations, p_values, pvals_corrected, selected_mask.astype(float)
        ])

    def train_all_models(self, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str] = None) -> Dict:
        """
        Train and evaluate all models.

        Parameters
        ----------
        X : ndarray
            Feature matrix
        y : ndarray
            Target values
        feature_names : list, optional
            Feature names for reporting

        Returns
        -------
        results : dict
            Results for all models
        """
        print("\n" + "="*60)
        print("Training Performance Prediction Models")
        print("="*60)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['feature_scaler'] = scaler

        selected_mask, selection_results = self.select_features_spearman_fdr(
            X_scaled, y, feature_names=feature_names
        )
        X_selected = X_scaled[:, selected_mask]
        self.feature_mask = selected_mask

        n_selected = int(np.sum(selected_mask))
        print(f"Feature selection: {n_selected}/{X.shape[1]} features retained "
              f"(Spearman + permutation + FDR)")

        if feature_names:
            kept = [f for f, m in zip(feature_names, selected_mask) if m]
            self.selected_feature_names = kept
            print(f"Selected features: {kept}")

        models = self.build_models()

        results = {}
        for model_name, model in models.items():
            model_results = self.evaluate_model(model, X_selected, y, model_name)
            results[model_name] = model_results
            self.models[model_name] = model

        return results

    def train_classifier(
        self,
        X_selected: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.65,
    ) -> Dict:
        """
        Train a RandomForestClassifier to predict high vs low BCI performers.

        The classifier is evaluated with Leave-One-Out Cross-Validation (LOOCV)
        to match the methodology used in the EDA notebook.

        Parameters
        ----------
        X_selected : ndarray
            Scaled and feature-selected matrix (n_subjects, n_selected_features)
        y : ndarray
            Continuous target accuracies (n_subjects,)
        threshold : float
            Decoder accuracy threshold separating high (>=) from low (<) performers

        Returns
        -------
        results : dict
            LOOCV evaluation metrics and per-subject predictions
        """
        y_class = (y >= threshold).astype(int)
        n_high = int(y_class.sum())
        n_low = int(len(y_class) - n_high)

        print("\n" + "=" * 60)
        print("Training RF Classifier (High / Low Performer)")
        print("=" * 60)
        print(f"  Threshold: {threshold}")
        print(f"  Class distribution: HIGH={n_high}, LOW={n_low}")
        print(f"  class_weight: {{0: 2, 1: 1}} (penalize missing low performers 2x)")

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            max_features=10,
            class_weight={0: 2, 1: 1},
            random_state=self.random_state,
            n_jobs=-1,
        )

        loo = LeaveOneOut()
        y_pred_loo = np.zeros(len(y_class), dtype=int)
        y_proba_loo = np.zeros(len(y_class))

        for train_idx, test_idx in loo.split(X_selected):
            clf.fit(X_selected[train_idx], y_class[train_idx])
            y_pred_loo[test_idx] = clf.predict(X_selected[test_idx])
            y_proba_loo[test_idx] = clf.predict_proba(X_selected[test_idx])[:, 1]

        acc = accuracy_score(y_class, y_pred_loo)
        cm = confusion_matrix(y_class, y_pred_loo)
        report = classification_report(y_class, y_pred_loo, target_names=["LOW", "HIGH"], output_dict=True)

        print(f"\n  LOOCV Accuracy: {acc:.4f}")
        print(f"  Confusion Matrix (rows=actual, cols=predicted):")
        print(f"    {cm}")
        print(f"  Classification Report:")
        print(f"    LOW  — precision={report['LOW']['precision']:.3f}  "
              f"recall={report['LOW']['recall']:.3f}  f1={report['LOW']['f1-score']:.3f}")
        print(f"    HIGH — precision={report['HIGH']['precision']:.3f}  "
              f"recall={report['HIGH']['recall']:.3f}  f1={report['HIGH']['f1-score']:.3f}")

        clf.fit(X_selected, y_class)
        self.models['RF Classifier'] = clf

        results = {
            'threshold': threshold,
            'class_distribution': {'HIGH': n_high, 'LOW': n_low},
            'loocv_accuracy': float(acc),
            'confusion_matrix': cm.tolist(),
            'classification_report': {
                k: v for k, v in report.items() if k not in ('accuracy',)
            },
            'predictions': y_pred_loo.tolist(),
            'probabilities': y_proba_loo.tolist(),
            'actual_classes': y_class.tolist(),
        }

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

    def save_models(self, output_dir: str, classifier_results: Dict = None):
        """
        Save trained models, scaler, feature selection mask, and classifier metadata.

        Parameters
        ----------
        output_dir : str
            Directory to save models
        classifier_results : dict, optional
            LOOCV evaluation results from train_classifier() to persist as metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            model_filename = model_name.lower().replace(' ', '_') + '_model.pkl'
            model_path = output_path / model_filename
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")

        scaler_path = output_path / 'feature_scaler.pkl'
        joblib.dump(self.scalers['feature_scaler'], scaler_path)
        print(f"Saved feature scaler to {scaler_path}")

        selector_path = output_path / 'feature_selector.pkl'
        joblib.dump(self.feature_mask, selector_path)
        print(f"Saved feature selection mask to {selector_path}")

        if classifier_results is not None:
            clf_pkl_path = output_path / 'rf_classifier.pkl'
            joblib.dump(self.models['RF Classifier'], clf_pkl_path)
            print(f"Saved RF Classifier to {clf_pkl_path}")

            metadata = {
                'generated_at': datetime.now().isoformat(),
                'algorithm': 'RandomForestClassifier',
                'hyperparameters': {
                    'n_estimators': 200,
                    'max_depth': 4,
                    'max_features': 10,
                    'class_weight': {0: 2, 1: 1},
                    'random_state': self.random_state,
                },
                'threshold': classifier_results['threshold'],
                'class_distribution': classifier_results['class_distribution'],
                'loocv_accuracy': classifier_results['loocv_accuracy'],
                'confusion_matrix': classifier_results['confusion_matrix'],
                'classification_report': classifier_results['classification_report'],
            }
            meta_path = output_path / 'rf_classifier_metadata.json'
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Saved RF Classifier metadata to {meta_path}")

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
                'best_model': best_model,
                'feature_selection_method': 'spearman_permutation_fdr',
                'selected_features': getattr(self, 'selected_feature_names', []),
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
    merged_path = 'src/results/early_trial_features_merged.json'
    base_path = 'src/results/early_trial_features.json'
    ground_truth_path = 'src/results/ground_truth_labels.json'
    output_dir = 'src/results/models'
    results_path = 'src/results/model_evaluation.json'

    # Prefer merged features (includes resting-state EEG features from EDA analysis)
    if os.path.exists(merged_path):
        features_path = merged_path
        print(f"Using merged features: {merged_path}")
    elif os.path.exists(base_path):
        features_path = base_path
        print(f"Merged features not found, using base: {base_path}")
    else:
        print(f"Error: No features file found. Run extract_early_trial_features.py first.")
        return

    if not os.path.exists(ground_truth_path):
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        print("Please run generate_ground_truth_labels.py first (Phase 1).")
        return

    predictor = BCIPerformancePredictor(n_folds=5, random_state=42)

    X, y, subject_ids = predictor.load_early_trial_features(
        features_path, ground_truth_path
    )

    with open(features_path, 'r') as f:
        feature_names = json.load(f)['metadata'].get('feature_names', None)

    results = predictor.train_all_models(X, y, feature_names=feature_names)

    X_scaled = predictor.scalers['feature_scaler'].transform(X)
    X_selected = X_scaled[:, predictor.feature_mask]
    classifier_results = predictor.train_classifier(X_selected, y, threshold=0.65)

    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    comparison_df = predictor.generate_comparison_report(results)
    print(comparison_df.to_string(index=False))

    predictor.save_models(output_dir, classifier_results=classifier_results)
    predictor.save_results(results, results_path, subject_ids)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nFeature selection: Spearman correlation + permutation tests + FDR")
    print("(Benjamini-Hochberg). Pearson r is computed out-of-fold for honest")
    print("evaluation.")


if __name__ == '__main__':
    main()
