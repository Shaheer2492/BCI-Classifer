
import pandas as pd
import json
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import os

# Paths
FEATURE_FILE = 'EEG-Project/eeg_features_andrew_compact.csv'
GROUND_TRUTH_PATH = 'src/results/ground_truth_labels.json'
BASELINE_R2 = 0.24 # From methodology.html

def load_data():
    print(f"Loading features from {FEATURE_FILE}...")
    df = pd.read_csv(FEATURE_FILE)
    
    # Clean subject_id (S001 -> 1)
    df['subject_id'] = df['subject_id'].apply(lambda x: int(x.replace('S', '')))
    
    print(f"Loading ground truth from {GROUND_TRUTH_PATH}...")
    with open(GROUND_TRUTH_PATH, 'r') as f:
        gt_data = json.load(f)
        
    subjects_gt = []
    for data in gt_data['subjects']:
        if data['success']:
            subjects_gt.append({
                'subject_id': int(data['subject_id']),
                'accuracy': data['accuracy']
            })
    
    df_gt = pd.DataFrame(subjects_gt)
    
    # Merge
    merged = pd.merge(df, df_gt, on='subject_id')
    print(f"Merged data shape: {merged.shape}")
    
    return merged

def evaluate_features(df):
    # Prepare X and y
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'accuracy']]
    X = df[feature_cols].values
    y = df['accuracy'].values
    
    print(f"Training on {len(feature_cols)} features: {feature_cols[:3]}...")
    
    # Initialize Model (Same params as baseline for fair comparison)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 5-Fold Grid CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_r2 = cross_val_score(rf, X, y, cv=kf, scoring='r2')
    cv_mae = cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_absolute_error')
    
    # Train predictor for correlation check
    rf.fit(X, y)
    y_pred = rf.predict(X)
    corr, _ = pearsonr(y, y_pred)
    
    print("\n" + "="*40)
    print("RESULTS: Andrew Compact Features")
    print("="*40)
    print(f"R² (CV Mean):       {cv_r2.mean():.4f} (Baseline: {BASELINE_R2})")
    print(f"R² (CV Std):        {cv_r2.std():.4f}")
    print(f"MAE (CV Mean):      {-cv_mae.mean():.4f}")
    print(f"Pearson Correlation: {corr:.4f}")
    print("="*40)
    
    if cv_r2.mean() > BASELINE_R2 + 0.05:
        print("CONCLUSION: SIGNIFICANT IMPROVEMENT. Recommend switching.")
    elif cv_r2.mean() > BASELINE_R2:
        print("CONCLUSION: MARGINAL IMPROVEMENT.")
    else:
        print("CONCLUSION: NO IMPROVEMENT. Stick to baseline.")

if __name__ == "__main__":
    try:
        data = load_data()
        evaluate_features(data)
    except Exception as e:
        print(f"Error: {e}")
