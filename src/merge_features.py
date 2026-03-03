"""
Merge EEG-Project features with early trial features.
--------------------------------------------------------------
Combines pre-computed features from three sources:
  1. Phase 2 early trial features (23 task-based features)
  2. Daniel's EDA features (RPL, SMR strength, ERD magnitudes)
  3. Andrew's features (spectral entropy, LZC, TAR)
  4. Resting-state EEG features from Daniel's notebook analysis
     (rpl_alpha, theta_alpha_ratio, pse_avg) -- statistically validated
     predictors of BCI performance via Spearman + FDR correction.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


DANIEL_FEATURES = [
    'rpl',
    'smr_strength',
    'mu_erd_real_C3',
    'beta_erd_real_C3',
    'mu_erd_imagined_C3',
]

ANDREW_FEATURES = [
    'LRF_beta_entropy_imagined',
    'LRF_beta_lzc_imagined',
    'LRF_beta_lzc_gap',
    'LRF_tar_imagined',
    'FF_beta_entropy_imagined',
    'FF_beta_lzc_imagined',
    'FF_tar_imagined',
]

RESTING_STATE_FEATURES_RAW = [
    'rpl_alpha',
    'theta_alpha_ratio',
    'pse_avg',
]

RESTING_STATE_FEATURES_RENAMED = [
    'resting_rpl_alpha',
    'resting_tar',
    'resting_pse_avg',
]


def load_early_trial_features(path):
    """Load Phase 2 early trial features JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def load_eeg_project_csvs(andrew_path, daniel_path):
    """Load and process EEG-Project CSV files."""
    andrew_df = pd.read_csv(andrew_path)
    daniel_df = pd.read_csv(daniel_path)

    andrew_df['subject_id_int'] = andrew_df['subject_id'].apply(lambda x: int(x[1:]))
    daniel_df['subject_id_int'] = daniel_df['subject_id'].apply(lambda x: int(x[1:]))

    return andrew_df, daniel_df


def load_resting_state_features(resting_path):
    """
    Load resting-state EEG features identified as significant BCI performance
    predictors in the EDA notebook (Spearman + permutation tests + FDR).

    Source columns: rpl_alpha, theta_alpha_ratio, pse_avg
    Renamed to resting_* prefix to avoid collision with Phase 2 features.
    """
    df = pd.read_csv(resting_path)
    df['subject_id_int'] = df['subject_id'].apply(lambda x: int(x[1:]))
    return df


def merge_features(features_data, andrew_df, daniel_df, resting_df=None):
    """Merge all feature sources into the Phase 2 features structure."""
    andrew_lookup = {}
    for _, row in andrew_df.iterrows():
        sid = row['subject_id_int']
        andrew_lookup[sid] = [float(row[col]) for col in ANDREW_FEATURES]

    daniel_lookup = {}
    for _, row in daniel_df.iterrows():
        sid = row['subject_id_int']
        daniel_lookup[sid] = [float(row[col]) for col in DANIEL_FEATURES]

    resting_lookup = {}
    if resting_df is not None:
        for _, row in resting_df.iterrows():
            sid = row['subject_id_int']
            resting_lookup[sid] = [float(row[col]) for col in RESTING_STATE_FEATURES_RAW]

    existing_feature_names = features_data['metadata']['feature_names']
    new_feature_names = (
        existing_feature_names
        + DANIEL_FEATURES
        + ANDREW_FEATURES
        + RESTING_STATE_FEATURES_RENAMED
    )

    daniel_values = np.array(list(daniel_lookup.values()))
    andrew_values = np.array(list(andrew_lookup.values()))
    daniel_medians = np.median(daniel_values, axis=0).tolist()
    andrew_medians = np.median(andrew_values, axis=0).tolist()

    resting_medians = [0.0] * len(RESTING_STATE_FEATURES_RAW)
    if resting_lookup:
        resting_values = np.array(list(resting_lookup.values()))
        resting_medians = np.median(resting_values, axis=0).tolist()

    merged_subjects = []
    n_merged = 0
    n_imputed = 0

    for subject in features_data['subjects']:
        sid = subject['subject_id']

        daniel_feats = daniel_lookup.get(sid, daniel_medians)
        andrew_feats = andrew_lookup.get(sid, andrew_medians)
        resting_feats = resting_lookup.get(sid, resting_medians)

        if (sid not in daniel_lookup or sid not in andrew_lookup
                or (resting_lookup and sid not in resting_lookup)):
            n_imputed += 1

        daniel_feats = [daniel_medians[i] if np.isnan(v) else v
                        for i, v in enumerate(daniel_feats)]
        andrew_feats = [andrew_medians[i] if np.isnan(v) else v
                        for i, v in enumerate(andrew_feats)]
        resting_feats = [resting_medians[i] if np.isnan(v) else v
                         for i, v in enumerate(resting_feats)]

        merged_features = (
            subject['features'] + daniel_feats + andrew_feats + resting_feats
        )

        merged_subject = dict(subject)
        merged_subject['features'] = merged_features
        merged_subject['feature_dict'] = dict(zip(new_feature_names, merged_features))
        merged_subjects.append(merged_subject)
        n_merged += 1

    merged_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_subjects': features_data['metadata']['n_subjects'],
            'n_early_trials': features_data['metadata']['n_early_trials'],
            'mi_runs': features_data['metadata']['mi_runs'],
            'frequency_band': features_data['metadata']['frequency_band'],
            'mu_band': features_data['metadata']['mu_band'],
            'beta_band': features_data['metadata']['beta_band'],
            'motor_channels': features_data['metadata']['motor_channels'],
            'n_csp_components': features_data['metadata']['n_csp_components'],
            'feature_names': new_feature_names,
            'n_features': len(new_feature_names),
            'merged_sources': {
                'phase2_features': len(existing_feature_names),
                'daniel_features': len(DANIEL_FEATURES),
                'andrew_features': len(ANDREW_FEATURES),
                'resting_state_features': len(RESTING_STATE_FEATURES_RENAMED),
            },
            'notes': (
                "Andrew's features use all trials (not just first 15). "
                "Daniel's RPL and SMR come from resting-state runs (no trial dependency). "
                "Daniel's ERD features use movement runs. "
                "Resting-state features (rpl_alpha, tar, pse_avg) are from "
                "Daniel's EDA notebook -- statistically validated predictors "
                "of BCI performance (Spearman + permutation tests + FDR)."
            )
        },
        'summary': {
            'total_subjects': len(merged_subjects),
            'successful_subjects': sum(1 for s in merged_subjects if s['success']),
            'failed_subjects': sum(1 for s in merged_subjects if not s['success']),
            'n_features': len(new_feature_names),
            'feature_names': new_feature_names,
            'n_imputed_subjects': n_imputed,
        },
        'subjects': merged_subjects
    }

    return merged_data


def main():
    """Main execution function."""
    features_path = 'src/results/early_trial_features.json'
    andrew_path = 'EEG-Project/eeg_features_andrew_compact.csv'
    daniel_path = 'EEG-Project/eeg_features_daniel.csv'
    resting_path = 'EEG-Project/eeg_features.csv'
    output_path = 'src/results/early_trial_features_merged.json'

    for path, name in [
        (features_path, 'Phase 2 features'),
        (andrew_path, "Andrew's features CSV"),
        (daniel_path, "Daniel's features CSV"),
        (resting_path, "Resting-state features CSV"),
    ]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            return

    print("Loading Phase 2 early trial features...")
    features_data = load_early_trial_features(features_path)
    existing_n = len(features_data['metadata']['feature_names'])
    print(f"  {existing_n} existing features, "
          f"{len(features_data['subjects'])} subjects")

    print("Loading EEG-Project CSVs...")
    andrew_df, daniel_df = load_eeg_project_csvs(andrew_path, daniel_path)
    print(f"  Andrew: {len(andrew_df)} subjects, selecting {len(ANDREW_FEATURES)} features")
    print(f"  Daniel: {len(daniel_df)} subjects, selecting {len(DANIEL_FEATURES)} features")

    print("Loading resting-state features (from EDA notebook analysis)...")
    resting_df = load_resting_state_features(resting_path)
    print(f"  Resting: {len(resting_df)} subjects, selecting "
          f"{len(RESTING_STATE_FEATURES_RAW)} features")

    print("\nMerging features...")
    merged_data = merge_features(features_data, andrew_df, daniel_df, resting_df)

    total_features = merged_data['metadata']['n_features']
    print(f"  Merged: {total_features} total features per subject")
    print(f"    Phase 2:       {existing_n}")
    print(f"    Daniel:        {len(DANIEL_FEATURES)}")
    print(f"    Andrew:        {len(ANDREW_FEATURES)}")
    print(f"    Resting-state: {len(RESTING_STATE_FEATURES_RENAMED)}")
    print(f"  Subjects with imputed values: "
          f"{merged_data['summary']['n_imputed_subjects']}")

    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"\nSaved merged features to {output_path}")


if __name__ == '__main__':
    main()
