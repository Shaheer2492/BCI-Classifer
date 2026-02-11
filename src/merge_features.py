"""
Merge EEG-Project features with Phase 2 early trial features.
--------------------------------------------------------------
Combines pre-computed features from Andrew's (spectral entropy, LZC, TAR)
and Daniel's (RPL, SMR strength, ERD magnitudes) EDA work with the
Phase 2 early trial features to create an enriched feature set.

Note: Andrew's features use all trials (not just the first 15).
Daniel's resting-state features (RPL, SMR) come from separate baseline runs
and have no trial dependency. This is acceptable for initial integration;
a follow-up step will re-extract Andrew's features for first-15-trials only.
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd


# Features to select from each source
DANIEL_FEATURES = [
    'rpl',                  # Resting alpha power level
    'smr_strength',         # SMR baseline strength
    'mu_erd_real_C3',       # Strongest ERD indicator
    'beta_erd_real_C3',     # Beta ERD magnitude
    'mu_erd_imagined_C3',   # Imagined ERD
]

ANDREW_FEATURES = [
    'LRF_beta_entropy_imagined',   # Beta spectral entropy during imagined movement
    'LRF_beta_lzc_imagined',       # Beta LZC during imagined movement
    'LRF_beta_lzc_gap',            # Real vs imagined LZC gap
    'LRF_tar_imagined',            # Theta/alpha ratio during imagined movement
    'FF_beta_entropy_imagined',    # Same for fists/feet task
    'FF_beta_lzc_imagined',        # Same for fists/feet task
    'FF_tar_imagined',             # Same for fists/feet task
]


def load_early_trial_features(path):
    """Load Phase 2 early trial features JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def load_eeg_project_csvs(andrew_path, daniel_path):
    """Load and process EEG-Project CSV files."""
    andrew_df = pd.read_csv(andrew_path)
    daniel_df = pd.read_csv(daniel_path)

    # Convert subject IDs: "S001" -> 1
    andrew_df['subject_id_int'] = andrew_df['subject_id'].apply(lambda x: int(x[1:]))
    daniel_df['subject_id_int'] = daniel_df['subject_id'].apply(lambda x: int(x[1:]))

    return andrew_df, daniel_df


def merge_features(features_data, andrew_df, daniel_df):
    """Merge EEG-Project features into the Phase 2 features structure."""
    # Build lookup tables for EEG-Project features
    andrew_lookup = {}
    for _, row in andrew_df.iterrows():
        sid = row['subject_id_int']
        andrew_lookup[sid] = [float(row[col]) for col in ANDREW_FEATURES]

    daniel_lookup = {}
    for _, row in daniel_df.iterrows():
        sid = row['subject_id_int']
        daniel_lookup[sid] = [float(row[col]) for col in DANIEL_FEATURES]

    # Get existing feature names
    existing_feature_names = features_data['metadata']['feature_names']
    new_feature_names = existing_feature_names + DANIEL_FEATURES + ANDREW_FEATURES

    # Compute medians for imputation (in case of missing subjects)
    daniel_values = np.array(list(daniel_lookup.values()))
    andrew_values = np.array(list(andrew_lookup.values()))
    daniel_medians = np.median(daniel_values, axis=0).tolist()
    andrew_medians = np.median(andrew_values, axis=0).tolist()

    # Merge features for each subject
    merged_subjects = []
    n_merged = 0
    n_imputed = 0

    for subject in features_data['subjects']:
        sid = subject['subject_id']

        # Get EEG-Project features (or use medians if subject missing)
        daniel_feats = daniel_lookup.get(sid, daniel_medians)
        andrew_feats = andrew_lookup.get(sid, andrew_medians)

        if sid not in daniel_lookup or sid not in andrew_lookup:
            n_imputed += 1

        # Handle any NaN values via median imputation
        daniel_feats = [daniel_medians[i] if np.isnan(v) else v
                        for i, v in enumerate(daniel_feats)]
        andrew_feats = [andrew_medians[i] if np.isnan(v) else v
                        for i, v in enumerate(andrew_feats)]

        # Append new features to existing feature vector
        merged_features = subject['features'] + daniel_feats + andrew_feats

        merged_subject = dict(subject)
        merged_subject['features'] = merged_features
        merged_subject['feature_dict'] = dict(zip(new_feature_names, merged_features))
        merged_subjects.append(merged_subject)
        n_merged += 1

    # Build merged output
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
            },
            'notes': (
                "Andrew's features use all trials (not just first 15). "
                "Daniel's RPL and SMR come from resting-state runs (no trial dependency). "
                "Daniel's ERD features use movement runs."
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
    # Paths
    features_path = 'src/results/early_trial_features.json'
    andrew_path = 'EEG-Project/eeg_features_andrew_compact.csv'
    daniel_path = 'EEG-Project/eeg_features_daniel.csv'
    output_path = 'src/results/early_trial_features_merged.json'

    # Check files exist
    for path, name in [(features_path, 'Phase 2 features'),
                       (andrew_path, "Andrew's features CSV"),
                       (daniel_path, "Daniel's features CSV")]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            return

    # Load data
    print("Loading Phase 2 early trial features...")
    features_data = load_early_trial_features(features_path)
    existing_n = len(features_data['metadata']['feature_names'])
    print(f"  {existing_n} existing features, "
          f"{len(features_data['subjects'])} subjects")

    print("Loading EEG-Project CSVs...")
    andrew_df, daniel_df = load_eeg_project_csvs(andrew_path, daniel_path)
    print(f"  Andrew: {len(andrew_df)} subjects, selecting {len(ANDREW_FEATURES)} features")
    print(f"  Daniel: {len(daniel_df)} subjects, selecting {len(DANIEL_FEATURES)} features")

    # Merge
    print("\nMerging features...")
    merged_data = merge_features(features_data, andrew_df, daniel_df)

    total_features = merged_data['metadata']['n_features']
    print(f"  Merged: {total_features} total features per subject "
          f"({existing_n} Phase 2 + {len(DANIEL_FEATURES)} Daniel + {len(ANDREW_FEATURES)} Andrew)")
    print(f"  Subjects with imputed values: {merged_data['summary']['n_imputed_subjects']}")

    # Save
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"\nSaved merged features to {output_path}")


if __name__ == '__main__':
    main()
