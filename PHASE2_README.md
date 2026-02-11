# Phase 2: Early Trial Feature Extraction

This phase focuses on extracting neurophysiological features from the first few trials of a BCI session to enable early prediction of decoder performance.

## Overview

- **Input**: Raw EEG data (PhysioNet MI dataset)
- **Selection**: First **15 trials** (approx. 2-3 minutes of data)
- **Task**: Left vs Right Hand Motor Imagery (Runs 4, 8, 12)
- **Features**: 23 features per subject
- **Output**: JSON file with extracted features

## Features Extracted

We extract features across 5 categories, focusing on the Sensorimotor Rhythms (SMR):

### 1. Band Power (6 features)
Absolute power in Mu (8-13 Hz) and Beta (13-30 Hz) bands for channels C3, Cz, C4.
- `band_power_mu_C3`, `band_power_beta_C3`, etc.

### 2. CSP Pattern Quality (4 features)
Properties of the Common Spatial Patterns (CSP) filters fitted on early trials.
- `csp_eigenvalue_ratio`: Ratio of largest to smallest eigenvalue (separability)
- `csp_class_separability`: Distance between class means in CSP space
- `csp_feature_mean/std`: Statistical properties of CSP filtered signals

### 3. ERD/ERS Magnitude (6 features)
Event-Related Desynchronization/Synchronization relative to baseline.
- `erdrs_mu_C3`: Suppression of Mu rhythm during motor imagery
- `erdrs_beta_C3`: Suppression of Beta rhythm

### 4. Trial Variability (4 features)
Consistency of neural responses across the 15 trials.
- `variability_inter_trial_corr`: Average correlation between single-trial responses
- `variability_cv_mean`: Coefficient of variation

### 5. Signal-to-Noise Ratio (3 features)
- `snr_mean`, `snr_max`: Ratio of task power to baseline power

## Usage

### Run Feature Extraction

```bash
python src/extract_early_trial_features.py
```

This script:
1. Loads the first 15 trials for each subject.
2. Extracts all 23 features.
3. Saves results to `results/early_trial_features.json`.

## Output Format

```json
{
  "metadata": "...",
  "subjects": [
    {
      "subject_id": 1,
      "n_early_trials_used": 15,
      "features": [1.2, 0.5, ...],
      "feature_dict": {
        "band_power_mu_C3": 1.2,
        ...
      }
    }
  ]
}
```
