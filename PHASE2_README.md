# Phase 2: Early Trial Feature Extraction

This phase extracts neurophysiological features from the **first 15 trials** of each subject to enable early prediction of final BCI decoder performance.

## Overview

- **Dataset**: PhysionetMI (109 subjects)
- **Early Trials**: First 15 trials per subject
- **Features**: 27 neurophysiological features across 5 categories
- **Purpose**: Predict final BCI performance without using the full dataset
- **Output**: JSON file with per-subject feature vectors

## Why Phase 2 is Critical

**The Problem with the Original Approach:**

The initial implementation (before Phase 2) achieved R²=0.95 but was using **circular reasoning**:
- Features were extracted from fold accuracies (computed on ALL trials)
- Target was the overall accuracy (from ALL trials)
- Result: Predicting final accuracy using information derived from final accuracy

**Phase 2 Solution:**

Extract features from **ONLY the first 15 trials** to enable genuine early prediction:
- Input: Features from trials 1-15
- Output: Final accuracy (from all trials)
- Result: True predictive power, expected R²=0.5-0.8

## Feature Categories

### 1. Band Power Features (6 features)

Spectral power in motor-relevant frequency bands for motor cortex channels.

**Features:**
- Mu band (8-13 Hz) power: C3, Cz, C4
- Beta band (13-30 Hz) power: C3, Cz, C4

**Method:** Welch's periodogram

**Neurophysiological Interpretation:**
- **Mu rhythm**: Associated with motor cortex activity, suppressed during motor imagery (ERD)
- **Beta rhythm**: Related to motor preparation and execution
- **Motor cortex channels**: C3 (left), Cz (central), C4 (right) - primary motor areas

**Why it predicts performance:**
- Strong baseline mu/beta power indicates good signal quality
- Clear spectral peaks suggest well-defined motor rhythms
- Higher power in motor areas correlates with better BCI control

### 2. CSP Pattern Features (4 features)

Spatial filter quality metrics from Common Spatial Patterns fitted on early trials.

**Features:**
- `csp_eigenvalue_ratio`: Log ratio of largest to smallest eigenvalue
- `csp_feature_mean`: Mean absolute value of CSP-transformed features
- `csp_feature_std`: Standard deviation of CSP features
- `csp_class_separability`: Euclidean distance between class means in CSP space

**Method:** MetaBCI CSP (4 components) fitted on first 15 trials

**Neurophysiological Interpretation:**
- **Eigenvalue ratio**: Measures how well spatial filters separate classes
- **Class separability**: Direct measure of discriminability in spatial domain
- **Feature statistics**: Indicate consistency of spatial patterns

**Why it predicts performance:**
- High eigenvalue ratio = strong spatial patterns = better classification
- Good class separability in early trials indicates subject has clear motor imagery patterns
- Consistent CSP features suggest stable neural responses

### 3. ERD/ERS Features (6 features)

Event-Related Desynchronization/Synchronization - relative power changes from baseline to task.

**Features:**
- ERD/ERS in mu band: C3, Cz, C4
- ERD/ERS in beta band: C3, Cz, C4

**Method:** 
- Baseline: -1 to 0 seconds (pre-cue)
- Task: 1 to 4 seconds (post-cue)
- ERD/ERS = (task_power - baseline_power) / baseline_power

**Neurophysiological Interpretation:**
- **ERD (negative values)**: Power decrease during motor imagery (desynchronization)
- **ERS (positive values)**: Power increase (synchronization)
- **Expected pattern**: Mu ERD in contralateral hemisphere during motor imagery

**Why it predicts performance:**
- Strong ERD indicates active motor imagery
- Clear baseline-to-task modulation = better signal
- Lateralized ERD (stronger in one hemisphere) = better class separation

### 4. Trial Variability Features (4 features)

Inter-trial consistency and signal stability metrics.

**Features:**
- `variability_cv_mean`: Mean coefficient of variation across channels
- `variability_cv_std`: Standard deviation of coefficient of variation
- `variability_inter_trial_corr`: Average pairwise correlation between trials
- `variability_stability`: Log-transformed inverse of trial variance

**Method:** Statistical measures across first 15 trials

**Neurophysiological Interpretation:**
- **Low variability**: Consistent neural responses across trials
- **High inter-trial correlation**: Reproducible motor imagery patterns
- **High stability**: Reliable signal, less noise

**Why it predicts performance:**
- Consistent trials = easier to learn patterns = better classification
- High variability suggests subject has difficulty maintaining motor imagery
- Stable signals indicate good concentration and task engagement

### 5. Signal-to-Noise Ratio (3 features)

Task-related signal quality compared to baseline noise.

**Features:**
- `snr_mean`: Average SNR across motor cortex channels
- `snr_std`: Standard deviation of SNR
- `snr_max`: Maximum SNR across channels

**Method:** SNR = task_power / baseline_power

**Neurophysiological Interpretation:**
- **High SNR**: Strong task-related activity relative to noise
- **Low SNR**: Weak signal or high noise/artifacts

**Why it predicts performance:**
- High SNR = cleaner signal = easier classification
- Consistent SNR across channels indicates good overall signal quality
- Low SNR suggests artifacts, poor electrode contact, or weak motor imagery

## Setup

### 1. Prerequisites

Ensure Phase 1 (ground truth generation) is complete:

```bash
# Check if ground truth exists
ls src/results/ground_truth_labels.json
```

If not, run Phase 1 first:

```bash
python src/generate_ground_truth_labels.py
```

### 2. Install Dependencies

All dependencies should already be installed from Phase 1. If needed:

```bash
pip install -r requirements.txt
```

## Usage

### Run Feature Extraction

```bash
python src/extract_early_trial_features.py
```

This will:
1. Process all 109 subjects
2. For each subject:
   - Load EEG data from motor imagery runs (4, 8, 12)
   - Extract **only the first 15 trials**
   - Apply bandpass filter (7-30 Hz)
   - Compute all 27 features
3. Save results to `src/results/early_trial_features.json`

### Expected Runtime

- **Per subject**: ~2-3 seconds
- **Total (109 subjects)**: ~5-10 minutes

Much faster than Phase 1 because we only process 15 trials per subject!

## Output Format

The script generates `src/results/early_trial_features.json`:

```json
{
  "metadata": {
    "generated_at": "2026-01-28T10:30:00",
    "n_subjects": 109,
    "n_early_trials": 15,
    "mi_runs": [4, 8, 12],
    "frequency_band": [7.0, 30.0],
    "mu_band": [8.0, 13.0],
    "beta_band": [13.0, 30.0],
    "motor_channels": ["C3", "Cz", "C4"],
    "n_csp_components": 4,
    "feature_names": [
      "band_power_mu_C3",
      "band_power_mu_Cz",
      "band_power_mu_C4",
      "band_power_beta_C3",
      "band_power_beta_Cz",
      "band_power_beta_C4",
      "csp_eigenvalue_ratio",
      "csp_feature_mean",
      "csp_feature_std",
      "csp_class_separability",
      "erdrs_mu_C3",
      "erdrs_mu_Cz",
      "erdrs_mu_C4",
      "erdrs_beta_C3",
      "erdrs_beta_Cz",
      "erdrs_beta_C4",
      "variability_cv_mean",
      "variability_cv_std",
      "variability_inter_trial_corr",
      "variability_stability",
      "snr_mean",
      "snr_std",
      "snr_max"
    ],
    "n_features": 23
  },
  "summary": {
    "total_subjects": 109,
    "successful_subjects": 99,
    "failed_subjects": 10,
    "n_features": 23
  },
  "subjects": [
    {
      "subject_id": 1,
      "n_early_trials_used": 15,
      "n_channels": 64,
      "features": [0.45, 0.52, 0.48, ...],
      "feature_dict": {
        "band_power_mu_C3": 0.45,
        "band_power_mu_Cz": 0.52,
        ...
      },
      "success": true
    }
  ]
}
```

## Feature Validation

### Check Feature Ranges

Expected ranges for features:

| Feature Category | Expected Range | Interpretation |
|-----------------|----------------|----------------|
| Band Power | 0.01 - 10.0 | Power spectral density (μV²/Hz) |
| CSP Eigenvalue Ratio | 0.0 - 5.0 | Log-transformed ratio |
| CSP Separability | 0.0 - 10.0 | Euclidean distance |
| ERD/ERS | -1.0 - 1.0 | Relative change (negative = ERD) |
| Variability CV | 0.0 - 2.0 | Coefficient of variation |
| Inter-trial Corr | -1.0 - 1.0 | Pearson correlation |
| SNR | 0.5 - 5.0 | Power ratio |

### Verify No Data Leakage

**Critical Check:** Ensure features are computed from ONLY first 15 trials.

```python
# In extract_early_trial_features.py, line ~145:
epochs_early = epochs[:actual_n_trials]  # Slice to first N trials
```

This guarantees no information from later trials leaks into features.

## Integration with Phase 3

Phase 3 (ML model training) uses these features:

```bash
python src/train_performance_predictor.py
```

The predictor will:
1. Load early trial features (Phase 2)
2. Load ground truth labels (Phase 1)
3. Match subjects by ID
4. Train models: X = early features, y = final accuracy
5. Evaluate with cross-validation

**Expected Performance:**
- R² = 0.5 - 0.8 (realistic for early prediction)
- Lower than the original 0.95 (which was circular)
- **This is correct!** We're doing real prediction now.

## Technical Details

### Preprocessing

Same as Phase 1:
- Bandpass filter: 7-30 Hz (mu + beta rhythms)
- Task window: 1-4 seconds after cue onset
- Baseline window: -1 to 0 seconds (for ERD/ERS)

### Channel Selection

**Motor cortex channels (C3, Cz, C4):**
- C3: Left motor cortex (right hand control)
- Cz: Central motor cortex
- C4: Right motor cortex (left hand control)

These channels show the strongest motor imagery signals.

### CSP Configuration

- **Components**: 4 (same as Phase 1 decoder)
- **Fitted on**: First 15 trials only
- **Purpose**: Measure spatial filter quality, not used for classification

### Frequency Bands

- **Mu band (8-13 Hz)**: Primary motor rhythm
- **Beta band (13-30 Hz)**: Motor preparation/execution
- **Full band (7-30 Hz)**: For general filtering

## Common Issues

### Issue: Subject has fewer than 15 trials

**Solution:** Script automatically uses all available trials if < 15.

```python
if len(epochs) < self.n_early_trials:
    actual_n_trials = len(epochs)
```

### Issue: Only one class in early trials

**Solution:** CSP features default to neutral values (0.5, 0.0).

### Issue: High variability in features

**Cause:** Normal! Early trials are inherently more variable.

**Impact:** This variability is informative - it's a feature, not a bug!

## Next Steps

After Phase 2:

1. **Verify output**: Check `src/results/early_trial_features.json`
2. **Run Phase 3**: Train ML models on early features
3. **Compare performance**: Expect R² = 0.5-0.8 (honest prediction)
4. **Analyze features**: Which features are most predictive?

## Research Implications

### What We're Proving

✅ **BCI performance can be predicted from early trials**
- No need to wait for full calibration
- Faster BCI setup and user screening

✅ **Neurophysiological features are informative**
- Band power, CSP patterns, ERD/ERS all contribute
- Multi-modal features capture different aspects of BCI aptitude

✅ **Honest evaluation of early prediction**
- R² = 0.5-0.8 is realistic and valuable
- Much better than random (R² = 0.0)
- Significant time savings (15 trials vs. 45+ trials)

### Practical Applications

1. **User Screening**: Identify good BCI users after 15 trials (~2 minutes)
2. **Adaptive Training**: Adjust protocols based on early predictions
3. **Clinical BCI**: Optimize rehabilitation for patients quickly
4. **Research**: Understand what makes a good BCI user

## References

### Motor Imagery BCI

- Pfurtscheller, G., & Neuper, C. (2001). Motor imagery and direct brain-computer communication. *Proceedings of the IEEE*, 89(7), 1123-1134.

### Common Spatial Patterns

- Ramoser, H., Muller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. *IEEE Transactions on Rehabilitation Engineering*, 8(4), 441-446.

### ERD/ERS

- Pfurtscheller, G., & Lopes da Silva, F. H. (1999). Event-related EEG/MEG synchronization and desynchronization: basic principles. *Clinical Neurophysiology*, 110(11), 1842-1857.

### Early BCI Performance Prediction

- Blankertz, B., et al. (2010). Neurophysiological predictor of SMR-based BCI performance. *NeuroImage*, 51(4), 1303-1309.
- Ahn, M., & Jun, S. C. (2015). Performance variation in motor imagery brain-computer interface: A brief review. *Journal of Neuroscience Methods*, 243, 103-110.

## License

MIT License

## Contributors

- Shaheer Khan (shk021@ucsd.edu)
