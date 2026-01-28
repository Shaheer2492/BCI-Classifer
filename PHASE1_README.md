# Phase 1: Ground Truth Label Generation

This phase generates ground truth labels by running CSP+LDA decoders on the PhysionetMI dataset.

## Overview

- **Dataset**: PhysionetMI (109 subjects)
- **Tasks**: Motor imagery (left/right hand, hands/feet)
- **Runs**: 4, 8, 12 (MI tasks)
- **Decoder**: MetaBCI CSP (4 components) + sklearn LDA
- **Cross-validation**: 5-fold stratified within-subject
- **Output**: JSON file with per-subject accuracy scores

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

The PhysionetMI dataset will be automatically downloaded by MNE when you run the script for the first time. The dataset will be stored in `~/mne_data/` (approximately 2.5 GB).

## Usage

### Run Ground Truth Generation

```bash
python src/generate_ground_truth_labels.py
```

This will:
1. Process all 109 subjects
2. For each subject:
   - Load EEG data from motor imagery runs (4, 8, 12)
   - Apply bandpass filter (7-30 Hz)
   - Epoch the data (1-4 seconds after cue)
   - Train CSP+LDA decoder using 5-fold cross-validation
   - Extract accuracy metrics
3. Save results to `results/ground_truth_labels.json`

### Expected Runtime

- **Per subject**: ~30-60 seconds
- **Total (109 subjects)**: ~1-2 hours

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "metadata": {
    "generated_at": "2026-01-14T10:30:00",
    "n_subjects": 109,
    "n_folds": 5,
    "mi_runs": [4, 8, 12],
    "n_csp_components": 4,
    "frequency_band": [7.0, 30.0],
    "time_window": [1.0, 4.0],
    "decoder": "MetaBCI CSP + sklearn LDA",
    "cv_strategy": "5-fold stratified cross-validation"
  },
  "summary": {
    "total_subjects": 109,
    "successful_subjects": 103,
    "failed_subjects": 6,
    "mean_accuracy": 0.65,
    "std_accuracy": 0.12,
    "min_accuracy": 0.45,
    "max_accuracy": 0.95,
    "median_accuracy": 0.64
  },
  "subjects": [
    {
      "subject_id": 1,
      "n_trials": 45,
      "n_channels": 64,
      "class_distribution": {"0": 23, "1": 22},
      "accuracy": 0.711,
      "accuracy_std": 0.082,
      "fold_accuracies": [0.67, 0.78, 0.67, 0.78, 0.67],
      "class_accuracies": {"0": 0.696, "1": 0.727},
      "cv_folds": 5,
      "runs_used": [4, 8, 12],
      "success": true
    },
    ...
  ]
}
```

## Key Metrics Extracted

For each subject:
- **accuracy**: Overall classification accuracy (mean across folds)
- **accuracy_std**: Standard deviation of accuracy across folds
- **fold_accuracies**: Accuracy for each individual fold
- **class_accuracies**: Per-class accuracy (class 0 and class 1)
- **n_trials**: Total number of trials
- **class_distribution**: Number of trials per class

## Cross-Validation Strategy

We use **within-subject stratified 5-fold cross-validation** because:

1. **Within-subject**: Each subject's data is evaluated independently
2. **Stratified**: Maintains class balance in each fold
3. **5-fold**: Provides reliable estimates with 80/20 train/test split
4. **Appropriate for Phase 1**: We're measuring individual subject performance, not cross-subject generalization

## Technical Details

### Preprocessing
- Bandpass filter: 7-30 Hz (mu + beta rhythms)
- Epoch window: 1-4 seconds after cue onset
- Baseline: None (relative to task period)

### MetaBCI CSP Parameters
- Number of components: 4 (2 per class)
- Uses MetaBCI's optimized CSP implementation
- Automatic eigenvalue decomposition and spatial pattern extraction

### sklearn LDA Parameters
- Solver: lsqr (suitable for high-dimensional data)
- Shrinkage: auto (automatic selection)

## Next Steps

After generating ground truth labels:
1. Analyze subject performance distribution
2. Optionally classify subjects into High/Medium/Low categories
3. Proceed to Phase 2: Feature extraction

## References

**Cross-Validation Strategy Research:**
- [Multi-scale convolutional transformer network for motor imagery BCI](https://www.nature.com/articles/s41598-025-96611-5)
- [Increasing accessibility to PhysionetMI dataset](https://www.sciencedirect.com/science/article/pii/S2352340924001525)
- [The Necessity of Leave One Subject Out (LOSO) Cross Validation](https://link.springer.com/chapter/10.1007/978-3-030-86993-9_50)
