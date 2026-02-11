# Methodology

## Overview
The motor imagery classification pipeline uses a standard Machine Learning approach for Brain-Computer Interfaces (BCI), specifically the Common Spatial Patterns (CSP) algorithm for feature extraction followed by Linear Discriminant Analysis (LDA) for classification.

## Data Preprocessing
1.  **Loading**: EEG data is loaded from the PhysioNet MI dataset (EDF+ format).
2.  **Electrode Selection**: 64 EEG channels are selected, excluding EOG and stimulation channels.
3.  **Filtering**: A Finite Impulse Response (FIR) bandpass filter is applied to the raw data.
    *   **Frequency Band**: [8, 13] Hz (Mu Band)
4.  **Epoching**: Data is segmented into trials based on event markers (T1 vs T2).
    *   **Time Window**: [0.5, 4.0] seconds relative to the event onset.
    *   **Baseline Correction**: None applied (based on literature best practices for CSP).

## Feature Extraction: Common Spatial Patterns (CSP)
CSP is a spatial filtering algorithm that maximizes the variance of one class while minimizing the variance of the other. It is highly effective for discriminating Event-Related Desynchronization/Synchronization (ERD/ERS) patterns associated with motor imagery.

*   **Components**: We select the first `k` and last `k` spatial filters (sorted by eigenvalue), where `2k = 8` components.
*   **Log-Variance**: The variance of the spatially filtered signals is computed and log-transformed to obtain the feature vector for each trial.

## Classification: Linear Discriminant Analysis (LDA)
LDA is used to find a linear decision boundary that separates the two classes in the feature space.

*   **Solver**: Least Squares (`lsqr`) is used, which is efficient for high-dimensional data.
*   **Regularization (Shrinkage)**: 0.5 shrinkage is applied to improve covariance matrix estimation and prevent overfitting.

## Evaluation Strategy
*   **Cross-Validation**: 5-fold stratified cross-validation is performed for each subject. This ensures that the train/test split preserves the percentage of samples for each class.
*   **Metrics**:
    *   Accuracy: The proportion of correctly classified trials.
    *   Standard Deviation: To measure the stability of the classifier across folds.
*   **Subject-Specific Models**: A separate model is trained and evaluated for each subject (no transfer learning).

## Optimization
We performed a grid search to optimize the following hyperparameters:
*   Frequency Band
*   Number of CSP Components
*   LDA Shrinkage
*   Time Window

The final parameters were selected based on the mean accuracy across a representative subset of subjects.
