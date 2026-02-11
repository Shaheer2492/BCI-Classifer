"""
Phase 2: Extract Early Trial Features
--------------------------------------
Extract neurophysiological features from the first N trials of each subject
to predict final BCI decoder performance.

Features extracted:
1. Band Power (mu/beta bands, motor cortex channels)
2. CSP Pattern Features (spatial filter quality from early trials)
3. ERD/ERS Magnitude (event-related desynchronization)
4. Trial Variability (inter-trial consistency)
5. Signal-to-Noise Ratio (task vs baseline)
"""

import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from scipy import signal
from scipy.stats import pearsonr
from metabci.brainda.algorithms.decomposition import CSP
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


class EarlyTrialFeatureExtractor:
    """Extract features from early trials to predict BCI performance.
    
    This class extracts neurophysiological features from only the first
    n_early_trials of each subject's data, enabling early prediction of
    final decoder performance without using the full dataset.
    """
    
    def __init__(self, n_early_trials=15, n_subjects=109, random_state=42):
        """
        Initialize the early trial feature extractor.
        
        Parameters
        ----------
        n_early_trials : int
            Number of early trials to use for feature extraction (default: 15)
        n_subjects : int
            Number of subjects to process (default: 109)
        random_state : int
            Random seed for reproducibility
        """
        self.n_early_trials = n_early_trials
        self.n_subjects = n_subjects
        self.random_state = random_state
        
        # PhysionetMI motor imagery runs
        self.mi_runs = [4, 8, 12]
        
        # EEG parameters
        self.tmin, self.tmax = 1.0, 4.0  # Task window
        self.baseline_tmin, self.baseline_tmax = -1.0, 0.0  # Baseline window
        self.fmin, self.fmax = 7.0, 30.0  # Full band
        
        # Frequency bands
        self.mu_band = (8.0, 13.0)  # Mu rhythm
        self.beta_band = (13.0, 30.0)  # Beta rhythm
        
        # Motor cortex channels (for band power and ERD/ERS)
        self.motor_channels = ['C3', 'Cz', 'C4']
        
        # CSP parameters
        self.n_csp_components = 8  # Matched to optimized Ground Truth model
        
    def load_subject_data(self, subject_id):
        """
        Load and preprocess EEG data for a single subject.
        
        Parameters
        ----------
        subject_id : int
            Subject ID (1-109)
            
        Returns
        -------
        epochs : mne.Epochs
            Epoched EEG data (all trials)
        epochs_baseline : mne.Epochs
            Baseline epochs (for ERD/ERS computation)
        y : ndarray
            Labels for all trials
        """
        raw_fnames = []
        
        # Load motor imagery runs for this subject
        for run in self.mi_runs:
            try:
                fnames = eegbci.load_data(subject_id, run, update_path=True)
                raw_fnames.extend(fnames)
            except Exception as e:
                print(f"  Warning: Could not load run {run} for subject {subject_id}: {e}")
                continue
        
        if not raw_fnames:
            raise ValueError(f"No valid data files found for subject {subject_id}")
        
        # Read and concatenate all runs
        raws = [read_raw_edf(fname, preload=True, verbose=False) for fname in raw_fnames]
        raw = concatenate_raws(raws)

        # Standardize channel names (PhysioNet EDFs have trailing dots like "C3.", "Cz.")
        eegbci.standardize(raw)

        # Apply bandpass filter
        raw.filter(self.fmin, self.fmax, fir_design='firwin', verbose=False)
        
        # Extract events
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Find T1 and T2 event IDs
        t1_id = event_id.get('T1', None)
        t2_id = event_id.get('T2', None)
        
        if t1_id is None or t2_id is None:
            raise ValueError(f"Could not find T1/T2 events for subject {subject_id}")
        
        # Keep only T1 and T2 events
        mask = np.isin(events[:, 2], [t1_id, t2_id])
        events = events[mask]
        
        # Create binary labels
        labels = np.where(events[:, 2] == t1_id, 0, 1)
        
        # Pick only EEG channels
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

        # Epoch the data (task window)
        epochs = mne.Epochs(
            raw, events,
            tmin=self.tmin, tmax=self.tmax,
            proj=True, picks=picks,
            baseline=None, preload=True,
            verbose=False
        )

        # Epoch baseline data (for ERD/ERS)
        epochs_baseline = mne.Epochs(
            raw, events,
            tmin=self.baseline_tmin, tmax=self.baseline_tmax,
            proj=True, picks=picks,
            baseline=None, preload=True,
            verbose=False
        )

        # Recompute labels from surviving epochs (some may be dropped during epoching)
        labels = np.where(epochs.events[:, 2] == t1_id, 0, 1)

        return epochs, epochs_baseline, labels
    
    def extract_band_power(self, epochs_early, channel_names):
        """
        Extract band power features from early trials.
        
        Computes mu and beta band power for motor cortex channels (C3, Cz, C4).
        
        Parameters
        ----------
        epochs_early : mne.Epochs
            Early trial epochs
        channel_names : list
            List of channel names
            
        Returns
        -------
        features : dict
            Band power features
        """
        features = {}
        
        # Get data for motor cortex channels
        for ch_name in self.motor_channels:
            if ch_name in channel_names:
                ch_idx = channel_names.index(ch_name)
                
                # Get data for this channel across all early trials
                ch_data = epochs_early.get_data()[:, ch_idx, :]  # (n_trials, n_times)
                
                # Compute average power spectrum using Welch's method
                freqs, psd = signal.welch(
                    ch_data,
                    fs=epochs_early.info['sfreq'],
                    nperseg=min(256, ch_data.shape[1]),
                    axis=1
                )
                
                # Average across trials
                psd_mean = np.mean(psd, axis=0)
                
                # Extract mu band power
                mu_mask = (freqs >= self.mu_band[0]) & (freqs <= self.mu_band[1])
                mu_power = np.mean(psd_mean[mu_mask])
                features[f'band_power_mu_{ch_name}'] = float(mu_power)
                
                # Extract beta band power
                beta_mask = (freqs >= self.beta_band[0]) & (freqs <= self.beta_band[1])
                beta_power = np.mean(psd_mean[beta_mask])
                features[f'band_power_beta_{ch_name}'] = float(beta_power)
        
        return features
    
    def extract_csp_features(self, X_early, y_early):
        """
        Extract CSP pattern features from early trials.
        
        Fits CSP on early trials and extracts spatial filter quality metrics.
        
        Parameters
        ----------
        X_early : ndarray, shape (n_trials, n_channels, n_times)
            Early trial data
        y_early : ndarray, shape (n_trials,)
            Early trial labels
            
        Returns
        -------
        features : dict
            CSP-based features
        """
        features = {}
        
        try:
            # Check if we have both classes in early trials
            unique_classes = np.unique(y_early)
            if len(unique_classes) < 2:
                # If only one class, return default values
                features['csp_eigenvalue_ratio'] = 0.5
                features['csp_feature_mean'] = 0.0
                features['csp_feature_std'] = 0.0
                features['csp_class_separability'] = 0.0
                return features
            
            # Fit CSP on early trials
            csp = CSP(n_components=self.n_csp_components)
            csp.fit(X_early, y_early)
            
            # Transform early trials
            X_csp = csp.transform(X_early)
            
            # Compute eigenvalue ratio (measure of class separability)
            # Higher ratio = better spatial filters
            if hasattr(csp, 'filters_'):
                # Get eigenvalues from CSP filters
                eigenvalues = np.var(X_csp, axis=0)
                if len(eigenvalues) >= 2:
                    eigenvalue_ratio = eigenvalues[0] / (eigenvalues[-1] + 1e-10)
                    features['csp_eigenvalue_ratio'] = float(np.log(eigenvalue_ratio + 1))
                else:
                    features['csp_eigenvalue_ratio'] = 0.0
            else:
                features['csp_eigenvalue_ratio'] = 0.0
            
            # CSP feature statistics
            features['csp_feature_mean'] = float(np.mean(np.abs(X_csp)))
            features['csp_feature_std'] = float(np.std(X_csp))
            
            # Class separability (distance between class means in CSP space)
            class_0_mask = y_early == 0
            class_1_mask = y_early == 1
            
            if np.sum(class_0_mask) > 0 and np.sum(class_1_mask) > 0:
                mean_0 = np.mean(X_csp[class_0_mask], axis=0)
                mean_1 = np.mean(X_csp[class_1_mask], axis=0)
                separability = np.linalg.norm(mean_0 - mean_1)
                features['csp_class_separability'] = float(separability)
            else:
                features['csp_class_separability'] = 0.0
            
        except Exception as e:
            print(f"    Warning: CSP feature extraction failed: {e}")
            features['csp_eigenvalue_ratio'] = 0.0
            features['csp_feature_mean'] = 0.0
            features['csp_feature_std'] = 0.0
            features['csp_class_separability'] = 0.0
        
        return features
    
    def extract_erdrs_features(self, epochs_task_early, epochs_baseline_early, channel_names):
        """
        Extract ERD/ERS (Event-Related Desynchronization/Synchronization) features.
        
        Computes the relative change in power from baseline to task period.
        
        Parameters
        ----------
        epochs_task_early : mne.Epochs
            Early trial task epochs
        epochs_baseline_early : mne.Epochs
            Early trial baseline epochs
        channel_names : list
            List of channel names
            
        Returns
        -------
        features : dict
            ERD/ERS features
        """
        features = {}
        
        for ch_name in self.motor_channels:
            if ch_name in channel_names:
                ch_idx = channel_names.index(ch_name)
                
                # Get baseline and task data
                baseline_data = epochs_baseline_early.get_data()[:, ch_idx, :]
                task_data = epochs_task_early.get_data()[:, ch_idx, :]
                
                # Compute power spectra
                freqs_baseline, psd_baseline = signal.welch(
                    baseline_data,
                    fs=epochs_baseline_early.info['sfreq'],
                    nperseg=min(256, baseline_data.shape[1]),
                    axis=1
                )
                
                freqs_task, psd_task = signal.welch(
                    task_data,
                    fs=epochs_task_early.info['sfreq'],
                    nperseg=min(256, task_data.shape[1]),
                    axis=1
                )
                
                # Average across trials
                psd_baseline_mean = np.mean(psd_baseline, axis=0)
                psd_task_mean = np.mean(psd_task, axis=0)

                # Use separate frequency masks (baseline and task may have different
                # frequency resolutions due to different epoch lengths)
                mu_mask_task = (freqs_task >= self.mu_band[0]) & (freqs_task <= self.mu_band[1])
                mu_mask_base = (freqs_baseline >= self.mu_band[0]) & (freqs_baseline <= self.mu_band[1])
                baseline_mu = np.mean(psd_baseline_mean[mu_mask_base])
                task_mu = np.mean(psd_task_mean[mu_mask_task])
                erdrs_mu = (task_mu - baseline_mu) / (baseline_mu + 1e-10)
                features[f'erdrs_mu_{ch_name}'] = float(erdrs_mu)

                beta_mask_task = (freqs_task >= self.beta_band[0]) & (freqs_task <= self.beta_band[1])
                beta_mask_base = (freqs_baseline >= self.beta_band[0]) & (freqs_baseline <= self.beta_band[1])
                baseline_beta = np.mean(psd_baseline_mean[beta_mask_base])
                task_beta = np.mean(psd_task_mean[beta_mask_task])
                erdrs_beta = (task_beta - baseline_beta) / (baseline_beta + 1e-10)
                features[f'erdrs_beta_{ch_name}'] = float(erdrs_beta)
        
        return features
    
    def extract_variability_features(self, X_early):
        """
        Extract trial variability features.
        
        Measures inter-trial consistency and signal stability.
        
        Parameters
        ----------
        X_early : ndarray, shape (n_trials, n_channels, n_times)
            Early trial data
            
        Returns
        -------
        features : dict
            Variability features
        """
        features = {}
        
        # Compute trial-averaged signal
        trial_means = np.mean(X_early, axis=2)  # (n_trials, n_channels)
        
        # Coefficient of variation across trials
        cv = np.std(trial_means, axis=0) / (np.mean(np.abs(trial_means), axis=0) + 1e-10)
        features['variability_cv_mean'] = float(np.mean(cv))
        features['variability_cv_std'] = float(np.std(cv))
        
        # Inter-trial correlation (average pairwise correlation)
        n_trials = X_early.shape[0]
        if n_trials >= 2:
            correlations = []
            for i in range(min(n_trials, 10)):  # Sample up to 10 trials for efficiency
                for j in range(i + 1, min(n_trials, 10)):
                    # Flatten trials and compute correlation
                    trial_i = X_early[i].flatten()
                    trial_j = X_early[j].flatten()
                    if len(trial_i) > 1:
                        corr, _ = pearsonr(trial_i, trial_j)
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            if correlations:
                features['variability_inter_trial_corr'] = float(np.mean(correlations))
            else:
                features['variability_inter_trial_corr'] = 0.0
        else:
            features['variability_inter_trial_corr'] = 0.0
        
        # Signal stability index (inverse of variance across trials)
        trial_variance = np.var(trial_means, axis=0)
        stability = 1.0 / (np.mean(trial_variance) + 1e-10)
        features['variability_stability'] = float(np.log(stability + 1))
        
        return features
    
    def extract_snr_features(self, epochs_task_early, epochs_baseline_early, channel_names):
        """
        Extract Signal-to-Noise Ratio features.
        
        Compares task-related activity to baseline noise.
        
        Parameters
        ----------
        epochs_task_early : mne.Epochs
            Early trial task epochs
        epochs_baseline_early : mne.Epochs
            Early trial baseline epochs
        channel_names : list
            List of channel names
            
        Returns
        -------
        features : dict
            SNR features
        """
        features = {}
        
        snr_values = []
        
        for ch_name in self.motor_channels:
            if ch_name in channel_names:
                ch_idx = channel_names.index(ch_name)
                
                # Get baseline and task data
                baseline_data = epochs_baseline_early.get_data()[:, ch_idx, :]
                task_data = epochs_task_early.get_data()[:, ch_idx, :]
                
                # Compute power
                baseline_power = np.mean(baseline_data ** 2)
                task_power = np.mean(task_data ** 2)
                
                # SNR = task_power / baseline_power
                snr = task_power / (baseline_power + 1e-10)
                snr_values.append(snr)
        
        # Overall SNR statistics
        if snr_values:
            features['snr_mean'] = float(np.mean(snr_values))
            features['snr_std'] = float(np.std(snr_values))
            features['snr_max'] = float(np.max(snr_values))
        else:
            features['snr_mean'] = 1.0
            features['snr_std'] = 0.0
            features['snr_max'] = 1.0
        
        return features
    
    def extract_subject_features(self, subject_id):
        """
        Extract all features for a single subject from early trials.
        
        Parameters
        ----------
        subject_id : int
            Subject ID (1-109)
            
        Returns
        -------
        results : dict
            Dictionary containing all extracted features
        """
        try:
            # Load data
            epochs, epochs_baseline, labels = self.load_subject_data(subject_id)
            
            # Check if we have enough trials
            if len(epochs) < self.n_early_trials:
                print(f"  Warning: Subject {subject_id} has only {len(epochs)} trials (need {self.n_early_trials})")
                actual_n_trials = len(epochs)
            else:
                actual_n_trials = self.n_early_trials
            
            # Extract only early trials
            epochs_early = epochs[:actual_n_trials]
            epochs_baseline_early = epochs_baseline[:actual_n_trials]
            y_early = labels[:actual_n_trials]
            
            # Get channel names
            channel_names = epochs_early.ch_names
            
            # Get data as numpy array
            X_early = epochs_early.get_data()  # (n_trials, n_channels, n_times)
            
            # Extract all features
            features = {}
            
            # 1. Band Power Features (6 features)
            band_power_features = self.extract_band_power(epochs_early, channel_names)
            features.update(band_power_features)
            
            # 2. CSP Pattern Features (4 features)
            csp_features = self.extract_csp_features(X_early, y_early)
            features.update(csp_features)
            
            # 3. ERD/ERS Features (6 features)
            erdrs_features = self.extract_erdrs_features(
                epochs_early, epochs_baseline_early, channel_names
            )
            features.update(erdrs_features)
            
            # 4. Trial Variability Features (4 features)
            variability_features = self.extract_variability_features(X_early)
            features.update(variability_features)
            
            # 5. SNR Features (3 features)
            snr_features = self.extract_snr_features(
                epochs_early, epochs_baseline_early, channel_names
            )
            features.update(snr_features)
            
            # Compile results
            results = {
                'subject_id': int(subject_id),
                'n_early_trials_used': int(actual_n_trials),
                'n_channels': int(X_early.shape[1]),
                'features': list(features.values()),
                'feature_dict': features,
                'success': True
            }
            
            return results
            
        except Exception as e:
            print(f"  Error processing subject {subject_id}: {e}")
            return {
                'subject_id': int(subject_id),
                'success': False,
                'error': str(e)
            }
    
    def extract_all_features(self, output_path='results/early_trial_features.json'):
        """
        Extract features from early trials for all subjects.
        
        Parameters
        ----------
        output_path : str
            Path to save results JSON file
            
        Returns
        -------
        all_results : list
            List of result dictionaries for all subjects
        """
        print(f"Extracting early trial features for {self.n_subjects} subjects...")
        print(f"Using first {self.n_early_trials} trials per subject")
        print(f"Motor imagery runs: {self.mi_runs}")
        print("-" * 60)
        
        all_results = []
        
        # Process each subject
        for subject_id in tqdm(range(1, self.n_subjects + 1), desc="Processing subjects"):
            print(f"\nSubject {subject_id}/{self.n_subjects}")
            results = self.extract_subject_features(subject_id)
            
            if results is not None:
                all_results.append(results)
                
                if results['success']:
                    n_features = len(results['features'])
                    print(f"  ✓ Extracted {n_features} features from {results['n_early_trials_used']} trials")
                else:
                    print(f"  ✗ Failed: {results.get('error', 'Unknown error')}")
        
        # Calculate summary statistics
        successful_subjects = [r for r in all_results if r['success']]
        
        # Get feature names from first successful subject
        feature_names = []
        if successful_subjects:
            feature_names = list(successful_subjects[0]['feature_dict'].keys())
        
        summary = {
            'total_subjects': self.n_subjects,
            'successful_subjects': len(successful_subjects),
            'failed_subjects': self.n_subjects - len(successful_subjects),
            'n_features': len(feature_names),
            'feature_names': feature_names
        }
        
        # Compile final output
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'n_subjects': self.n_subjects,
                'n_early_trials': self.n_early_trials,
                'mi_runs': self.mi_runs,
                'frequency_band': [self.fmin, self.fmax],
                'mu_band': list(self.mu_band),
                'beta_band': list(self.beta_band),
                'motor_channels': self.motor_channels,
                'n_csp_components': self.n_csp_components,
                'feature_names': feature_names,
                'n_features': len(feature_names)
            },
            'summary': summary,
            'subjects': all_results
        }
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Successfully processed: {summary['successful_subjects']}/{self.n_subjects} subjects")
        print(f"Features extracted: {summary['n_features']}")
        print(f"Feature names: {', '.join(feature_names[:5])}...")
        print(f"\nResults saved to: {output_path.absolute()}")
        
        return all_results


def main():
    """Main execution function."""
    # Create feature extractor
    extractor = EarlyTrialFeatureExtractor(
        n_early_trials=15,
        n_subjects=109,
        random_state=42
    )
    
    # Extract features
    results = extractor.extract_all_features(
        output_path='results/early_trial_features.json'
    )
    
    return results


if __name__ == '__main__':
    main()
