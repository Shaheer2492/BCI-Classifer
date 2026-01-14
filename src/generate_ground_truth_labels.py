"""
Phase 1: Generate Ground Truth Labels
--------------------------------------
Run MetaBCI CSP+LDA decoder on PhysionetMI dataset to extract
per-subject accuracy scores for all 109 subjects.

Uses stratified 5-fold cross-validation within each subject.
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
from tqdm import tqdm

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


class PhysionetMIGroundTruthGenerator:
    """Generate ground truth labels using CSP+LDA on PhysionetMI dataset."""

    def __init__(self, n_subjects=109, n_folds=5, random_state=42):
        """
        Initialize the ground truth generator.

        Parameters
        ----------
        n_subjects : int
            Number of subjects to process (default: 109)
        n_folds : int
            Number of folds for cross-validation (default: 5)
        random_state : int
            Random seed for reproducibility
        """
        self.n_subjects = n_subjects
        self.n_folds = n_folds
        self.random_state = random_state

        # PhysionetMI motor imagery runs
        # Run 4: left hand vs right hand (class 0 vs 1)
        # Run 8: hands vs feet (class 0 vs 1)
        # Run 12: left hand vs right hand (class 0 vs 1)
        self.mi_runs = [4, 8, 12]

        # EEG parameters
        self.tmin, self.tmax = 1.0, 4.0  # Time window for MI (skip first 1s)
        self.fmin, self.fmax = 7.0, 30.0  # Frequency band (mu + beta)

        # CSP parameters
        self.n_components = 4  # Number of CSP components

    def load_subject_data(self, subject_id):
        """
        Load and preprocess EEG data for a single subject.

        Parameters
        ----------
        subject_id : int
            Subject ID (1-109)

        Returns
        -------
        X : ndarray, shape (n_trials, n_channels, n_times)
            EEG data
        y : ndarray, shape (n_trials,)
            Labels
        run_info : list
            Information about which run each trial belongs to
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

        # Apply bandpass filter
        raw.filter(self.fmin, self.fmax, fir_design='firwin', verbose=False)

        # Extract events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # For PhysionetMI:
        # T1 = left fist (or both fists for runs with hands vs feet)
        # T2 = right fist (or both feet for runs with hands vs feet)
        # We'll use T1 vs T2 as our two-class problem

        # Map events to binary labels (0 vs 1)
        # T1 (left/hands) -> 0, T2 (right/feet) -> 1
        event_id_binary = {'T1': 0, 'T2': 1}

        # Find which event IDs correspond to T1 and T2
        t1_id = event_id.get('T1', None)
        t2_id = event_id.get('T2', None)

        if t1_id is None or t2_id is None:
            raise ValueError(f"Could not find T1/T2 events for subject {subject_id}")

        # Keep only T1 and T2 events
        mask = np.isin(events[:, 2], [t1_id, t2_id])
        events = events[mask]

        # Create binary labels
        labels = np.where(events[:, 2] == t1_id, 0, 1)

        # Pick only EEG channels (excluding EOG)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False)

        # Epoch the data
        epochs = mne.Epochs(
            raw, events,
            tmin=self.tmin, tmax=self.tmax,
            proj=True, picks=picks,
            baseline=None, preload=True,
            verbose=False
        )

        # Get data as numpy array
        X = epochs.get_data()  # (n_trials, n_channels, n_times)
        y = labels

        # Store run information (which run each trial came from)
        run_info = self._get_run_info(events, raw_fnames)

        return X, y, run_info

    def _get_run_info(self, events, raw_fnames):
        """Get information about which run each trial belongs to."""
        # Simplified: just return run indices
        # In a more sophisticated version, we'd track exact run boundaries
        n_trials = len(events)
        n_runs = len(self.mi_runs)
        trials_per_run = n_trials // n_runs

        run_info = []
        for i in range(n_trials):
            run_idx = min(i // trials_per_run, n_runs - 1)
            run_info.append(self.mi_runs[run_idx])

        return run_info

    def build_decoder(self):
        """
        Build CSP + LDA pipeline.

        Returns
        -------
        pipeline : sklearn.pipeline.Pipeline
            CSP + LDA decoder pipeline
        """
        pipeline = Pipeline([
            ('CSP', CSP(n_components=self.n_components,
                       reg=None,
                       log=True,
                       norm_trace=False)),
            ('Scaler', StandardScaler()),
            ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ])

        return pipeline

    def evaluate_subject(self, subject_id):
        """
        Evaluate decoder performance for a single subject using cross-validation.

        Parameters
        ----------
        subject_id : int
            Subject ID (1-109)

        Returns
        -------
        results : dict
            Dictionary containing performance metrics
        """
        try:
            # Load data
            X, y, run_info = self.load_subject_data(subject_id)

            # Check if we have enough trials
            if len(X) < self.n_folds:
                print(f"  Warning: Subject {subject_id} has too few trials ({len(X)})")
                return None

            # Check class balance
            unique, counts = np.unique(y, return_counts=True)
            if len(unique) < 2:
                print(f"  Warning: Subject {subject_id} has only one class")
                return None

            # Initialize cross-validation
            cv = StratifiedKFold(n_splits=self.n_folds,
                                shuffle=True,
                                random_state=self.random_state)

            # Store results for each fold
            fold_accuracies = []
            fold_predictions = []
            fold_true_labels = []

            # Perform cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Build and train decoder
                decoder = self.build_decoder()
                decoder.fit(X_train, y_train)

                # Predict on test set
                y_pred = decoder.predict(X_test)

                # Calculate accuracy
                accuracy = np.mean(y_pred == y_test)
                fold_accuracies.append(accuracy)
                fold_predictions.extend(y_pred.tolist())
                fold_true_labels.extend(y_test.tolist())

            # Calculate overall metrics
            overall_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)

            # Calculate per-class accuracy
            fold_predictions = np.array(fold_predictions)
            fold_true_labels = np.array(fold_true_labels)

            class_accuracies = {}
            for class_id in unique:
                class_mask = fold_true_labels == class_id
                class_acc = np.mean(fold_predictions[class_mask] == fold_true_labels[class_mask])
                class_accuracies[int(class_id)] = float(class_acc)

            # Compile results
            results = {
                'subject_id': int(subject_id),
                'n_trials': int(len(X)),
                'n_channels': int(X.shape[1]),
                'class_distribution': {int(c): int(cnt) for c, cnt in zip(unique, counts)},
                'accuracy': float(overall_accuracy),
                'accuracy_std': float(std_accuracy),
                'fold_accuracies': [float(acc) for acc in fold_accuracies],
                'class_accuracies': class_accuracies,
                'cv_folds': self.n_folds,
                'runs_used': self.mi_runs,
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

    def generate_all_labels(self, output_path='results/ground_truth_labels.json'):
        """
        Generate ground truth labels for all subjects.

        Parameters
        ----------
        output_path : str
            Path to save results JSON file

        Returns
        -------
        all_results : list
            List of result dictionaries for all subjects
        """
        print(f"Generating ground truth labels for {self.n_subjects} subjects...")
        print(f"Using {self.n_folds}-fold stratified cross-validation")
        print(f"Motor imagery runs: {self.mi_runs}")
        print(f"CSP components: {self.n_components}")
        print("-" * 60)

        all_results = []

        # Process each subject
        for subject_id in tqdm(range(1, self.n_subjects + 1), desc="Processing subjects"):
            print(f"\nSubject {subject_id}/{self.n_subjects}")
            results = self.evaluate_subject(subject_id)

            if results is not None:
                all_results.append(results)

                if results['success']:
                    print(f"  ✓ Accuracy: {results['accuracy']:.3f} ± {results['accuracy_std']:.3f}")
                else:
                    print(f"  ✗ Failed: {results.get('error', 'Unknown error')}")

        # Calculate summary statistics
        successful_subjects = [r for r in all_results if r['success']]
        accuracies = [r['accuracy'] for r in successful_subjects]

        summary = {
            'total_subjects': self.n_subjects,
            'successful_subjects': len(successful_subjects),
            'failed_subjects': self.n_subjects - len(successful_subjects),
            'mean_accuracy': float(np.mean(accuracies)) if accuracies else 0.0,
            'std_accuracy': float(np.std(accuracies)) if accuracies else 0.0,
            'min_accuracy': float(np.min(accuracies)) if accuracies else 0.0,
            'max_accuracy': float(np.max(accuracies)) if accuracies else 0.0,
            'median_accuracy': float(np.median(accuracies)) if accuracies else 0.0,
        }

        # Compile final output
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'n_subjects': self.n_subjects,
                'n_folds': self.n_folds,
                'mi_runs': self.mi_runs,
                'n_csp_components': self.n_components,
                'frequency_band': [self.fmin, self.fmax],
                'time_window': [self.tmin, self.tmax],
                'decoder': 'CSP + LDA',
                'cv_strategy': f'{self.n_folds}-fold stratified cross-validation',
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
        print(f"Mean accuracy: {summary['mean_accuracy']:.3f} ± {summary['std_accuracy']:.3f}")
        print(f"Accuracy range: [{summary['min_accuracy']:.3f}, {summary['max_accuracy']:.3f}]")
        print(f"Median accuracy: {summary['median_accuracy']:.3f}")
        print(f"\nResults saved to: {output_path.absolute()}")

        return all_results


def main():
    """Main execution function."""
    # Create generator
    generator = PhysionetMIGroundTruthGenerator(
        n_subjects=109,
        n_folds=5,
        random_state=42
    )

    # Generate labels
    results = generator.generate_all_labels(
        output_path='results/ground_truth_labels.json'
    )

    return results


if __name__ == '__main__':
    main()
