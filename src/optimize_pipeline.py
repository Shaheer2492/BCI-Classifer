import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from generate_ground_truth_labels import PhysionetMIGroundTruthGenerator
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_optimization(n_subjects=10):
    """
    Run grid search for hyperparameter optimization on a subset of subjects.
    """
    print(f"Running hyperparameter optimization on first {n_subjects} subjects...")
    
    # Define search space
    param_grid = {
        'freq_band': [[7, 30], [8, 13], [13, 30], [4, 40]],
        'n_csp_components': [2, 4, 6, 8],
        'lda_shrinkage': ['auto', None, 0.1, 0.5],
        'time_window': [[1.0, 4.0], [0.5, 4.0]]
    }
    
    grid = ParameterGrid(param_grid)
    print(f"Total combinations to test: {len(grid)}")
    
    # Initialize generator (will re-configure it in the loop)
    generator = PhysionetMIGroundTruthGenerator(n_subjects=n_subjects)
    
    # Cache data loading to avoid reloading for every parameter combination
    # We only need Left/Right task (runs 4, 8, 12) for now as it's the baseline
    runs = [4, 8, 12]
    subjects_data = {}
    
    print("Pre-loading subject data...")
    for sub_id in range(1, n_subjects + 1):
        try:
            # Load with default params initially, but we need raw access or 
            # flexible loading. The current load_subject_data relies on 
            # self.fmin/fmax/tmin/tmax. 
            # We need to modify the generator's state BEFORE loading 
            # if we want to change filtering/epoching.
            # OR we load raw once and filter/epoch dynamically.
            # Given the structure of PhysionetMIGroundTruthGenerator, 
            # it filters and epochs inside load_subject_data.
            # So we cannot easily cache the 'X' if freq/time changes.
            # We will have to reload/reprocess for time/freq changes.
            pass
        except Exception as e:
            print(f"Skipping subject {sub_id}: {e}")

    results = []
    
    for params in tqdm(grid, desc="Grid Search"):
        # Update generator parameters
        generator.fmin, generator.fmax = params['freq_band']
        generator.n_components = params['n_csp_components']
        generator.tmin, generator.tmax = params['time_window']
        
        # We need to hack/patch the build_decoder method or the class to accept shrinkage
        # The current build_decoder hardcodes shrinkage='auto'
        # Let's verify this by checking the file content again or assuming (it was 'auto').
        # We'll subclass or monkeypatch for this script.
        
        # Monkeypatch build_decoder to use current shrinkage param
        original_build_decoder = generator.build_decoder
        
        def custom_build_decoder(shrinkage=params['lda_shrinkage']):
            from sklearn.pipeline import Pipeline
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.preprocessing import StandardScaler
            from metabci.brainda.algorithms.decomposition import CSP
            
            return Pipeline([
                ('CSP', CSP(n_components=generator.n_components)),
                ('Scaler', StandardScaler()),
                ('LDA', LinearDiscriminantAnalysis(solver='lsqr', shrinkage=shrinkage))
            ])
            
        generator.build_decoder = custom_build_decoder
        
        # Run evaluation
        accuracies = []
        for sub_id in range(1, n_subjects + 1):
            try:
                # Reload data because freq/time params changed
                # This is slow but necessary with current class design
                res = generator.evaluate_subject(sub_id, runs)
                if res and res['success']:
                    accuracies.append(res['accuracy'])
            except Exception as e:
                pass
        
        if accuracies:
            mean_acc = np.mean(accuracies)
            results.append({
                **params,
                'mean_accuracy': mean_acc,
                'n_subjects': len(accuracies)
            })
            
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('mean_accuracy', ascending=False)
    
    print("\nTop 5 Configurations:")
    print(df.head(5).to_string())
    
    # Save results
    df.to_csv('results/hyperparameter_optimization_results.csv', index=False)
    print("\nResults saved to results/hyperparameter_optimization_results.csv")
    
    return df.iloc[0]

if __name__ == "__main__":
    run_optimization(n_subjects=5) # Start with 5 for speed
