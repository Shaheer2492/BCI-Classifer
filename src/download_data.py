
import mne
from mne.datasets import eegbci
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

def download_subject(subject_id):
    """Download all MI runs for a subject."""
    # Runs 4, 8, 12 (Left/Right) and 6, 10, 14 (Hands/Feet)
    runs = [4, 6, 8, 10, 12, 14]
    try:
        # load_data downloads if not present
        eegbci.load_data(subject_id, runs, update_path=True, verbose=False)
        return True
    except Exception as e:
        print(f"Error downloading S{subject_id:03d}: {e}")
        return False

def main():
    n_subjects = 109
    
    # Use ThreadPoolExecutor for parallel downloads
    # Adjust max_workers based on network/system limits. 4 is safer for PhysioNet.
    max_workers = 4
    
    print(f"Downloading data for {n_subjects} subjects in parallel (workers={max_workers})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(download_subject, range(1, n_subjects + 1)),
            total=n_subjects,
            desc="Downloading"
        ))
        
    success_count = sum(results)
    print(f"\nDownload complete. Successfully processed: {success_count}/{n_subjects}")

if __name__ == "__main__":
    mne.set_log_level('ERROR')
    main()
