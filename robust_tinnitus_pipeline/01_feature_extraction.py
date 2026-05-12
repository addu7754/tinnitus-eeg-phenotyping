import os
import glob
import numpy as np
import pandas as pd
import mne
import warnings
from typing import List, Dict
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

def process_file_global(set_file_path: str, bands: Dict[str, tuple], regions: Dict[str, list]) -> pd.DataFrame:
    return extract_features_mne_multitaper(set_file_path, bands, regions)

def extract_features_mne_multitaper(set_file_path: str, bands: Dict[str, tuple], regions: Dict[str, list]) -> pd.DataFrame:
    """
    Extracts absolute and relative PSD features from an EEGLAB .set file using 
    MNE's Multitaper Spectral Estimation.
    """
    try:
        # Try loading as pre-epoched data first
        try:
            data = mne.io.read_epochs_eeglab(set_file_path, verbose=False)
        except Exception:
            # If that fails, it's likely continuous raw data. Load and create default 2s epochs.
            raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
            events = mne.make_fixed_length_events(raw, duration=2.0)
            data = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False)
    except Exception as e:
        # Silencing load failures for multiprocessing
        return pd.DataFrame()

    # Get actual channel names to map indices dynamically
    ch_names = data.ch_names
    region_indices = {}
    for r_name, r_channels in regions.items():
        # Find indices of the channels that exist in this file
        region_indices[r_name] = [ch_names.index(ch) for ch in r_channels if ch in ch_names]

    # Calculate PSD using Multitaper
    fmin, fmax = min(b[0] for b in bands.values()), max(b[1] for b in bands.values())
    spectrum = data.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, verbose=False)
    
    psds, freqs = spectrum.get_data(return_freqs=True)
    subject_id = os.path.basename(set_file_path).replace('.set', '')
    
    features = []
    for epoch_idx in range(psds.shape[0]):
        feat_row = {'subject_id': subject_id, 'epoch_id': epoch_idx}
        
        for r_name, r_idx in region_indices.items():
            if not r_idx:
                continue
                
            region_psd = psds[epoch_idx, r_idx, :].mean(axis=0)
            total_power = np.sum(region_psd)
            
            for b_name, (b_low, b_high) in bands.items():
                freq_mask = np.logical_and(freqs >= b_low, freqs <= b_high)
                band_power = region_psd[freq_mask].sum()
                
                feat_row[f'{b_name}_{r_name}_abs'] = band_power
                feat_row[f'{b_name}_{r_name}_rel'] = band_power / (total_power + 1e-10)
                
        features.append(feat_row)
        
    return pd.DataFrame(features)

def main():
    bands = {
        'delta': (0.5, 4), 'theta': (4, 8), 'alpha1': (8, 10), 'alpha2': (10, 12),
        'beta1': (12, 20), 'beta2': (20, 30), 'gamma': (30, 45)
    }
    # Mapped explicitly from the dataset's channel names
    regions = {
        'Frontal': ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8'],
        'Central': ['C3', 'C4'],
        'Parietal': ['P7', 'Pz', 'P8'],
        'Temporal': ['T7', 'T8'],
        'Occipital': ['O1', 'O2']
    }
    
    dataset_dir = r"data/Acoustic Therapies for Tinnitus Treatment An EEG Database/TA_Database_set"
    set_files = list(glob.glob(os.path.join(dataset_dir, "**", "*.set"), recursive=True))
    
    if not set_files:
         print(f"No .set files found in {dataset_dir}")
         return

    print("--- Running First Bit: Multitaper Feature Extraction ---")
    print(f"Total .set files available: {len(set_files)}")
    print("Extracting features from all datasets. This will use joblib to safely speed it up...\n")
    
    all_features = []
    
    # Using joblib with loky backend which is much safer for scientific stack (numpy/mne)
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_file_global)(f, bands, regions) for f in tqdm(set_files, desc="Processing EEG files")
    )
    
    for df in results:
        if not df.empty:
            all_features.append(df)
            
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        print("\n--- Extraction Successful ---")
        print(f"Total shape: {final_df.shape}")
        
        # Save output for next steps
        out_path = "robust_tinnitus_pipeline/01_extracted_features.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\nFeatures saved to: {out_path}")

if __name__ == '__main__':
    main()