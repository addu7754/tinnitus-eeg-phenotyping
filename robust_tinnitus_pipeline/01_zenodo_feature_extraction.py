import os
import glob
import numpy as np
import pandas as pd
import mne
import warnings
from typing import Dict
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')

def process_zenodo_file(file_path: str, bands: Dict[str, tuple], regions: Dict[str, list]) -> pd.DataFrame:
    try:
        # Zenodo RAW EEG-PRO files are already epoched. We'll load them securely.
        data = mne.io.read_epochs_eeglab(file_path, verbose=False)
    except Exception as e:
        return pd.DataFrame()

    ch_names = data.ch_names
    region_indices = {}
    for r_name, r_channels in regions.items():
        region_indices[r_name] = [ch_names.index(ch) for ch in r_channels if ch in ch_names]

    # Multitaper Spectral Analysis
    fmin, fmax = min(b[0] for b in bands.values()), max(b[1] for b in bands.values())
    spectrum = data.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, verbose=False)
    
    psds, freqs = spectrum.get_data(return_freqs=True)
    subject_id = os.path.basename(file_path).replace('.set', '')
    
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
    print("--- Zenodo Pipeline Bit 1: Feature Extraction ---")
    
    bands = {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
        'beta': (13, 30), 'gamma': (30, 45)
    }
    
    # 128-channel Geodesic Sensor Net approximation mapping
    regions = {
        'Frontal': [f'E{i}' for i in range(1, 33)],
        'Central': [f'E{i}' for i in range(33, 50)] + ['Cz'],
        'Parietal': [f'E{i}' for i in range(50, 70)],
        'Temporal': [f'E{i}' for i in range(100, 120)],
        'Occipital': [f'E{i}' for i in range(70, 95)]
    }
    
    dataset_dir = r"data/EEG-PRO_data/EEG-PRO_data/RAW/Tinnitus Dataset"
    set_files = list(glob.glob(os.path.join(dataset_dir, "*.set")))
    print(f"Located {len(set_files)} unique patient .set files.")
    
    # Extract using JobLib natively!
    print("Parallel processing Zenodo files (this may take a few moments)...")
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_zenodo_file)(f, bands, regions) for f in tqdm(set_files, desc="Zenodo Subjects")
    )
    
    all_features = [df for df in results if not df.empty]
    
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        print("\n--- Extraction Successful ---")
        print(f"Total shape: {final_df.shape}")
        
        out_path = "robust_tinnitus_pipeline/zenodo_01_extracted_features.csv"
        final_df.to_csv(out_path, index=False)
        print(f"Data saved strictly to: {out_path}")

if __name__ == '__main__':
    main()