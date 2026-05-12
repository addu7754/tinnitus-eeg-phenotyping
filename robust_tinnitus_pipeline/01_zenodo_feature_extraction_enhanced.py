"""
Zenodo Enhanced Predictive Validation Pipeline
===============================================
For the Zenodo EEG-PRO Tinnitus dataset.
Differences from main pipeline:
  - 128-channel Geodesic Sensor Net mapping
  - Single .set file per patient (not per session)
  - Patient-level subtyping (KMeans on averaged features)
  - 4-class classification of discovered subtypes
"""

import os
import numpy as np
import pandas as pd
import mne
import warnings
from typing import Dict, List
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


# =============================================================================
# TIME-DOMAIN FEATURE FUNCTIONS
# =============================================================================

def compute_hjorth_parameters(signal: np.ndarray) -> tuple:
    """Compute Hjorth mobility, complexity, and activity."""
    var = np.var(signal)
    if var == 0:
        return 0.0, 0.0, 0.0
    diff_signal = np.diff(signal)
    diff_var = np.var(diff_signal)
    mobility = np.sqrt(diff_var / var) if diff_var > 0 else 0.0
    diff2_signal = np.diff(diff_signal)
    diff2_var = np.var(diff2_signal)
    mobility_deriv = np.sqrt(diff2_var / diff_var) if diff2_var > 0 and diff_var > 0 else 0.0
    complexity = mobility_deriv / mobility if mobility > 0 else 0.0
    activity = var
    return mobility, complexity, activity


def compute_rms(signal: np.ndarray) -> float:
    return np.sqrt(np.mean(signal ** 2))


def compute_zcr(signal: np.ndarray) -> float:
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / (len(signal) - 1)


# =============================================================================
# SPECTRAL FEATURE FUNCTIONS
# =============================================================================

def compute_spectral_entropy(psd: np.ndarray) -> float:
    psd_norm = psd / (psd.sum() + 1e-10)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-10))


def compute_spectral_slope(psd: np.ndarray, freqs: np.ndarray) -> float:
    valid = (psd > 0) & (freqs > 0)
    if valid.sum() < 2:
        return 0.0
    log_freqs = np.log10(freqs[valid])
    log_psd = np.log10(psd[valid])
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    return slope


def compute_spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    total_power = psd.sum() + 1e-10
    return np.sum(freqs * psd) / total_power


def compute_spectral_flux(psd: np.ndarray) -> float:
    return np.sum(np.diff(psd) ** 2)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def process_zenodo_file(file_path: str, bands: Dict[str, tuple], 
                         regions: Dict[str, list]) -> pd.DataFrame:
    """Enhanced feature extraction for Zenodo EEG-PRO files."""
    try:
        data = mne.io.read_epochs_eeglab(file_path, verbose=False)
    except Exception as e:
        return pd.DataFrame()

    ch_names = data.ch_names
    sfreq = data.info['sfreq']
    
    region_indices = {}
    for r_name, r_channels in regions.items():
        region_indices[r_name] = [ch_names.index(ch) for ch in r_channels if ch in ch_names]

    # PSD
    fmin, fmax = min(b[0] for b in bands.values()), max(b[1] for b in bands.values())
    spectrum = data.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # Raw epoch data for time-domain features
    epoch_data = data.get_data()
    
    # Subject ID from filename (e.g., "01_01_sub001.set" -> "01_01_sub001")
    subject_id = os.path.basename(file_path).replace('.set', '')
    
    features = []
    for epoch_idx in range(psds.shape[0]):
        feat_row = {'subject_id': subject_id, 'epoch_id': epoch_idx}
        
        for r_name, r_idx in region_indices.items():
            if not r_idx:
                continue
                
            region_psd = psds[epoch_idx, r_idx, :].mean(axis=0)
            total_power = np.sum(region_psd) + 1e-10
            
            # PSD band features
            for b_name, (b_low, b_high) in bands.items():
                freq_mask = np.logical_and(freqs >= b_low, freqs <= b_high)
                band_power = region_psd[freq_mask].sum()
                feat_row[f'{b_name}_{r_name}_abs'] = band_power
                feat_row[f'{b_name}_{r_name}_rel'] = band_power / total_power
            
            # Spectral features
            feat_row[f'spectral_entropy_{r_name}'] = compute_spectral_entropy(region_psd)
            feat_row[f'spectral_slope_{r_name}'] = compute_spectral_slope(region_psd, freqs)
            feat_row[f'spectral_centroid_{r_name}'] = compute_spectral_centroid(region_psd, freqs)
            feat_row[f'spectral_flux_{r_name}'] = compute_spectral_flux(region_psd)
            
            # Time-domain features
            region_signal = epoch_data[epoch_idx, r_idx, :].mean(axis=0)
            mobility, complexity, activity = compute_hjorth_parameters(region_signal)
            feat_row[f'hjorth_mobility_{r_name}'] = mobility
            feat_row[f'hjorth_complexity_{r_name}'] = complexity
            feat_row[f'hjorth_activity_{r_name}'] = activity
            feat_row[f'kurtosis_{r_name}'] = kurtosis(region_signal)
            feat_row[f'skewness_{r_name}'] = skew(region_signal)
            feat_row[f'rms_{r_name}'] = compute_rms(region_signal)
            feat_row[f'zcr_{r_name}'] = compute_zcr(region_signal)
        
        features.append(feat_row)
    
    return pd.DataFrame(features)


def main():
    print("--- Zenodo Enhanced Pipeline: Feature Extraction & Validation ---\n")
    
    bands = {
        'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 
        'beta': (13, 30), 'gamma': (30, 45)
    }
    
    regions = {
        'Frontal': [f'E{i}' for i in range(1, 33)],
        'Central': [f'E{i}' for i in range(33, 50)] + ['Cz'],
        'Parietal': [f'E{i}' for i in range(50, 70)],
        'Temporal': [f'E{i}' for i in range(100, 120)],
        'Occipital': [f'E{i}' for i in range(70, 95)]
    }
    
    dataset_dir = r"data/EEG-PRO_data/EEG-PRO_data/RAW/Tinnitus Dataset"
    set_files = sorted(glob.glob(os.path.join(dataset_dir, "*.set")))
    print(f"Located {len(set_files)} unique patient .set files.")
    
    # Count total features that will be extracted per epoch
    n_psd = len(bands) * len(regions) * 2
    n_spectral = 4 * len(regions)
    n_time = 8 * len(regions)
    print(f"\nExpected features per epoch:")
    print(f"  PSD bands (abs+rel): {n_psd}")
    print(f"  Spectral features: {n_spectral}")
    print(f"  Time-domain features: {n_time}")
    print(f"  Total: {n_psd + n_spectral + n_time} (+ PLV connectivity if added)")
    
    print("\nExtracting features (parallel processing)...")
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_zenodo_file)(f, bands, regions) for f in tqdm(set_files, desc="Zenodo")
    )
    
    all_features = [df for df in results if not df.empty]
    
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        print(f"\n--- Enhanced Extraction Complete ---")
        print(f"Shape: {final_df.shape}")
        
        out_path = "robust_tinnitus_pipeline/zenodo_01_extracted_features_enhanced.csv"
        final_df.to_csv(out_path, index=False)
        print(f"Saved to: {out_path}")
    else:
        print("ERROR: No features extracted!")


if __name__ == '__main__':
    main()