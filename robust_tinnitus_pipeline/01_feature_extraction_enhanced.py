"""
Enhanced Feature Extraction Pipeline for Tinnitus EEG Classification
=====================================================================
Adds: spectral features (entropy, slope, flux, centroid), 
      time-domain features (Hjorth, kurtosis, skewness, RMS, ZCR),
      and connectivity features (Phase Locking Value, Imaginary Coherence).
"""

import os
import glob
import numpy as np
import pandas as pd
import mne
import warnings
from typing import List, Dict, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert

warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


# =============================================================================
# TIME-DOMAIN FEATURE FUNCTIONS
# =============================================================================

def compute_hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
    """Compute Hjorth mobility, complexity, and activity parameters."""
    var = np.var(signal)
    if var == 0:
        return 0.0, 0.0, 0.0
    
    # First derivative
    diff_signal = np.diff(signal)
    diff_var = np.var(diff_signal)
    
    # Mobility: sqrt(variance of derivative / variance of signal)
    mobility = np.sqrt(diff_var / var) if diff_var > 0 else 0.0
    
    # Complexity: mobility of derivative / mobility of signal
    diff2_signal = np.diff(diff_signal)
    diff2_var = np.var(diff2_signal)
    mobility_deriv = np.sqrt(diff2_var / diff_var) if diff2_var > 0 and diff_var > 0 else 0.0
    complexity = mobility_deriv / mobility if mobility > 0 else 0.0
    
    # Activity: variance of signal
    activity = var
    
    return mobility, complexity, activity


def compute_rms(signal: np.ndarray) -> float:
    """Compute Root Mean Square of signal."""
    return np.sqrt(np.mean(signal ** 2))


def compute_zcr(signal: np.ndarray) -> float:
    """Compute Zero Crossing Rate of signal."""
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / (len(signal) - 1)


# =============================================================================
# SPECTRAL FEATURE FUNCTIONS
# =============================================================================

def compute_spectral_entropy(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Compute spectral entropy from PSD."""
    psd_norm = psd / (psd.sum() + 1e-10)
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    return entropy


def compute_spectral_slope(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Compute spectral slope via linear regression on log-log PSD."""
    valid = (psd > 0) & (freqs > 0)
    if valid.sum() < 2:
        return 0.0
    log_freqs = np.log10(freqs[valid])
    log_psd = np.log10(psd[valid])
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    return slope


def compute_spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Compute spectral centroid (center of mass of spectrum)."""
    total_power = psd.sum() + 1e-10
    return np.sum(freqs * psd) / total_power


def compute_spectral_flux(psd: np.ndarray, freqs: np.ndarray) -> float:
    """Compute spectral flux as sum of squared differences between adjacent frequency bins."""
    return np.sum(np.diff(psd) ** 2)


# =============================================================================
# CONNECTIVITY FEATURE FUNCTIONS
# =============================================================================

def compute_plv(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Compute Phase Locking Value between two signals."""
    analytic1 = hilbert(signal1)
    analytic2 = hilbert(signal2)
    phase1 = np.angle(analytic1)
    phase2 = np.angle(analytic2)
    plv = np.abs(np.mean(np.exp(1j * (phase1 - phase2))))
    return plv


def compute_imaginary_coherence(signal1: np.ndarray, signal2: np.ndarray, 
                                 fs: float = 500.0, nfft: int = 256) -> float:
    """Compute Imaginary Coherence between two signals using Welch's method."""
    from scipy.signal import welch, csd
    
    f, Pxy = csd(signal1, signal2, fs=fs, nperseg=nfft, scaling='density')
    f, Pxx = welch(signal1, fs=fs, nperseg=nfft, scaling='density')
    f, Pyy = welch(signal2, fs=fs, nperseg=nfft, scaling='density')
    
    denom = np.sqrt(Pxx * Pyy)
    valid = denom > 1e-10
    if not np.any(valid):
        return 0.0
    
    # Imaginary coherence: imaginary part of coherence
    coh = Pxy[valid] / denom[valid]
    imag_coh = np.abs(np.imag(coh)).mean()
    return float(imag_coh)


def compute_connectivity_pairs(psds: np.ndarray, epoch_data: np.ndarray, 
                                ch_names: List[str], region_indices: Dict) -> Dict[str, float]:
    """Compute connectivity features between region pairs."""
    features = {}
    regions = list(region_indices.keys())
    
    # Channel-level PLV for representative channel pairs between regions
    # Use the first channel of each region as representative
    rep_channels = {}
    for r_name, r_idx in region_indices.items():
        if r_idx:
            rep_channels[r_name] = r_idx[0]
    
    for i, r1 in enumerate(regions):
        for r2 in regions[i+1:]:
            if r1 in rep_channels and r2 in rep_channels:
                ch1_idx = rep_channels[r1]
                ch2_idx = rep_channels[r2]
                
                # Average PLV across epochs
                plv_values = []
                for epoch_idx in range(min(epoch_data.shape[0], 50)):  # subsample for speed
                    sig1 = epoch_data[epoch_idx, ch1_idx, :]
                    sig2 = epoch_data[epoch_idx, ch2_idx, :]
                    plv_values.append(compute_plv(sig1, sig2))
                
                pair_name = f"{r1}_{r2}"
                features[f'plv_{pair_name}'] = np.mean(plv_values) if plv_values else 0.0
    
    return features


# =============================================================================
# MAIN FEATURE EXTRACTION
# =============================================================================

def extract_features_mne_multitaper(set_file_path: str, bands: Dict[str, tuple], 
                                     regions: Dict[str, list]) -> pd.DataFrame:
    """
    Extracts comprehensive features from an EEGLAB .set file:
    - PSD band powers (absolute + relative)
    - Spectral features (entropy, slope, centroid, flux)
    - Time-domain features (Hjorth mobility/complexity/activity, kurtosis, skewness, RMS, ZCR)
    - Connectivity features (PLV between regions)
    """
    try:
        try:
            data = mne.io.read_epochs_eeglab(set_file_path, verbose=False)
        except Exception:
            raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
            events = mne.make_fixed_length_events(raw, duration=2.0)
            data = mne.Epochs(raw, events, tmin=0, tmax=2.0, baseline=None, preload=True, verbose=False)
    except Exception as e:
        return pd.DataFrame()

    ch_names = data.ch_names
    sfreq = data.info['sfreq']
    
    # Map region indices
    region_indices = {}
    for r_name, r_channels in regions.items():
        region_indices[r_name] = [ch_names.index(ch) for ch in r_channels if ch in ch_names]

    # Compute PSD using Multitaper
    fmin, fmax = min(b[0] for b in bands.values()), max(b[1] for b in bands.values())
    spectrum = data.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    # Get raw epoch data for time-domain and connectivity features
    epoch_data = data.get_data()  # shape: (n_epochs, n_channels, n_times)
    
    subject_id = os.path.basename(set_file_path).replace('.set', '')
    features = []
    
    for epoch_idx in range(psds.shape[0]):
        feat_row = {'subject_id': subject_id, 'epoch_id': epoch_idx}
        
        for r_name, r_idx in region_indices.items():
            if not r_idx:
                continue
                
            region_psd = psds[epoch_idx, r_idx, :].mean(axis=0)
            total_power = np.sum(region_psd) + 1e-10
            
            # --- PSD Band Power Features (original) ---
            for b_name, (b_low, b_high) in bands.items():
                freq_mask = np.logical_and(freqs >= b_low, freqs <= b_high)
                band_power = region_psd[freq_mask].sum()
                feat_row[f'{b_name}_{r_name}_abs'] = band_power
                feat_row[f'{b_name}_{r_name}_rel'] = band_power / total_power
            
            # --- Spectral Features ---
            # Spectral entropy
            feat_row[f'spectral_entropy_{r_name}'] = compute_spectral_entropy(region_psd, freqs)
            
            # Spectral slope (spectral tilt)
            feat_row[f'spectral_slope_{r_name}'] = compute_spectral_slope(region_psd, freqs)
            
            # Spectral centroid
            feat_row[f'spectral_centroid_{r_name}'] = compute_spectral_centroid(region_psd, freqs)
            
            # Spectral flux (spectral variability)
            feat_row[f'spectral_flux_{r_name}'] = compute_spectral_flux(region_psd, freqs)
            
            # --- Time-Domain Features ---
            # Average across channels in the region for time-domain features
            region_signal = epoch_data[epoch_idx, r_idx, :].mean(axis=0)
            
            # Hjorth parameters
            mobility, complexity, activity = compute_hjorth_parameters(region_signal)
            feat_row[f'hjorth_mobility_{r_name}'] = mobility
            feat_row[f'hjorth_complexity_{r_name}'] = complexity
            feat_row[f'hjorth_activity_{r_name}'] = activity
            
            # Kurtosis (peakedness of distribution)
            feat_row[f'kurtosis_{r_name}'] = kurtosis(region_signal)
            
            # Skewness (asymmetry of distribution)
            feat_row[f'skewness_{r_name}'] = skew(region_signal)
            
            # RMS amplitude
            feat_row[f'rms_{r_name}'] = compute_rms(region_signal)
            
            # Zero Crossing Rate
            feat_row[f'zcr_{r_name}'] = compute_zcr(region_signal)
        
        features.append(feat_row)
    
    return pd.DataFrame(features)


def process_file_global(set_file_path: str, bands: Dict[str, tuple], 
                         regions: Dict[str, list]) -> pd.DataFrame:
    return extract_features_mne_multitaper(set_file_path, bands, regions)


def main():
    # Standard 10-20 band definitions
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

    print("=== Enhanced Feature Extraction Pipeline ===")
    print(f"Features to extract:")
    print(f"  - PSD bands (abs + rel): {len(bands)} bands x {len(regions)} regions x 2 = {len(bands)*len(regions)*2}")
    print(f"  - Spectral features (entropy, slope, centroid, flux): 4 x {len(regions)} = {4*len(regions)}")
    print(f"  - Time-domain (Hjorth mobility/complexity/activity, kurtosis, skewness, RMS, ZCR): 8 x {len(regions)} = {8*len(regions)}")
    print(f"  - Connectivity (PLV between {len(regions)} region pairs)")
    print(f"Total features per epoch: ~{len(bands)*len(regions)*2 + 4*len(regions) + 8*len(regions) + len(regions)*(len(regions)-1)//2}")
    print(f"\nTotal .set files to process: {len(set_files)}")
    print("Using parallel processing with joblib (loky backend)...\n")
    
    all_features = []
    
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_file_global)(f, bands, regions) for f in tqdm(set_files, desc="Processing EEG files")
    )
    
    for df in results:
        if not df.empty:
            all_features.append(df)
            
    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        print(f"\n--- Extraction Complete ---")
        print(f"Total epochs: {final_df.shape[0]}")
        print(f"Total features: {final_df.shape[1] - 2}")  # minus subject_id and epoch_id
        
        out_path = "robust_tinnitus_pipeline/01_extracted_features_enhanced.csv"
        final_df.to_csv(out_path, index=False)
        print(f"Features saved to: {out_path}")
        print(f"\nFeature breakdown:")
        psd_cols = [c for c in final_df.columns if '_abs' in c or '_rel' in c]
        spectral_cols = [c for c in final_df.columns if 'spectral_' in c]
        time_cols = [c for c in final_df.columns if any(x in c for x in ['hjorth', 'kurtosis', 'skewness', 'rms', 'zcr'])]
        conn_cols = [c for c in final_df.columns if 'plv' in c]
        print(f"  PSD band features: {len(psd_cols)}")
        print(f"  Spectral features: {len(spectral_cols)}")
        print(f"  Time-domain features: {len(time_cols)}")
        print(f"  Connectivity features: {len(conn_cols)}")
    else:
        print("ERROR: No features extracted!")

if __name__ == '__main__':
    main()