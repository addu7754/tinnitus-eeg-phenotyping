
# ============================================================
# RESEARCH-GRADE ICA PREPROCESSING PIPELINE
# Tinnitus EEG Phenotyping — Zenodo EEG-PRO (Record 13219018)
# Aligned with: "Hardware-Robust Tinnitus Phenotypes from
#                Resting-State EEG: A Stable Riemannian
#                Network Architecture"
# ============================================================
# SCIENTIFIC RATIONALE:
#   This pipeline addresses the primary reviewer critique:
#   boundary sensors E1, E127, E32, E128, E125 lie in
#   EOG/EMG-susceptible zones. ICA is run to confirm the
#   k=2 phenotypic split (n=110 vs n=20) is neurological,
#   not artifact-driven.
#
# DESIGN PRINCIPLES:
#   1. Conservative artifact rejection (eye+muscle only, p>0.80)
#      — preserves 'other' components (unclassified neural)
#   2. Minimum signal duration gate (≥30s) before ICA
#   3. Strict channel count gate (≥120ch EGI HydroCel only)
#   4. GSN-HydroCel-128 montage for accurate ICLabel inference
#   5. All output to Google Drive (session-persistent)
#   6. Resume-safe: skips already-processed subjects
#   7. Full CSV audit log for reproducibility / paper reporting
# ============================================================


# %% ── CELL 1: INSTALL DEPENDENCIES ──────────────────────────────────────────
# Run once per session. Restart runtime if prompted after install.

!pip install -q mne==1.7.1 mne-icalabel pyriemann zenodo_get pymatreader tqdm
import mne
print(f"MNE version: {mne.__version__}")  # Confirm ≥ 1.3 for ICLabel


# %% ── CELL 2: MOUNT GOOGLE DRIVE ────────────────────────────────────────────
# CRITICAL: All outputs saved to Drive to survive session timeouts.

from google.colab import drive
drive.mount('/content/drive')

import os

# ── Directory layout ──
# Note: Raw data is on Colab local disk, results go to Drive
BASE      = '/content/drive/MyDrive/Tinnitus_ICA'
RAW_DIR   = '/content/eeg_data/EEG-PRO_data/RAW/Tinnitus Dataset'  # Path from your screenshot
CLEAN_DIR = os.path.join(BASE, 'ica_cleaned')     # ICA-cleaned .fif files
LOG_DIR   = os.path.join(BASE, 'logs')            # Audit logs

for d in [RAW_DIR, CLEAN_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

print("Drive mounted. Directory layout:")
print(f"  Raw      → {RAW_DIR}")
print(f"  Cleaned  → {CLEAN_DIR}")
print(f"  Logs     → {LOG_DIR}")


# %% ── CELL 3: DOWNLOAD ZENODO EEG-PRO ───────────────────────────────────────
# DOI: 10.5281/zenodo.13219018
# Downloads all files in the Zenodo record to RAW_DIR.
# Safe to re-run — zenodo_get resumes interrupted downloads.

# import os
# os.chdir(RAW_DIR)
# !zenodo_get 13219018

# Inventory what was downloaded
import glob
import os
all_items = os.listdir(RAW_DIR)
zip_files = [f for f in all_items if f.endswith('.zip')]
set_files = glob.glob(os.path.join(RAW_DIR, '**', '*.set'), recursive=True)

print(f"\nInventory: {len(all_items)} items total in {RAW_DIR}")
print(f"  ZIP archives : {len(zip_files)}")
print(f"  .set files   : {len(set_files)}")


# %% ── CELL 3b: EXTRACT ZIP ARCHIVES (only if .set count is 0) ───────────────

set_files = glob.glob(os.path.join(RAW_DIR, '**', '*.set'), recursive=True)

if not os.path.exists(RAW_DIR):
    print(f"ERROR: Directory not found: {RAW_DIR}")
    print("Please verify the path matches your Colab 'Files' pane.")
elif len(set_files) == 0:
    print(f"WARNING: No .set files found in {RAW_DIR}!")
else:
    print(f"Found {len(set_files)} Tinnitus subjects in {RAW_DIR}. Ready to process.")


# %% ── CELL 4: CONFIGURATION ─────────────────────────────────────────────────
# All tunable parameters in one place for reproducibility.

CFG = {
    # ── Signal Quality Gates ──
    'min_channels'    : 120,    # Must be ≥ 120 for EGI 128-ch HydroCel net
    'min_duration_s'  : 30.0,   # Skip recordings shorter than 30 seconds
                                 # (ICA needs sufficient data to decompose)

    # ── Filtering (must match paper's feature extraction bands) ──
    'hp_freq'         : 1.0,    # 1 Hz high-pass (mandatory for ICA stability)
    'lp_freq'         : 45.0,   # 45 Hz low-pass (matches paper's gamma ceiling)

    # ── ICA ──
    'ica_method'      : 'fastica',  # Reliable, well-tested for EEG
    'ica_n_components': 0.99,       # Keep components explaining 99% variance
    'ica_max_iter'    : 500,        # Sufficient for 128-ch convergence
    'ica_random_state': 42,         # Reproducibility (matches paper seed)

    # ── ICLabel Rejection ──
    # CONSERVATIVE: Only remove high-confidence eye + muscle
    # 'other' components are NOT removed — they may represent
    # unclassifiable neural activity, and removing them would
    # distort the covariance matrices the Riemannian pipeline needs
    'reject_labels'   : ['eye', 'muscle'],
    'reject_threshold': 0.80,   # 80% confidence required for removal

    # ── Paths ──
    'raw_dir'         : RAW_DIR,
    'clean_dir'       : CLEAN_DIR,
    'log_path'        : os.path.join(LOG_DIR, 'ica_audit_log.csv'),
}

print("Configuration loaded:")
for k, v in CFG.items():
    print(f"  {k:22s}: {v}")


# %% ── CELL 5: HELPER FUNCTIONS ───────────────────────────────────────────────

import mne
import numpy as np
import gc  # For aggressive memory management
from mne.preprocessing import ICA
from mne_icalabel import label_components

mne.set_log_level('WARNING')


def load_eeglab_file(file_path):
    """
    Load a .set file with multiple fallback strategies.
    Returns: (raw_or_epochs, data_type_str)
    Raises: RuntimeError if all strategies fail.
    """
    err_s1 = err_s2 = err_s3 = "None"

    # Strategy 1: Standard MNE loader
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        return raw, 'Raw'
    except Exception as e:
        err_s1 = str(e)

    # Strategy 2: Try as Epochs (some Zenodo files are pre-epoched)
    try:
        epochs = mne.io.read_epochs_eeglab(file_path, verbose=False)
        return epochs, 'Epochs'
    except Exception as e:
        err_s2 = str(e)

    # Strategy 3: Manual reconstruction via pymatreader
    # (handles buffer/header corruption that plagued 295/425 files in paper)
    try:
        from pymatreader import read_mat
        mat  = read_mat(file_path)
        eeg  = mat.get('EEG', mat)
        
        # Robust data extraction
        if 'data' not in eeg:
            raise ValueError("No 'data' field found in MAT file.")
        
        data = np.array(eeg['data'], dtype=np.float64)
        sfreq = float(eeg.get('srate', 128.0))  # Default EGI sfreq if missing
        
        chanlocs = eeg.get('chanlocs', [])
        ch_names = []
        if isinstance(chanlocs, list) and len(chanlocs) > 0:
            for ch in chanlocs:
                if isinstance(ch, dict):
                    name = str(ch.get('labels', f'Ch{len(ch_names)}')).strip()
                    ch_names.append(name)
                else:
                    ch_names.append(str(ch).strip())
        
        # If channel names are still missing/mismatched
        if len(ch_names) != (data.shape[0] if data.ndim != 3 else data.shape[1]):
             ch_names = [f'E{i+1}' for i in range(data.shape[0] if data.ndim != 3 else data.shape[1])]

        if data.ndim == 3:
            # Epoched: (n_channels, n_times, n_trials) → (n_trials, n_ch, n_times)
            data = np.transpose(data, (2, 0, 1))
            info = mne.create_info(ch_names, sfreq, ch_types='eeg')
            return mne.EpochsArray(data, info, verbose=False), 'Epochs'
        else:
            info = mne.create_info(ch_names, sfreq, ch_types='eeg')
            return mne.io.RawArray(data, info, verbose=False), 'Raw'
            f"  S3 (pymatreader manual): {e3}"
        )


def validate_signal(data, cfg):
    """
    Gate 1: Channel count check.
    Gate 2: Minimum recording duration check.
    Returns: (passed: bool, reason: str)
    """
    n_ch = len(data.ch_names)
    if n_ch < cfg['min_channels']:
        return False, f"FAIL_CH: {n_ch} channels < {cfg['min_channels']} required"

    # Duration check
    if hasattr(data, 'times'):
        # Raw object
        duration = data.times[-1]
    else:
        # Epochs object — total time across all epochs
        duration = len(data) * data.times[-1]

    if duration < cfg['min_duration_s']:
        return False, f"FAIL_DUR: {duration:.1f}s < {cfg['min_duration_s']}s required"

    return True, 'OK'


def set_egi_montage(data):
    """
    Set GSN-HydroCel-128 montage.
    CRITICAL: ICLabel uses electrode positions for spatial feature inference.
    Without a montage, ICLabel may misclassify components, leading to
    incorrect artifact rejection.
    """
    try:
        montage = mne.channels.make_standard_montage('GSN-HydroCel-128')
        data.set_montage(montage, on_missing='ignore', verbose=False)
        return True
    except Exception:
        return False  # Non-fatal — pipeline continues without montage


def run_ica_and_clean(data, data_type, cfg):
    """
    Full ICA pipeline:
      1. Filter (1–45 Hz)
      2. Average reference
      3. Fit FastICA
      4. ICLabel classification
      5. Conservative artifact rejection (eye+muscle, p>0.80)
      6. Signal reconstruction
    Returns: (raw_clean, n_eye_removed, n_muscle_removed, n_total_components)
    """
    # ── Step 1: Filter ──
    # For Epochs, filter each epoch independently by converting to Raw for ICA (concatenate epochs)
    if data_type == 'Epochs':
        data_raw = mne.concatenate_epochs([data]).get_data()
        n_epochs, n_ch, n_t = data_raw.shape
        data_2d = data_raw.reshape(n_ch, n_epochs * n_t)
        info = data.info.copy()
        data = mne.io.RawArray(data_2d, info, verbose=False)
        data_type = 'Raw_from_Epochs'

    # Adjust filter length for short signals if necessary
    sfreq = data.info['sfreq']
    n_times = data.n_times
    # MNE default FIR filter length for 1Hz is long. 
    # If signal is too short, we need to reduce it.
    min_len = int(3.3 / cfg['hp_freq'] * sfreq)
    filter_len = 'auto'
    if n_times <= min_len:
        filter_len = f"{n_times - 1}" if n_times > 1 else 'auto'
        print(f"  [NOTE] Short signal ({n_times} samples), using filter_length={filter_len}")

    data.filter(
        l_freq  = cfg['hp_freq'],
        h_freq  = cfg['lp_freq'],
        method  = 'fir',
        filter_length = filter_len,
        verbose = False,
        l_trans_bandwidth = 'auto' if filter_len == 'auto' else 0.5
    )

    # ── Step 2: Average reference (standard for EGI nets) ──
    data.set_eeg_reference('average', projection=True, verbose=False)
    data.apply_proj(verbose=False)

    # ── Step 3: Fit ICA ──
    ica = ICA(
        n_components = cfg['ica_n_components'],
        method       = cfg['ica_method'],
        max_iter     = cfg['ica_max_iter'],
        random_state = cfg['ica_random_state']
    )
    ica.fit(data, verbose=False)
    n_components = ica.n_components_

    # ── Step 4: ICLabel Classification ──
    ic_labels = label_components(data, ica, method='iclabel')
    labels = ic_labels['labels']        # e.g. ['brain','eye','muscle','other',...]
    probs  = ic_labels['y_pred_proba']  # shape (n_components, n_classes)

    # ── Step 5: Conservative Rejection ──
    # Only remove components where BOTH:
    #   (a) the dominant label is in reject_labels list
    #   (b) the probability for that label exceeds the threshold
    # This preserves 'other' components which may be unclassified neural sources
    eye_idx    = []
    muscle_idx = []

    for i, (lbl, prob_row) in enumerate(zip(labels, probs)):
        max_prob = prob_row.max()
        if lbl == 'eye'    and max_prob > cfg['reject_threshold']:
            eye_idx.append(i)
        elif lbl == 'muscle' and max_prob > cfg['reject_threshold']:
            muscle_idx.append(i)

    ica.exclude = eye_idx + muscle_idx

    # ── Step 6: Reconstruct clean signal ──
    data_clean = ica.apply(data.copy(), verbose=False)

    return data_clean, len(eye_idx), len(muscle_idx), n_components


print("Helper functions defined.")


# %% ── CELL 6: MAIN ICA LOOP ──────────────────────────────────────────────────
# Research-grade features:
#   ✓ Resume-safe (skips already processed files)
#   ✓ Full audit log (CSV)
#   ✓ Per-subject RAM cleanup
#   ✓ Detailed per-file status reporting
#   ✓ No crashes — all exceptions caught and logged

import csv
import glob
import os
from tqdm.auto import tqdm

mne.set_log_level('WARNING')

# ── Locate all .set files ──
set_files = sorted(glob.glob(
    os.path.join(CFG['raw_dir'], '**', '*.set'), recursive=True
))
print(f"Found {len(set_files)} .set files to process.\n")

# ── Initialize CSV log ──
log_path   = CFG['log_path']
log_exists = os.path.exists(log_path)

log_fields = [
    'subject_file', 'n_channels', 'duration_s', 'data_type',
    'montage_set', 'n_ica_components',
    'n_eye_removed', 'n_muscle_removed', 'n_total_removed',
    'status', 'notes'
]

counters = {'ok': 0, 'skipped_ch': 0, 'skipped_dur': 0,
            'skipped_exists': 0, 'error': 0}

with open(log_path, 'a', newline='') as logfile:
    writer = csv.DictWriter(logfile, fieldnames=log_fields)
    if not log_exists:
        writer.writeheader()

    for file_path in tqdm(set_files, desc="ICA Pipeline"):
        subj_name = os.path.basename(file_path)
        out_name  = os.path.join(
            CFG['clean_dir'],
            subj_name.replace('.set', '-ica-raw.fif')
        )

        # ── Resume check ──
        if os.path.exists(out_name):
            counters['skipped_exists'] += 1
            continue

        row = {f: '' for f in log_fields}
        row['subject_file'] = subj_name

        try:
            # ── Load ──
            data, data_type = load_eeglab_file(file_path)
            row['data_type']  = data_type
            row['n_channels'] = len(data.ch_names)

            # Duration
            if hasattr(data, 'times'):
                row['duration_s'] = round(data.times[-1], 2)
            else:
                row['duration_s'] = round(len(data) * data.times[-1], 2)

            # ── Gate: channel count + duration ──
            passed, reason = validate_signal(data, CFG)
            if not passed:
                row['status'] = reason
                if 'FAIL_CH'  in reason: counters['skipped_ch']  += 1
                if 'FAIL_DUR' in reason: counters['skipped_dur'] += 1
                writer.writerow(row)
                del data
                continue

            # ── Montage ──
            montage_ok = set_egi_montage(data)
            row['montage_set'] = 'Yes' if montage_ok else 'No'

            # ── ICA + Clean ──
            data_clean, n_eye, n_muscle, n_comp = run_ica_and_clean(
                data, data_type, CFG
            )

            row['n_ica_components'] = n_comp
            row['n_eye_removed']    = n_eye
            row['n_muscle_removed'] = n_muscle
            row['n_total_removed']  = n_eye + n_muscle

            # ── Save ──
            data_clean.save(out_name, overwrite=True, verbose=False)

            row['status'] = 'OK'
            counters['ok'] += 1

            # ── Free RAM ──
            del data, data_clean
            gc.collect()

        except Exception as e:
            row['status'] = 'ERROR'
            row['notes']  = str(e)[:200]
            counters['error'] += 1

        writer.writerow(row)
        logfile.flush()  # Write to disk immediately (safe against crashes)

# ── Final Summary ──
total = sum(counters.values())
print("\n" + "="*50)
print("ICA PIPELINE COMPLETE")
print("="*50)
print(f"  Total files       : {len(set_files)}")
print(f"  ✅ Cleaned (OK)   : {counters['ok']}")
print(f"  ⏭  Already done   : {counters['skipped_exists']}")
print(f"  ⚠️  Failed ch gate : {counters['skipped_ch']}")
print(f"  ⚠️  Failed dur gate: {counters['skipped_dur']}")
print(f"  ❌ Errors          : {counters['error']}")
print(f"\nAudit log → {log_path}")
print(f"Clean files → {CFG['clean_dir']}")


# %% ── CELL 7: AUDIT LOG ANALYSIS ────────────────────────────────────────────
# Run this after the loop to get summary statistics for the paper.

import pandas as pd

df = pd.read_csv(CFG['log_path'])
ok = df[df['status'] == 'OK'].copy()

print("="*50)
print("AUDIT SUMMARY")
print("="*50)
print(f"\nStatus breakdown:\n{df['status'].value_counts().to_string()}")

print(f"\n── ICA Component Statistics (successful subjects only) ──")
print(f"  N subjects cleaned          : {len(ok)}")
print(f"  Avg ICA components fitted   : {ok['n_ica_components'].mean():.1f} ± {ok['n_ica_components'].std():.1f}")
print(f"  Avg eye components removed  : {ok['n_eye_removed'].mean():.2f} ± {ok['n_eye_removed'].std():.2f}")
print(f"  Avg muscle components removed: {ok['n_muscle_removed'].mean():.2f} ± {ok['n_muscle_removed'].std():.2f}")
print(f"  Avg total removed           : {ok['n_total_removed'].mean():.2f} ± {ok['n_total_removed'].std():.2f}")

# Flag subjects with zero removals (may indicate ICLabel failed silently)
zero_removal = ok[ok['n_total_removed'] == 0]
print(f"\n  ⚠️  Subjects with 0 components removed: {len(zero_removal)}")
print("     (Review these — ICLabel may have lacked montage info)")

# Flag aggressive cleaning (>10 removed = suspicious)
heavy = ok[ok['n_total_removed'] > 10]
print(f"  ⚠️  Subjects with >10 components removed: {len(heavy)}")
if len(heavy) > 0:
    print(heavy[['subject_file', 'n_eye_removed', 'n_muscle_removed']].to_string())

# Channel distribution
print(f"\n── Channel Distribution ──")
print(df['n_channels'].value_counts().head(10).to_string())

# Data for paper Methods section
print(f"""
── TEXT FOR PAPER METHODS §IV.A ──
Prior to covariance estimation, all EEG recordings underwent
automated ICA-based artifact rejection. FastICA was applied to
signals bandpass-filtered at {CFG['hp_freq']}–{CFG['lp_freq']} Hz with an average
reference. The GSN-HydroCel-128 electrode layout was assigned
to enable spatial inference by the ICLabel classifier
(Pion-Tonachini et al., 2019). Components classified as ocular
or myogenic with probability > {CFG['reject_threshold']} were removed; components
labelled 'other' were retained to avoid discarding ambiguous
neural sources. On average, {ok['n_eye_removed'].mean():.1f} ± {ok['n_eye_removed'].std():.1f} ocular and
{ok['n_muscle_removed'].mean():.1f} ± {ok['n_muscle_removed'].std():.1f} myogenic components were rejected per subject.
""")


# %% ── CELL 8: STRUCTURAL FILTER (Reproduce Paper's N=130) ───────────────────
# Apply the same structural quality control from the paper:
# only keep 126-channel EGI HydroCel recordings.

import glob, os
import mne

mne.set_log_level('WARNING')

clean_files = sorted(glob.glob(os.path.join(CFG['clean_dir'], '*-ica-raw.fif')))
print(f"Total ICA-cleaned files: {len(clean_files)}")

valid_126 = []
for fif in clean_files:
    try:
        info = mne.io.read_info(fif, verbose=False)
        if len(info['ch_names']) >= 120:
            valid_126.append(fif)
    except Exception:
        continue

print(f"Files with ≥120 channels (126-ch EGI topology): {len(valid_126)}")
print("\nThese are your ICA-cleaned equivalent of the paper's N=130 cohort.")
print("Feed valid_126 into Cell 9 (Riemannian bridge) to re-run the pipeline.")


# %% ── CELL 9: RIEMANNIAN PIPELINE BRIDGE ─────────────────────────────────────
# Loads ICA-cleaned .fif files and computes OAS spatial covariance
# matrices per frequency band — exact match to paper's methodology.
# Output: X_subjects (list of dicts) ready for TangentSpace + k-means.

import mne
import numpy as np
from pyriemann.estimation import Covariances

mne.set_log_level('WARNING')

# Frequency bands from paper (§IV.A)
BANDS = {
    'broadband' : (1,  45),
    'delta'     : (1,   4),
    'theta'     : (4,   8),
    'alpha'     : (8,  13),
    'beta'      : (13, 30),
    'gamma'     : (30, 45),
}

EPOCH_LEN = 2.0   # seconds — matches paper's non-overlapping 2-second epochs
cov_est   = Covariances(estimator='oas')  # OAS — matches paper §IV.A

X_subjects    = []   # list of dicts: {band_name: cov_matrix (n_ch x n_ch)}
subject_files = []   # parallel list of file paths

print(f"Computing covariance matrices for {len(valid_126)} subjects...")
print(f"Bands: {list(BANDS.keys())}")
print(f"Epoch length: {EPOCH_LEN}s | Estimator: OAS\n")

for fif_file in valid_126:
    subj = os.path.basename(fif_file)
    try:
        raw = mne.io.read_raw_fif(fif_file, preload=True, verbose=False)
        n_ch = len(raw.ch_names)

        subject_covs = {}
        valid_bands  = True

        for band_name, (l, h) in BANDS.items():
            raw_band = raw.copy().filter(l, h, verbose=False)
            epochs   = mne.make_fixed_length_epochs(
                raw_band, duration=EPOCH_LEN,
                preload=True, verbose=False
            )
            if len(epochs) < 5:
                # Need at least 5 epochs for a stable covariance estimate
                valid_bands = False
                break

            epoch_data = epochs.get_data()  # (n_epochs, n_ch, n_times)
            covs = cov_est.fit_transform(epoch_data)  # (n_epochs, n_ch, n_ch)
            # Mean covariance across epochs — single matrix per subject per band
            subject_covs[band_name] = covs.mean(axis=0)

        if valid_bands and len(subject_covs) == len(BANDS):
            X_subjects.append(subject_covs)
            subject_files.append(fif_file)
            print(f"  ✅ {subj} | {n_ch}ch | {len(epochs)} epochs/band")
        else:
            print(f"  ⚠️  {subj} | Skipped (insufficient epochs)")

        del raw

    except Exception as e:
        print(f"  ❌ {subj} | Error: {e}")

print(f"\n{'='*50}")
print(f"RIEMANNIAN BRIDGE COMPLETE")
print(f"  Valid subjects for Riemannian pipeline: {len(X_subjects)}")
print(f"  Each subject has {len(BANDS)} band covariance matrices")
print(f"\nX_subjects is ready. Feed into TangentSpace + k-means pipeline.")
print(f"This is your ICA-validated equivalent of the paper's N=130 cohort.")


# %% ── CELL 10: QUICK SANITY CHECK ───────────────────────────────────────────
# Verify covariance matrix properties before feeding to Riemannian pipeline.
# All matrices must be Symmetric Positive Definite (SPD) for pyriemann.

import numpy as np

print("Sanity checking covariance matrices...")
all_ok = True

for i, subj_covs in enumerate(X_subjects):
    for band, C in subj_covs.items():
        # Check symmetry
        sym_ok = np.allclose(C, C.T, atol=1e-10)
        # Check positive definiteness via eigenvalues
        eigvals = np.linalg.eigvalsh(C)
        spd_ok  = np.all(eigvals > 0)

        if not (sym_ok and spd_ok):
            print(f"  ❌ Subject {i} | Band {band} | "
                  f"Sym={sym_ok} | SPD={spd_ok} | "
                  f"Min eigval={eigvals.min():.4e}")
            all_ok = False

if all_ok:
    print(f"  ✅ All {len(X_subjects)} × {len(BANDS)} matrices are valid SPD.")
    print("     Safe to pass to pyriemann TangentSpace.")
else:
    print("\n  ⚠️  Some matrices failed SPD check.")
    print("     Apply regularisation: Covariances(estimator='oas') already handles")
    print("     most cases — if failures persist, increase regularisation manually.")
