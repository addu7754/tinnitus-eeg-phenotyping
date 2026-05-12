"""
Riemannian Biomarker Extraction Pipeline
Generates topographical weights and Fréchet mean differences for clinical interpretation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

import mne
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

OUT_DIR = Path(__file__).resolve().parent / "riemannian_results"
CACHE_PATH = OUT_DIR / "zenodo_subject_covariances.npz"
LABELS_PATH = OUT_DIR / "selected_subject_phenotypes.csv"

def extract_biomarkers():
    if not CACHE_PATH.exists():
        print("Covariance cache missing. Run 04_riemannian_hardening.py first.")
        return

    # Load spatial covariance data
    cache = np.load(CACHE_PATH, allow_pickle=True)
    covs = cache["covariances"]
    bands = list(cache["bands"])
    channels = cache["channels"]
    subject_ids = cache["subject_ids"]

    # Load phenotype labels
    labels_df = pd.read_csv(LABELS_PATH)
    labels_dict = dict(zip(labels_df["subject_id"], labels_df["riemannian_subtype"]))
    
    # Filter arrays identically
    valid_idx = [i for i, sid in enumerate(subject_ids) if sid in labels_dict]
    covs = covs[valid_idx]
    y = np.array([labels_dict[sid] for sid in subject_ids[valid_idx]])

    print(f"Loaded {len(y)} subjects with labels.")

    # 1. Compute Haufe's Inverse Linear Weights in Tangent Space (per band for clarity)
    # We will use broadband or alpha as the primary example, or all bands
    print("\n--- Computing Linear Spatial Weights (Haufe's method) ---")
    results = []
    
    for band_idx, band_name in enumerate(bands):
        cov_band = covs[:, band_idx, :, :]
        ts = TangentSpace(metric='riemann')
        X_tangent = ts.fit_transform(cov_band)
        
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', LinearSVC(C=0.1, class_weight='balanced', dual="auto", random_state=42))
        ])
        clf.fit(X_tangent, y)
        
        # Haufe Transform: A = Cov(X) * W * Cov(S)^-1
        # For simple linear models, A ~ Cov(X) @ weights
        weights = clf.named_steps['svm'].coef_[0]
        cov_X = np.cov(X_tangent, rowvar=False)
        activation_pattern = cov_X.dot(weights)
        
        # The pyriemann tangent space vectorizes symmetric matrices such that the first N elements
        # correspond exactly to the diagonal entries (node variance).
        N_channels = len(channels)
        node_importance = np.abs(activation_pattern[:N_channels])
        
        top_indices = np.argsort(node_importance)[-5:][::-1]
        top_channels = [channels[i] for i in top_indices]
        
        results.append({
            'band': band_name,
            'top_nodes': ", ".join(top_channels),
            'top_importance_scores': ", ".join([f"{node_importance[i]:.4f}" for i in top_indices])
        })
        
        print(f"Band: {band_name:9s} | Top Electrodes driving separation: {', '.join(top_channels)}")

    pd.DataFrame(results).to_csv(OUT_DIR / "biomarker_haufe_weights.csv", index=False)

    # 2. Fréchet Mean differences
    print("\n--- Computing True Riemannian Fréchet Means ---")
    band_idx = bands.index("broadband")
    cov_broad = covs[:, band_idx, :, :]
    
    # We only take a subset if it's too slow computationally, but 130 is 
    # definitely tractable for univariate Fréchet means (not the K-means search).
    cov_group0 = cov_broad[y == 0]
    cov_group1 = cov_broad[y == 1]
    
    from pyriemann.utils.mean import mean_covariance
    print(f"Computing Geometric Center for Subtype 0 ({len(cov_group0)} subjects)...")
    mean_0 = mean_covariance(cov_group0, metric='riemann')
    print(f"Computing Geometric Center for Subtype 1 ({len(cov_group1)} subjects)...")
    mean_1 = mean_covariance(cov_group1, metric='riemann')
    
    # Riemannian distance between means
    dist = distance_riemann(mean_0, mean_1)
    print(f"Riemannian Tangent Distance between clinical subtypes: {dist:.4f}")
    
    # The actual spatial variance difference: 
    log_diff = np.log(np.diag(mean_1)) - np.log(np.diag(mean_0))
    top_diff_idx = np.argsort(np.abs(log_diff))[-10:][::-1]
    
    diff_report = [f"{channels[i]} (Diff: {log_diff[i]:.4f})" for i in top_diff_idx]
    print("\nTop Broadband Power DIFFERENCES (Log Ratio Subtype 1 / Subtype 0):")
    for r in diff_report:
        print(f" - {r}")
        
    np.savez_compressed(OUT_DIR / "riemannian_frechet_means.npz", 
                        mean_0=mean_0, mean_1=mean_1, channels=channels, log_diff=log_diff)
    print("\nBiomarker data exported fully for LaTeX Topography generation.")

if __name__ == "__main__":
    extract_biomarkers()
