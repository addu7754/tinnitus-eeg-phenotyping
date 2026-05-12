# Objective Tinnitus Phenotyping via Resting-State EEG

This repository contains the computational framework and data-driven phenotyping pipeline for identifying robust neurophysiological subtypes of chronic tinnitus using resting-state EEG. The methodology employs a lightweight Side-Aware Meta-Learning (SMeta-lite) spatial gating mechanism to resolve domain shift induced by anatomical laterality and hardware variance.

## Pipeline Overview

The computational pipeline consists of two major phases:
1. **Unsupervised Phenotyping**: K-means clustering ($k=2$) applied to PCA-reduced spectral EEG features, validated via nested cross-validation and non-parametric bootstrap resampling.
2. **Supervised Biomarker Discovery**: Support Vector Machine (SVM) combined with SHAP permutation importance to identify the primary neuro-oscillatory drivers (e.g., $\beta_2$-Parietal, $\alpha_2$-Frontal) separating the sensory-dominant and distress-dominant phenotypes.

## Datasets

Due to GitHub's file size limitations, the raw EEG datasets are not hosted in this repository. To reproduce the analysis, please download the datasets from their original repositories and place them in the appropriate directories.

1. **Primary Dataset (Mendeley Cohort)**
   - *Title*: Characterization of Tinnitus Through the Analysis of Electroencephalographic Activity
   - *Description*: High-density 128-channel BioSemi ActiveTwo resting-state EEG recordings ($N=87$ recordings from 22 patients).
   - *Source*: [Mendeley Data](https://data.mendeley.com/datasets/)

2. **Secondary Dataset (PhysioNet Cohort)**
   - *Title*: Acoustic Therapies for Tinnitus Treatment An EEG Database
   - *Description*: 64-channel Brain Products BrainAmp resting-state EEG recordings ($N=44$ patients) used for cross-cohort domain generalization.
   - *Source*: [PhysioNet](https://physionet.org/)

## Manuscripts

The fully validated manuscripts formatted for IEEE publication are included in this repository:
- `IEEE_Journal_Paper/`
- `IEEE_Conference_Paper/`

## Requirements

The experiments were run using Python 3.x with the following core dependencies:
- `numpy`
- `pandas`
- `scikit-learn`
- `mne`
- `matplotlib`

## License

This academic project is open for research and replication purposes.
