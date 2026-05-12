import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path

OUT_DIR = Path('d:/Adnan_Project/gn7754/robust_tinnitus_pipeline/riemannian_results')
FIG_DIR = Path('d:/Adnan_Project/gn7754/IEEE_Conference_Paper')

# 1. PCA Scatter Plot
features_df = pd.read_csv(OUT_DIR / 'zenodo_filterbank_tangent_features.csv')
labels_df = pd.read_csv(OUT_DIR / 'selected_subject_phenotypes.csv')

# Only broadband or all features? We can just do PCA on all features or just broadband.
broadband_cols = [c for c in features_df.columns if c.startswith('broadband')]
X = features_df[broadband_cols].values
y = labels_df['riemannian_subtype'].values

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=['#1f77b4', '#ff7f0e'], s=100, alpha=0.8, edgecolor='w')
plt.title('Riemannian Tangent Space (PCA Projection)\nn=110 vs n=20 Phenotypes', fontsize=14, pad=15)
plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
# Ensure we check unique y to label properly
unique_y = np.unique(y)
labels = ['Primary (n=110)', 'Secondary (n=20)']
plt.legend(title='Subtype', labels=labels, loc='upper right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'pca_scatter.png', dpi=300)
plt.close()

# 2. Haufe Biomarkers Bar Chart
biomarkers = pd.read_csv(OUT_DIR / 'biomarker_haufe_weights.csv')
broadband_row = biomarkers[biomarkers['band'] == 'broadband'].iloc[0]
nodes = broadband_row['top_nodes'].split(', ')
scores = [float(s) for s in broadband_row['top_importance_scores'].split(', ')]

plt.figure(figsize=(8, 5))
# Set palette to match seaborn version without warning
sns.barplot(x=nodes, y=scores, hue=nodes, legend=False, palette='viridis')
plt.title('Top Spatial Network Drivers (Haufe Transformed)', fontsize=14, pad=15)
plt.xlabel('EGI HydroCel EEG Sensor', fontsize=12)
plt.ylabel('Haufe Forward Activation Weight', fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / 'haufe_biomarkers.png', dpi=300)
plt.close()

print('Figures generated successfully in IEEE_Conference_Paper!')
