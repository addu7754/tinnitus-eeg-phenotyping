import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("--- Running Bit 2: Strict Unsupervised Clustering Evaluation ---")
    df = pd.read_csv("robust_tinnitus_pipeline/01_extracted_features.csv")
    
    initial_len = len(df)
    df = df.dropna()
    print(f"Dropped {initial_len - len(df)} rows containing NaNs.")
    
    # We only cluster on the features, not the IDs.
    feature_cols = [c for c in df.columns if c not in ['subject_id', 'epoch_id']]
    X_raw = df[feature_cols].copy()
    
    # Create the Scikit-Learn Pipeline
    # 1. Zero-leakage scaling (StandardScaler computes mean/std strictly on what it is given)
    # 2. PCA to remove colinearity (95% variance)
    # 3. K-Means clustering (testing k=2,3,4)
    
    # We are evaluating internal cluster metrics *strictly* after processing.
    # No Support Vector Machines are used here for validation to avoid circular logic.
    
    results = []
    
    for k in [2, 3, 4, 5]:
        print(f"\nEvaluating K={k}...")
        
        # Define pipeline structure up to clustering
        preprocessor = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=42))
        ])
        
        # Transform data through the pipeline securely
        X_processed = preprocessor.fit_transform(X_raw)
        
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_processed)
        
        # Since this dataset is quite large (180k epochs), computing silhouette score 
        # on the entire set can take hours. We will use a random stratified sub-sample of 20,000 
        # epochs *only* for the silhouette metric calculation to save time, preserving labels.
        if len(X_processed) > 20000:
            sample_indices = np.random.choice(len(X_processed), size=20000, replace=False)
            x_samp = X_processed[sample_indices]
            l_samp = labels[sample_indices]
        else:
            x_samp = X_processed
            l_samp = labels
            
        # Calculate Unsupervised Metrics
        silhouette = silhouette_score(x_samp, l_samp)
        # Davies-Bouldin is faster and can usually run on the full dataset without issue
        db_score = davies_bouldin_score(X_processed, labels)
        
        print(f"Silhouette Score: {silhouette:.3f} (closer to 1 is better)")
        print(f"Davies-Bouldin Index: {db_score:.3f} (closer to 0 is better)")
        
        results.append({
            'k': k,
            'silhouette': silhouette,
            'davies_bouldin': db_score,
            'inertia': kmeans.inertia_
        })

    # Save cluster evaluation results
    res_df = pd.DataFrame(results)
    res_df.to_csv("robust_tinnitus_pipeline/02_clustering_metrics.csv", index=False)
    print("\n--- Clustering Evaluation Complete ---")
    print("Metrics saved to robust_tinnitus_pipeline/02_clustering_metrics.csv")
    
if __name__ == '__main__':
    main()