import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(__file__)
FEATURE_PATH = os.path.join(BASE_DIR, "zenodo_01_extracted_features.csv")
OUT_PATH = os.path.join(BASE_DIR, "zenodo_02_clustering_metrics.csv")


def load_subject_level_relative_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)
    rel_cols = [c for c in df.columns if c.endswith("_rel")]

    if not rel_cols:
        raise ValueError("No relative-power columns found. Run 01_zenodo_feature_extraction.py first.")

    before = len(df)
    df = df.dropna(subset=["subject_id", *rel_cols])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with missing relative-power features.")

    subject_df = df.groupby("subject_id", as_index=False)[rel_cols].mean()
    return subject_df


def main():
    print("--- Zenodo Bit 2: Subject-Level Unsupervised Tinnitus Subtyping ---")
    subject_df = load_subject_level_relative_features()
    feature_cols = [c for c in subject_df.columns if c != "subject_id"]

    print(
        f"Clustering {len(subject_df)} unique tinnitus subjects using "
        f"{len(feature_cols)} relative spectral features."
    )

    preprocessor = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ]
    )
    x_processed = preprocessor.fit_transform(subject_df[feature_cols])
    print(f"PCA retained {x_processed.shape[1]} components at 95% variance.")

    results = []
    for k in [2, 3, 4, 5]:
        print(f"\nEvaluating subject phenotypes: k={k}")
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = kmeans.fit_predict(x_processed)

        silhouette = silhouette_score(x_processed, labels)
        db_score = davies_bouldin_score(x_processed, labels)
        sizes = np.bincount(labels, minlength=k).tolist()

        print(f"  Silhouette:    {silhouette:.4f}")
        print(f"  Davies-Bouldin:{db_score:.4f}")
        print(f"  Cluster sizes: {sizes}")

        results.append(
            {
                "k": k,
                "n_subjects": len(subject_df),
                "n_features": len(feature_cols),
                "pca_components": x_processed.shape[1],
                "silhouette": silhouette,
                "davies_bouldin": db_score,
                "inertia": kmeans.inertia_,
                "cluster_sizes": "|".join(map(str, sizes)),
            }
        )

    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"\nSaved subject-level clustering metrics to: {OUT_PATH}")


if __name__ == "__main__":
    main()
