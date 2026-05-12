"""
Scientifically hardened Zenodo phenotype validation.

This replaces the previous 4-class epoch-level benchmark with a patient-level
analysis that matches the conference paper's core claim: stable binary tinnitus
phenotypes discovered from resting-state EEG spectral topology.

Key safeguards:
- Aggregates windows/epochs to subject-level features before phenotyping.
- Uses relative power by default to reduce hardware/amplitude confounding.
- Fits imputation, scaling, PCA, and KMeans inside each validation fold.
- Evaluates classifier agreement with train-only propagated cluster labels.
- Reports dummy baselines, cluster stability, and permutation-tested silhouette.
"""

from __future__ import annotations

import itertools
import os
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = os.path.dirname(__file__)
FEATURE_PATH = os.path.join(BASE_DIR, "zenodo_01_extracted_features.csv")
OUT_DIR = os.path.join(BASE_DIR, "hardened_results")
FINAL_REPORT_PATH = os.path.join(BASE_DIR, "ZENODO_FINAL_METRICS.txt")


@dataclass
class FoldMetrics:
    classifier: str
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    adjusted_rand: float
    test_cluster_sizes: str


def log(message: str, report: list[str]) -> None:
    print(message)
    report.append(message)


def load_subject_features() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURE_PATH)
    df = df.replace([np.inf, -np.inf], np.nan)

    rel_cols = [c for c in df.columns if c.endswith("_rel")]
    if not rel_cols:
        raise ValueError("No *_rel columns found. Run the Zenodo feature extraction step first.")

    before = len(df)
    df = df.dropna(subset=["subject_id", *rel_cols])
    if before != len(df):
        print(f"Dropped {before - len(df)} rows with missing relative-power values.")

    subject_df = df.groupby("subject_id", as_index=False)[rel_cols].mean()
    return subject_df, rel_cols


def make_unsupervised_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ]
    )


def make_classifiers() -> dict[str, Pipeline | DummyClassifier]:
    return {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearSVC(
                        C=0.1,
                        class_weight="balanced",
                        dual="auto",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=1.0,
                        gamma="scale",
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def align_labels(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    best_score = -1.0
    best = y_pred
    for perm in itertools.permutations(range(n_classes)):
        mapped = np.array([perm[int(label)] for label in y_pred])
        score = (mapped == y_true).mean()
        if score > best_score:
            best_score = score
            best = mapped
    return best


def k_scan(x_processed: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE).fit_predict(x_processed)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(x_processed, labels),
                "davies_bouldin": davies_bouldin_score(x_processed, labels),
                "cluster_sizes": "|".join(map(str, np.bincount(labels, minlength=k).tolist())),
            }
        )
    return pd.DataFrame(rows)


def permutation_silhouette_p_value(
    x_processed: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1000,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    observed = silhouette_score(x_processed, labels)
    null_scores = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        null_scores.append(silhouette_score(x_processed, shuffled))
    null = np.asarray(null_scores)
    p_value = (np.sum(null >= observed) + 1.0) / (len(null) + 1.0)
    return observed, float(null.mean()), float(p_value)


def mean_best_jaccard(y_ref: np.ndarray, y_boot: np.ndarray, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        ref_set = set(np.where(y_ref == label)[0])
        best = 0.0
        for other in range(n_classes):
            boot_set = set(np.where(y_boot == other)[0])
            union = ref_set | boot_set
            if union:
                best = max(best, len(ref_set & boot_set) / len(union))
        scores.append(best)
    return float(np.mean(scores))


def bootstrap_stability(
    x_subjects: pd.DataFrame,
    reference_labels: np.ndarray,
    n_clusters: int = 2,
    n_bootstrap: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    n_subjects = len(x_subjects)

    for _ in range(n_bootstrap):
        sample_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        unique_idx = np.unique(sample_idx)

        preprocessor = make_unsupervised_preprocessor()
        x_boot = preprocessor.fit_transform(x_subjects.iloc[sample_idx])
        boot_labels_all = KMeans(
            n_clusters=n_clusters,
            n_init=50,
            random_state=RANDOM_STATE,
        ).fit_predict(x_boot)

        first_boot_label = {}
        for position, original_idx in enumerate(sample_idx):
            first_boot_label.setdefault(original_idx, boot_labels_all[position])

        y_ref = reference_labels[unique_idx]
        y_boot = np.array([first_boot_label[idx] for idx in unique_idx])
        scores.append(mean_best_jaccard(y_ref, y_boot, n_clusters))

    return float(np.mean(scores)), float(np.std(scores))


def evaluate_global_label_separability(
    x_subjects: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    rows = []

    for name, model in make_classifiers().items():
        results = cross_validate(
            model,
            x_subjects,
            labels,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        rows.append(
            {
                "classifier": name,
                "accuracy_mean": results["test_accuracy"].mean(),
                "accuracy_std": results["test_accuracy"].std(),
                "balanced_accuracy_mean": results["test_balanced_accuracy"].mean(),
                "f1_macro_mean": results["test_f1_macro"].mean(),
            }
        )
    return pd.DataFrame(rows)


def evaluate_train_only_cluster_propagation(
    x_subjects: pd.DataFrame,
    n_clusters: int = 2,
) -> pd.DataFrame:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[FoldMetrics] = []

    for name, model in make_classifiers().items():
        if name == "dummy_majority":
            continue

        for train_idx, test_idx in cv.split(x_subjects):
            x_train = x_subjects.iloc[train_idx]
            x_test = x_subjects.iloc[test_idx]

            preprocessor = make_unsupervised_preprocessor()
            x_train_pca = preprocessor.fit_transform(x_train)
            x_test_pca = preprocessor.transform(x_test)

            kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=RANDOM_STATE)
            y_train = kmeans.fit_predict(x_train_pca)
            y_test = kmeans.predict(x_test_pca)

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred_aligned = align_labels(y_test, y_pred, n_clusters)

            rows.append(
                FoldMetrics(
                    classifier=name,
                    accuracy=float((y_pred_aligned == y_test).mean()),
                    balanced_accuracy=float(balanced_accuracy_score(y_test, y_pred_aligned)),
                    f1_macro=float(f1_score(y_test, y_pred_aligned, average="macro", zero_division=0)),
                    adjusted_rand=float(adjusted_rand_score(y_test, y_pred)),
                    test_cluster_sizes="|".join(map(str, np.bincount(y_test, minlength=n_clusters).tolist())),
                )
            )

    return pd.DataFrame([row.__dict__ for row in rows])


def biomarker_table(
    subject_df: pd.DataFrame,
    feature_cols: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    x = subject_df[feature_cols]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(x, labels)
    coef = model.named_steps["classifier"].coef_[0]
    means = subject_df.assign(subtype=labels).groupby("subtype")[feature_cols].mean()

    rows = []
    for feature, weight in zip(feature_cols, coef):
        direction = "higher_in_subtype_1" if means.loc[1, feature] > means.loc[0, feature] else "higher_in_subtype_0"
        rows.append(
            {
                "feature": feature,
                "abs_logistic_weight": abs(float(weight)),
                "signed_logistic_weight": float(weight),
                "subtype_0_mean": float(means.loc[0, feature]),
                "subtype_1_mean": float(means.loc[1, feature]),
                "direction": direction,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_logistic_weight", ascending=False)


def summarize_fold_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_df.groupby("classifier")
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            adjusted_rand_mean=("adjusted_rand", "mean"),
            adjusted_rand_std=("adjusted_rand", "std"),
        )
        .reset_index()
        .sort_values("balanced_accuracy_mean", ascending=False)
    )


def format_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    return df.loc[:, list(columns)].to_string(index=False, float_format=lambda x: f"{x:.4f}")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    report: list[str] = []

    log("=" * 72, report)
    log(" ZENODO SUBJECT-LEVEL PHENOTYPE VALIDATION (HARDENED)", report)
    log("=" * 72, report)

    subject_df, feature_cols = load_subject_features()
    x_subjects = subject_df[feature_cols]
    log(f"Subjects: {len(subject_df)}", report)
    log(f"Feature space: {len(feature_cols)} relative-power features", report)
    log("Validation unit: subject, not epoch/window", report)

    preprocessor = make_unsupervised_preprocessor()
    x_pca = preprocessor.fit_transform(x_subjects)
    log(f"PCA components retained: {x_pca.shape[1]} (95% variance)", report)

    scan = k_scan(x_pca)
    scan.to_csv(os.path.join(OUT_DIR, "k_scan_subject_level.csv"), index=False)
    log("\nK scan:", report)
    log(format_table(scan, ["k", "silhouette", "davies_bouldin", "cluster_sizes"]), report)

    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(x_pca)
    cluster_sizes = np.bincount(labels, minlength=n_clusters).tolist()

    observed_sil, null_sil, p_value = permutation_silhouette_p_value(x_pca, labels)
    mean_jaccard, std_jaccard = bootstrap_stability(
        x_subjects,
        labels,
        n_clusters=n_clusters,
        n_bootstrap=500,
    )

    log("\nPrimary k=2 phenotype result:", report)
    log(f"  Cluster sizes: {cluster_sizes}", report)
    log(f"  Silhouette: {observed_sil:.4f}", report)
    log(f"  Permutation null silhouette mean: {null_sil:.4f}", report)
    log(f"  Permutation p-value: {p_value:.4f}", report)
    log(f"  Bootstrap Jaccard stability: {mean_jaccard:.4f} +/- {std_jaccard:.4f}", report)

    subject_labels = subject_df[["subject_id"]].copy()
    subject_labels["subtype_label"] = labels
    subject_labels.to_csv(os.path.join(OUT_DIR, "subject_phenotypes_k2.csv"), index=False)

    global_sep = evaluate_global_label_separability(x_subjects, labels)
    global_sep.to_csv(os.path.join(OUT_DIR, "global_label_separability.csv"), index=False)
    log("\nDescriptive separability against full-cohort k=2 labels:", report)
    log(
        format_table(
            global_sep,
            [
                "classifier",
                "accuracy_mean",
                "accuracy_std",
                "balanced_accuracy_mean",
                "f1_macro_mean",
            ],
        ),
        report,
    )

    fold_metrics = evaluate_train_only_cluster_propagation(x_subjects, n_clusters=n_clusters)
    fold_metrics.to_csv(os.path.join(OUT_DIR, "train_only_cluster_propagation_folds.csv"), index=False)
    fold_summary = summarize_fold_metrics(fold_metrics)
    fold_summary.to_csv(os.path.join(OUT_DIR, "train_only_cluster_propagation_summary.csv"), index=False)

    log("\nTrain-only cluster propagation validation:", report)
    log(
        format_table(
            fold_summary,
            [
                "classifier",
                "accuracy_mean",
                "accuracy_std",
                "balanced_accuracy_mean",
                "f1_macro_mean",
                "adjusted_rand_mean",
            ],
        ),
        report,
    )

    biomarkers = biomarker_table(subject_df, feature_cols, labels)
    biomarkers.to_csv(os.path.join(OUT_DIR, "top_biomarkers_logistic_k2.csv"), index=False)
    log("\nTop relative-power biomarkers:", report)
    log(
        format_table(
            biomarkers.head(10),
            [
                "feature",
                "abs_logistic_weight",
                "signed_logistic_weight",
                "direction",
            ],
        ),
        report,
    )

    best = fold_summary.iloc[0]
    log("\nPaper-safe headline from this script:", report)
    log(
        "  Subject-level k=2 phenotype propagation "
        f"({best['classifier']}): accuracy={best['accuracy_mean']:.3f} +/- "
        f"{best['accuracy_std']:.3f}, balanced accuracy={best['balanced_accuracy_mean']:.3f}.",
        report,
    )
    log(
        "  dont forget to report this as agreement with train-only propagated unsupervised labels, "
        "not as clinical diagnostic accuracy.",
        report,
    )

    report_text = "\n".join(report)
    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(os.path.join(OUT_DIR, "report_subject_level_validation.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved final report to: {FINAL_REPORT_PATH}")
    print(f"Saved detailed hardened outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
