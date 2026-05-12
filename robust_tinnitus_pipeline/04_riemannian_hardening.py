"""
Riemannian hardening pipeline for Zenodo tinnitus EEG.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt
from sklearn.cluster import KMeans
from sklearn.covariance import OAS
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from pyriemann.clustering import Kmeans as RiemannianKMeans
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SET_DIR = (
    BASE_DIR.parent
    / "data"
    / "EEG-PRO_data"
    / "EEG-PRO_data"
    / "RAW"
    / "Tinnitus Dataset"
)
OUT_DIR = BASE_DIR / "riemannian_results"
CACHE_PATH = OUT_DIR / "zenodo_subject_covariances.npz"
TANGENT_PATH = OUT_DIR / "zenodo_filterbank_tangent_features.csv"
QC_PATH = OUT_DIR / "subject_loading_qc.csv"
REPORT_PATH = OUT_DIR / "final_riemannian_report.txt"

BANDS = {
    "broadband": (1.0, 45.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class CacheData:
    subject_ids: np.ndarray
    covariances: np.ndarray
    bands: list[str]
    channels: list[str]


def log(message: str, report: list[str]) -> None:
    print(message)
    report.append(message)


def expected_egi_channels() -> list[str]:
    return [f"E{i}" for i in range(1, 128)]


def regularize_covariance(data: np.ndarray, trace_normalize: bool = True) -> np.ndarray:
    cov = OAS(store_precision=False, assume_centered=False).fit(data.T).covariance_
    cov = (cov + cov.T) / 2.0
    if trace_normalize:
        trace = np.trace(cov)
        if trace > 0:
            cov = cov / trace * cov.shape[0]
    cov += np.eye(cov.shape[0]) * 1e-7
    return cov.astype(np.float32)


def bandpass_epochs(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    nyquist = sfreq / 2.0
    high = min(high, nyquist - 1.0)
    sos = butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def process_set_file(set_path: Path, channels: list[str]) -> tuple[str, dict[str, np.ndarray] | None, dict]:
    subject_id = set_path.stem
    qc = {
        "subject_id": subject_id,
        "file": str(set_path),
        "status": "failed",
        "n_epochs": 0,
        "n_channels": 0,
        "sfreq": np.nan,
        "message": "",
    }

    try:
        epochs = mne.io.read_epochs_eeglab(str(set_path), verbose=False)
        available = [ch for ch in channels if ch in epochs.ch_names]
        missing = sorted(set(channels) - set(available))
        if missing:
            raise ValueError(f"Missing expected EGI channels: {missing[:5]} ... ({len(missing)} total)")

        epochs.pick(available)
        data = epochs.get_data(copy=True).astype(np.float64)
        data -= data.mean(axis=-1, keepdims=True)
        sfreq = float(epochs.info["sfreq"])

        qc.update(
            {
                "status": "ok",
                "n_epochs": int(data.shape[0]),
                "n_channels": int(data.shape[1]),
                "sfreq": sfreq,
            }
        )

        covs = {}
        for band_name, (low, high) in BANDS.items():
            filtered = bandpass_epochs(data, sfreq, low, high)
            subject_matrix = filtered.transpose(1, 0, 2).reshape(filtered.shape[1], -1)
            covs[band_name] = regularize_covariance(subject_matrix)

        return subject_id, covs, qc
    except Exception as exc:
        qc["message"] = f"{type(exc).__name__}: {exc}"
        return subject_id, None, qc


def build_covariance_cache(
    set_dir: Path,
    limit: int | None,
    n_jobs: int,
    force: bool,
) -> CacheData:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    channels = expected_egi_channels()

    if CACHE_PATH.exists() and QC_PATH.exists() and not force and limit is None:
        return load_covariance_cache()

    set_files = sorted(set_dir.glob("*.set"))
    if limit is not None:
        set_files = set_files[:limit]
    if not set_files:
        raise FileNotFoundError(f"No .set files found in {set_dir}")

    print(f"Computing covariance cache from {len(set_files)} .set files...")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_set_file)(path, channels) for path in set_files
    )

    qc_rows = [qc for _, _, qc in results]
    pd.DataFrame(qc_rows).to_csv(QC_PATH, index=False)

    ok_results = [(sid, covs) for sid, covs, qc in results if covs is not None and qc["status"] == "ok"]
    if not ok_results:
        raise RuntimeError("No EEG files could be converted into covariance matrices.")

    subject_ids = np.array([sid for sid, _ in ok_results], dtype=object)
    band_names = list(BANDS.keys())
    covariances = np.stack(
        [np.stack([covs[band] for band in band_names], axis=0) for _, covs in ok_results],
        axis=0,
    )

    np.savez_compressed(
        CACHE_PATH,
        subject_ids=subject_ids,
        covariances=covariances,
        bands=np.array(band_names, dtype=object),
        channels=np.array(channels, dtype=object),
    )

    return CacheData(subject_ids=subject_ids, covariances=covariances, bands=band_names, channels=channels)


def load_covariance_cache() -> CacheData:
    cached = np.load(CACHE_PATH, allow_pickle=True)
    return CacheData(
        subject_ids=cached["subject_ids"],
        covariances=cached["covariances"],
        bands=[str(x) for x in cached["bands"].tolist()],
        channels=[str(x) for x in cached["channels"].tolist()],
    )


def fit_filterbank_tangent(covs: np.ndarray, bands: list[str]) -> tuple[np.ndarray, list[TangentSpace], list[str]]:
    features = []
    transformers = []
    feature_names = []

    for band_idx, band_name in enumerate(bands):
        ts = TangentSpace(metric="riemann")
        x_band = ts.fit_transform(covs[:, band_idx])
        features.append(x_band)
        transformers.append(ts)
        feature_names.extend([f"{band_name}_ts_{i:04d}" for i in range(x_band.shape[1])])

    x = np.concatenate(features, axis=1).astype(np.float32)
    return x, transformers, feature_names


def transform_filterbank_tangent(
    covs: np.ndarray,
    transformers: list[TangentSpace],
) -> np.ndarray:
    features = [ts.transform(covs[:, band_idx]) for band_idx, ts in enumerate(transformers)]
    return np.concatenate(features, axis=1).astype(np.float32)


def make_cluster_space(x: np.ndarray, variance: float = 0.95) -> tuple[np.ndarray, Pipeline]:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=variance, random_state=RANDOM_STATE)),
        ]
    )
    return pipeline.fit_transform(x), pipeline


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


def best_jaccard(y_ref: np.ndarray, y_boot: np.ndarray, n_classes: int) -> float:
    scores = []
    for label in range(n_classes):
        ref = set(np.where(y_ref == label)[0])
        best = 0.0
        for other in range(n_classes):
            boot = set(np.where(y_boot == other)[0])
            union = ref | boot
            if union:
                best = max(best, len(ref & boot) / len(union))
        scores.append(best)
    return float(np.mean(scores))


def permutation_silhouette(x: np.ndarray, labels: np.ndarray, n_permutations: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    observed = silhouette_score(x, labels)
    null = np.empty(n_permutations, dtype=float)
    for i in trange(n_permutations, desc="  Permutations", leave=False, position=1):
        null[i] = silhouette_score(x, rng.permutation(labels))
    p_value = (np.sum(null >= observed) + 1.0) / (n_permutations + 1.0)
    return float(observed), float(null.mean()), float(p_value)


def bootstrap_stability(
    x: np.ndarray,
    labels_ref: np.ndarray,
    n_clusters: int,
    n_bootstrap: int,
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    ari_scores = []
    jaccard_scores = []
    n_subjects = len(x)

    for _ in trange(n_bootstrap, desc="  Bootstrap", leave=False, position=1):
        sample_idx = rng.choice(n_subjects, size=n_subjects, replace=True)
        unique_idx = np.unique(sample_idx)
        boot_labels_all = KMeans(
            n_clusters=n_clusters,
            n_init=50,
            random_state=RANDOM_STATE,
        ).fit_predict(x[sample_idx])

        first_label = {}
        for position, original_idx in enumerate(sample_idx):
            first_label.setdefault(original_idx, boot_labels_all[position])

        y_ref = labels_ref[unique_idx]
        y_boot = np.array([first_label[idx] for idx in unique_idx])
        ari_scores.append(adjusted_rand_score(y_ref, y_boot))
        jaccard_scores.append(best_jaccard(y_ref, y_boot, n_clusters))

    return (
        float(np.mean(ari_scores)),
        float(np.std(ari_scores)),
        float(np.mean(jaccard_scores)),
        float(np.std(jaccard_scores)),
    )


def evaluate_kmeans_scan(
    x_cluster: np.ndarray,
    n_subjects: int,
    n_permutations: int,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    rows = []
    labels_by_k = {}
    min_cluster_size = max(int(np.ceil(0.10 * n_subjects)), 20)

    for k in tqdm(range(2, 7), desc="KMeans Scan", position=0):
        km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
        labels = km.fit_predict(x_cluster)
        labels_by_k[k] = labels
        sizes = np.bincount(labels, minlength=k)
        sil, null_sil, p_value = permutation_silhouette(x_cluster, labels, n_permutations)
        db = davies_bouldin_score(x_cluster, labels)
        ari_mean, ari_std, jac_mean, jac_std = bootstrap_stability(x_cluster, labels, k, n_bootstrap)
        valid = bool(sizes.min() >= min_cluster_size and p_value < 0.05)

        rows.append(
            {
                "method": "filterbank_tangent_kmeans",
                "k": k,
                "n_subjects": n_subjects,
                "min_required_cluster_size": min_cluster_size,
                "cluster_sizes": "|".join(map(str, sizes.tolist())),
                "min_cluster_size": int(sizes.min()),
                "silhouette": sil,
                "permutation_null_silhouette_mean": null_sil,
                "permutation_p": p_value,
                "davies_bouldin": db,
                "bootstrap_ari_mean": ari_mean,
                "bootstrap_ari_std": ari_std,
                "bootstrap_jaccard_mean": jac_mean,
                "bootstrap_jaccard_std": jac_std,
                "valid": valid,
            }
        )

    return pd.DataFrame(rows), labels_by_k


def evaluate_gmm_scan(x_cluster: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in tqdm(range(2, 7), desc="GMM Scan"):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE, n_init=10, reg_covar=1e-3)
            labels = gmm.fit_predict(x_cluster)
            sizes = np.bincount(labels, minlength=k)
            rows.append(
                {
                    "method": "filterbank_tangent_gmm",
                    "k": k,
                    "cluster_sizes": "|".join(map(str, sizes.tolist())),
                    "silhouette": silhouette_score(x_cluster, labels),
                    "davies_bouldin": davies_bouldin_score(x_cluster, labels),
                    "bic": gmm.bic(x_cluster),
                    "aic": gmm.aic(x_cluster),
                }
            )
        except Exception:
            pass
    return pd.DataFrame(rows)


def evaluate_broadband_riemannian(covs: np.ndarray, bands: list[str]) -> pd.DataFrame:
    rows = []
    broadband_idx = bands.index("broadband")
    x_broad = covs[:, broadband_idx]
    ts = TangentSpace(metric="riemann")
    x_broad_ts = ts.fit_transform(x_broad)
    x_broad_cluster, _ = make_cluster_space(x_broad_ts)

    for k in tqdm(range(2, 7), desc="Riemannian Scan"):
        rk = RiemannianKMeans(n_clusters=k, metric="riemann", n_init=10, random_state=RANDOM_STATE, n_jobs=1)
        labels = rk.fit_predict(x_broad)
        sizes = np.bincount(labels, minlength=k)
        rows.append(
            {
                "method": "broadband_riemannian_kmeans",
                "k": k,
                "cluster_sizes": "|".join(map(str, sizes.tolist())),
                "silhouette_tangent_space": silhouette_score(x_broad_cluster, labels),
                "davies_bouldin_tangent_space": davies_bouldin_score(x_broad_cluster, labels),
            }
        )
    return pd.DataFrame(rows)


def select_final_k(scan: pd.DataFrame) -> int:
    valid = scan[scan["valid"]].copy()
    if valid.empty:
        return 2

    valid = valid.sort_values(
        by=["bootstrap_ari_mean", "bootstrap_jaccard_mean", "silhouette", "davies_bouldin"],
        ascending=[False, False, False, True],
    )
    return int(valid.iloc[0]["k"])


def simple_models() -> dict[str, object]:
    return {
        "logistic_l2": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearSVC(C=0.1, class_weight="balanced", dual="auto", random_state=RANDOM_STATE)),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }


def tri_ensemble() -> StackingClassifier:
    estimators = [
        (
            "catboost",
            CatBoostClassifier(
                iterations=150,
                depth=4,
                learning_rate=0.05,
                auto_class_weights="Balanced",
                loss_function="MultiClass",
                verbose=False,
                random_seed=RANDOM_STATE,
                allow_writing_files=False,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=350,
                max_features="sqrt",
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                l2_regularization=0.1,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        cv=3,
        n_jobs=1,
        passthrough=False,
    )


def descriptive_global_validation(x_cluster: np.ndarray, labels: np.ndarray, k: int) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
    }
    rows = []
    models = simple_models()
    models["tri_ensemble"] = tri_ensemble()

    for name, model in tqdm(models.items(), desc="Global Descriptive CV"):
        result = cross_validate(model, x_cluster, labels, cv=cv, scoring=scoring, n_jobs=1)
        rows.append(
            {
                "validation": "global_label_descriptive",
                "model": name,
                "k": k,
                "accuracy_mean": result["test_accuracy"].mean(),
                "accuracy_std": result["test_accuracy"].std(),
                "balanced_accuracy_mean": result["test_balanced_accuracy"].mean(),
                "balanced_accuracy_std": result["test_balanced_accuracy"].std(),
                "f1_macro_mean": result["test_f1_macro"].mean(),
                "f1_macro_std": result["test_f1_macro"].std(),
            }
        )
    return pd.DataFrame(rows)


def train_only_propagation_validation(covs: np.ndarray, bands: list[str], k: int) -> pd.DataFrame:
    rows = []
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = simple_models()
    models["tri_ensemble"] = tri_ensemble()

    for model_name, model in tqdm(models.items(), desc="Train-Only Prop CV", position=0):
        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(cv.split(covs), total=5, desc=f"  {model_name} folds", leave=False, position=1), start=1):
            cov_train = covs[train_idx]
            cov_test = covs[test_idx]

            x_train_ts, transformers, _ = fit_filterbank_tangent(cov_train, bands)
            x_test_ts = transform_filterbank_tangent(cov_test, transformers)

            cluster_pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
                ]
            )
            x_train_cluster = cluster_pipe.fit_transform(x_train_ts)
            x_test_cluster = cluster_pipe.transform(x_test_ts)

            km = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
            y_train = km.fit_predict(x_train_cluster)
            y_test = km.predict(x_test_cluster)

            model.fit(x_train_cluster, y_train)
            pred = model.predict(x_test_cluster)
            pred_aligned = align_labels(y_test, pred, k)

            rows.append(
                {
                    "validation": "train_only_cluster_propagation",
                    "model": model_name,
                    "fold": fold_idx,
                    "k": k,
                    "test_cluster_sizes": "|".join(map(str, np.bincount(y_test, minlength=k).tolist())),
                    "accuracy": float((pred_aligned == y_test).mean()),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, pred_aligned)),
                    "f1_macro": float(f1_score(y_test, pred_aligned, average="macro", zero_division=0)),
                    "adjusted_rand": float(adjusted_rand_score(y_test, pred)),
                }
            )
    return pd.DataFrame(rows)


def summarize_fold_validation(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby(["validation", "model", "k"])
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
        .sort_values(["validation", "balanced_accuracy_mean"], ascending=[True, False])
    )


def format_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_string(index=False, float_format=lambda x: f"{x:.4f}")


def write_tangent_csv(subject_ids: np.ndarray, x_tangent: np.ndarray, feature_names: list[str]) -> None:
    df = pd.DataFrame(x_tangent, columns=feature_names)
    df.insert(0, "subject_id", subject_ids)
    df.to_csv(TANGENT_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Riemannian EEG phenotype discovery and validation.")
    parser.add_argument("--set-dir", type=Path, default=DEFAULT_SET_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N .set files for smoke tests.")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--bootstraps", type=int, default=100)
    parser.add_argument("--skip-ensemble", action="store_true", help="Only run clustering and simple models.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    log("=" * 78, report)
    log(" RIEMANNIAN EEG PHENOTYPE HARDENING PIPELINE", report)
    log("=" * 78, report)
    log(f"Input .set directory: {args.set_dir}", report)

    cache = build_covariance_cache(args.set_dir, args.limit, args.n_jobs, args.force_cache)
    n_subjects = len(cache.subject_ids)
    log(f"Subjects with valid covariance matrices: {n_subjects}", report)
    log(f"Bands: {', '.join(cache.bands)}", report)
    log(f"Channels: {len(cache.channels)} EGI channels", report)

    x_tangent, _, feature_names = fit_filterbank_tangent(cache.covariances, cache.bands)
    write_tangent_csv(cache.subject_ids, x_tangent, feature_names)
    x_cluster, cluster_pipe = make_cluster_space(x_tangent)
    log(f"Filter-bank tangent feature shape: {x_tangent.shape}", report)
    log(f"Cluster PCA shape: {x_cluster.shape}", report)

    kmeans_scan, labels_by_k = evaluate_kmeans_scan(
        x_cluster,
        n_subjects=n_subjects,
        n_permutations=args.permutations,
        n_bootstrap=args.bootstraps,
    )
    kmeans_scan.to_csv(OUT_DIR / "kmeans_stability_scan.csv", index=False)
    log("\nFilter-bank tangent KMeans scan:", report)
    log(
        format_table(
            kmeans_scan[
                [
                    "k",
                    "cluster_sizes",
                    "silhouette",
                    "permutation_p",
                    "davies_bouldin",
                    "bootstrap_ari_mean",
                    "bootstrap_jaccard_mean",
                    "valid",
                ]
            ]
        ),
        report,
    )

    gmm_scan = evaluate_gmm_scan(x_cluster)
    gmm_scan.to_csv(OUT_DIR / "gmm_ablation_scan.csv", index=False)

    # SKIP the brute-force Riemannian KMeans scan as 126x126 Fréchet mean is too slow
    # riemannian_scan = evaluate_broadband_riemannian(cache.covariances, cache.bands)
    # riemannian_scan.to_csv(OUT_DIR / "broadband_riemannian_kmeans_ablation.csv", index=False)

    selected_k = select_final_k(kmeans_scan)
    selected_row = kmeans_scan[kmeans_scan["k"] == selected_k].iloc[0].to_dict()
    selected_labels = labels_by_k[selected_k]

    subject_labels = pd.DataFrame(
        {
            "subject_id": cache.subject_ids,
            "riemannian_subtype": selected_labels,
        }
    )
    subject_labels.to_csv(OUT_DIR / "selected_subject_phenotypes.csv", index=False)

    log("\nSelected phenotype architecture:", report)
    log(f"  selected k: {selected_k}", report)
    log(f"  cluster sizes: {selected_row['cluster_sizes']}", report)
    log(f"  silhouette: {selected_row['silhouette']:.4f}", report)
    log(f"  permutation p: {selected_row['permutation_p']:.4f}", report)
    log(f"  bootstrap ARI: {selected_row['bootstrap_ari_mean']:.4f} +/- {selected_row['bootstrap_ari_std']:.4f}", report)
    log(
        f"  bootstrap Jaccard: {selected_row['bootstrap_jaccard_mean']:.4f} +/- "
        f"{selected_row['bootstrap_jaccard_std']:.4f}",
        report,
    )

    if args.skip_ensemble:
        log("\nValidation skipped by --skip-ensemble.", report)
    else:
        global_validation = descriptive_global_validation(x_cluster, selected_labels, selected_k)
        global_validation.to_csv(OUT_DIR / "global_label_descriptive_validation.csv", index=False)
        log("\nDescriptive validation against full-cohort labels:", report)
        log(format_table(global_validation), report)

        fold_validation = train_only_propagation_validation(cache.covariances, cache.bands, selected_k)
        fold_validation.to_csv(OUT_DIR / "train_only_cluster_propagation_folds.csv", index=False)
        fold_summary = summarize_fold_validation(fold_validation)
        fold_summary.to_csv(OUT_DIR / "train_only_cluster_propagation_summary.csv", index=False)
        log("\nTrain-only cluster propagation validation:", report)
        log(format_table(fold_summary), report)

    supports = "multi-cluster" if selected_k > 2 else "binary macroscopic"
    log("\nPaper-ready interpretation:", report)
    log(f"  The Riemannian pipeline supports a {supports} phenotype architecture (k={selected_k}).", report)
    log("  Ensemble metrics are phenotype-label agreement metrics, not clinical diagnostic accuracy.", report)

    metadata = {
        "selected_k": selected_k,
        "selected_row": selected_row,
        "n_subjects": n_subjects,
        "bands": cache.bands,
        "n_channels": len(cache.channels),
        "cache_path": str(CACHE_PATH),
        "tangent_path": str(TANGENT_PATH),
    }
    with open(OUT_DIR / "final_selection_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report_text = "\n".join(report)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nSaved final report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
