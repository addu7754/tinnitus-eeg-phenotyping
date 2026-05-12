"""
Enhanced Predictive Validation Pipeline for Tinnitus EEG Classification
========================================================================
Improvements over baseline:
  1. Additional spectral, time-domain, and connectivity features
  2. Feature selection via mutual information
  3. Class imbalance handling (SMOTE + class_weight)
  4. Hyperparameter optimization (RandomizedSearchCV)
  5. Ensemble classifiers (Random Forest + Gradient Boosting + SVM)
  6. Comprehensive cross-validation with multiple metrics
  7. Visualization and detailed reporting
"""

import os
import numpy as np
import pandas as pd
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# Must import matplotlib before other heavy imports to avoid backend issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import (
    StratifiedGroupKFold, cross_validate, RandomizedSearchCV,
    learning_curve, StratifiedKFold
)
from sklearn.ensemble import (
    RandomForestClassifier, HistGradientBoostingClassifier, 
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, cohen_kappa_score,
    balanced_accuracy_score, make_scorer
)
from sklearn.calibration import calibration_curve

# SMOTE must be imported conditionally
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("WARNING: imbalanced-learn not installed. Install via: pip install imbalanced-learn")


def load_and_prepare_data(feature_path: str, task: str = '4class'):
    """Load features and prepare X, y, groups for classification."""
    df = pd.read_csv(feature_path)
    df = df.dropna()

    # Extract metadata from subject_id
    df['patient_id'] = df['subject_id'].apply(lambda x: x.split('_')[0])
    df['clinical_state'] = df['subject_id'].apply(
        lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown'
    )
    df['session'] = df['subject_id'].apply(
        lambda x: x.split('_')[2] if len(x.split('_')) > 2 else 'S1'
    )

    feature_cols = [c for c in df.columns if c not in 
                    ['subject_id', 'epoch_id', 'patient_id', 'clinical_state', 'session']]
    X = df[feature_cols].copy()
    
    if task == '4class':
        # Use all 4 clinical states for 4-class classification
        state_map = {'Baseline': 0, 'Passive': 1, 'Active': 2, 'Sham': 3}
        valid_states = ['Baseline', 'Passive', 'Active', 'Sham']
        task_df = df[df['clinical_state'].isin(valid_states)].copy()
        y = task_df['clinical_state'].map(state_map)
    elif task == 'baseline_vs_active':
        # Binary: Baseline vs Active
        task_df = df[df['clinical_state'].isin(['Baseline', 'Active'])].copy()
        y = task_df['clinical_state'].map({'Baseline': 0, 'Active': 1})
    elif task == 'baseline_vs_treatment':
        # Binary: Baseline vs (Active + Sham + Passive)
        task_df = df[df['clinical_state'].isin(['Baseline', 'Active', 'Sham', 'Passive'])].copy()
        y = task_df['clinical_state'].map({
            'Baseline': 0, 'Active': 1, 'Sham': 1, 'Passive': 1
        })
    else:
        raise ValueError(f"Unknown task: {task}")
    
    groups = task_df['patient_id']
    X = task_df[feature_cols].copy()
    
    print(f"\nTask: {task}")
    print(f"  Total epochs: {len(y)}")
    print(f"  Total unique patients: {groups.nunique()}")
    print(f"  Class distribution: {dict(y.value_counts())}")
    print(f"  Total features: {X.shape[1]}")
    
    return X, y, groups, feature_cols


class FeatureSelector:
    """Mutual Information-based feature selector with visualization."""
    
    def __init__(self, n_features: int = 30, random_state: int = 42):
        self.n_features = n_features
        self.random_state = random_state
        self.selector = None
        self.selected_features = None
    
    def fit_select(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select top features based on mutual information."""
        self.selector = SelectKBest(score_func=mutual_info_classif, 
                                     k=min(self.n_features, X.shape[1]))
        X_selected = self.selector.fit_transform(X, y)
        selected_mask = self.selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        
        # Print feature importance
        scores = self.selector.scores_
        sorted_idx = np.argsort(scores)[::-1]
        print(f"\nTop {min(15, len(scores))} features by mutual information:")
        for i, idx in enumerate(sorted_idx[:15]):
            print(f"  {i+1}. {X.columns[idx]}: {scores[idx]:.4f}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features)


def create_classifiers(random_state: int = 42):
    """Create base classifiers with hyperparameter grids."""
    
    classifiers = {}
    
    # Random Forest
    rf = RandomForestClassifier(random_state=random_state, class_weight='balanced')
    rf_params = {
        'n_estimators': [100, 200, 400, 600],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5, None],
        'n_jobs': [-1]
    }
    classifiers['RandomForest'] = (rf, rf_params)
    
    # Histogram Gradient Boosting
    hgb = HistGradientBoostingClassifier(random_state=random_state)
    hgb_params = {
        'max_iter': [100, 200, 400, 600],
        'max_depth': [3, 5, 8, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [5, 10, 20, 50],
        'l2_regularization': [0, 0.1, 1, 10]
    }
    classifiers['HistGradientBoosting'] = (hgb, hgb_params)
    
    # SVM
    svc = SVC(random_state=random_state)
    svc_params = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'class_weight': ['balanced', None]
    }
    classifiers['SVM'] = (svc, svc_params)
    
    return classifiers


def run_hyperparameter_search(X, y, groups, base_classifier, param_grid, 
                               n_iter=50, cv_splits=5, random_state=42):
    """Run randomized search with StratifiedGroupKFold."""
    
    cv = StratifiedGroupKFold(n_splits=cv_splits)
    
    scorer = make_scorer(balanced_accuracy_score)
    
    search = RandomizedSearchCV(
        base_classifier,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        error_score='raise'
    )
    
    search.fit(X, y, groups=groups)
    
    return search


def evaluate_classifier(X, y, groups, pipeline, cv_splits=5, n_repeats=1):
    """Comprehensive cross-validation evaluation."""
    
    cv = StratifiedGroupKFold(n_splits=cv_splits)
    
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'roc_auc_ovr': 'roc_auc_ovr',
        'cohen_kappa': make_scorer(cohen_kappa_score)
    }
    
    all_results = {}
    fold_predictions = []
    fold_labels = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        fold_predictions.extend(y_pred)
        fold_labels.extend(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro', zero_division=0
        )
        f1_w, *_ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        kappa = cohen_kappa_score(y_test, y_pred)
        
        all_results.setdefault('accuracy', []).append(acc)
        all_results.setdefault('balanced_accuracy', []).append(bal_acc)
        all_results.setdefault('f1_macro', []).append(f1)
        all_results.setdefault('f1_weighted', []).append(f1_w)
        all_results.setdefault('precision_macro', []).append(prec)
        all_results.setdefault('recall_macro', []).append(rec)
        all_results.setdefault('cohen_kappa', []).append(kappa)
    
    # Per-class metrics
    per_class = precision_recall_fscore_support(
        fold_labels, fold_predictions, labels=list(set(y)), zero_division=0
    )
    
    return all_results, fold_labels, fold_predictions, per_class


def create_visualizations(X, y, groups, pipeline, feature_names, 
                          output_dir: str, task_name: str, results: dict):
    """Create comprehensive visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    cv = StratifiedGroupKFold(n_splits=5)
    
    # 1. Feature importance bar chart
    if hasattr(pipeline.named_steps.get('classifier'), 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].feature_importances_
        sorted_idx = np.argsort(importances)[::-1][:30]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(range(30), importances[sorted_idx])
        ax.set_yticks(range(30))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Top 30 Feature Importances - {task_name}')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'feature_importance_{task_name}.png'), dpi=150)
        plt.close()
        print(f"  Saved: feature_importance_{task_name}.png")
    
    # 2. Confusion matrix heatmap
    all_true, all_pred = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        all_true.extend(y_test)
        all_pred.extend(preds)
    
    cm = confusion_matrix(all_true, all_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Baseline', 'Passive', 'Active', 'Sham'],
                yticklabels=['Baseline', 'Passive', 'Active', 'Sham'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {task_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{task_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: confusion_matrix_{task_name}.png")
    
    # 3. Metrics comparison bar chart
    metrics_names = list(results.keys())
    means = [np.mean(results[m]) for m in metrics_names]
    stds = [np.std(results[m]) for m in metrics_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(metrics_names))
    ax.bar(x_pos, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title(f'Cross-Validation Metrics - {task_name}')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'metrics_{task_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: metrics_{task_name}.png")
    
    # 4. Box plot of accuracy across folds
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot([results['accuracy']], labels=['Accuracy'])
    ax.set_ylabel('Score')
    ax.set_title(f'Accuracy Distribution - {task_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'accuracy_boxplot_{task_name}.png'), dpi=150)
    plt.close()
    print(f"  Saved: accuracy_boxplot_{task_name}.png")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "robust_tinnitus_pipeline/results_enhanced"
    
    print("=" * 70)
    print("  ENHANCED TINNITUS EEG CLASSIFICATION PIPELINE")
    print(f"  Timestamp: {timestamp}")
    print("=" * 70)
    
    # ===== STEP 1: Load Data =====
    print("\n[Step 1] Loading features...")
    base_path = "robust_tinnitus_pipeline"
    
    # Try enhanced features first, fall back to original
    if os.path.exists(os.path.join(base_path, "01_extracted_features_enhanced.csv")):
        feature_path = os.path.join(base_path, "01_extracted_features_enhanced.csv")
        print("  Using ENHANCED feature set")
    elif os.path.exists(os.path.join(base_path, "01_extracted_features.csv")):
        feature_path = os.path.join(base_path, "01_extracted_features.csv")
        print("  Using ORIGINAL feature set (enhanced not found)")
    else:
        raise FileNotFoundError("No feature file found! Extract features first.")
    
    # ===== STEP 2: Feature Selection =====
    print("\n[Step 2] Feature selection with Mutual Information...")
    
    # Run for both binary and 4-class tasks
    tasks = {
        'baseline_vs_active': 'Baseline vs Active (Binary)',
        '4class': '4-Class Classification'
    }
    
    all_results = {}
    
    for task_key, task_name in tasks.items():
        print(f"\n{'='*70}")
        print(f"  TASK: {task_name}")
        print(f"{'='*70}")
        
        X, y, groups, feature_names = load_and_prepare_data(feature_path, task=task_key)
        
        # Feature Selection
        n_select = min(50, X.shape[1])  # Select top 50 features
        selector = FeatureSelector(n_features=n_select)
        X_selected_df = selector.fit_select(X, y)
        selected_features = selector.selected_features
        
        print(f"\nSelected {len(selected_features)} features from {X.shape[1]} original features")
        
        # ===== STEP 3: Handle Class Imbalance =====
        print("\n[Step 3] Handling class imbalance...")
        
        # Get class distribution
        class_counts = y.value_counts()
        print(f"  Original class distribution: {dict(class_counts)}")
        
        # Determine if SMOTE is needed
        use_smote = HAS_IMBLEARN and (class_counts.min() / class_counts.max()) < 0.5
        
        if use_smote:
            print("  Applying SMOTE for class balancing...")
        else:
            print("  Using class_weight='balanced' in classifier (no SMOTE needed or not available)")
        
        # ===== STEP 4: Classifier Selection & Hyperparameter Optimization =====
        print("\n[Step 4] Training classifiers with hyperparameter optimization...")
        
        classifiers = create_classifiers()
        best_models = {}
        task_results = {}
        
        for clf_name, (base_clf, param_grid) in classifiers.items():
            print(f"\n  --- {clf_name} ---")
            
            # Create pipeline
            if use_smote and HAS_IMBLEARN:
                pipeline = ImbPipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', base_clf)
                ])
                # SMOTE in pipeline
                smote_pipeline = ImbPipeline([
                    ('smote', SMOTE(random_state=42)),
                    ('scaler', StandardScaler()),
                    ('classifier', base_clf)
                ])
                pipe_to_search = smote_pipeline
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', base_clf)
                ])
                pipe_to_search = pipeline
            
            # For hyperparameter search, we need to prefix param names
            prefixed_params = {}
            for key, val in param_grid.items():
                prefixed_params[f'classifier__{key}'] = val
            
            # Run RandomizedSearchCV
            n_iter = min(30, len(param_grid) * 3)  # Limit iterations
            
            try:
                search = run_hyperparameter_search(
                    X_selected_df, y, groups,
                    pipe_to_search, prefixed_params,
                    n_iter=n_iter, cv_splits=5, random_state=42
                )
                
                best_params = search.best_params_
                best_score = search.best_score_
                print(f"    Best balanced accuracy: {best_score:.4f}")
                print(f"    Best params: {best_params}")
                
                best_models[clf_name] = search.best_estimator_
                
                # Evaluate with cross-validation
                print(f"    Running final cross-validation...")
                final_results, true_labels, pred_labels, per_class = evaluate_classifier(
                    X_selected_df, y, groups, search.best_estimator_, cv_splits=5
                )
                
                task_results[clf_name] = {
                    'results': final_results,
                    'best_params': best_params,
                    'best_search_score': best_score,
                    'per_class': per_class
                }
                
                print(f"    Final Accuracy: {np.mean(final_results['accuracy']):.4f} ± {np.std(final_results['accuracy']):.4f}")
                print(f"    Final F1 (macro): {np.mean(final_results['f1_macro']):.4f} ± {np.std(final_results['f1_macro']):.4f}")
                
            except Exception as e:
                print(f"    ERROR during {clf_name}: {e}")
                continue
        
        # ===== STEP 5: Ensemble =====
        if len(best_models) >= 2:
            print("\n  --- Ensemble (Voting) ---")
            try:
                ensemble_estimators = []
                for name, model in best_models.items():
                    # Rename for pipeline compatibility
                    pipeline_clone = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', model.named_steps['classifier'] 
                         if hasattr(model, 'named_steps') else model)
                    ])
                    ensemble_estimators.append((name.lower(), pipeline_clone))
                
                if len(ensemble_estimators) >= 2:
                    ensemble = VotingClassifier(
                        estimators=ensemble_estimators, 
                        voting='soft'
                    )
                    ensemble.fit(X_selected_df, y)
                    
                    final_results, true_labels, pred_labels, per_class = evaluate_classifier(
                        X_selected_df, y, groups, ensemble, cv_splits=5
                    )
                    
                    task_results['Ensemble'] = {
                        'results': final_results,
                        'per_class': per_class
                    }
                    
                    print(f"    Accuracy: {np.mean(final_results['accuracy']):.4f} ± {np.std(final_results['accuracy']):.4f}")
                    print(f"    F1 (macro): {np.mean(final_results['f1_macro']):.4f} ± {np.std(final_results['f1_macro']):.4f}")
            except Exception as e:
                print(f"    ERROR during ensemble: {e}")
        
        # Store results
        all_results[task_key] = {
            'task_name': task_name,
            'task_results': task_results,
            'selected_features': selected_features,
            'true_labels': true_labels if 'true_labels' in dir() else [],
            'pred_labels': pred_labels if 'pred_labels' in dir() else []
        }
        
        # ===== STEP 6: Generate Visualizations =====
        print(f"\n[Step 5] Generating visualizations for {task_name}...")
        
        for clf_name, tr in task_results.items():
            if isinstance(tr['results'], dict):
                # Find best model's pipeline
                if clf_name in best_models:
                    best_pipe = Pipeline([
                        ('scaler', StandardScaler()),
                        ('classifier', best_models[clf_name].named_steps['classifier'] 
                         if hasattr(best_models[clf_name], 'named_steps') else best_models[clf_name])
                    ])
                    create_visualizations(
                        X_selected_df, y, groups, best_pipe, 
                        selected_features, output_dir, 
                        f"{task_key}_{clf_name}", tr['results']
                    )
    
    # ===== STEP 7: Save Results =====
    print(f"\n{'='*70}")
    print("  SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save detailed metrics
    summary = {}
    for task_key, task_data in all_results.items():
        summary[task_key] = {
            'task_name': task_data['task_name'],
            'selected_features_count': len(task_data['selected_features']),
            'selected_features': task_data['selected_features']
        }
        for clf_name, tr in task_data['task_results'].items():
            if isinstance(tr['results'], dict):
                summary[task_key][clf_name] = {
                    'mean_accuracy': float(np.mean(tr['results']['accuracy'])),
                    'std_accuracy': float(np.std(tr['results']['accuracy'])),
                    'mean_balanced_accuracy': float(np.mean(tr['results']['balanced_accuracy'])),
                    'mean_f1_macro': float(np.mean(tr['results']['f1_macro'])),
                    'std_f1_macro': float(np.std(tr['results']['f1_macro'])),
                    'mean_precision_macro': float(np.mean(tr['results']['precision_macro'])),
                    'mean_recall_macro': float(np.mean(tr['results']['recall_macro'])),
                    'mean_cohen_kappa': float(np.mean(tr['results']['cohen_kappa'])),
                    'best_search_score': float(tr.get('best_search_score', 0)),
                    'best_params': {k: str(v) for k, v in tr.get('best_params', {}).items()}
                }
    
    # Summary JSON
    with open(os.path.join(output_dir, f'results_summary_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: results_summary_{timestamp}.json")
    
    # Detailed metrics CSV
    rows = []
    for task_key, task_data in all_results.items():
        for clf_name, tr in task_data['task_results'].items():
            if isinstance(tr['results'], dict):
                row = {
                    'task': task_key,
                    'classifier': clf_name
                }
                for metric, values in tr['results'].items():
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values)
                rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, f'detailed_metrics_{timestamp}.csv'), index=False)
    print(f"  Saved: detailed_metrics_{timestamp}.csv")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    for task_key, task_data in all_results.items():
        print(f"\n  {task_data['task_name']}:")
        for clf_name, tr in task_data['task_results'].items():
            if isinstance(tr['results'], dict):
                acc = np.mean(tr['results']['accuracy'])
                f1 = np.mean(tr['results']['f1_macro'])
                kappa = np.mean(tr['results']['cohen_kappa'])
                bal_acc = np.mean(tr['results']['balanced_accuracy'])
                print(f"    {clf_name:25s} | Acc: {acc:.4f} | Bal Acc: {bal_acc:.4f} | F1: {f1:.4f} | κ: {kappa:.4f}")
    
    print(f"\nAll results saved to: {output_dir}/")
    print("Pipeline complete!")


if __name__ == '__main__':
    main()