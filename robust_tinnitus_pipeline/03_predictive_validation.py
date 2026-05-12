import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

def main():
    print("--- Running Bit 3: Rigorous Predictive Cross-Validation ---")
    
    # 1. Load the leak-free extracted features
    df = pd.read_csv("robust_tinnitus_pipeline/01_extracted_features.csv")
    df = df.dropna()
    
    # 2. Extract strictly tracked Subject IDs and Target Labels
    # File names look like: 'P1G1_Baseline_S1' or 'P10G1_Active_S2_1Res'
    # The true human is 'P1G1' or 'P10G1'
    df['patient_id'] = df['subject_id'].apply(lambda x: x.split('_')[0])
    
    # The clinical state is the second component: 'Baseline', 'Passive', or 'Active'
    df['clinical_state'] = df['subject_id'].apply(lambda x: x.split('_')[1] if len(x.split('_')) > 1 else 'Unknown')
    
    # For a clean, peer-review-ready benchmark, let's classify: Baseline vs. Active Treatment
    # This proves whether the EEG actually captures a difference when treatment is applied.
    task_df = df[df['clinical_state'].isin(['Baseline', 'Active'])].copy()
    
    # Target (y) and Groups (Patient ID to enforce zero-leakage splits)
    y = task_df['clinical_state'].map({'Baseline': 0, 'Active': 1})
    groups = task_df['patient_id']
    
    feature_cols = [c for c in task_df.columns if c not in ['subject_id', 'epoch_id', 'patient_id', 'clinical_state']]
    X = task_df[feature_cols]
    
    print(f"Task: Classify Baseline (0) vs. Active Treatment (1)")
    print(f"Total Epochs: {len(y)}")
    print(f"Total Unique Patients: {groups.nunique()}")
    print(f"Class Distribution:\n{y.value_counts()}")
    
    # 3. Create the strictly isolated ML Pipeline
    # Scaling is computed ONLY on the training folds dynamically
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42))
    ])
    
    # 4. Stratified Group K-Fold 
    # - Stratified: Maintains the ratio of Baseline/Active in train/test
    # - Group: NEVER lets the same patient_id appear in both train and test!
    cv = StratifiedGroupKFold(n_splits=5)
    
    print("\nExecuting Stratified Group K-Fold Validation (Zero-Leakage)...")
    
    # 5. Run Cross Validation
    cv_results = cross_validate(
        pipeline, X, y, 
        groups=groups, 
        cv=cv, 
        scoring=['accuracy', 'roc_auc', 'f1', 'precision', 'recall'],
        return_train_score=False,
        n_jobs=-1
    )
    
    print("\n--- Final Validation Metrics ---")
    print(f"Accuracy:  {np.mean(cv_results['test_accuracy']):.3f} ± {np.std(cv_results['test_accuracy']):.3f}")
    print(f"ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.3f} ± {np.std(cv_results['test_roc_auc']):.3f}")
    print(f"F1 Score:  {np.mean(cv_results['test_f1']):.3f} ± {np.std(cv_results['test_f1']):.3f}")
    print(f"Precision: {np.mean(cv_results['test_precision']):.3f} ± {np.std(cv_results['test_precision']):.3f}")
    print(f"Recall:    {np.mean(cv_results['test_recall']):.3f} ± {np.std(cv_results['test_recall']):.3f}")
    
    # Save the strict CV metrics for the manuscript
    metrics_df = pd.DataFrame(cv_results)
    metrics_df.to_csv("robust_tinnitus_pipeline/03_cv_metrics.csv", index=False)
    print("\nMetrics explicitly saved to robust_tinnitus_pipeline/03_cv_metrics.csv")

if __name__ == '__main__':
    main()