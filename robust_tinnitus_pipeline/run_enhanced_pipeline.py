"""
Unified Launcher for Enhanced Tinnitus Pipeline
================================================
Run all steps: feature extraction -> clustering -> classification.
"""

import subprocess
import sys
import os

def run_script(script_path, description):
    print(f"\n{'='*70}")
    print(f"  RUNNING: {description}")
    print(f"{'='*70}")
    result = subprocess.run([sys.executable, script_path], cwd=os.getcwd())
    return result.returncode


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Tinnitus Pipeline Launcher')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'extract', 'classify', 'zenodo-extract', 'zenodo-classify'],
                        help='Which step to run')
    parser.add_argument('--fast', action='store_true',
                        help='Use reduced hyperparameter search (faster)')
    args = parser.parse_args()
    
    scripts_dir = "robust_tinnitus_pipeline"
    
    if args.step == 'all':
        run_script(f"{scripts_dir}/01_feature_extraction_enhanced.py", 
                    "Enhanced Feature Extraction (Main Dataset)")
        run_script(f"{scripts_dir}/02_clustering_evaluation.py",
                    "Clustering Evaluation")
        run_script(f"{scripts_dir}/03_predictive_validation_enhanced.py",
                    "Enhanced Predictive Validation")
        run_script(f"{scripts_dir}/01_zenodo_feature_extraction_enhanced.py",
                    "Enhanced Feature Extraction (Zenodo)")
        run_script(f"{scripts_dir}/02_zenodo_clustering.py",
                    "Subject-Level Zenodo Clustering")
        run_script(f"{scripts_dir}/03_zenodo_novel_validation.py",
                    "Hardened Subject-Level Zenodo Validation")
    
    elif args.step == 'extract':
        run_script(f"{scripts_dir}/01_feature_extraction_enhanced.py", 
                    "Enhanced Feature Extraction")
    
    elif args.step == 'classify':
        run_script(f"{scripts_dir}/03_predictive_validation_enhanced.py",
                    "Enhanced Predictive Validation")
    
    elif args.step == 'zenodo-extract':
        run_script(f"{scripts_dir}/01_zenodo_feature_extraction_enhanced.py",
                    "Enhanced Zenodo Feature Extraction")
    
    elif args.step == 'zenodo-classify':
        run_script(f"{scripts_dir}/02_zenodo_clustering.py",
                    "Subject-Level Zenodo Clustering")
        run_script(f"{scripts_dir}/03_zenodo_novel_validation.py",
                    "Hardened Subject-Level Zenodo Validation")
    
    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETE")
    print("=" * 70)
