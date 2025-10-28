#!/usr/bin/env python3
"""
Run optimized experiments with tuned thresholds.
"""

import subprocess
import os

DATA = "data/prompts_hf_augmented.csv"

# Optimized configurations based on tuning results
EXPERIMENTS = [
    # Individual components with optimal thresholds
    ("opt_classifier_balanced", "classifier", 0.1),  # 58.7% TPR, 4.8% FPR
    ("opt_classifier_precise", "classifier", 0.5),  # 25.8% TPR, 0% FPR
    ("opt_nemo_balanced", "nemo", 0.3),             # 34.2% TPR, 2.7% FPR
    ("opt_nemo_precise", "nemo", 0.6),              # 6.8% TPR, 0% FPR
    
    # Best combinations
    ("opt_sig_clf_best", "signature,classifier", 0.1),      # 92.0% TPR, 4.8% FPR (Best F1)
    ("opt_sig_clf_precise", "signature,classifier", 0.5),   # 86.5% TPR, 0% FPR (Zero FP)
    ("opt_sig_rules_clf", "signature,rules,classifier", 0.6),  # 85.9% TPR, 3.5% FPR
    ("opt_all_balanced", "signature,rules,classifier,nemo", 0.3),  # All defenses
    
    # Previous best for comparison
    ("baseline_sig_only", "signature", 0.5),
    ("baseline_sig_rules_clf", "signature,rules,classifier", 0.5),
]

def run_experiment(name, pipeline, threshold):
    """Run a single experiment configuration."""
    cmd = [
        ".venv/Scripts/python.exe",
        "src/run_experiment.py",
        "--data", DATA,
        "--pipeline", pipeline,
        "--threshold", str(threshold),
        "--out", f"results/{name}"
    ]
    
    print(f"\nRunning: {name} (pipeline={pipeline}, threshold={threshold})")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
    except Exception as e:
        print(f"Failed: {e}")

def main():
    print("="*60)
    print("OPTIMIZED EXPERIMENTS WITH TUNED THRESHOLDS")
    print("="*60)
    print(f"Dataset: {DATA} (2000 samples)")
    print(f"Experiments: {len(EXPERIMENTS)}\n")
    
    os.makedirs("results", exist_ok=True)
    
    for name, pipeline, threshold in EXPERIMENTS:
        run_experiment(name, pipeline, threshold)
    
    print(f"\n{'='*60}")
    print("All optimized experiments complete!")
    print(f"{'='*60}")
    print("\nKey Results:")
    print("  opt_sig_clf_best: Expected 92.0% TPR, 4.8% FPR")
    print("  opt_sig_clf_precise: Expected 86.5% TPR, 0% FPR")
    print("\nTo analyze:")
    print("  python src/analyze_results.py --results results --output analysis_output")

if __name__ == "__main__":
    main()
