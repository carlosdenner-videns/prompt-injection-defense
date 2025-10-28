#!/usr/bin/env python3
"""
Run comprehensive experiments on the HuggingFace dataset (2000 samples).
"""

import subprocess
import os

# Dataset to use
DATA = "data/prompts_hf_augmented.csv"

# Experiment configurations: (name, pipeline, threshold)
EXPERIMENTS = [
    ("hf_signature_only", "signature", 0.5),
    ("hf_rules_only", "rules", 0.5),
    ("hf_classifier_only", "classifier", 0.5),
    ("hf_classifier_low", "classifier", 0.3),
    ("hf_nemo_only", "nemo", 0.5),
    ("hf_sig_rules", "signature,rules", 0.5),
    ("hf_sig_clf", "signature,classifier", 0.5),
    ("hf_rules_clf", "rules,classifier", 0.5),
    ("hf_sig_rules_clf", "signature,rules,classifier", 0.5),
    ("hf_sig_rules_clf_oracle", "signature,rules,classifier", 0.5),  # with --oracle
    ("hf_all_nemo", "signature,rules,classifier,nemo", 0.5),
]

def run_experiment(name, pipeline, threshold, oracle=False):
    """Run a single experiment configuration."""
    cmd = [
        ".venv/Scripts/python.exe",
        "src/run_experiment.py",
        "--data", DATA,
        "--pipeline", pipeline,
        "--threshold", str(threshold),
        "--out", f"results/{name}"
    ]
    
    if oracle or "oracle" in name:
        cmd.append("--oracle")
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Pipeline: {pipeline}, Threshold: {threshold}, Oracle: {oracle or 'oracle' in name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {name}:")
        print(e.stderr)
    except FileNotFoundError:
        print(f"Error: Python executable not found. Using system python...")
        cmd[0] = "python"
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except Exception as e2:
            print(f"Failed: {e2}")

def main():
    print("Starting comprehensive experiment suite on HuggingFace dataset...")
    print(f"Dataset: {DATA} (2000 samples: 1000 attacks + 1000 benign)")
    print(f"Total experiments: {len(EXPERIMENTS)}")
    
    os.makedirs("results", exist_ok=True)
    
    for name, pipeline, threshold in EXPERIMENTS:
        oracle = "oracle" in name
        run_experiment(name, pipeline, threshold, oracle)
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"{'='*60}")
    print("\nTo analyze results, run:")
    print("  python src/analyze_results.py --results results --output analysis_output")
    print("\nTo compare specific experiments:")
    print("  python src/analyze_results.py --results results --compare hf_signature_only hf_sig_rules hf_sig_rules_clf --output analysis_output")

if __name__ == "__main__":
    main()
