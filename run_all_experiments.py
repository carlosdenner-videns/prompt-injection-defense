#!/usr/bin/env python3
"""
Run comprehensive experiments across multiple defense configurations.
"""

import subprocess
import os

# Dataset to use
DATA = "data/prompts_base_large.csv"

# Experiment configurations: (name, pipeline, threshold)
EXPERIMENTS = [
    ("signature_only", "signature", 0.5),
    ("rules_only", "rules", 0.5),
    ("classifier_only", "classifier", 0.5),
    ("classifier_low", "classifier", 0.3),
    ("nemo_only", "nemo", 0.5),
    ("nemo_low", "nemo", 0.3),
    ("sig_rules", "signature,rules", 0.5),
    ("sig_clf", "signature,classifier", 0.5),
    ("rules_clf", "rules,classifier", 0.5),
    ("sig_rules_clf", "signature,rules,classifier", 0.5),
    ("sig_rules_clf_oracle", "signature,rules,classifier", 0.5),  # with --oracle
    ("all_nemo", "signature,rules,classifier,nemo", 0.5),
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
        # Fallback to system python
        cmd[0] = "python"
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
        except Exception as e2:
            print(f"Failed: {e2}")

def main():
    print("Starting comprehensive experiment suite...")
    print(f"Dataset: {DATA}")
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

if __name__ == "__main__":
    main()
