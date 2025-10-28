#!/usr/bin/env python3
"""
Tune thresholds for classifier and NeMo Guardrails on HuggingFace dataset.
Tests multiple threshold values to find optimal TPR/FPR trade-offs.
"""

import subprocess
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset to use
DATA = "data/prompts_hf_augmented.csv"

# Threshold values to test
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def run_experiment(pipeline, threshold):
    """Run experiment and extract metrics."""
    result_name = f"tune_{pipeline.replace(',', '_')}_t{int(threshold*100)}"
    cmd = [
        ".venv/Scripts/python.exe",
        "src/run_experiment.py",
        "--data", DATA,
        "--pipeline", pipeline,
        "--threshold", str(threshold),
        "--out", f"results/{result_name}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse metrics from summary file
        summary_file = f"results/{result_name}/summary.csv"
        if os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            overall = df[df['family'] == 'overall'].iloc[0]
            return {
                'threshold': threshold,
                'tpr': overall['TPR'],
                'fpr': overall['FPR'],
                'p50_ms': overall['p50_ms'],
                'p95_ms': overall['p95_ms']
            }
    except Exception as e:
        print(f"Error running threshold {threshold}: {e}")
    
    return None

def tune_pipeline(pipeline, name):
    """Tune thresholds for a specific pipeline."""
    print(f"\n{'='*60}")
    print(f"Tuning: {name}")
    print(f"Pipeline: {pipeline}")
    print(f"{'='*60}")
    
    results = []
    for threshold in THRESHOLDS:
        print(f"Testing threshold: {threshold:.1f}...", end=" ", flush=True)
        metrics = run_experiment(pipeline, threshold)
        if metrics:
            results.append(metrics)
            print(f"TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
        else:
            print("FAILED")
    
    return pd.DataFrame(results)

def plot_threshold_curves(results_dict, output_dir):
    """Plot TPR/FPR vs threshold for each pipeline."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Tuning Results on HuggingFace Dataset', fontsize=14, fontweight='bold')
    
    # Plot 1: TPR vs Threshold
    ax = axes[0, 0]
    for name, df in results_dict.items():
        ax.plot(df['threshold'], df['tpr'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title('Detection Rate vs Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: FPR vs Threshold
    ax = axes[0, 1]
    for name, df in results_dict.items():
        ax.plot(df['threshold'], df['fpr'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_title('False Positive Rate vs Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 3: ROC-style (FPR vs TPR)
    ax = axes[1, 0]
    for name, df in results_dict.items():
        # Sort by FPR for proper ROC curve
        df_sorted = df.sort_values('fpr')
        ax.plot(df_sorted['fpr'], df_sorted['tpr'], marker='o', label=name, linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title('ROC-style Curve (Different Thresholds)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 4: F1 Score vs Threshold
    ax = axes[1, 1]
    for name, df in results_dict.items():
        # Calculate F1 = 2 * (precision * recall) / (precision + recall)
        # TPR = recall, precision = TP / (TP + FP)
        # Assuming 1000 attacks and 1000 benign
        tp = df['tpr'] * 1000
        fp = df['fpr'] * 1000
        precision = tp / (tp + fp + 1e-10)  # avoid division by zero
        recall = df['tpr']
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        ax.plot(df['threshold'], f1, marker='o', label=name, linewidth=2)
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score vs Threshold', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'threshold_tuning.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nThreshold tuning plot saved to: {output_path}")

def find_best_thresholds(results_dict):
    """Find optimal thresholds for each pipeline."""
    print(f"\n{'='*60}")
    print("OPTIMAL THRESHOLDS")
    print(f"{'='*60}")
    
    for name, df in results_dict.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Best F1 score
        tp = df['tpr'] * 1000
        fp = df['fpr'] * 1000
        precision = tp / (tp + fp + 1e-10)
        recall = df['tpr']
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        df['f1'] = f1
        
        best_f1_idx = f1.argmax()
        best_f1_row = df.iloc[best_f1_idx]
        print(f"Best F1 ({best_f1_row['f1']:.3f}) at threshold {best_f1_row['threshold']:.1f}:")
        print(f"  TPR={best_f1_row['tpr']:.3f}, FPR={best_f1_row['fpr']:.3f}, latency={best_f1_row['p50_ms']:.2f}ms")
        
        # Best TPR with FPR <= 0.05
        low_fpr = df[df['fpr'] <= 0.05]
        if not low_fpr.empty:
            best_tpr_idx = low_fpr['tpr'].argmax()
            best_tpr_row = low_fpr.iloc[best_tpr_idx]
            print(f"Best TPR with FPR≤5% at threshold {best_tpr_row['threshold']:.1f}:")
            print(f"  TPR={best_tpr_row['tpr']:.3f}, FPR={best_tpr_row['fpr']:.3f}, latency={best_tpr_row['p50_ms']:.2f}ms")
        
        # Zero FPR (if any)
        zero_fpr = df[df['fpr'] == 0.0]
        if not zero_fpr.empty:
            best_zero_idx = zero_fpr['tpr'].argmax()
            best_zero_row = zero_fpr.iloc[best_zero_idx]
            print(f"Best TPR with FPR=0% at threshold {best_zero_row['threshold']:.1f}:")
            print(f"  TPR={best_zero_row['tpr']:.3f}, FPR={best_zero_row['fpr']:.3f}, latency={best_zero_row['p50_ms']:.2f}ms")

def main():
    print("Starting threshold tuning on HuggingFace dataset (2000 samples)...")
    print(f"Testing thresholds: {THRESHOLDS}")
    
    os.makedirs("results", exist_ok=True)
    os.makedirs("analysis_output", exist_ok=True)
    
    # Pipelines to tune
    pipelines = {
        'classifier': 'classifier',
        'nemo': 'nemo',
        'sig+clf': 'signature,classifier',
        'sig+rules+clf': 'signature,rules,classifier',
    }
    
    results = {}
    for name, pipeline in pipelines.items():
        df = tune_pipeline(pipeline, name)
        if not df.empty:
            results[name] = df
            # Save results
            df.to_csv(f"analysis_output/tune_{name}.csv", index=False)
    
    if results:
        # Plot results
        plot_threshold_curves(results, "analysis_output")
        
        # Find and display best thresholds
        find_best_thresholds(results)
        
        print(f"\n{'='*60}")
        print("Threshold tuning complete!")
        print(f"{'='*60}")
        print("\nResults saved to:")
        for name in results.keys():
            print(f"  - analysis_output/tune_{name}.csv")
        print("  - analysis_output/threshold_tuning.png")
    else:
        print("\nNo results collected. Check for errors above.")

if __name__ == "__main__":
    main()
