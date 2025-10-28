#!/usr/bin/env python3
"""
Batch Runner for Cross-Model Validation

Runs cross-model validation with multiple configurations to explore:
- Different sample sizes
- Different thresholds
- Different model subsets

Generates comparative analysis across configurations.

Usage:
    python batch_cross_model_validation.py --mode quick
    python batch_cross_model_validation.py --mode full
    python batch_cross_model_validation.py --mode threshold-sweep
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


class BatchRunner:
    """Run cross-model validation in batch mode."""
    
    def __init__(self, base_output_dir: str = "results/batch_cross_model"):
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def run_validation(self, config: dict) -> bool:
        """
        Run validation with specific configuration.
        
        Args:
            config: Dictionary with validation parameters
            
        Returns:
            True if successful, False otherwise
        """
        # Build command
        cmd = [
            sys.executable,
            "run_cross_model_validation.py",
            "--data", config.get("data", "data/prompts_hf_augmented.csv"),
            "--max-samples", str(config.get("max_samples", 50)),
            "--threshold", str(config.get("threshold", 0.5)),
            "--output", config["output"],
            "--rate-limit", str(config.get("rate_limit", 0.5))
        ]
        
        print(f"\n{'='*80}")
        print(f"Running: {config['name']}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")
        
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed: {config['name']}")
            return False
    
    def run_analysis(self, input_dir: str, output_name: str) -> bool:
        """Run analysis on validation results."""
        cmd = [
            sys.executable,
            "analyze_cross_model_results.py",
            "--input", input_dir,
            "--output", f"{self.base_output_dir}/{output_name}_summary.csv",
            "--latex", f"{self.base_output_dir}/{output_name}_table.tex"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def run_visualization(self, summary_file: str, output_name: str) -> bool:
        """Run visualization."""
        cmd = [
            sys.executable,
            "visualize_cross_model.py",
            "--input", summary_file,
            "--output-dir", f"{self.base_output_dir}/figures_{output_name}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def quick_test(self):
        """Quick test with minimal samples."""
        print("\n" + "="*80)
        print("QUICK TEST MODE")
        print("Testing with 20 samples per model")
        print("="*80)
        
        config = {
            "name": "Quick Test (20 samples)",
            "max_samples": 20,
            "threshold": 0.5,
            "rate_limit": 1.0,
            "output": f"{self.base_output_dir}/quick_test"
        }
        
        if self.run_validation(config):
            self.run_analysis(config["output"], "quick_test")
            self.run_visualization(
                f"{self.base_output_dir}/quick_test_summary.csv",
                "quick_test"
            )
    
    def full_validation(self):
        """Full validation with recommended sample size."""
        print("\n" + "="*80)
        print("FULL VALIDATION MODE")
        print("Testing with 100 samples per model")
        print("="*80)
        
        config = {
            "name": "Full Validation (100 samples)",
            "max_samples": 100,
            "threshold": 0.5,
            "rate_limit": 0.5,
            "output": f"{self.base_output_dir}/full_validation"
        }
        
        if self.run_validation(config):
            self.run_analysis(config["output"], "full_validation")
            self.run_visualization(
                f"{self.base_output_dir}/full_validation_summary.csv",
                "full_validation"
            )
    
    def threshold_sweep(self):
        """Test multiple thresholds to find optimal."""
        print("\n" + "="*80)
        print("THRESHOLD SWEEP MODE")
        print("Testing thresholds: 0.3, 0.4, 0.5, 0.6, 0.7")
        print("="*80)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        for threshold in thresholds:
            config = {
                "name": f"Threshold {threshold}",
                "max_samples": 50,
                "threshold": threshold,
                "rate_limit": 0.5,
                "output": f"{self.base_output_dir}/threshold_{int(threshold*10)}"
            }
            
            if self.run_validation(config):
                self.run_analysis(
                    config["output"],
                    f"threshold_{int(threshold*10)}"
                )
        
        # Generate comparative analysis
        self._compare_thresholds(thresholds)
    
    def _compare_thresholds(self, thresholds: list):
        """Compare results across different thresholds."""
        import pandas as pd
        
        print("\n" + "="*80)
        print("THRESHOLD COMPARISON")
        print("="*80)
        
        results = []
        for threshold in thresholds:
            summary_file = f"{self.base_output_dir}/threshold_{int(threshold*10)}_summary.csv"
            if Path(summary_file).exists():
                df = pd.read_csv(summary_file)
                df['threshold'] = threshold
                results.append(df)
        
        if not results:
            print("❌ No threshold results to compare")
            return
        
        combined = pd.concat(results, ignore_index=True)
        
        # Calculate averages by threshold
        threshold_summary = combined.groupby('threshold').agg({
            'TPR': 'mean',
            'FPR': 'mean',
            'accuracy': 'mean',
            'F1_score': 'mean'
        }).round(4)
        
        print("\nAverage Performance by Threshold:")
        print(threshold_summary.to_string())
        
        # Find optimal threshold (maximize F1)
        best_threshold = threshold_summary['F1_score'].idxmax()
        print(f"\n✅ Optimal threshold: {best_threshold}")
        print(f"   F1 Score: {threshold_summary.loc[best_threshold, 'F1_score']:.4f}")
        
        # Save comparison
        output_file = f"{self.base_output_dir}/threshold_comparison.csv"
        combined.to_csv(output_file, index=False)
        print(f"\n✅ Saved threshold comparison to: {output_file}")
    
    def sample_size_study(self):
        """Test different sample sizes to assess stability."""
        print("\n" + "="*80)
        print("SAMPLE SIZE STUDY")
        print("Testing sample sizes: 20, 50, 100, 200")
        print("="*80)
        
        sample_sizes = [20, 50, 100, 200]
        
        for size in sample_sizes:
            config = {
                "name": f"Sample Size {size}",
                "max_samples": size,
                "threshold": 0.5,
                "rate_limit": 0.5,
                "output": f"{self.base_output_dir}/samples_{size}"
            }
            
            if self.run_validation(config):
                self.run_analysis(
                    config["output"],
                    f"samples_{size}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for cross-model validation"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "threshold-sweep", "sample-size"],
        default="quick",
        help="Batch mode to run"
    )
    parser.add_argument(
        "--output-dir",
        default="results/batch_cross_model",
        help="Base output directory"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("BATCH CROSS-MODEL VALIDATION RUNNER")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize runner
    runner = BatchRunner(base_output_dir=args.output_dir)
    
    # Run selected mode
    if args.mode == "quick":
        runner.quick_test()
    elif args.mode == "full":
        runner.full_validation()
    elif args.mode == "threshold-sweep":
        runner.threshold_sweep()
    elif args.mode == "sample-size":
        runner.sample_size_study()
    
    print("\n" + "="*80)
    print("✅ BATCH VALIDATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated outputs:")
    print("  - *_summary.csv: Summary metrics for each run")
    print("  - *_table.tex: LaTeX tables")
    print("  - figures_*/: Visualization figures")
    
    if args.mode == "threshold-sweep":
        print("  - threshold_comparison.csv: Comparison across thresholds")


if __name__ == "__main__":
    main()
