#!/usr/bin/env python3
"""
Quick Test Script for Cross-Model Validation

Tests the full pipeline with a minimal number of samples to verify:
1. API connections work
2. Defense components function correctly
3. Analysis and visualization scripts run without errors

This is a smoke test - use small sample size to minimize costs.

Usage:
    python test_cross_model_pipeline.py
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path


def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def check_dependencies():
    """Check if required packages are installed."""
    print_header("CHECKING DEPENDENCIES")
    
    required = ['pandas', 'openai', 'anthropic', 'matplotlib', 'seaborn', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg} - NOT INSTALLED")
            missing.append(pkg)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed")
    return True


def check_api_keys():
    """Check if API keys are configured."""
    print_header("CHECKING API KEYS")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    issues = []
    
    if openai_key:
        print(f"✅ OPENAI_API_KEY found ({openai_key[:10]}...)")
    else:
        print(f"❌ OPENAI_API_KEY not found")
        issues.append("OpenAI")
    
    if anthropic_key:
        print(f"✅ ANTHROPIC_API_KEY found ({anthropic_key[:10]}...)")
    else:
        print(f"❌ ANTHROPIC_API_KEY not found")
        issues.append("Anthropic")
    
    if issues:
        print(f"\n⚠️  Missing API keys for: {', '.join(issues)}")
        print(f"Add them to .env file in project root")
        return False
    
    print("\n✅ All API keys configured")
    return True


def check_data_file():
    """Check if test data exists."""
    print_header("CHECKING DATA FILE")
    
    data_file = "data/prompts_hf_augmented.csv"
    
    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return False
    
    df = pd.read_csv(data_file)
    print(f"✅ Data file found: {data_file}")
    print(f"   Total prompts: {len(df)}")
    print(f"   Attacks: {len(df[df['label'] == 'attack'])}")
    print(f"   Benign: {len(df[df['label'] == 'benign'])}")
    
    return True


def run_validation():
    """Run cross-model validation with minimal samples."""
    print_header("RUNNING CROSS-MODEL VALIDATION (10 samples)")
    
    cmd = [
        sys.executable,
        "run_cross_model_validation.py",
        "--data", "data/prompts_hf_augmented.csv",
        "--max-samples", "10",
        "--rate-limit", "1.0",
        "--output", "results/cross_model_test"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n✅ Validation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Validation failed with error code {e.returncode}")
        return False


def run_analysis():
    """Run analysis on validation results."""
    print_header("RUNNING ANALYSIS")
    
    cmd = [
        sys.executable,
        "analyze_cross_model_results.py",
        "--input", "results/cross_model_test",
        "--output", "results/cross_model_summary_test.csv",
        "--latex", "results/cross_model_table_test.tex"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n✅ Analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Analysis failed with error code {e.returncode}")
        return False


def run_visualization():
    """Run visualization."""
    print_header("RUNNING VISUALIZATION")
    
    cmd = [
        sys.executable,
        "visualize_cross_model.py",
        "--input", "results/cross_model_summary_test.csv",
        "--output-dir", "results/figures_test"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print("\n✅ Visualization completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Visualization failed with error code {e.returncode}")
        return False


def verify_outputs():
    """Verify expected outputs were created."""
    print_header("VERIFYING OUTPUTS")
    
    expected_files = [
        "results/cross_model_test/all_models_raw.csv",
        "results/cross_model_summary_test.csv",
        "results/cross_model_table_test.tex",
        "results/figures_test/model_generalization.png",
        "results/figures_test/performance_consistency.png",
        "results/figures_test/detailed_comparison_heatmap.png"
    ]
    
    all_exist = True
    for file_path in expected_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def show_summary():
    """Show summary of test results."""
    print_header("TEST SUMMARY")
    
    summary_file = "results/cross_model_summary_test.csv"
    
    if not Path(summary_file).exists():
        print("❌ Summary file not found")
        return
    
    df = pd.read_csv(summary_file)
    
    print("Results by Model:")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"\n{row['model_name']} ({row['model_vendor']}):")
        print(f"  TPR: {row['TPR']*100:.1f}%")
        print(f"  FPR: {row['FPR']*100:.1f}%")
        print(f"  Accuracy: {row['accuracy']*100:.1f}%")
        print(f"  Total Latency (p50): {row['total_latency_p50_ms']:.0f}ms")
    
    print("\n" + "-" * 80)
    print(f"\nOverall Statistics:")
    print(f"  TPR Range: {df['TPR'].min()*100:.1f}% - {df['TPR'].max()*100:.1f}%")
    print(f"  FPR Range: {df['FPR'].min()*100:.1f}% - {df['FPR'].max()*100:.1f}%")
    print(f"  TPR Std Dev: {df['TPR'].std()*100:.2f}%")
    print(f"  FPR Std Dev: {df['FPR'].std()*100:.2f}%")
    
    if df['TPR'].std() < 0.05 and df['FPR'].std() < 0.05:
        print("\n✅ GOOD GENERALIZATION: Low variance across models")
    else:
        print("\n⚠️  HIGH VARIANCE: Performance varies significantly by model")


def main():
    """Run full test pipeline."""
    print("\n" + "=" * 80)
    print("  CROSS-MODEL VALIDATION PIPELINE TEST")
    print("  This will test with 10 samples (5 attacks + 5 benign)")
    print("  Estimated cost: < $0.05")
    print("=" * 80)
    
    # Pre-flight checks
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return 1
    
    if not check_api_keys():
        print("\n❌ API key check failed. Please configure API keys.")
        return 1
    
    if not check_data_file():
        print("\n❌ Data file check failed. Please verify data file exists.")
        return 1
    
    # Run pipeline
    print_header("STARTING PIPELINE TEST")
    
    steps = [
        ("Validation", run_validation),
        ("Analysis", run_analysis),
        ("Visualization", run_visualization),
        ("Output Verification", verify_outputs)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Pipeline failed at step: {step_name}")
            return 1
    
    # Show results
    show_summary()
    
    # Final message
    print_header("✅ ALL TESTS PASSED!")
    
    print("Test outputs saved to:")
    print("  - results/cross_model_test/")
    print("  - results/cross_model_summary_test.csv")
    print("  - results/figures_test/")
    print()
    print("To run full validation with more samples:")
    print("  python run_cross_model_validation.py --max-samples 100")
    print()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
