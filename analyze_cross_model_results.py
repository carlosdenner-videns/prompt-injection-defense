#!/usr/bin/env python3
"""
Analyze Cross-Model Validation Results

Aggregates results from cross-model validation runs to create a comprehensive
summary comparing performance across different LLM vendors and models.

Produces:
- results/cross_model_summary.csv: Comprehensive comparison table
- Console output with detailed analysis

Usage:
    python analyze_cross_model_results.py --input results/cross_model
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def load_all_model_results(base_dir: str) -> pd.DataFrame:
    """
    Load results from all model subdirectories.
    
    Args:
        base_dir: Base directory containing model subdirectories
        
    Returns:
        Combined DataFrame with all model results
    """
    all_results = []
    
    # Look for summary.csv files in subdirectories
    base_path = Path(base_dir)
    
    for model_dir in base_path.iterdir():
        if not model_dir.is_dir():
            continue
        
        summary_file = model_dir / "summary.csv"
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            all_results.append(df)
            print(f"✅ Loaded: {model_dir.name}")
    
    if not all_results:
        raise ValueError(f"No summary.csv files found in {base_dir}")
    
    return pd.concat(all_results, ignore_index=True)


def calculate_aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate aggregate metrics across all models.
    
    Args:
        df: DataFrame with per-model summaries
        
    Returns:
        DataFrame with aggregate statistics
    """
    # Add derived metrics
    df['F1_score'] = 2 * (
        (1 - df['FPR']) * df['TPR']
    ) / ((1 - df['FPR']) + df['TPR'] + 1e-10)
    
    # Latency overhead (defense latency as % of total)
    df['defense_overhead_pct'] = (
        df['defense_latency_p50_ms'] / 
        (df['total_latency_p50_ms'] + 1e-10) * 100
    )
    
    return df


def print_comparison_table(df: pd.DataFrame):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("CROSS-MODEL PERFORMANCE COMPARISON")
    print("=" * 100)
    print()
    
    # Select key columns for display
    display_cols = [
        'model_name',
        'model_vendor',
        'TPR',
        'FPR',
        'accuracy',
        'F1_score',
        'defense_latency_p50_ms',
        'total_latency_p50_ms',
        'defense_overhead_pct'
    ]
    
    display_df = df[display_cols].copy()
    
    # Format percentages
    for col in ['TPR', 'FPR', 'accuracy', 'F1_score']:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + '%'
    
    display_df['defense_overhead_pct'] = (
        display_df['defense_overhead_pct'].round(2).astype(str) + '%'
    )
    
    # Round latencies
    for col in ['defense_latency_p50_ms', 'total_latency_p50_ms']:
        display_df[col] = display_df[col].round(2)
    
    print(display_df.to_string(index=False))
    print()


def print_vendor_summary(df: pd.DataFrame):
    """Print summary by vendor."""
    print("\n" + "=" * 100)
    print("VENDOR-LEVEL SUMMARY")
    print("=" * 100)
    print()
    
    vendor_summary = df.groupby('model_vendor').agg({
        'TPR': ['mean', 'std', 'min', 'max'],
        'FPR': ['mean', 'std', 'min', 'max'],
        'accuracy': ['mean', 'std', 'min', 'max'],
        'total_latency_p50_ms': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(vendor_summary)
    print()


def print_key_insights(df: pd.DataFrame):
    """Print key insights from the analysis."""
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    print()
    
    # Best overall model
    best_f1 = df.loc[df['F1_score'].idxmax()]
    print(f"🏆 Best F1 Score: {best_f1['model_name']} "
          f"(F1={best_f1['F1_score']:.4f}, TPR={best_f1['TPR']:.4f}, FPR={best_f1['FPR']:.4f})")
    
    # Highest TPR
    best_tpr = df.loc[df['TPR'].idxmax()]
    print(f"🎯 Highest TPR: {best_tpr['model_name']} (TPR={best_tpr['TPR']:.4f})")
    
    # Lowest FPR
    best_fpr = df.loc[df['FPR'].idxmin()]
    print(f"✅ Lowest FPR: {best_fpr['model_name']} (FPR={best_fpr['FPR']:.4f})")
    
    # Fastest defense
    fastest_defense = df.loc[df['defense_latency_p50_ms'].idxmin()]
    print(f"⚡ Fastest Defense: {fastest_defense['model_name']} "
          f"({fastest_defense['defense_latency_p50_ms']:.2f}ms)")
    
    # Fastest total
    fastest_total = df.loc[df['total_latency_p50_ms'].idxmin()]
    print(f"🚀 Fastest Overall: {fastest_total['model_name']} "
          f"({fastest_total['total_latency_p50_ms']:.2f}ms)")
    
    print()
    
    # Variance analysis
    tpr_variance = df['TPR'].std()
    fpr_variance = df['FPR'].std()
    
    print(f"📊 Performance Variance:")
    print(f"   TPR std dev: {tpr_variance:.4f} ({'Low' if tpr_variance < 0.05 else 'High'} variance)")
    print(f"   FPR std dev: {fpr_variance:.4f} ({'Low' if fpr_variance < 0.05 else 'High'} variance)")
    
    if tpr_variance < 0.05 and fpr_variance < 0.05:
        print(f"   ✅ GOOD GENERALIZATION: Defense performs consistently across models")
    else:
        print(f"   ⚠️  VARIABLE PERFORMANCE: Defense effectiveness varies by model")
    
    print()
    
    # Latency delta analysis
    latency_variance = df['total_latency_p50_ms'].std()
    print(f"⏱️  Latency Variance:")
    print(f"   Total latency std dev: {latency_variance:.2f}ms")
    print(f"   Range: {df['total_latency_p50_ms'].min():.2f}ms - "
          f"{df['total_latency_p50_ms'].max():.2f}ms")
    print(f"   Delta: {df['total_latency_p50_ms'].max() - df['total_latency_p50_ms'].min():.2f}ms")
    
    print()


def save_summary(df: pd.DataFrame, output_path: str):
    """
    Save comprehensive summary to CSV.
    
    Args:
        df: DataFrame with aggregate metrics
        output_path: Path to save summary CSV
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sort by F1 score (descending)
    df_sorted = df.sort_values('F1_score', ascending=False)
    
    # Save
    df_sorted.to_csv(output_path, index=False)
    print(f"✅ Saved summary to: {output_path}")


def generate_latex_table(df: pd.DataFrame, output_path: str):
    """
    Generate LaTeX table for paper inclusion.
    
    Args:
        df: DataFrame with results
        output_path: Path to save LaTeX file
    """
    latex_cols = [
        'model_name',
        'model_vendor',
        'TPR',
        'FPR',
        'F1_score',
        'total_latency_p50_ms'
    ]
    
    latex_df = df[latex_cols].copy()
    
    # Format for LaTeX
    latex_df['TPR'] = (latex_df['TPR'] * 100).round(1).astype(str)
    latex_df['FPR'] = (latex_df['FPR'] * 100).round(1).astype(str)
    latex_df['F1_score'] = (latex_df['F1_score'] * 100).round(1).astype(str)
    latex_df['total_latency_p50_ms'] = latex_df['total_latency_p50_ms'].round(0).astype(int)
    
    # Rename columns
    latex_df.columns = ['Model', 'Vendor', 'TPR (%)', 'FPR (%)', 'F1 (%)', 'Latency (ms)']
    
    # Generate LaTeX
    latex_str = latex_df.to_latex(
        index=False,
        caption="Cross-model performance of signature + classifier defense pipeline",
        label="tab:cross_model",
        escape=False
    )
    
    # Save
    with open(output_path, 'w') as f:
        f.write(latex_str)
    
    print(f"✅ Saved LaTeX table to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-model validation results"
    )
    parser.add_argument(
        "--input",
        default="results/cross_model",
        help="Input directory with model subdirectories"
    )
    parser.add_argument(
        "--output",
        default="results/cross_model_summary.csv",
        help="Output path for summary CSV"
    )
    parser.add_argument(
        "--latex",
        default="results/cross_model_table.tex",
        help="Output path for LaTeX table"
    )
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("CROSS-MODEL RESULTS ANALYSIS")
    print("=" * 100)
    print(f"Input directory: {args.input}")
    print(f"Output summary: {args.output}")
    print(f"Output LaTeX: {args.latex}")
    print()
    
    # Load results
    print("Loading model results...")
    df = load_all_model_results(args.input)
    print(f"Loaded {len(df)} model results\n")
    
    # Calculate aggregate metrics
    df = calculate_aggregate_metrics(df)
    
    # Print analysis
    print_comparison_table(df)
    print_vendor_summary(df)
    print_key_insights(df)
    
    # Save outputs
    save_summary(df, args.output)
    generate_latex_table(df, args.latex)
    
    print("\n" + "=" * 100)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 100)
    print("\nNext step:")
    print(f"  python visualize_cross_model.py --input {args.output}")


if __name__ == "__main__":
    main()
