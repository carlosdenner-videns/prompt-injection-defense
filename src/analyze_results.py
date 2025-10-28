#!/usr/bin/env python3
"""
Statistical analysis and visualization for prompt injection experiments.
Computes confidence intervals, McNemar tests, and Pareto frontiers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import bootstrap
import json
import os
from pathlib import Path


def bootstrap_ci(data, stat_func, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval for a statistic."""
    if len(data) == 0:
        return (np.nan, np.nan)
    
    rng = np.random.default_rng(seed=42)
    res = bootstrap(
        (data,),
        stat_func,
        n_resamples=n_bootstrap,
        confidence_level=confidence,
        random_state=rng
    )
    return (res.confidence_interval.low, res.confidence_interval.high)


def compute_metrics_with_ci(df, n_bootstrap=1000):
    """Compute TPR, FPR, F1 with confidence intervals."""
    attacks = df[df['label'] == 'attack']
    benign = df[df['label'] == 'benign']
    
    # TPR (True Positive Rate / Recall)
    if len(attacks) > 0:
        tpr = attacks['flagged'].mean()
        tpr_ci = bootstrap_ci(
            attacks['flagged'].values,
            np.mean,
            n_bootstrap=n_bootstrap
        )
    else:
        tpr, tpr_ci = np.nan, (np.nan, np.nan)
    
    # FPR (False Positive Rate)
    if len(benign) > 0:
        fpr = benign['flagged'].mean()
        fpr_ci = bootstrap_ci(
            benign['flagged'].values,
            np.mean,
            n_bootstrap=n_bootstrap
        )
    else:
        fpr, fpr_ci = np.nan, (np.nan, np.nan)
    
    # Precision
    flagged = df[df['flagged'] == True]
    if len(flagged) > 0:
        precision = (flagged['label'] == 'attack').mean()
        precision_ci = bootstrap_ci(
            (flagged['label'] == 'attack').values.astype(float),
            np.mean,
            n_bootstrap=n_bootstrap
        )
    else:
        precision, precision_ci = np.nan, (np.nan, np.nan)
    
    # F1 Score
    if precision > 0 and tpr > 0:
        f1 = 2 * (precision * tpr) / (precision + tpr)
    else:
        f1 = 0.0
    
    # Latency metrics
    p50 = df['lat_ms'].median()
    p95 = df['lat_ms'].quantile(0.95)
    
    return {
        'tpr': tpr,
        'tpr_ci_low': tpr_ci[0],
        'tpr_ci_high': tpr_ci[1],
        'fpr': fpr,
        'fpr_ci_low': fpr_ci[0],
        'fpr_ci_high': fpr_ci[1],
        'precision': precision,
        'precision_ci_low': precision_ci[0],
        'precision_ci_high': precision_ci[1],
        'f1': f1,
        'p50_ms': p50,
        'p95_ms': p95,
        'n_attacks': len(attacks),
        'n_benign': len(benign)
    }


def mcnemar_test(df1, df2, name1="Model 1", name2="Model 2"):
    """
    Perform McNemar's test to compare two models.
    Tests if detection rates are significantly different.
    """
    # Ensure same prompts
    merged = pd.merge(
        df1[['id', 'label', 'flagged']],
        df2[['id', 'label', 'flagged']],
        on=['id', 'label'],
        suffixes=('_1', '_2')
    )
    
    # Create contingency table
    # Both correct, both wrong, 1 correct 2 wrong, 1 wrong 2 correct
    attacks = merged[merged['label'] == 'attack']
    
    a = sum(attacks['flagged_1'] & attacks['flagged_2'])  # Both detect
    b = sum(attacks['flagged_1'] & ~attacks['flagged_2'])  # Only 1 detects
    c = sum(~attacks['flagged_1'] & attacks['flagged_2'])  # Only 2 detects
    d = sum(~attacks['flagged_1'] & ~attacks['flagged_2'])  # Neither detects
    
    # McNemar's test
    if b + c > 0:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
    else:
        statistic = 0
        p_value = 1.0
    
    print(f"\n=== McNemar's Test: {name1} vs {name2} ===")
    print(f"Contingency table (attacks only):")
    print(f"  Both detect: {a}")
    print(f"  Only {name1} detects: {b}")
    print(f"  Only {name2} detects: {c}")
    print(f"  Neither detects: {d}")
    print(f"  Chi-square statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  Result: Significant difference (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'contingency': {'a': a, 'b': b, 'c': c, 'd': d}
    }


def plot_pareto_frontier(results_dir, output_file="pareto_frontier.png"):
    """
    Plot Pareto frontier: TPR vs FPR trade-off across different defenses.
    """
    # Find all result directories
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    # Collect metrics from all experiments
    experiments = []
    for exp_dir in results_path.iterdir():
        if not exp_dir.is_dir():
            continue
        
        summary_file = exp_dir / "summary.csv"
        if not summary_file.exists():
            continue
        
        # Read summary
        summary = pd.read_csv(summary_file)
        overall = summary[summary['family'] == 'overall']
        
        if len(overall) > 0:
            experiments.append({
                'name': exp_dir.name,
                'tpr': overall.iloc[0]['TPR'],
                'fpr': overall.iloc[0]['FPR'],
                'p50_ms': overall.iloc[0]['p50_ms'],
                'p95_ms': overall.iloc[0]['p95_ms']
            })
    
    if len(experiments) == 0:
        print("No experiment results found")
        return
    
    df_exp = pd.DataFrame(experiments)
    
    # Create Pareto frontier plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: TPR vs FPR (main Pareto frontier)
    ax1 = axes[0]
    scatter = ax1.scatter(
        df_exp['fpr'], 
        df_exp['tpr'],
        s=100,
        c=df_exp['p50_ms'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    # Identify Pareto optimal points
    pareto_mask = np.ones(len(df_exp), dtype=bool)
    for i in range(len(df_exp)):
        for j in range(len(df_exp)):
            if i != j:
                # Point j dominates point i if it has higher TPR and lower/equal FPR
                if (df_exp.iloc[j]['tpr'] >= df_exp.iloc[i]['tpr'] and 
                    df_exp.iloc[j]['fpr'] <= df_exp.iloc[i]['fpr'] and
                    (df_exp.iloc[j]['tpr'] > df_exp.iloc[i]['tpr'] or 
                     df_exp.iloc[j]['fpr'] < df_exp.iloc[i]['fpr'])):
                    pareto_mask[i] = False
                    break
    
    # Highlight Pareto optimal points
    pareto_points = df_exp[pareto_mask]
    if len(pareto_points) > 0:
        # Sort by FPR for drawing the frontier
        pareto_points = pareto_points.sort_values('fpr')
        ax1.plot(
            pareto_points['fpr'],
            pareto_points['tpr'],
            'r--',
            linewidth=2,
            alpha=0.5,
            label='Pareto Frontier'
        )
        ax1.scatter(
            pareto_points['fpr'],
            pareto_points['tpr'],
            s=200,
            facecolors='none',
            edgecolors='red',
            linewidth=2.5,
            label='Pareto Optimal'
        )
    
    # Annotate points
    for idx, row in df_exp.iterrows():
        ax1.annotate(
            row['name'],
            (row['fpr'], row['tpr']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax1.set_title('Defense Performance: TPR vs FPR Trade-off', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Median Latency (ms)', rotation=270, labelpad=20)
    
    # Ideal corner annotation
    ax1.plot([0], [1], 'g*', markersize=20, label='Ideal (TPR=1, FPR=0)')
    
    # Plot 2: TPR vs Latency trade-off
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        df_exp['p50_ms'],
        df_exp['tpr'],
        s=100,
        c=df_exp['fpr'],
        cmap='RdYlGn_r',
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    for idx, row in df_exp.iterrows():
        ax2.annotate(
            row['name'],
            (row['p50_ms'], row['tpr']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    ax2.set_xlabel('Median Latency (ms)', fontsize=12)
    ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax2.set_title('Performance vs Speed Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('FPR', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPareto frontier plot saved to: {output_file}")
    plt.close()
    
    # Print Pareto optimal defenses
    print("\n=== Pareto Optimal Defenses ===")
    if len(pareto_points) > 0:
        print(pareto_points[['name', 'tpr', 'fpr', 'p50_ms']].to_string(index=False))
    else:
        print("No clear Pareto optimal points identified")
    
    return df_exp


def compare_experiments(exp_dirs, names=None):
    """Compare multiple experiments with statistical tests."""
    if names is None:
        names = [Path(d).name for d in exp_dirs]
    
    # Load all predictions
    dfs = []
    for exp_dir in exp_dirs:
        pred_file = Path(exp_dir) / "predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            dfs.append(df)
        else:
            print(f"Warning: {pred_file} not found")
            dfs.append(None)
    
    # Compute metrics with CIs
    print("\n=== Performance Metrics with 95% Confidence Intervals ===")
    for i, (df, name) in enumerate(zip(dfs, names)):
        if df is None:
            continue
        print(f"\n{name}:")
        metrics = compute_metrics_with_ci(df)
        print(f"  TPR: {metrics['tpr']:.3f} [{metrics['tpr_ci_low']:.3f}, {metrics['tpr_ci_high']:.3f}]")
        print(f"  FPR: {metrics['fpr']:.3f} [{metrics['fpr_ci_low']:.3f}, {metrics['fpr_ci_high']:.3f}]")
        print(f"  Precision: {metrics['precision']:.3f} [{metrics['precision_ci_low']:.3f}, {metrics['precision_ci_high']:.3f}]")
        print(f"  F1: {metrics['f1']:.3f}")
        print(f"  Latency p50: {metrics['p50_ms']:.2f}ms, p95: {metrics['p95_ms']:.2f}ms")
    
    # Pairwise McNemar tests
    print("\n=== Pairwise Statistical Comparisons (McNemar's Test) ===")
    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            if dfs[i] is not None and dfs[j] is not None:
                mcnemar_test(dfs[i], dfs[j], names[i], names[j])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze and visualize experiment results")
    parser.add_argument("--results", default="results", help="Results directory")
    parser.add_argument("--compare", nargs="+", help="Specific experiments to compare")
    parser.add_argument("--output", default="analysis_output", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Plot Pareto frontier
    print("Generating Pareto frontier plot...")
    df_exp = plot_pareto_frontier(args.results, f"{args.output}/pareto_frontier.png")
    
    # Compare specific experiments if specified
    if args.compare:
        exp_dirs = [f"{args.results}/{name}" for name in args.compare]
        compare_experiments(exp_dirs, args.compare)
