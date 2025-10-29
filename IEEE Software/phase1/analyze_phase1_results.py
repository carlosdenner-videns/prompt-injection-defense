"""
Phase 1 Results Analysis Script

This script analyzes the baseline performance results and generates:
1. Comparative visualizations
2. Summary statistics for IEEE Software paper
3. Key findings and insights
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(results_dir: str = "results") -> dict:
    """Load Phase 1 results from JSON file."""
    json_path = Path(results_dir) / "phase1_results_full.json"
    
    if not json_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {json_path}\n"
            "Please run run_phase1_experiments.py first."
        )
    
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_comparison_plot(results: list, output_dir: str = "results"):
    """Generate bar chart comparing all defenses."""
    # Extract data
    defenses = [r['defense'] for r in results]
    tpr_values = [r['metrics']['tpr'] for r in results]
    fpr_values = [r['metrics']['fpr'] for r in results]
    f1_values = [r['metrics']['f1'] for r in results]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Phase 1: Baseline Defense Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: TPR comparison
    axes[0].bar(defenses, tpr_values, color='skyblue', edgecolor='black')
    axes[0].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axes[0].set_title('Detection Rate')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(tpr_values):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold')
    
    # Plot 2: FPR comparison
    axes[1].bar(defenses, fpr_values, color='salmon', edgecolor='black')
    axes[1].set_ylabel('False Positive Rate (FPR)', fontsize=12)
    axes[1].set_title('False Alarm Rate')
    axes[1].set_ylim([0, 0.15])
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(fpr_values):
        axes[1].text(i, v + 0.005, f'{v:.1%}', ha='center', fontweight='bold')
    
    # Plot 3: F1 Score comparison
    axes[2].bar(defenses, f1_values, color='lightgreen', edgecolor='black')
    axes[2].set_ylabel('F1 Score', fontsize=12)
    axes[2].set_title('Overall Performance')
    axes[2].set_ylim([0, 1])
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(f1_values):
        axes[2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / "phase1_comparison_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved comparison plot: {output_path}")
    
    plt.close()


def generate_tradeoff_plot(results: list, output_dir: str = "results"):
    """Generate TPR vs FPR trade-off scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract data
    defenses = [r['defense'] for r in results]
    tpr_values = [r['metrics']['tpr'] for r in results]
    fpr_values = [r['metrics']['fpr'] for r in results]
    
    # Plot scatter with labels
    colors = ['blue', 'red', 'green', 'orange']
    for i, defense in enumerate(defenses):
        ax.scatter(fpr_values[i], tpr_values[i], s=200, c=colors[i % len(colors)],
                  alpha=0.7, edgecolor='black', linewidth=2, label=defense)
        
        # Add text label
        ax.annotate(defense, (fpr_values[i], tpr_values[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7))
    
    # Plot ideal point (top-left corner)
    ax.scatter(0, 1, s=300, c='gold', marker='*', edgecolor='black',
              linewidth=2, label='Ideal (TPR=1, FPR=0)', zorder=5)
    
    # Formatting
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax.set_title('Phase 1: TPR vs FPR Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim([-0.02, max(fpr_values) + 0.05])
    ax.set_ylim([min(tpr_values) - 0.1, 1.05])
    
    # Save figure
    output_path = Path(output_dir) / "phase1_tradeoff_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved trade-off plot: {output_path}")
    
    plt.close()


def generate_summary_report(results: list, output_dir: str = "results"):
    """Generate text summary report for IEEE Software paper."""
    report_lines = []
    
    report_lines.append("="*70)
    report_lines.append("PHASE 1: BASELINES AND PRIOR ART COMPARISON - SUMMARY REPORT")
    report_lines.append("="*70)
    report_lines.append("")
    
    # Sort by F1 score (descending)
    sorted_results = sorted(results, key=lambda x: x['metrics']['f1'], reverse=True)
    
    report_lines.append("PERFORMANCE RANKING (by F1 Score):")
    report_lines.append("-" * 70)
    
    for rank, r in enumerate(sorted_results, 1):
        defense = r['defense']
        metrics = r['metrics']
        latency = r['latency']
        
        report_lines.append(f"\n{rank}. {defense}")
        report_lines.append(f"   TPR:       {metrics['tpr']:.1%} [{metrics['tpr_ci_95'][0]:.1%}, {metrics['tpr_ci_95'][1]:.1%}]")
        report_lines.append(f"   FPR:       {metrics['fpr']:.1%} [{metrics['fpr_ci_95'][0]:.1%}, {metrics['fpr_ci_95'][1]:.1%}]")
        report_lines.append(f"   Precision: {metrics['precision']:.1%}")
        report_lines.append(f"   F1 Score:  {metrics['f1']:.3f}")
        report_lines.append(f"   Latency:   {latency['p50_ms']:.3f} ms (p50)")
    
    report_lines.append("\n" + "="*70)
    report_lines.append("KEY FINDINGS FOR IEEE SOFTWARE PAPER")
    report_lines.append("="*70)
    
    # Find best/worst performers
    best = sorted_results[0]
    worst = sorted_results[-1]
    
    # Find fastest
    fastest = min(results, key=lambda x: x['latency']['p50_ms'])
    
    # Find lowest FPR
    lowest_fpr = min(results, key=lambda x: x['metrics']['fpr'])
    
    report_lines.append("\n1. BEST OVERALL PERFORMANCE:")
    report_lines.append(f"   {best['defense']}: F1={best['metrics']['f1']:.3f}")
    report_lines.append(f"   - Achieves {best['metrics']['tpr']:.1%} TPR with {best['metrics']['fpr']:.1%} FPR")
    
    report_lines.append("\n2. LOWEST FALSE POSITIVE RATE:")
    report_lines.append(f"   {lowest_fpr['defense']}: FPR={lowest_fpr['metrics']['fpr']:.1%}")
    report_lines.append(f"   - Critical for production deployment (user experience)")
    
    report_lines.append("\n3. FASTEST DEFENSE:")
    report_lines.append(f"   {fastest['defense']}: {fastest['latency']['p50_ms']:.3f} ms")
    report_lines.append(f"   - Near-zero overhead for high-throughput applications")
    
    report_lines.append("\n4. BASELINE COMPARISON:")
    report_lines.append("   - Traditional rules (NeMo): Moderate performance baseline")
    report_lines.append("   - Commercial API: Real-world comparator (if tested)")
    report_lines.append("   - Signature-only: High TPR but limited scope")
    
    report_lines.append("\n5. RESEARCH IMPLICATIONS:")
    
    # Calculate improvement potential
    best_tpr = max(r['metrics']['tpr'] for r in results)
    best_fpr = min(r['metrics']['fpr'] for r in results)
    
    report_lines.append(f"   - Current best TPR: {best_tpr:.1%}")
    report_lines.append(f"   - Current best FPR: {best_fpr:.1%}")
    report_lines.append(f"   - Gap to ideal (TPR=100%, FPR=0%): {1-best_tpr:.1%} TPR gap")
    report_lines.append("   - Motivation for data-driven pattern discovery (Phases 2-6)")
    
    report_lines.append("\n" + "="*70)
    report_lines.append("REPRODUCTION NOTES")
    report_lines.append("="*70)
    report_lines.append("\nDataset: HuggingFace prompt-injections (2,000 samples)")
    report_lines.append("Attacks: 1,000 (50%)")
    report_lines.append("Benign:  1,000 (50%)")
    report_lines.append("Seed:    42 (reproducible)")
    report_lines.append("\nStatistical validation:")
    report_lines.append("- Bootstrap confidence intervals: 1,000 iterations, 95% confidence")
    report_lines.append("- All results include [lower, upper] CI bounds")
    
    report_lines.append("\n" + "="*70)
    report_lines.append("END OF REPORT")
    report_lines.append("="*70)
    
    # Write to file
    report_text = "\n".join(report_lines)
    output_path = Path(output_dir) / "phase1_summary_report.txt"
    with open(output_path, 'w') as f:
        f.write(report_text)
    
    print(f"  Saved summary report: {output_path}")
    
    # Also print to console
    print("\n" + report_text)


def generate_latex_snippet(results: list, output_dir: str = "results"):
    """Generate LaTeX snippet for findings section."""
    latex_lines = []
    
    latex_lines.append("% Phase 1 Key Findings - LaTeX Snippet")
    latex_lines.append("% Insert into IEEE Software paper")
    latex_lines.append("")
    latex_lines.append("\\subsection{Baseline Performance}")
    latex_lines.append("")
    latex_lines.append("We evaluated four baseline defenses to establish reference points:")
    latex_lines.append("\\begin{itemize}")
    
    for r in results:
        defense = r['defense']
        tpr = r['metrics']['tpr']
        fpr = r['metrics']['fpr']
        f1 = r['metrics']['f1']
        
        latex_lines.append(f"  \\item \\textbf{{{defense}}}: "
                          f"TPR={tpr:.1%}, FPR={fpr:.1%}, F1={f1:.3f}")
    
    latex_lines.append("\\end{itemize}")
    latex_lines.append("")
    latex_lines.append("Table~\\ref{tab:phase1-baselines} shows the complete performance comparison.")
    
    # Write to file
    latex_text = "\n".join(latex_lines)
    output_path = Path(output_dir) / "phase1_findings_snippet.tex"
    with open(output_path, 'w') as f:
        f.write(latex_text)
    
    print(f"  Saved LaTeX snippet: {output_path}")


def main():
    print("="*70)
    print("PHASE 1 RESULTS ANALYSIS")
    print("="*70)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"  Loaded {len(results)} defense results")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_comparison_plot(results)
    generate_tradeoff_plot(results)
    
    # Generate reports
    print("\nGenerating reports...")
    generate_summary_report(results)
    generate_latex_snippet(results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/phase1_comparison_plot.png")
    print("  - results/phase1_tradeoff_plot.png")
    print("  - results/phase1_summary_report.txt")
    print("  - results/phase1_findings_snippet.tex")
    print("\nUse these files for the IEEE Software paper.")


if __name__ == '__main__':
    main()
