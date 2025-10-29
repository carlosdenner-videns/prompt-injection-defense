"""
Enhanced visualization with error bars and family-specific analysis.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


def plot_performance_with_error_bars(results_file: str = 'results/phase1_test_results.json'):
    """Create performance plots with 95% confidence intervals."""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract data
    defenses = []
    tpr_vals = []
    tpr_lower = []
    tpr_upper = []
    fpr_vals = []
    fpr_lower = []
    fpr_upper = []
    f1_vals = []
    f1_lower = []
    f1_upper = []
    latencies = []
    
    for defense_name, defense_results in results.items():
        defenses.append(defense_name)
        
        tpr_vals.append(defense_results['tpr'])
        tpr_lower.append(defense_results['tpr_ci_lower'])
        tpr_upper.append(defense_results['tpr_ci_upper'])
        
        fpr_vals.append(defense_results['fpr'])
        fpr_lower.append(defense_results['fpr_ci_lower'])
        fpr_upper.append(defense_results['fpr_ci_upper'])
        
        f1_vals.append(defense_results['f1'])
        f1_lower.append(defense_results['f1_ci_lower'])
        f1_upper.append(defense_results['f1_ci_upper'])
        
        latencies.append(defense_results['avg_latency_ms'])
    
    # Convert to numpy arrays
    tpr_vals = np.array(tpr_vals)
    tpr_errors = np.array([tpr_vals - np.array(tpr_lower), np.array(tpr_upper) - tpr_vals])
    
    fpr_vals = np.array(fpr_vals)
    fpr_errors = np.array([fpr_vals - np.array(fpr_lower), np.array(fpr_upper) - fpr_vals])
    
    f1_vals = np.array(f1_vals)
    f1_errors = np.array([f1_vals - np.array(f1_lower), np.array(f1_upper) - f1_vals])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: TPR with error bars
    ax1 = axes[0, 0]
    x_pos = np.arange(len(defenses))
    bars1 = ax1.bar(x_pos, tpr_vals, alpha=0.7, color='green')
    ax1.errorbar(x_pos, tpr_vals, yerr=tpr_errors, fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax1.set_title('Attack Detection Rate (TPR) with 95% CI', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(defenses, rotation=45, ha='right')
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (v, upper, lower) in enumerate(zip(tpr_vals, tpr_upper, tpr_lower)):
        ax1.text(i, v + tpr_errors[1][i] + 0.03, f'{v:.3f}\n[{lower:.3f}, {upper:.3f}]', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: FPR with error bars
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x_pos, fpr_vals, alpha=0.7, color='red')
    ax2.errorbar(x_pos, fpr_vals, yerr=fpr_errors, fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax2.set_ylabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax2.set_title('False Alarm Rate (FPR) with 95% CI', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(defenses, rotation=45, ha='right')
    ax2.set_ylim(0, max(fpr_vals) * 1.5 if max(fpr_vals) > 0 else 0.1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (v, upper, lower) in enumerate(zip(fpr_vals, fpr_upper, fpr_lower)):
        ax2.text(i, v + fpr_errors[1][i] + max(fpr_vals)*0.05, f'{v:.3f}\n[{lower:.3f}, {upper:.3f}]', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 3: F1 Score with error bars
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x_pos, f1_vals, alpha=0.7, color='blue')
    ax3.errorbar(x_pos, f1_vals, yerr=f1_errors, fmt='none', ecolor='black', capsize=5, linewidth=2)
    ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1 Score with 95% CI', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(defenses, rotation=45, ha='right')
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (v, upper, lower) in enumerate(zip(f1_vals, f1_upper, f1_lower)):
        ax3.text(i, v + f1_errors[1][i] + 0.03, f'{v:.3f}\n[{lower:.3f}, {upper:.3f}]', 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Latency (log scale)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, latencies, alpha=0.7, color='orange')
    ax4.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax4.set_title('Detection Latency (log scale)', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(defenses, rotation=45, ha='right')
    ax4.set_yscale('log')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(latencies):
        ax4.text(i, v * 1.2, f'{v:.2f}ms', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/phase1_performance_with_error_bars.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/phase1_performance_with_error_bars.png")
    
    return fig


def plot_family_analysis(family_file: str = 'results/phase1_family_analysis.csv'):
    """Create heatmap of performance by attack family."""
    
    df = pd.read_csv(family_file)
    
    # Prepare data for heatmap
    tpr_columns = [col for col in df.columns if col.endswith('_tpr')]
    defenses = [col.replace('_tpr', '') for col in tpr_columns]
    
    # Create matrix
    heatmap_data = df[tpr_columns].T
    heatmap_data.index = defenses
    heatmap_data.columns = df['family'].values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'True Positive Rate'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('TPR by Attack Family and Defense', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Attack Family', fontsize=12, fontweight='bold')
    ax.set_ylabel('Defense Method', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/phase1_family_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/phase1_family_heatmap.png")
    
    return fig


def plot_roc_comparison(results_file: str = 'results/phase1_test_results.json'):
    """Create ROC-style scatter plot with error bars."""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['green', 'blue', 'red', 'orange', 'purple']
    
    for idx, (defense_name, defense_results) in enumerate(results.items()):
        fpr = defense_results['fpr']
        tpr = defense_results['tpr']
        
        # Error bars
        tpr_err = [[tpr - defense_results['tpr_ci_lower']], 
                   [defense_results['tpr_ci_upper'] - tpr]]
        fpr_err = [[fpr - defense_results['fpr_ci_lower']], 
                   [defense_results['fpr_ci_upper'] - fpr]]
        
        ax.errorbar(
            fpr, tpr,
            xerr=fpr_err,
            yerr=tpr_err,
            fmt='o',
            markersize=10,
            label=defense_name,
            color=colors[idx % len(colors)],
            capsize=5,
            capthick=2,
            elinewidth=2
        )
    
    # Add diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    
    # Ideal point
    ax.plot(0, 1, 'g*', markersize=20, label='Ideal (0% FPR, 100% TPR)')
    
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
    ax.set_title('ROC Comparison with 95% Confidence Intervals', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.05, max([r['fpr'] for r in results.values()]) * 1.5)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig('results/phase1_roc_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/phase1_roc_comparison.png")
    
    return fig


def main():
    print("Generating enhanced visualizations...")
    print("="*60)
    
    # Check if results exist
    if not Path('results/phase1_test_results.json').exists():
        print("Error: results/phase1_test_results.json not found")
        print("Run `python run_enhanced_experiments.py` first")
        return
    
    # Create visualizations
    plot_performance_with_error_bars()
    plot_roc_comparison()
    
    if Path('results/phase1_family_analysis.csv').exists():
        plot_family_analysis()
    
    print("\n✅ All visualizations created successfully!")
    print("\nGenerated files:")
    print("  - results/phase1_performance_with_error_bars.png")
    print("  - results/phase1_roc_comparison.png")
    print("  - results/phase1_family_heatmap.png")


if __name__ == '__main__':
    main()
