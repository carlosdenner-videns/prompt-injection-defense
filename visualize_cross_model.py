#!/usr/bin/env python3
"""
Visualize Cross-Model Validation Results

Creates publication-quality visualizations comparing defense performance
across different LLM vendors and models.

Generates:
- Model Generalization figure showing TPR/FPR/Latency comparison
- Vendor comparison charts
- Performance consistency plots

Usage:
    python visualize_cross_model.py --input results/cross_model_summary.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_model_generalization(df: pd.DataFrame, output_path: str):
    """
    Main figure: Model Generalization
    
    Creates a multi-panel figure showing:
    - TPR/FPR comparison across models
    - Latency comparison
    - F1 score ranking
    
    Args:
        df: DataFrame with cross-model summary
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Generalization: Defense Performance Across LLM Vendors', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Color mapping by vendor
    vendor_colors = {
        'openai': '#10A37F',  # OpenAI green
        'anthropic': '#D97757'  # Anthropic orange
    }
    colors = [vendor_colors.get(v, '#888888') for v in df['model_vendor']]
    
    # Panel 1: TPR vs FPR scatter
    ax1 = axes[0, 0]
    for vendor in df['model_vendor'].unique():
        vendor_df = df[df['model_vendor'] == vendor]
        ax1.scatter(
            vendor_df['FPR'] * 100,
            vendor_df['TPR'] * 100,
            s=200,
            alpha=0.7,
            label=vendor.capitalize(),
            color=vendor_colors.get(vendor, '#888888'),
            edgecolors='black',
            linewidths=1.5
        )
        
        # Add labels
        for _, row in vendor_df.iterrows():
            model_label = row['model_name'].replace('claude-', '').replace('gpt-', '')
            ax1.annotate(
                model_label,
                xy=(row['FPR'] * 100, row['TPR'] * 100),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
    
    # Add ideal point (TPR=100%, FPR=0%)
    ax1.scatter([0], [100], s=300, marker='*', color='gold', 
                edgecolors='black', linewidths=2, label='Ideal', zorder=10)
    
    ax1.set_xlabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A) Detection Performance: TPR vs FPR', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, max(df['FPR'].max() * 100 + 5, 20))
    ax1.set_ylim(min(df['TPR'].min() * 100 - 5, 70), 105)
    
    # Panel 2: F1 Score bar chart
    ax2 = axes[0, 1]
    df_sorted = df.sort_values('F1_score', ascending=False)
    bars = ax2.barh(
        range(len(df_sorted)),
        df_sorted['F1_score'] * 100,
        color=[vendor_colors.get(v, '#888888') for v in df_sorted['model_vendor']],
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    
    ax2.set_yticks(range(len(df_sorted)))
    ax2.set_yticklabels(df_sorted['model_name'], fontsize=10)
    ax2.set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('B) Overall Performance: F1 Score Ranking', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim(0, 100)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax2.text(
            row['F1_score'] * 100 + 1,
            i,
            f"{row['F1_score'] * 100:.1f}%",
            va='center',
            fontsize=9,
            fontweight='bold'
        )
    
    # Panel 3: Latency comparison
    ax3 = axes[1, 0]
    x = np.arange(len(df))
    width = 0.35
    
    defense_bars = ax3.bar(
        x - width/2,
        df['defense_latency_p50_ms'],
        width,
        label='Defense Overhead',
        color='#3498db',
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    llm_bars = ax3.bar(
        x + width/2,
        df['llm_latency_p50_ms'],
        width,
        label='LLM Latency',
        color='#e74c3c',
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )
    
    ax3.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax3.set_title('C) Latency Breakdown by Component', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['model_name'], rotation=45, ha='right', fontsize=9)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in defense_bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.0f}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    for bar in llm_bars:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{height:.0f}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    # Panel 4: Vendor-level summary
    ax4 = axes[1, 1]
    vendor_stats = df.groupby('model_vendor').agg({
        'TPR': 'mean',
        'FPR': 'mean',
        'F1_score': 'mean',
        'total_latency_p50_ms': 'mean'
    })
    
    # Normalize for radar chart
    vendor_stats_norm = vendor_stats.copy()
    vendor_stats_norm['TPR'] = vendor_stats_norm['TPR'] * 100
    vendor_stats_norm['FPR'] = 100 - (vendor_stats_norm['FPR'] * 100)  # Invert (higher is better)
    vendor_stats_norm['F1_score'] = vendor_stats_norm['F1_score'] * 100
    vendor_stats_norm['Speed'] = 100 - (
        vendor_stats_norm['total_latency_p50_ms'] / 
        vendor_stats_norm['total_latency_p50_ms'].max() * 100
    )  # Invert (higher is better)
    
    # Radar chart
    categories = ['TPR', 'FPR\n(inverted)', 'F1 Score', 'Speed']
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    ax4 = plt.subplot(2, 2, 4, projection='polar')
    
    for vendor in vendor_stats_norm.index:
        values = [
            vendor_stats_norm.loc[vendor, 'TPR'],
            vendor_stats_norm.loc[vendor, 'FPR'],
            vendor_stats_norm.loc[vendor, 'F1_score'],
            vendor_stats_norm.loc[vendor, 'Speed']
        ]
        values += values[:1]
        
        ax4.plot(
            angles,
            values,
            'o-',
            linewidth=2,
            label=vendor.capitalize(),
            color=vendor_colors.get(vendor, '#888888')
        )
        ax4.fill(angles, values, alpha=0.15, color=vendor_colors.get(vendor, '#888888'))
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.set_yticks([25, 50, 75, 100])
    ax4.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax4.set_title('D) Vendor-Level Summary (Average)', fontsize=13, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {output_path}")
    plt.close()


def plot_variance_analysis(df: pd.DataFrame, output_path: str):
    """
    Create variance analysis plot showing consistency across models.
    
    Args:
        df: DataFrame with cross-model summary
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Performance Consistency Across Models', 
                 fontsize=14, fontweight='bold')
    
    # TPR/FPR box plots by vendor
    ax1 = axes[0]
    
    tpr_data = [
        df[df['model_vendor'] == 'openai']['TPR'] * 100,
        df[df['model_vendor'] == 'anthropic']['TPR'] * 100
    ]
    fpr_data = [
        df[df['model_vendor'] == 'openai']['FPR'] * 100,
        df[df['model_vendor'] == 'anthropic']['FPR'] * 100
    ]
    
    bp1 = ax1.boxplot(
        tpr_data,
        positions=[1, 2],
        widths=0.6,
        patch_artist=True,
        labels=['OpenAI', 'Anthropic'],
        boxprops=dict(facecolor='#10A37F', alpha=0.7),
        medianprops=dict(color='black', linewidth=2)
    )
    
    bp2 = ax1.boxplot(
        fpr_data,
        positions=[4, 5],
        widths=0.6,
        patch_artist=True,
        labels=['OpenAI', 'Anthropic'],
        boxprops=dict(facecolor='#D97757', alpha=0.7),
        medianprops=dict(color='black', linewidth=2)
    )
    
    ax1.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('TPR and FPR Distribution by Vendor', fontsize=12, fontweight='bold')
    ax1.set_xticks([1.5, 4.5])
    ax1.set_xticklabels(['TPR', 'FPR'], fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#10A37F', alpha=0.7, label='OpenAI'),
        Patch(facecolor='#D97757', alpha=0.7, label='Anthropic')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Latency distribution
    ax2 = axes[1]
    
    for vendor in df['model_vendor'].unique():
        vendor_df = df[df['model_vendor'] == vendor]
        ax2.scatter(
            vendor_df['defense_latency_p50_ms'],
            vendor_df['total_latency_p50_ms'],
            s=200,
            alpha=0.7,
            label=vendor.capitalize(),
            edgecolors='black',
            linewidth=1.5
        )
        
        # Add labels
        for _, row in vendor_df.iterrows():
            model_label = row['model_name'].replace('claude-', '').replace('gpt-', '')
            ax2.annotate(
                model_label,
                xy=(row['defense_latency_p50_ms'], row['total_latency_p50_ms']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.8
            )
    
    ax2.set_xlabel('Defense Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Latency (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Latency Relationship: Defense vs Total', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {output_path}")
    plt.close()


def plot_detailed_comparison(df: pd.DataFrame, output_path: str):
    """
    Create detailed comparison heatmap.
    
    Args:
        df: DataFrame with cross-model summary
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Select metrics for heatmap
    metrics = ['TPR', 'FPR', 'accuracy', 'F1_score', 'defense_latency_p50_ms', 'total_latency_p50_ms']
    heatmap_data = df.set_index('model_name')[metrics].T
    
    # Normalize to 0-100 scale for visualization
    heatmap_norm = heatmap_data.copy()
    for metric in metrics:
        if metric in ['TPR', 'accuracy', 'F1_score']:
            # Higher is better - scale directly
            heatmap_norm.loc[metric] = heatmap_data.loc[metric] * 100
        elif metric == 'FPR':
            # Lower is better - invert
            heatmap_norm.loc[metric] = (1 - heatmap_data.loc[metric]) * 100
        else:
            # Latency - normalize to 0-100 (lower is better)
            max_val = heatmap_data.loc[metric].max()
            heatmap_norm.loc[metric] = (1 - heatmap_data.loc[metric] / max_val) * 100
    
    # Create heatmap
    sns.heatmap(
        heatmap_norm,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Normalized Score (0-100)'},
        ax=ax,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_title('Detailed Performance Heatmap\n(All metrics normalized: higher = better)', 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    # Better labels
    metric_labels = {
        'TPR': 'True Positive Rate',
        'FPR': 'False Positive Rate (inv)',
        'accuracy': 'Accuracy',
        'F1_score': 'F1 Score',
        'defense_latency_p50_ms': 'Defense Speed',
        'total_latency_p50_ms': 'Total Speed'
    }
    ax.set_yticklabels([metric_labels.get(m, m) for m in metrics], rotation=0, fontsize=10)
    ax.set_xticklabels(heatmap_norm.columns, rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved figure: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize cross-model validation results"
    )
    parser.add_argument(
        "--input",
        default="results/cross_model_summary.csv",
        help="Input CSV with cross-model summary"
    )
    parser.add_argument(
        "--output-dir",
        default="results/figures",
        help="Output directory for figures"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CROSS-MODEL VISUALIZATION")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load data
    if not Path(args.input).exists():
        print(f"❌ Error: Input file not found: {args.input}")
        print("\nPlease run analyze_cross_model_results.py first:")
        print("  python analyze_cross_model_results.py --input results/cross_model")
        return
    
    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df)} model results\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("Generating visualizations...")
    print()
    
    # Main figure: Model Generalization
    plot_model_generalization(
        df,
        str(output_dir / "model_generalization.png")
    )
    
    # Variance analysis
    plot_variance_analysis(
        df,
        str(output_dir / "performance_consistency.png")
    )
    
    # Detailed comparison heatmap
    plot_detailed_comparison(
        df,
        str(output_dir / "detailed_comparison_heatmap.png")
    )
    
    print()
    print("=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated figures in: {args.output_dir}")
    print("  1. model_generalization.png - Main cross-model comparison")
    print("  2. performance_consistency.png - Variance analysis")
    print("  3. detailed_comparison_heatmap.png - Metric heatmap")


if __name__ == "__main__":
    main()
