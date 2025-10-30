"""
Generate publication-quality visualizations for Phase 1 results.

Creates 3 main plots:
1. Performance comparison (4-panel: TPR, FPR, F1, Latency) with error bars
2. ROC scatter plot comparing INPUT vs OUTPUT detection
3. Attack family heatmap showing per-family TPR

Usage:
    python visualize_phase1_final.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("colorblind")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Create output directory
output_dir = Path('results/plots')
output_dir.mkdir(exist_ok=True, parents=True)

print("="*60)
print("Phase 1: Publication-Quality Visualizations")
print("="*60)

# Load INPUT detection results (from earlier experiments)
print("\nLoading INPUT detection results...")
input_df = pd.read_csv('results/phase1_baseline_performance.csv')
print(f"  Loaded {len(input_df)} INPUT defenses")
print(f"  Defenses: {input_df['Defense'].tolist()}")

# Load OUTPUT detection results (fixed methodology)
print("\nLoading OUTPUT detection results...")
output_df = pd.read_csv('results/phase1_output_fixed_results.csv')
print(f"  Loaded {len(output_df)} OUTPUT defenses")
print(f"  Defenses: {output_df['defense'].tolist()}")

# Standardize column names
input_df = input_df.rename(columns={
    'Defense': 'defense',
    'TPR': 'tpr',
    'FPR': 'fpr',
    'Precision': 'precision',
    'F1': 'f1'
})

# Add paradigm labels
input_df['paradigm'] = 'INPUT'
output_df['paradigm'] = 'OUTPUT'

# ============================================================
# PLOT 1: 4-Panel Performance Comparison
# ============================================================
print("\n" + "="*60)
print("Creating Plot 1: 4-Panel Performance Comparison")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Phase 1: Defense Performance Comparison (INPUT vs OUTPUT Detection)', 
             fontsize=14, fontweight='bold')

# Prepare data for plotting
defenses = ['Signature-Only', 'Rules-Only', 'NeMo-Baseline', 'OpenAI-Moderation']
colors = sns.color_palette("colorblind", n_colors=4)

# Panel 1: TPR with error bars
ax = axes[0, 0]
x_pos = np.arange(len(defenses))
width = 0.35

for i, defense in enumerate(defenses):
    # INPUT data
    input_row = input_df[input_df['defense'] == defense]
    if not input_row.empty:
        tpr_input = input_row['tpr'].values[0]
        ax.bar(i - width/2, tpr_input, width, label='INPUT' if i == 0 else '', 
               color=colors[0], alpha=0.8)
    
    # OUTPUT data
    output_row = output_df[output_df['defense'] == defense]
    if not output_row.empty:
        tpr_output = output_row['tpr'].values[0]
        tpr_ci_lower = output_row['tpr_ci_lower'].values[0]
        tpr_ci_upper = output_row['tpr_ci_upper'].values[0]
        err = [[tpr_output - tpr_ci_lower], [tpr_ci_upper - tpr_output]]
        
        ax.bar(i + width/2, tpr_output, width, yerr=err, capsize=5,
               label='OUTPUT' if i == 0 else '', color=colors[1], alpha=0.8)

ax.set_ylabel('True Positive Rate (TPR)', fontweight='bold')
ax.set_title('A) Detection Rate', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(defenses, rotation=45, ha='right')
ax.set_ylim(0, 1.0)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)

# Panel 2: FPR with error bars
ax = axes[0, 1]

for i, defense in enumerate(defenses):
    # INPUT data
    input_row = input_df[input_df['defense'] == defense]
    if not input_row.empty:
        fpr_input = input_row['fpr'].values[0]
        ax.bar(i - width/2, fpr_input, width, color=colors[0], alpha=0.8)
    
    # OUTPUT data
    output_row = output_df[output_df['defense'] == defense]
    if not output_row.empty:
        fpr_output = output_row['fpr'].values[0]
        fpr_ci_lower = output_row['fpr_ci_lower'].values[0]
        fpr_ci_upper = output_row['fpr_ci_upper'].values[0]
        err = [[fpr_output - fpr_ci_lower], [fpr_ci_upper - fpr_output]]
        
        ax.bar(i + width/2, fpr_output, width, yerr=err, capsize=5,
               color=colors[1], alpha=0.8)

ax.set_ylabel('False Positive Rate (FPR)', fontweight='bold')
ax.set_title('B) False Alarm Rate', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(defenses, rotation=45, ha='right')
ax.set_ylim(0, 0.15)
ax.grid(axis='y', alpha=0.3)

# Panel 3: F1 Score with error bars
ax = axes[1, 0]

for i, defense in enumerate(defenses):
    # INPUT data
    input_row = input_df[input_df['defense'] == defense]
    if not input_row.empty:
        f1_input = input_row['f1'].values[0]
        ax.bar(i - width/2, f1_input, width, color=colors[0], alpha=0.8)
    
    # OUTPUT data
    output_row = output_df[output_df['defense'] == defense]
    if not output_row.empty:
        f1_output = output_row['f1'].values[0]
        f1_ci_lower = output_row['f1_ci_lower'].values[0]
        f1_ci_upper = output_row['f1_ci_upper'].values[0]
        err = [[f1_output - f1_ci_lower], [f1_ci_upper - f1_output]]
        
        ax.bar(i + width/2, f1_output, width, yerr=err, capsize=5,
               color=colors[1], alpha=0.8)

ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('C) Overall Performance', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(defenses, rotation=45, ha='right')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Panel 4: Latency comparison
ax = axes[1, 1]

for i, defense in enumerate(defenses):
    # INPUT data
    input_row = input_df[input_df['defense'] == defense]
    if not input_row.empty and 'Latency_p50_ms' in input_df.columns:
        lat_input = input_row['Latency_p50_ms'].values[0]
        ax.bar(i - width/2, lat_input, width, color=colors[0], alpha=0.8)
    
    # OUTPUT data
    output_row = output_df[output_df['defense'] == defense]
    if not output_row.empty:
        lat_output = output_row['latency_p50'].values[0]
        ax.bar(i + width/2, lat_output, width, color=colors[1], alpha=0.8)

ax.set_ylabel('Latency (ms, p50)', fontweight='bold')
ax.set_title('D) Response Time', fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(defenses, rotation=45, ha='right')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3, which='both')

plt.tight_layout()
plot1_path = output_dir / 'phase1_performance_comparison.png'
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {plot1_path}")
plt.close()

# ============================================================
# PLOT 2: ROC Scatter (TPR vs FPR)
# ============================================================
print("\n" + "="*60)
print("Creating Plot 2: ROC Scatter with Error Regions")
print("="*60)

fig, ax = plt.subplots(figsize=(10, 8))

markers = ['o', 's', '^', 'D']
marker_size = 120

for i, defense in enumerate(defenses):
    # INPUT point
    input_row = input_df[input_df['defense'] == defense]
    if not input_row.empty:
        fpr_in = input_row['fpr'].values[0]
        tpr_in = input_row['tpr'].values[0]
        ax.scatter(fpr_in, tpr_in, s=marker_size, marker=markers[i], 
                  color=colors[0], alpha=0.7, edgecolors='black', linewidth=1.5,
                  label=f'{defense} (INPUT)', zorder=3)
    
    # OUTPUT point with error region
    output_row = output_df[output_df['defense'] == defense]
    if not output_row.empty:
        fpr_out = output_row['fpr'].values[0]
        tpr_out = output_row['tpr'].values[0]
        
        fpr_ci_lower = output_row['fpr_ci_lower'].values[0]
        fpr_ci_upper = output_row['fpr_ci_upper'].values[0]
        tpr_ci_lower = output_row['tpr_ci_lower'].values[0]
        tpr_ci_upper = output_row['tpr_ci_upper'].values[0]
        
        # Plot error rectangle
        width_err = fpr_ci_upper - fpr_ci_lower
        height_err = tpr_ci_upper - tpr_ci_lower
        
        from matplotlib.patches import Rectangle
        rect = Rectangle((fpr_ci_lower, tpr_ci_lower), width_err, height_err,
                         linewidth=0, edgecolor=None, facecolor=colors[1], alpha=0.2)
        ax.add_patch(rect)
        
        # Plot point
        ax.scatter(fpr_out, tpr_out, s=marker_size, marker=markers[i],
                  color=colors[1], alpha=0.7, edgecolors='black', linewidth=1.5,
                  label=f'{defense} (OUTPUT)', zorder=3)

# Reference lines
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Random Classifier')
ax.plot([0, 0, 1], [0, 1, 1], 'g--', alpha=0.3, linewidth=1, label='Perfect Classifier')

ax.set_xlabel('False Positive Rate (FPR)', fontweight='bold', fontsize=12)
ax.set_ylabel('True Positive Rate (TPR)', fontweight='bold', fontsize=12)
ax.set_title('Phase 1: ROC Comparison (INPUT vs OUTPUT Detection)',
             fontweight='bold', fontsize=13)
ax.set_xlim(-0.02, 0.20)
ax.set_ylim(-0.02, 1.02)
ax.legend(loc='lower right', fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot2_path = output_dir / 'phase1_roc_comparison.png'
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {plot2_path}")
plt.close()

# ============================================================
# PLOT 3: Attack Family Heatmap (INPUT only - no OUTPUT data)
# ============================================================
print("\n" + "="*60)
print("Creating Plot 3: Attack Family Heatmap")
print("="*60)

# Load family analysis (from INPUT detection experiments)
family_path = Path('results/phase1_family_analysis.csv')
if family_path.exists():
    family_df = pd.read_csv(family_path)
    
    # Reshape data for heatmap (wide to long format)
    family_df = family_df.set_index('family')
    family_df = family_df.drop(columns=['count'], errors='ignore')
    
    # Rename columns to remove '_tpr' suffix
    family_df.columns = [col.replace('_tpr', '') for col in family_df.columns]
    
    # Reorder columns
    available_defenses = [d for d in defenses if d in family_df.columns]
    heatmap_data = family_df[available_defenses]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'TPR'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_title('Phase 1: Per-Family Attack Detection (INPUT Detection)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Defense Strategy', fontweight='bold', fontsize=11)
    ax.set_ylabel('Attack Family', fontweight='bold', fontsize=11)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plot3_path = output_dir / 'phase1_family_heatmap.png'
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {plot3_path}")
    plt.close()
else:
    print(f"  ⚠️  Family analysis not found: {family_path}")
    print("     Skipping attack family heatmap")

# ============================================================
# BONUS PLOT: Key Finding Summary
# ============================================================
print("\n" + "="*60)
print("Creating Bonus Plot: Key Findings Summary")
print("="*60)

fig, ax = plt.subplots(figsize=(12, 6))

# Create summary table
summary_data = []

for defense in defenses:
    input_row = input_df[input_df['defense'] == defense]
    output_row = output_df[output_df['defense'] == defense]
    
    if not input_row.empty:
        tpr_in = input_row['tpr'].values[0]
        fpr_in = input_row['fpr'].values[0]
        f1_in = input_row['f1'].values[0]
    else:
        tpr_in = fpr_in = f1_in = 0.0
    
    if not output_row.empty:
        tpr_out = output_row['tpr'].values[0]
        fpr_out = output_row['fpr'].values[0]
        f1_out = output_row['f1'].values[0]
    else:
        tpr_out = fpr_out = f1_out = 0.0
    
    summary_data.append({
        'Defense': defense,
        'INPUT_TPR': f'{tpr_in:.1%}',
        'INPUT_FPR': f'{fpr_in:.1%}',
        'INPUT_F1': f'{f1_in:.3f}',
        'OUTPUT_TPR': f'{tpr_out:.1%}',
        'OUTPUT_FPR': f'{fpr_out:.1%}',
        'OUTPUT_F1': f'{f1_out:.3f}',
    })

summary_df = pd.DataFrame(summary_data)

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create table
table = ax.table(cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.20, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(len(summary_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4CAF50')
    cell.set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#f0f0f0')

ax.set_title('Phase 1 Results Summary: INPUT vs OUTPUT Detection',
             fontweight='bold', fontsize=14, pad=20)

# Add key findings text
findings_text = """
KEY FINDINGS:
• INPUT Detection: Signature-Only achieves 80% TPR with 0% FPR (best performance)
• OUTPUT Detection: All defenses show 0-5.5% TPR (Claude Haiku respects non-disclosure)
• Critical Insight: Modern LLMs do NOT leak canary tokens → OUTPUT signature detection NOT viable
• Rules-Only shows modest performance in both paradigms (pattern-based)
"""

fig.text(0.5, 0.02, findings_text, ha='center', fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.12, 1, 1])
plot4_path = output_dir / 'phase1_summary_table.png'
plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {plot4_path}")
plt.close()

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("✅ All Visualizations Complete!")
print("="*60)
print(f"\nSaved to: {output_dir}/")
print("  1. phase1_performance_comparison.png (4-panel plot)")
print("  2. phase1_roc_comparison.png (ROC scatter)")
if family_path.exists():
    print("  3. phase1_family_heatmap.png (attack family analysis)")
print("  4. phase1_summary_table.png (results summary)")
print("\n" + "="*60)
