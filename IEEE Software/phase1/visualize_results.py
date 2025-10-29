"""
Quick visualization of Phase 1 results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load results
results_file = Path("results/phase1_baseline_performance.csv")
df = pd.read_csv(results_file)

# Convert string percentages to floats if needed
for col in ['TPR', 'FPR', 'Precision', 'F1']:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(float)

print("Phase 1 Results Loaded:")
print(df.to_string(index=False))
print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Phase 1: Baseline Defense Performance Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: TPR Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(df['Defense'], df['TPR'], color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
ax1.set_title('Attack Detection Rate', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 1])
ax1.tick_params(axis='x', rotation=15)
for i, v in enumerate(df['TPR']):
    ax1.text(i, v + 0.03, f'{v:.1%}', ha='center', fontweight='bold', fontsize=10)

# Plot 2: FPR Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(df['Defense'], df['FPR'], color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
ax2.set_ylabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
ax2.set_title('False Alarm Rate (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 0.02])
ax2.tick_params(axis='x', rotation=15)
for i, v in enumerate(df['FPR']):
    ax2.text(i, v + 0.001, f'{v:.1%}', ha='center', fontweight='bold', fontsize=10)

# Plot 3: F1 Score Comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(df['Defense'], df['F1'], color=['#2ecc71', '#e74c3c', '#f39c12', '#3498db'])
ax3.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax3.set_title('Overall Performance (Harmonic Mean)', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 1])
ax3.tick_params(axis='x', rotation=15)
for i, v in enumerate(df['F1']):
    ax3.text(i, v + 0.03, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)

# Plot 4: TPR vs FPR Scatter
ax4 = axes[1, 1]
colors_scatter = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
for i, row in df.iterrows():
    ax4.scatter(row['FPR'], row['TPR'], s=300, c=colors_scatter[i], 
               alpha=0.7, edgecolor='black', linewidth=2)
    ax4.annotate(row['Defense'], (row['FPR'], row['TPR']),
                xytext=(10, 5), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# Add ideal point
ax4.scatter(0, 1, s=500, c='gold', marker='*', edgecolor='black',
           linewidth=2, label='Ideal (TPR=1, FPR=0)', zorder=10)

ax4.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
ax4.set_ylabel('True Positive Rate (TPR)', fontsize=11, fontweight='bold')
ax4.set_title('TPR vs FPR Trade-off', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right')
ax4.set_xlim([-0.002, 0.02])
ax4.set_ylim([0, 1.05])
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save
output_file = Path("results/phase1_visual_summary.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization: {output_file}")

# Show
plt.show()

# Print summary
print("\n" + "="*70)
print("PHASE 1 RESULTS SUMMARY")
print("="*70)
print("\n📊 Performance Ranking (by F1 Score):")
sorted_df = df.sort_values('F1', ascending=False)
for i, row in sorted_df.iterrows():
    print(f"\n{i+1}. {row['Defense']}")
    print(f"   F1: {row['F1']:.3f} | TPR: {row['TPR']:.1%} | FPR: {row['FPR']:.1%}")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print(f"\n✅ Best TPR: {df.loc[df['TPR'].idxmax(), 'Defense']} ({df['TPR'].max():.1%})")
print(f"✅ Best FPR: {df.loc[df['FPR'].idxmin(), 'Defense']} ({df['FPR'].min():.1%})")
print(f"✅ Best F1:  {df.loc[df['F1'].idxmax(), 'Defense']} ({df['F1'].max():.3f})")

print("\n📈 Performance Gap:")
print(f"   Best baseline TPR: {df['TPR'].max():.1%}")
print(f"   Gap to ideal (100%): {1 - df['TPR'].max():.1%}")
print("   → Motivates multi-strategy approach in Phases 2-6")

print("\n🎯 Next Steps:")
print("   1. Copy LaTeX table to IEEE Software paper")
print("   2. Include visualization in paper figures")
print("   3. Proceed to Phase 2 (simple combinations)")
