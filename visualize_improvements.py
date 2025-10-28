#!/usr/bin/env python3
"""
Create before/after comparison visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

# Data
before_after = {
    'Defense': ['Classifier', 'Classifier', 'NeMo', 'NeMo', 'Sig+Clf', 'Sig+Clf'],
    'Version': ['Before', 'After', 'Before', 'After', 'Before', 'After'],
    'TPR': [3.7, 58.7, 0.0, 34.2, 78.3, 91.4],
    'FPR': [3.5, 4.8, 0.0, 2.7, 0.0, 4.8],
}

df = pd.DataFrame(before_after)

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: TPR Comparison
ax = axes[0]
x_pos = [0, 1, 3, 4, 6, 7]
colors = ['#e74c3c', '#27ae60', '#e74c3c', '#27ae60', '#e74c3c', '#27ae60']

bars = ax.bar(x_pos, df['TPR'], color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (pos, val) in enumerate(zip(x_pos, df['TPR'])):
    ax.text(pos, val + 2, f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Annotations for improvements
ax.annotate('', xy=(1, 58.7), xytext=(0, 3.7),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(0.5, 31, '+1485%', ha='center', fontsize=10, color='green', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.8))

ax.annotate('', xy=(4, 34.2), xytext=(3, 0.0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(3.5, 17, 'NEW', ha='center', fontsize=10, color='green', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.8))

ax.annotate('', xy=(7, 91.4), xytext=(6, 78.3),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(6.5, 85, '+16.7%', ha='center', fontsize=10, color='green', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='green', alpha=0.8))

ax.set_ylabel('True Positive Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Detection Rate Improvements', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(['Before', 'After', 'Before', 'After', 'Before', 'After'])
ax.set_ylim(0, 105)
ax.grid(axis='y', alpha=0.3)

# Add defense labels
ax.text(0.5, -10, 'Classifier', ha='center', fontsize=11, fontweight='bold')
ax.text(3.5, -10, 'NeMo', ha='center', fontsize=11, fontweight='bold')
ax.text(6.5, -10, 'Sig+Clf', ha='center', fontsize=11, fontweight='bold')

# Plot 2: ROC-style comparison
ax = axes[1]

# Plot points
before_points = df[df['Version'] == 'Before']
after_points = df[df['Version'] == 'After']

ax.scatter(before_points['FPR'], before_points['TPR'], s=200, c='#e74c3c', 
           marker='o', label='Before', alpha=0.7, edgecolors='black', linewidth=2)
ax.scatter(after_points['FPR'], after_points['TPR'], s=200, c='#27ae60', 
           marker='s', label='After', alpha=0.7, edgecolors='black', linewidth=2)

# Add arrows showing improvements
for i in range(len(df)//2):
    before = before_points.iloc[i]
    after = after_points.iloc[i]
    ax.annotate('', xy=(after['FPR'], after['TPR']), 
                xytext=(before['FPR'], before['TPR']),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5, alpha=0.6))

# Add labels
for i, row in after_points.iterrows():
    defense = row['Defense']
    ax.annotate(defense, 
                xy=(row['FPR'], row['TPR']), 
                xytext=(10, 10), 
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Diagonal reference line
ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Random')

# Optimal region annotation
ax.fill_between([0, 5], [80, 80], [100, 100], alpha=0.1, color='green', label='Optimal Region')

ax.set_xlabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('ROC Space: Before vs After', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(-2, 25)
ax.set_ylim(-2, 102)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis_output/improvement_comparison.png', dpi=150, bbox_inches='tight')
print("Comparison plot saved to: analysis_output/improvement_comparison.png")

# Create detailed results table
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

results_data = {
    'Configuration': [
        'Classifier (t=0.1)',
        'Classifier (t=0.5)',
        'NeMo (t=0.3)',
        'Signature only',
        'Sig+Clf (t=0.1) ⭐',
        'Sig+Clf (t=0.5) 🎯',
        'Sig+Rules+Clf (t=0.5)',
        'All defenses (t=0.3)',
    ],
    'TPR (%)': [58.7, 25.8, 34.2, 81.0, 91.4, 85.8, 85.1, 91.2],
    'FPR (%)': [4.8, 0.0, 2.7, 0.0, 4.8, 0.0, 3.5, 8.1],
    'F1 Score': [0.718, 0.409, 0.478, 0.895, 0.935, 0.923, 0.918, 0.915],
    'Latency (ms)': [0.06, 0.06, 0.02, 0.00, 0.08, 0.07, 0.08, 0.11],
}

results_df = pd.DataFrame(results_data)
print("\n" + results_df.to_string(index=False))

print("\n" + "="*80)
print("KEY ACHIEVEMENTS")
print("="*80)
print("✅ Classifier improved from 3.7% → 58.7% TPR (+1485%)")
print("✅ NeMo improved from 0% → 34.2% TPR (∞)")
print("✅ Best combined: 91.4% TPR with only 4.8% FPR")
print("✅ Zero-FP option: 85.8% TPR with 0% FPR")
print("✅ All defenses fast: <0.1ms median latency")
print("\n⭐ RECOMMENDED: Sig+Clf (t=0.1) for best F1 score (0.935)")
print("🎯 RECOMMENDED: Sig+Clf (t=0.5) for zero false positives")
