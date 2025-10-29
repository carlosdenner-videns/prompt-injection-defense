"""
Confusion-cost analysis for production deployment scenarios.

Translates FPR to false alarms per 10k requests at realistic attack prevalence.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_production_costs(
    tpr: float,
    fpr: float,
    prevalence: float,
    requests_per_day: int = 10000,
    fp_cost: float = 1.0,  # Cost of blocking legitimate user
    fn_cost: float = 10.0,  # Cost of missing an attack
    tp_benefit: float = 9.0  # Benefit of catching an attack
) -> dict:
    """
    Calculate expected costs/benefits in production scenario.
    
    Args:
        tpr: True positive rate
        fpr: False positive rate
        prevalence: Attack prevalence (e.g., 0.01 = 1% of requests are attacks)
        requests_per_day: Expected daily traffic
        fp_cost: Cost of each false positive (blocked legitimate user)
        fn_cost: Cost of each false negative (missed attack)
        tp_benefit: Benefit of catching each attack
        
    Returns:
        Dictionary with cost analysis
    """
    # Expected daily counts
    daily_attacks = requests_per_day * prevalence
    daily_benign = requests_per_day * (1 - prevalence)
    
    # Expected outcomes
    true_positives = daily_attacks * tpr
    false_negatives = daily_attacks * (1 - tpr)
    true_negatives = daily_benign * (1 - fpr)
    false_positives = daily_benign * fpr
    
    # Costs
    fp_total_cost = false_positives * fp_cost
    fn_total_cost = false_negatives * fn_cost
    tp_total_benefit = true_positives * tp_benefit
    
    net_value = tp_total_benefit - fp_total_cost - fn_total_cost
    
    return {
        'prevalence': prevalence,
        'requests_per_day': requests_per_day,
        'daily_attacks': daily_attacks,
        'daily_benign': daily_benign,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'fp_cost_total': fp_total_cost,
        'fn_cost_total': fn_total_cost,
        'tp_benefit_total': tp_total_benefit,
        'net_value': net_value,
        'blocked_users': false_positives,
        'missed_attacks': false_negatives
    }


def analyze_all_defenses(
    results_file: str = 'results/phase1_test_results.json',
    prevalence_levels: list = [0.001, 0.005, 0.01, 0.02, 0.05]
) -> pd.DataFrame:
    """
    Analyze production costs for all defenses across prevalence levels.
    
    Args:
        results_file: Path to results JSON
        prevalence_levels: List of attack prevalence values to test
        
    Returns:
        DataFrame with cost analysis
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    analysis_rows = []
    
    for defense_name, defense_results in results.items():
        tpr = defense_results['tpr']
        fpr = defense_results['fpr']
        
        for prevalence in prevalence_levels:
            costs = calculate_production_costs(tpr, fpr, prevalence)
            
            row = {
                'Defense': defense_name,
                'Prevalence': f"{prevalence*100:.1f}%",
                'Prevalence_value': prevalence,
                'TPR': tpr,
                'FPR': fpr,
                'False_Alarms_per_10k': costs['false_positives'],
                'Missed_Attacks_per_10k': costs['false_negatives'],
                'Attacks_Caught_per_10k': costs['true_positives'],
                'Net_Value': costs['net_value'],
                'FP_Cost': costs['fp_cost_total'],
                'FN_Cost': costs['fn_cost_total']
            }
            
            analysis_rows.append(row)
    
    return pd.DataFrame(analysis_rows)


def create_cost_table(df: pd.DataFrame, prevalence: float = 0.01) -> pd.DataFrame:
    """
    Create summary table for a specific prevalence level.
    
    Args:
        df: Full analysis DataFrame
        prevalence: Prevalence level to filter
        
    Returns:
        Formatted table
    """
    filtered = df[df['Prevalence_value'] == prevalence].copy()
    
    # Format for display
    table = filtered[[
        'Defense', 'TPR', 'FPR',
        'False_Alarms_per_10k', 'Missed_Attacks_per_10k',
        'Attacks_Caught_per_10k', 'Net_Value'
    ]].copy()
    
    # Round values
    table['TPR'] = table['TPR'].apply(lambda x: f"{x:.3f}")
    table['FPR'] = table['FPR'].apply(lambda x: f"{x:.4f}")
    table['False_Alarms_per_10k'] = table['False_Alarms_per_10k'].apply(lambda x: f"{x:.1f}")
    table['Missed_Attacks_per_10k'] = table['Missed_Attacks_per_10k'].apply(lambda x: f"{x:.1f}")
    table['Attacks_Caught_per_10k'] = table['Attacks_Caught_per_10k'].apply(lambda x: f"{x:.1f}")
    table['Net_Value'] = table['Net_Value'].apply(lambda x: f"{x:.1f}")
    
    return table


def plot_cost_analysis(df: pd.DataFrame):
    """Create visualizations of cost analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: False alarms by prevalence
    ax1 = axes[0, 0]
    for defense in df['Defense'].unique():
        defense_data = df[df['Defense'] == defense]
        ax1.plot(
            defense_data['Prevalence_value'] * 100,
            defense_data['False_Alarms_per_10k'],
            marker='o',
            label=defense,
            linewidth=2
        )
    
    ax1.set_xlabel('Attack Prevalence (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('False Alarms per 10k Requests', fontsize=11, fontweight='bold')
    ax1.set_title('False Alarm Rate vs Attack Prevalence', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Missed attacks by prevalence
    ax2 = axes[0, 1]
    for defense in df['Defense'].unique():
        defense_data = df[df['Defense'] == defense]
        ax2.plot(
            defense_data['Prevalence_value'] * 100,
            defense_data['Missed_Attacks_per_10k'],
            marker='o',
            label=defense,
            linewidth=2
        )
    
    ax2.set_xlabel('Attack Prevalence (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Missed Attacks per 10k Requests', fontsize=11, fontweight='bold')
    ax2.set_title('Missed Attacks vs Attack Prevalence', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Plot 3: Net value by prevalence
    ax3 = axes[1, 0]
    for defense in df['Defense'].unique():
        defense_data = df[df['Defense'] == defense]
        ax3.plot(
            defense_data['Prevalence_value'] * 100,
            defense_data['Net_Value'],
            marker='o',
            label=defense,
            linewidth=2
        )
    
    ax3.set_xlabel('Attack Prevalence (%)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Net Value (Benefit - Cost)', fontsize=11, fontweight='bold')
    ax3.set_title('Economic Value vs Attack Prevalence', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 4: Cost breakdown at 1% prevalence
    ax4 = axes[1, 1]
    prevalence_1pct = df[df['Prevalence_value'] == 0.01]
    
    x = np.arange(len(prevalence_1pct))
    width = 0.25
    
    ax4.bar(x - width, prevalence_1pct['FP_Cost'], width, label='FP Cost', alpha=0.7, color='red')
    ax4.bar(x, prevalence_1pct['FN_Cost'], width, label='FN Cost', alpha=0.7, color='orange')
    ax4.bar(x + width, prevalence_1pct['Net_Value'], width, label='Net Value', alpha=0.7, color='green')
    
    ax4.set_xlabel('Defense Method', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax4.set_title('Cost/Benefit Breakdown at 1% Prevalence', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(prevalence_1pct['Defense'], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('results/phase1_cost_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: results/phase1_cost_analysis.png")


def generate_latex_cost_table(df: pd.DataFrame, prevalence: float = 0.01) -> str:
    """Generate LaTeX table for cost analysis."""
    
    filtered = df[df['Prevalence_value'] == prevalence]
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Production Cost Analysis at {prevalence*100:.1f}\\% Attack Prevalence (per 10k requests)}}")
    lines.append("\\label{tab:cost_analysis}")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Defense & TPR & FPR & False Alarms & Missed Attacks & Attacks Caught & Net Value \\\\")
    lines.append("\\midrule")
    
    for _, row in filtered.iterrows():
        line = (
            f"{row['Defense'].replace('_', ' ')} & "
            f"{row['TPR']:.3f} & "
            f"{row['FPR']:.4f} & "
            f"{row['False_Alarms_per_10k']:.1f} & "
            f"{row['Missed_Attacks_per_10k']:.1f} & "
            f"{row['Attacks_Caught_per_10k']:.1f} & "
            f"{row['Net_Value']:.1f} \\\\"
        )
        lines.append(line)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\\\[0.2cm]")
    lines.append("{\\small Note: FP cost=\\$1, FN cost=\\$10, TP benefit=\\$9}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def main():
    print("="*80)
    print("Production Cost/Benefit Analysis")
    print("="*80)
    
    # Check if results exist
    if not Path('results/phase1_test_results.json').exists():
        print("Error: results/phase1_test_results.json not found")
        print("Run `python run_enhanced_experiments.py` first")
        return
    
    # Analyze across prevalence levels
    print("\nAnalyzing costs at different attack prevalence levels...")
    prevalence_levels = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    df = analyze_all_defenses(prevalence_levels=prevalence_levels)
    
    # Save full analysis
    df.to_csv('results/phase1_cost_analysis.csv', index=False)
    print(f"Saved: results/phase1_cost_analysis.csv")
    
    # Display tables for key prevalence levels
    print("\n" + "="*80)
    print("Cost Analysis at 1% Attack Prevalence (100 attacks per 10k requests)")
    print("="*80)
    table_1pct = create_cost_table(df, 0.01)
    print(table_1pct.to_string(index=False))
    
    print("\n" + "="*80)
    print("Cost Analysis at 5% Attack Prevalence (500 attacks per 10k requests)")
    print("="*80)
    table_5pct = create_cost_table(df, 0.05)
    print(table_5pct.to_string(index=False))
    
    # Create visualizations
    print("\nGenerating cost analysis plots...")
    plot_cost_analysis(df)
    
    # Generate LaTeX table
    latex_table = generate_latex_cost_table(df, 0.01)
    with open('results/phase1_cost_table.tex', 'w') as f:
        f.write(latex_table)
    print(f"Saved: results/phase1_cost_table.tex")
    
    print("\n✅ Cost analysis completed!")
    print("\nKey insights:")
    print("  - FPR is critical at low prevalence (few attacks, many legit users)")
    print("  - TPR matters more at higher prevalence (more attacks to catch)")
    print("  - Net value shows economic viability of each defense")


if __name__ == '__main__':
    main()
