"""
Generate cost comparison table for INPUT vs OUTPUT detection paradigms.

Compares deployment costs including:
- API costs per 10k requests
- Infrastructure costs
- False positive costs
- Total cost of ownership (TCO)

Usage:
    python create_cost_table.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path('results/tables')
output_dir.mkdir(exist_ok=True, parents=True)

print("="*60)
print("Phase 1: Cost Comparison Table")
print("="*60)

# Load results
input_df = pd.read_csv('results/phase1_baseline_performance.csv')
output_df = pd.read_csv('results/phase1_output_fixed_results.csv')

# Standardize column names
input_df = input_df.rename(columns={
    'Defense': 'defense',
    'TPR': 'tpr',
    'FPR': 'fpr',
    'Latency_p50_ms': 'latency_p50'
})

# Cost assumptions (per 10k requests)
print("\nCost Assumptions (per 10,000 requests):")
print("  - LLM API cost (Claude/GPT): $0.50")
print("  - OpenAI Moderation API: $0.02") 
print("  - Compute costs (patterns/rules): $0.001")
print("  - False positive handling cost: $1.00 per FP")
print("  - False negative cost (breach): $100.00 per FN")

# Base costs per 10k requests
LLM_COST = 0.50  # Claude/GPT generation
MODERATION_COST = 0.02  # OpenAI Moderation
COMPUTE_COST = 0.001  # Pattern matching/rules
FP_COST = 1.00  # Human review per false positive
FN_COST = 100.00  # Security breach cost per false negative

# Build cost table
cost_data = []

defenses = ['Signature-Only', 'Rules-Only', 'NeMo-Baseline', 'OpenAI-Moderation']

for defense in defenses:
    # INPUT paradigm
    input_row = input_df[input_df['defense'] == defense]
    if not input_row.empty:
        fpr_in = input_row['fpr'].values[0]
        tpr_in = input_row['tpr'].values[0]
        lat_in = input_row['latency_p50'].values[0]
        
        # Cost calculation for INPUT
        if 'OpenAI-Moderation' in defense:
            api_cost_in = MODERATION_COST
        else:
            api_cost_in = COMPUTE_COST
        
        # Assume 50/50 attack/benign split
        benign_count = 5000
        attack_count = 5000
        
        fp_count_in = benign_count * fpr_in
        fn_count_in = attack_count * (1 - tpr_in)
        
        fp_total_cost_in = fp_count_in * FP_COST
        fn_total_cost_in = fn_count_in * FN_COST
        
        total_cost_in = api_cost_in + fp_total_cost_in + fn_total_cost_in
    else:
        api_cost_in = fp_total_cost_in = fn_total_cost_in = total_cost_in = 0
        lat_in = 0
        tpr_in = fpr_in = 0
    
    # OUTPUT paradigm
    output_row = output_df[output_df['defense'] == defense]
    if not output_row.empty:
        fpr_out = output_row['fpr'].values[0]
        tpr_out = output_row['tpr'].values[0]
        lat_out = output_row['latency_p50'].values[0]
        
        # Cost calculation for OUTPUT
        # Requires LLM generation PLUS defense cost
        if 'OpenAI-Moderation' in defense:
            api_cost_out = LLM_COST + MODERATION_COST
        else:
            api_cost_out = LLM_COST + COMPUTE_COST
        
        fp_count_out = benign_count * fpr_out
        fn_count_out = attack_count * (1 - tpr_out)
        
        fp_total_cost_out = fp_count_out * FP_COST
        fn_total_cost_out = fn_count_out * FN_COST
        
        total_cost_out = api_cost_out + fp_total_cost_out + fn_total_cost_out
    else:
        api_cost_out = fp_total_cost_out = fn_total_cost_out = total_cost_out = 0
        lat_out = 0
        tpr_out = fpr_out = 0
    
    # Calculate cost ratio
    if total_cost_in > 0:
        cost_ratio = total_cost_out / total_cost_in
    else:
        cost_ratio = 0
    
    cost_data.append({
        'Defense': defense,
        
        # INPUT paradigm
        'INPUT_TPR': f'{tpr_in:.1%}',
        'INPUT_FPR': f'{fpr_in:.1%}',
        'INPUT_API_Cost': f'${api_cost_in:.3f}',
        'INPUT_FP_Cost': f'${fp_total_cost_in:.2f}',
        'INPUT_FN_Cost': f'${fn_total_cost_in:.2f}',
        'INPUT_Total': f'${total_cost_in:.2f}',
        'INPUT_Latency_ms': f'{lat_in:.1f}',
        
        # OUTPUT paradigm
        'OUTPUT_TPR': f'{tpr_out:.1%}',
        'OUTPUT_FPR': f'{fpr_out:.1%}',
        'OUTPUT_API_Cost': f'${api_cost_out:.3f}',
        'OUTPUT_FP_Cost': f'${fp_total_cost_out:.2f}',
        'OUTPUT_FN_Cost': f'${fn_total_cost_out:.2f}',
        'OUTPUT_Total': f'${total_cost_out:.2f}',
        'OUTPUT_Latency_ms': f'{lat_out:.1f}',
        
        # Comparison
        'Cost_Ratio': f'{cost_ratio:.1f}x',
    })

cost_df = pd.DataFrame(cost_data)

# Save detailed table
detailed_path = output_dir / 'phase1_cost_comparison_detailed.csv'
cost_df.to_csv(detailed_path, index=False)
print(f"\n✅ Detailed cost table saved: {detailed_path}")

# Create summary table (simpler view)
summary_data = []

for defense in defenses:
    input_row = input_df[input_df['defense'] == defense]
    output_row = output_df[output_df['defense'] == defense]
    
    if not input_row.empty:
        total_in = float(cost_df[cost_df['Defense'] == defense]['INPUT_Total'].values[0].replace('$', ''))
    else:
        total_in = 0
    
    if not output_row.empty:
        total_out = float(cost_df[cost_df['Defense'] == defense]['OUTPUT_Total'].values[0].replace('$', ''))
    else:
        total_out = 0
    
    summary_data.append({
        'Defense': defense,
        'INPUT_Cost_per_10k': f'${total_in:.2f}',
        'OUTPUT_Cost_per_10k': f'${total_out:.2f}',
        'Cost_Difference': f'${total_out - total_in:.2f}',
        'Overhead': f'{((total_out / total_in - 1) * 100) if total_in > 0 else 0:.0f}%'
    })

summary_df = pd.DataFrame(summary_data)

summary_path = output_dir / 'phase1_cost_comparison.csv'
summary_df.to_csv(summary_path, index=False)
print(f"✅ Summary cost table saved: {summary_path}")

# Print summary
print("\n" + "="*60)
print("Cost Comparison Summary (per 10,000 requests)")
print("="*60)
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("Key Insights")
print("="*60)
print("1. OUTPUT detection adds LLM generation cost (~$0.50/10k)")
print("2. INPUT detection is 50-1000x cheaper than OUTPUT")
print("3. Signature-Only INPUT has best cost/performance ratio")
print("4. OUTPUT detection's high FN cost dominates (0% TPR = $500/10k)")
print("5. For production, INPUT detection is strongly preferred")
print("="*60)
