#!/usr/bin/env python3
"""
Analyze the NEW output detection results.
"""
import pandas as pd
import os

def analyze_new_output():
    """Analyze results with new output classifier."""
    result_dir = "results/test_new_output"
    
    # Load predictions from working models
    all_dfs = []
    for model in ["gpt-4o-mini", "gpt-4o", "claude-haiku"]:
        pred_file = os.path.join(result_dir, model, "predictions.csv")
        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file)
            # Filter out errors
            df = df[~df['llm_response'].str.contains('ERROR', na=False)]
            all_dfs.append(df)
    
    if not all_dfs:
        print("No results found!")
        return
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print("🆕 NEW OUTPUT DETECTION RESULTS")
    print("="*80)
    print(f"Total predictions (excluding errors): {len(combined)}")
    print(f"Models: {combined['model_name'].unique().tolist()}")
    
    # Variance analysis
    print("\n📊 Classifier Score Variance:")
    variance_data = []
    for prompt_id in combined['prompt_id'].unique():
        prompt_df = combined[combined['prompt_id'] == prompt_id]
        variance = prompt_df['classifier_score'].var()
        scores = prompt_df['classifier_score'].tolist()
        variance_data.append({
            'prompt_id': prompt_id,
            'variance': variance,
            'scores': scores
        })
    
    variance_df = pd.DataFrame(variance_data)
    avg_variance = variance_df['variance'].mean()
    zero_var_count = (variance_df['variance'] == 0).sum()
    pct_zero = zero_var_count / len(variance_df) * 100
    
    print(f"   Average variance: {avg_variance:.6f}")
    print(f"   Zero variance prompts: {zero_var_count}/{len(variance_df)} ({pct_zero:.1f}%)")
    print(f"   Non-zero variance: {len(variance_df) - zero_var_count}/{len(variance_df)} ({100-pct_zero:.1f}%)")
    
    # Show some examples
    print("\n📝 Sample Variance Examples:")
    for i, row in variance_df.head(10).iterrows():
        label = combined[combined['prompt_id'] == row['prompt_id']]['label'].iloc[0]
        print(f"   {row['prompt_id']} ({label}): var={row['variance']:.6f}, scores={[f'{s:.2f}' for s in row['scores']]}")
    
    # Performance by model
    print("\n🎯 Performance by Model (Output Mode):")
    for model in combined['model_name'].unique():
        model_df = combined[combined['model_name'] == model]
        attacks = model_df[model_df['label'] == 'attack']
        benign = model_df[model_df['label'] == 'benign']
        
        tpr = attacks['defense_blocked'].sum() / len(attacks) if len(attacks) > 0 else 0
        fpr = benign['defense_blocked'].sum() / len(benign) if len(benign) > 0 else 0
        acc = model_df['correct'].sum() / len(model_df)
        
        print(f"   {model}: TPR={tpr:.1%}, FPR={fpr:.1%}, Acc={acc:.1%}")
    
    # Compare to old output mode
    print("\n📈 Comparison to Old Classifier:")
    print("   OLD (input patterns only): 100% zero variance, TPR=0%")
    print(f"   NEW (output patterns): {pct_zero:.1f}% zero variance, TPR varies by model")
    
    if pct_zero < 100:
        print(f"\n   ✅ SUCCESS! Non-zero variance detected ({100-pct_zero:.1f}%)")
        print("   Output mode now shows model-dependent behavior!")
    else:
        print(f"\n   ⚠️  Still 100% zero variance")
        print("   Models may be responding too similarly")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    analyze_new_output()
