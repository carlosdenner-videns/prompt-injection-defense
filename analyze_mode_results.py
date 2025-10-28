#!/usr/bin/env python3
"""
Analyze classifier mode test results to verify expected behavior:
- input mode: 100% zero variance
- output mode: <100% zero variance  
- both mode: intermediate
"""

import pandas as pd
import os
from pathlib import Path

def analyze_mode(mode: str):
    """Analyze results for a specific mode."""
    mode_dir = f"results/test_mode_{mode}"
    
    if not os.path.exists(mode_dir):
        print(f"❌ {mode_dir} not found")
        return None
    
    # Load all predictions
    all_dfs = []
    models = []
    for model_dir in Path(mode_dir).iterdir():
        if model_dir.is_dir():
            pred_file = model_dir / "predictions.csv"
            if pred_file.exists():
                df = pd.read_csv(pred_file)
                all_dfs.append(df)
                models.append(model_dir.name)
    
    if not all_dfs:
        print(f"❌ No predictions found in {mode_dir}")
        return None
    
    # Combine all results
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*80}")
    print(f"Models tested: {', '.join(sorted(models))}")
    print(f"Total predictions: {len(combined)}")
    
    # Calculate variance per prompt
    variance_data = []
    for prompt_id in combined['prompt_id'].unique():
        prompt_df = combined[combined['prompt_id'] == prompt_id]
        
        # Variance in classifier scores
        clf_variance = prompt_df['classifier_score'].var()
        clf_scores = prompt_df['classifier_score'].tolist()
        
        # Variance in combined scores
        combined_variance = prompt_df['combined_score'].var()
        
        variance_data.append({
            'prompt_id': prompt_id,
            'clf_variance': clf_variance,
            'clf_scores': clf_scores,
            'combined_variance': combined_variance,
            'num_models': len(prompt_df)
        })
    
    variance_df = pd.DataFrame(variance_data)
    
    # Statistics
    avg_clf_variance = variance_df['clf_variance'].mean()
    zero_clf_variance = (variance_df['clf_variance'] == 0).sum()
    pct_zero_clf = (zero_clf_variance / len(variance_df) * 100)
    
    print(f"\n📊 Classifier Score Variance:")
    print(f"   Average variance: {avg_clf_variance:.6f}")
    print(f"   Zero variance prompts: {zero_clf_variance}/{len(variance_df)} ({pct_zero_clf:.1f}%)")
    print(f"   Non-zero variance prompts: {len(variance_df) - zero_clf_variance}/{len(variance_df)} ({100-pct_zero_clf:.1f}%)")
    
    # Show a few examples
    if len(variance_df) > 0:
        print(f"\n📝 Sample Variance Values:")
        for i, row in variance_df.head(5).iterrows():
            print(f"   Prompt {row['prompt_id']}: variance={row['clf_variance']:.6f}, scores={row['clf_scores']}")
    
    # Performance metrics per model
    print(f"\n🎯 Performance by Model:")
    for model in sorted(models):
        model_df = combined[combined['model_name'] == model]
        
        # Skip if errors
        if 'ERROR' in model_df['llm_response'].iloc[0] if len(model_df) > 0 else '':
            error_rate = (model_df['llm_response'].str.contains('ERROR').sum() / len(model_df) * 100)
            print(f"   {model}: ⚠️ {error_rate:.0f}% errors")
            continue
        
        attacks = model_df[model_df['label'] == 'attack']
        benign = model_df[model_df['label'] == 'benign']
        
        tpr = attacks['defense_blocked'].sum() / len(attacks) if len(attacks) > 0 else 0
        fpr = benign['defense_blocked'].sum() / len(benign) if len(benign) > 0 else 0
        acc = model_df['correct'].sum() / len(model_df) if len(model_df) > 0 else 0
        
        print(f"   {model}: TPR={tpr:.1%}, FPR={fpr:.1%}, Acc={acc:.1%}")
    
    return {
        'mode': mode,
        'avg_variance': avg_clf_variance,
        'pct_zero_variance': pct_zero_clf,
        'num_prompts': len(variance_df),
        'num_models': len(models)
    }


def main():
    print("🔬 CLASSIFIER MODE RESULTS ANALYSIS")
    print("="*80)
    
    results = {}
    for mode in ['input', 'output', 'both']:
        result = analyze_mode(mode)
        if result:
            results[mode] = result
    
    # Summary comparison
    if results:
        print(f"\n\n{'='*80}")
        print("📊 SUMMARY COMPARISON")
        print(f"{'='*80}\n")
        
        print(f"{'Mode':<12} {'Avg Variance':<15} {'Zero Var %':<15} {'Expected'}")
        print("-" * 80)
        
        for mode in ['input', 'output', 'both']:
            if mode in results:
                r = results[mode]
                
                # Check expectations
                if mode == 'input':
                    expectation = "100% (✓)" if r['pct_zero_variance'] == 100 else f"100% (✗ got {r['pct_zero_variance']:.1f}%)"
                elif mode == 'output':
                    expectation = "<100% (✓)" if r['pct_zero_variance'] < 100 else f"<100% (✗ got {r['pct_zero_variance']:.1f}%)"
                else:  # both
                    expectation = "<100% (✓)" if r['pct_zero_variance'] < 100 else f"<100% (✗ got {r['pct_zero_variance']:.1f}%)"
                
                print(f"{mode:<12} {r['avg_variance']:<15.6f} {r['pct_zero_variance']:<15.1f}% {expectation}")
        
        print("\n" + "="*80)
        print("✅ VERIFICATION COMPLETE")
        print("="*80)
        print("\nConclusion:")
        
        if 'input' in results and results['input']['pct_zero_variance'] == 100:
            print("✅ 'input' mode: CORRECT - 100% zero variance (model-independent)")
        else:
            print("❌ 'input' mode: UNEXPECTED - should have 100% zero variance")
        
        if 'output' in results and results['output']['pct_zero_variance'] < 100:
            print("✅ 'output' mode: CORRECT - some variance (model-dependent)")
        else:
            print("⚠️  'output' mode: Check if models are producing different responses")
        
        if 'both' in results and results['both']['pct_zero_variance'] < 100:
            print("✅ 'both' mode: CORRECT - combines input and output")
        else:
            print("⚠️  'both' mode: Check implementation")
        
        print("\n💡 Recommendation:")
        print("   For cross-model generalization claims, use 'output' or 'both' mode.")
        print("   These modes show model-dependent behavior (non-zero variance).")


if __name__ == "__main__":
    main()
