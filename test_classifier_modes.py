#!/usr/bin/env python3
"""
Quick test script to demonstrate the three classifier modes.

This script runs a small validation (10 samples) with each mode:
1. "input" - Classifier scores the input prompt (model-independent)
2. "output" - Classifier scores the LLM response (model-dependent)
3. "both" - Classifier scores both, combines with max()

Expected behavior:
- "input" mode: Zero variance across models (same prompt → same score)
- "output" mode: Some variance (different models → different responses → different scores)
- "both" mode: Some variance (combines input and output)
"""

import subprocess
import pandas as pd
import os

def run_mode(mode: str, samples: int = 10):
    """Run validation with specific classifier mode."""
    print(f"\n{'='*80}")
    print(f"Testing classifier_mode={mode}")
    print(f"{'='*80}\n")
    
    output_dir = f"results/test_mode_{mode}"
    
    cmd = [
        "python", "run_cross_model_validation.py",
        "--data", "data/prompts_hf_augmented.csv",
        "--max-samples", str(samples),
        "--classifier-mode", mode,
        "--output", output_dir,
        "--rate-limit", "0.3"
    ]
    
    subprocess.run(cmd)
    
    # Load and analyze results
    results_files = []
    for model in ["gpt-4o-mini", "gpt-4o", "claude-haiku"]:
        pred_file = os.path.join(output_dir, model, "predictions.csv")
        if os.path.exists(pred_file):
            results_files.append(pred_file)
    
    if results_files:
        # Combine all predictions
        all_results = pd.concat([pd.read_csv(f) for f in results_files])
        
        # Check variance in classifier scores
        variance_by_prompt = []
        for prompt_id in all_results['prompt_id'].unique():
            prompt_df = all_results[all_results['prompt_id'] == prompt_id]
            variance = prompt_df['classifier_score'].var()
            variance_by_prompt.append(variance)
        
        avg_variance = sum(variance_by_prompt) / len(variance_by_prompt) if variance_by_prompt else 0
        zero_variance_pct = (sum(1 for v in variance_by_prompt if v == 0) / len(variance_by_prompt) * 100) if variance_by_prompt else 0
        
        print(f"\n📊 Results for mode={mode}:")
        print(f"   Average classifier score variance: {avg_variance:.6f}")
        print(f"   Prompts with zero variance: {zero_variance_pct:.1f}%")
        
        # Show TPR/FPR per model
        for model in all_results['model_name'].unique():
            model_df = all_results[all_results['model_name'] == model]
            attacks = model_df[model_df['label'] == 'attack']
            benign = model_df[model_df['label'] == 'benign']
            
            tpr = attacks['defense_blocked'].sum() / len(attacks) if len(attacks) > 0 else 0
            fpr = benign['defense_blocked'].sum() / len(benign) if len(benign) > 0 else 0
            
            print(f"   {model}: TPR={tpr:.2%}, FPR={fpr:.2%}")
    
    return output_dir


def main():
    print("🔬 CLASSIFIER MODE COMPARISON TEST")
    print("=" * 80)
    print("\nThis test runs 10 samples with each classifier mode to show the difference:")
    print("- 'input': Scores the prompt (model-independent)")
    print("- 'output': Scores the LLM response (model-dependent)")
    print("- 'both': Scores both and combines with max()")
    print("\nExpected: 'input' has zero variance, 'output' has some variance")
    
    modes = ["input", "output", "both"]
    results = {}
    
    for mode in modes:
        results[mode] = run_mode(mode, samples=10)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("- 'input' mode should have 100% zero variance (same prompt → same score)")
    print("- 'output' mode should have <100% zero variance (different responses → different scores)")
    print("- 'both' mode combines both approaches")
    print("\nFor true cross-model generalization testing, use 'output' or 'both' mode.")
    print("\nResults saved to:")
    for mode, path in results.items():
        print(f"  {mode}: {path}")


if __name__ == "__main__":
    main()
