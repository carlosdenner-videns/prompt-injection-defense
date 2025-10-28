#!/usr/bin/env python3
"""
Debug script to investigate the suspiciously perfect zero variance.
This checks if models are actually being called independently.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("INVESTIGATING ZERO VARIANCE - IS IT TOO GOOD TO BE TRUE?")
print("=" * 80)

# Load raw results
df = pd.read_csv('results/cross_model/all_models_raw.csv')

print(f"\nTotal rows: {len(df)}")
print(f"Models: {df['model_name'].unique()}")
print(f"\nSamples per model:")
print(df['model_name'].value_counts())

print("\n" + "=" * 80)
print("1. CHECKING IF RESPONSES ARE ACTUALLY DIFFERENT")
print("=" * 80)

# Pick first 5 prompts and compare responses across models
for prompt_id in df['prompt_id'].unique()[:5]:
    sample_df = df[df['prompt_id'] == prompt_id]
    
    print(f"\n--- Prompt: {prompt_id} (Label: {sample_df.iloc[0]['label']}) ---")
    print(f"Prompt text: {sample_df.iloc[0]['original_prompt'][:80]}...")
    
    for _, row in sample_df.iterrows():
        resp = str(row['llm_response'])
        print(f"\n{row['model_name']}:")
        print(f"  LLM called: {row['llm_called']}")
        print(f"  Response length: {len(resp)}")
        print(f"  Response preview: {resp[:100]}...")
        print(f"  Signature score: {row['signature_score']}")
        print(f"  Classifier score: {row['classifier_score']}")
        print(f"  Combined score: {row['combined_score']}")
        print(f"  Blocked: {row['defense_blocked']}")

print("\n" + "=" * 80)
print("2. CHECKING IF CLASSIFIER SCORES VARY BY MODEL")
print("=" * 80)

# For each prompt, check if classifier scores differ across models
zero_var_prompts = 0
nonzero_var_prompts = 0

variance_data = []

for prompt_id in df['prompt_id'].unique():
    prompt_df = df[df['prompt_id'] == prompt_id]
    
    # Get classifier scores for this prompt across models
    clf_scores = prompt_df['classifier_score'].values
    variance = clf_scores.var()
    
    variance_data.append({
        'prompt_id': prompt_id,
        'variance': variance,
        'mean': clf_scores.mean(),
        'scores': list(clf_scores)
    })
    
    if variance == 0:
        zero_var_prompts += 1
    else:
        nonzero_var_prompts += 1

print(f"\nPrompts with ZERO variance in classifier scores: {zero_var_prompts}")
print(f"Prompts with NON-ZERO variance: {nonzero_var_prompts}")
print(f"Percentage with zero variance: {zero_var_prompts / len(df['prompt_id'].unique()) * 100:.1f}%")

variance_df = pd.DataFrame(variance_data)

if nonzero_var_prompts > 0:
    print("\nSample prompts with NON-ZERO variance:")
    print(variance_df[variance_df['variance'] > 0].head(10))

print("\n" + "=" * 80)
print("3. CHECKING THE SMOKING GUN: ARE CLASSIFIER SCORES INPUT-DEPENDENT?")
print("=" * 80)

# The classifier should score the PROMPT, not the response
# So same prompt should get same classifier score regardless of model
# This would explain zero variance!

print("\nKey question: Does the classifier look at the PROMPT or the RESPONSE?")
print("If it looks at the PROMPT only, then zero variance is EXPECTED!")
print("\nLet's check by looking at code flow...\n")

# Check if classifier scores are identical for same prompt
same_prompt_same_score = 0
same_prompt_diff_score = 0

for prompt_id in df['prompt_id'].unique():
    prompt_df = df[df['prompt_id'] == prompt_id]
    clf_scores = prompt_df['classifier_score'].unique()
    
    if len(clf_scores) == 1:
        same_prompt_same_score += 1
    else:
        same_prompt_diff_score += 1

print(f"Prompts with IDENTICAL classifier scores across all models: {same_prompt_same_score}")
print(f"Prompts with DIFFERENT classifier scores across models: {same_prompt_diff_score}")

if same_prompt_same_score == len(df['prompt_id'].unique()):
    print("\n⚠️  FOUND THE ISSUE! ⚠️")
    print("Classifier scores are IDENTICAL for each prompt across all models.")
    print("This means the classifier is scoring the INPUT PROMPT, not the model's RESPONSE.")
    print("\nThis explains the zero variance:")
    print("- Same prompt → Same classifier score")
    print("- Same threshold → Same blocking decision")
    print("- Same blocking decision → Identical TPR/FPR")
    print("\nThe 'model-agnostic' result is actually a BUG, not a feature!")

print("\n" + "=" * 80)
print("4. CHECKING SIGNATURE SCORES (SHOULD vary by response)")
print("=" * 80)

# Signature scores SHOULD vary because they check if token is in RESPONSE
for prompt_id in df['prompt_id'].unique()[:5]:
    prompt_df = df[df['prompt_id'] == prompt_id]
    sig_scores = prompt_df['signature_score'].values
    
    print(f"\nPrompt {prompt_id}:")
    print(f"  Signature scores: {sig_scores}")
    print(f"  Variance: {sig_scores.var():.4f}")
    print(f"  Unique scores: {len(set(sig_scores))}")

print("\n" + "=" * 80)
print("5. FINAL DIAGNOSIS")
print("=" * 80)

# Check what the actual decision flow is
print("\nLet's trace the decision logic:")

# Get one example of each outcome
blocked_example = df[df['defense_blocked'] == True].iloc[0] if len(df[df['defense_blocked'] == True]) > 0 else None
allowed_example = df[df['defense_blocked'] == False].iloc[0] if len(df[df['defense_blocked'] == False]) > 0 else None

if blocked_example is not None:
    print("\nExample BLOCKED request:")
    print(f"  Prompt: {blocked_example['original_prompt'][:80]}...")
    print(f"  Signature score: {blocked_example['signature_score']}")
    print(f"  Classifier score: {blocked_example['classifier_score']}")
    print(f"  Combined: (0.2 * {blocked_example['signature_score']}) + (0.8 * {blocked_example['classifier_score']}) = {blocked_example['combined_score']}")
    print(f"  Threshold: 0.5")
    print(f"  Blocked: {blocked_example['combined_score']} >= 0.5 = {blocked_example['defense_blocked']}")

if allowed_example is not None:
    print("\nExample ALLOWED request:")
    print(f"  Prompt: {allowed_example['original_prompt'][:80]}...")
    print(f"  Signature score: {allowed_example['signature_score']}")
    print(f"  Classifier score: {allowed_example['classifier_score']}")
    print(f"  Combined: (0.2 * {allowed_example['signature_score']}) + (0.8 * {allowed_example['classifier_score']}) = {allowed_example['combined_score']}")
    print(f"  Threshold: 0.5")
    print(f"  Blocked: {allowed_example['combined_score']} >= 0.5 = {allowed_example['defense_blocked']}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if same_prompt_same_score == len(df['prompt_id'].unique()):
    print("\n❌ ZERO VARIANCE IS A BUG, NOT A FEATURE!")
    print("\nThe classifier is evaluating the INPUT PROMPT, not the model's RESPONSE.")
    print("This means:")
    print("  - Same prompt always gets same score, regardless of which LLM is used")
    print("  - The 'model-agnostic generalization' is actually just scoring the input")
    print("  - We're not actually testing if different model RESPONSES are handled consistently")
    print("\nWhat we SHOULD be doing:")
    print("  - Classifier should score the LLM's RESPONSE to detect jailbreaks")
    print("  - Different models give different responses → different scores → some variance")
    print("  - True generalization = similar (not identical) performance across models")
else:
    print("\n✅ Classifier scores DO vary by model - investigating other causes...")
