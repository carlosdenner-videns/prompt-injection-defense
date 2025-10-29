"""
Run Phase 1 OUTPUT Detection experiments with real LLM responses.

This uses the ORIGINAL defense implementations that support OUTPUT detection.
"""

import os
import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add parent src to path for original defenses
parent_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(parent_src))

from defenses.signature_proxy import SignatureProxy
from defenses.rules import RegexRules  
from defenses.openai_adapter import OpenAIAdapter

print("Loading dataset...")
df = pd.read_csv('data/responses/test_combined_responses.csv')

# Convert labels if needed
if df['label'].dtype == 'object':
    label_map = {'attack': 1, 'benign': 0}
    df['label'] = df['label'].map(label_map)

# Filter to rows with responses
df = df[df['response'].notna()].reset_index(drop=True)

print(f"Dataset: {len(df)} samples")
print(f"  Attacks: {(df['label']==1).sum()}")
print(f"  Benign: {(df['label']==0).sum()}")

# Initialize defenses
print("\nInitializing defenses...")
signature = SignatureProxy()
rules = RegexRules(rules_path=str(parent_src.parent / "configs" / "rules.yml"), use_intent_classifier=False)
print("✅ Signature and Rules defenses initialized")

# Evaluate Signature-Only (OUTPUT detection)
print("\n" + "="*60)
print("Signature-Only Defense (OUTPUT Detection)")
print("="*60)

sig_results = []
for idx, row in df.iterrows():
    prompt = row['text']
    response = row['response']
    label = row['label']
    
    # Inject token
    modified_prompt, token = signature.inject(prompt)
    
    # Check if token leaked in response
    flagged, score, latency = signature.detect(response, token)
    
    sig_results.append({
        'true_label': label,
        'predicted': 1 if flagged else 0,
        'score': score
    })
    
sig_df = pd.DataFrame(sig_results)
tp = ((sig_df['true_label']==1) & (sig_df['predicted']==1)).sum()
fp = ((sig_df['true_label']==0) & (sig_df['predicted']==1)).sum()
tn = ((sig_df['true_label']==0) & (sig_df['predicted']==0)).sum()
fn = ((sig_df['true_label']==1) & (sig_df['predicted']==0)).sum()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2*precision*tpr/(precision+tpr) if (precision+tpr)>0 else 0

print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"TPR: {tpr:.1%}, FPR: {fpr:.1%}, Precision: {precision:.1%}, F1: {f1:.3f}")

# Evaluate Rules-Only (check response for suspicious patterns)
print("\n" + "="*60)
print("Rules-Only Defense (OUTPUT Detection)")
print("="*60)

rules_results = []
for idx, row in df.iterrows():
    response = row['response']
    label = row['label']
    
    # Check response for suspicious patterns
    flagged, score, latency = rules.detect(response or "")
    
    rules_results.append({
        'true_label': label,
        'predicted': 1 if flagged else 0,
        'score': score
    })

rules_df = pd.DataFrame(rules_results)
tp = ((rules_df['true_label']==1) & (rules_df['predicted']==1)).sum()
fp = ((rules_df['true_label']==0) & (rules_df['predicted']==1)).sum()
tn = ((rules_df['true_label']==0) & (rules_df['predicted']==0)).sum()
fn = ((rules_df['true_label']==1) & (rules_df['predicted']==0)).sum()

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2*precision*tpr/(precision+tpr) if (precision+tpr)>0 else 0

print(f"TP={tp}, FP={fp}, TN={tn}, FN={fn}")
print(f"TPR: {tpr:.1%}, FPR: {fpr:.1%}, Precision: {precision:.1%}, F1: {f1:.3f}")

# Save results
results = {
    'Signature-Only': {
        'TPR': float(sig_df.apply(lambda r: r['true_label']==1 and r['predicted']==1, axis=1).sum() / (sig_df['true_label']==1).sum()),
        'FPR': float(sig_df.apply(lambda r: r['true_label']==0 and r['predicted']==1, axis=1).sum() / (sig_df['true_label']==0).sum()),
    },
    'Rules-Only': {
        'TPR': float(rules_df.apply(lambda r: r['true_label']==1 and r['predicted']==1, axis=1).sum() / (rules_df['true_label']==1).sum()),
        'FPR': float(rules_df.apply(lambda r: r['true_label']==0 and r['predicted']==1, axis=1).sum() / (rules_df['true_label']==0).sum()),
    }
}

with open('results/phase1_output_detection_quick.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("✅ Results saved to: results/phase1_output_detection_quick.json")
print("="*60)
