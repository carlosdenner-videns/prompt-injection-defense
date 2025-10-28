#!/usr/bin/env python3
"""
Summarize defense test results with threshold adjustments.
"""

import pandas as pd
import sys

csv_file = sys.argv[1] if len(sys.argv) > 1 else "results/openai_test_30_t05.csv"
df = pd.read_csv(csv_file)

print("="*80)
print(f"DEFENSE RESULTS SUMMARY: {csv_file}")
print("="*80)

# Pre-LLM blocking stats
attacks_blocked = len(df[(df['label'] == 'attack') & (~df['llm_called'])])
benign_blocked = len(df[(df['label'] == 'benign') & (~df['llm_called'])])
total_attacks = len(df[df['label'] == 'attack'])
total_benign = len(df[df['label'] == 'benign'])

print(f"\n📊 PRE-LLM BLOCKING (Defense stopped these before OpenAI call):")
print(f"   Attacks blocked: {attacks_blocked} / {total_attacks} ({attacks_blocked/total_attacks*100:.1f}%)")
print(f"   Benign blocked:  {benign_blocked} / {total_benign} ({benign_blocked/total_benign*100:.1f}%)")
print(f"\n   ✅ TPR (True Positive Rate):  {attacks_blocked/total_attacks*100:.1f}%")
print(f"   ⚠️  FPR (False Positive Rate): {benign_blocked/total_benign*100:.1f}%")

# Reached LLM
attacks_reached = len(df[(df['label'] == 'attack') & (df['llm_called'])])
benign_reached = len(df[(df['label'] == 'benign') & (df['llm_called'])])

print(f"\n📨 REACHED OPENAI:")
print(f"   Attacks: {attacks_reached} / {total_attacks} ({attacks_reached/total_attacks*100:.1f}%)")
print(f"   Benign:  {benign_reached} / {total_benign} ({benign_reached/total_benign*100:.1f}%)")

# Show which attacks reached LLM
if attacks_reached > 0:
    print(f"\n⚠️  ATTACKS THAT REACHED OPENAI (need manual review):")
    attacks_llm = df[(df['label'] == 'attack') & (df['llm_called'])]
    for idx, row in attacks_llm.iterrows():
        print(f"\n   [{row['prompt_id']}]")
        print(f"   Prompt: {row['original_prompt'][:80]}...")
        print(f"   Response: {row['llm_response'][:80]}...")
        print(f"   Score: {row['defense_score']:.3f} (below threshold)")

# Show blocked attacks
if attacks_blocked > 0:
    print(f"\n✅ ATTACKS BLOCKED BY DEFENSE:")
    blocked = df[(df['label'] == 'attack') & (~df['llm_called'])]
    for idx, row in blocked.iterrows():
        print(f"\n   [{row['prompt_id']}]")
        print(f"   Prompt: {row['original_prompt'][:80]}...")
        print(f"   Score: {row['defense_score']:.3f} (above threshold)")

# Latency stats
if len(df[df['llm_called']]) > 0:
    avg_defense = df[df['llm_called']]['defense_latency_ms'].mean()
    avg_llm = df[df['llm_called']]['llm_latency_ms'].mean()
    print(f"\n⏱️  LATENCY (when LLM called):")
    print(f"   Defense overhead: {avg_defense:.2f}ms")
    print(f"   LLM response:     {avg_llm:.2f}ms")
    print(f"   Total:            {avg_defense + avg_llm:.2f}ms")

# Token usage
if len(df[df['llm_called']]) > 0:
    total_tokens = df[df['llm_called']]['llm_tokens'].sum()
    avg_tokens = df[df['llm_called']]['llm_tokens'].mean()
    print(f"\n💰 TOKEN USAGE:")
    print(f"   Total: {total_tokens}")
    print(f"   Average: {avg_tokens:.1f} per call")
    print(f"   Estimated cost: ${total_tokens * 0.00015:.4f} (at $0.15/1M tokens)")

print(f"\n{'='*80}")
print("INTERPRETATION")
print('='*80)
if benign_blocked == 0 and attacks_blocked > 0:
    print("✅ EXCELLENT: Blocking attacks without false positives!")
elif benign_blocked == 0:
    print("⚠️  WARNING: No false positives, but also not blocking attacks!")
elif attacks_blocked == total_attacks and benign_blocked < total_benign * 0.1:
    print("✅ GOOD: Blocking all attacks with low false positive rate")
elif attacks_blocked > total_attacks * 0.8:
    print("⚠️  FAIR: Blocking most attacks but with some false positives")
else:
    print("❌ NEEDS TUNING: Not blocking enough attacks")

print(f"\n💡 NEXT STEPS:")
if attacks_reached > 0:
    print(f"   1. Manually review {attacks_reached} attacks that reached OpenAI")
    print(f"   2. Determine how many actually succeeded in manipulation")
    print(f"   3. If many succeeded, lower threshold or improve patterns")
if benign_blocked > 0:
    print(f"   4. Review {benign_blocked} benign prompts that were blocked")
    print(f"   5. Adjust patterns/rules to reduce false positives")
if benign_blocked == 0 and attacks_blocked > 0:
    print(f"   ✅ Current configuration looks good! Monitor in production.")
