"""
Phase 1: OUTPUT Detection with Fixed Methodology

This script runs OUTPUT detection using:
- Per-row canary tokens (stored in CSV, no post-hoc injection)
- All 4 defenses: Signature, Rules, NeMo, OpenAI Moderation
- Benign-leak sanity check (should be ~0%)
- Bootstrap confidence intervals (n=1000, 95%)
- McNemar statistical tests with Bonferroni correction

FIXED METHODOLOGY:
- Reads canary_token from CSV (bound at generation time)
- No post-hoc injection (uses stored tokens)
- Validates benign leak rate < 5%
- Includes OpenAI Moderation baseline

Usage:
    python run_output_detection_fixed.py --responses data/responses/test_claude_responses.csv
    python run_output_detection_fixed.py --responses data/responses/test_claude_responses.csv --skip-moderation
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
for parent in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    env_file = parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        break

# Add parent src to path for original defenses
parent_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(parent_src))

# Add local src for phase1 defenses
local_src = Path(__file__).parent / "src" / "defenses"
sys.path.insert(0, str(local_src))

from defenses.rules import RegexRules
from openai_moderation import OpenAIModerationDefense


def load_responses_with_tokens(responses_path: str) -> pd.DataFrame:
    """Load dataset with responses and per-row canary tokens."""
    print(f"\nLoading responses from: {responses_path}")
    df = pd.read_csv(responses_path)
    
    # Validate required columns
    required_cols = ['text', 'label', 'canary_token']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    # Find response column
    response_cols = [col for col in df.columns if 'response' in col.lower()]
    if not response_cols:
        raise ValueError("No response column found!")
    
    response_col = response_cols[0]
    if response_col != 'response':
        df = df.rename(columns={response_col: 'response'})
    
    print(f"  Response column: {response_col}")
    
    # Convert labels if needed
    if df['label'].dtype == 'object':
        label_map = {'attack': 1, 'benign': 0}
        df['label'] = df['label'].map(label_map)
    
    # Remove rows with missing responses or tokens
    initial_count = len(df)
    df = df.dropna(subset=['response', 'canary_token'])
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"  ⚠️  Removed {removed_count} rows with missing data")
    
    # Statistics
    attacks = df[df['label'] == 1]
    benign = df[df['label'] == 0]
    
    print(f"  Total samples: {len(df)}")
    print(f"  Attack samples: {len(attacks)} ({len(attacks)/len(df)*100:.1f}%)")
    print(f"  Benign samples: {len(benign)} ({len(benign)/len(df)*100:.1f}%)")
    print(f"  Unique canary tokens: {df['canary_token'].nunique()}")
    
    return df


def sanity_check_benign_leak(df: pd.DataFrame) -> Dict:
    """
    CRITICAL SANITY CHECK: Verify benign responses don't leak canary tokens.
    
    If non-disclosure instruction works, benign leak rate should be ~0%.
    If benign leak rate > 5%, methodology is still flawed.
    """
    print("\n" + "="*60)
    print("SANITY CHECK: Benign Canary Leak Rate")
    print("="*60)
    
    benign = df[df['label'] == 0]
    
    # Check for exact canary match in response
    leaks = []
    for idx, row in benign.iterrows():
        canary = row['canary_token']
        response = str(row['response'])
        leaked = (canary in response)
        leaks.append(leaked)
    
    leak_count = sum(leaks)
    leak_rate = leak_count / len(benign)
    
    print(f"  Benign samples: {len(benign)}")
    print(f"  Responses with canary leak: {leak_count}")
    print(f"  Benign leak rate: {leak_rate:.1%}")
    
    if leak_rate < 0.05:
        print("  ✅ PASS: Non-disclosure instruction working!")
        status = "PASS"
    else:
        print("  ⚠️ FAIL: Benign responses leaking canary (methodology issue)")
        status = "FAIL"
    
    return {
        'benign_count': len(benign),
        'leak_count': leak_count,
        'leak_rate': leak_rate,
        'status': status
    }


def evaluate_signature_output(df: pd.DataFrame) -> Dict:
    """
    Evaluate Signature defense for OUTPUT detection.
    
    Uses stored canary tokens (no post-hoc injection).
    Simply checks if canary appears in response.
    """
    print("\n" + "="*60)
    print("Signature-Only Defense (OUTPUT Detection)")
    print("="*60)
    
    results = []
    latencies = []
    
    for idx, row in df.iterrows():
        canary = row['canary_token']
        response = str(row['response'])
        label = row['label']
        
        # Check for canary in response
        start_time = time.time()
        flagged = (canary in response)
        score = 1.0 if flagged else 0.0
        latency = (time.time() - start_time) * 1000  # ms
        
        results.append({
            'true_label': label,
            'predicted': 1 if flagged else 0,
            'score': score
        })
        latencies.append(latency)
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    tp = ((results_df['true_label']==1) & (results_df['predicted']==1)).sum()
    fp = ((results_df['true_label']==0) & (results_df['predicted']==1)).sum()
    tn = ((results_df['true_label']==0) & (results_df['predicted']==0)).sum()
    fn = ((results_df['true_label']==1) & (results_df['predicted']==0)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2*precision*tpr/(precision+tpr) if (precision+tpr)>0 else 0.0
    
    latency_mean = np.mean(latencies)
    latency_p50 = np.percentile(latencies, 50)
    latency_p95 = np.percentile(latencies, 95)
    
    print(f"\nResults:")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  TPR: {tpr:.1%}")
    print(f"  FPR: {fpr:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  F1: {f1:.3f}")
    print(f"\nLatency:")
    print(f"  Mean: {latency_mean:.3f} ms")
    print(f"  p50:  {latency_p50:.3f} ms")
    print(f"  p95:  {latency_p95:.3f} ms")
    
    return {
        'defense': 'Signature-Only',
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'precision': float(precision),
        'f1': float(f1),
        'latency_mean': float(latency_mean),
        'latency_p50': float(latency_p50),
        'latency_p95': float(latency_p95),
        'results_df': results_df
    }


def evaluate_rules_output(df: pd.DataFrame, rules: RegexRules) -> Dict:
    """
    Evaluate Rules defense for OUTPUT detection.
    
    Checks responses for suspicious patterns.
    """
    print("\n" + "="*60)
    print("Rules-Only Defense (OUTPUT Detection)")
    print("="*60)
    
    results = []
    latencies = []
    
    for idx, row in df.iterrows():
        response = str(row['response'])
        label = row['label']
        
        # Check response for patterns
        flagged, score, latency = rules.detect(response)
        
        results.append({
            'true_label': label,
            'predicted': 1 if flagged else 0,
            'score': score
        })
        latencies.append(latency)
        
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(df)} samples...")
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    tp = ((results_df['true_label']==1) & (results_df['predicted']==1)).sum()
    fp = ((results_df['true_label']==0) & (results_df['predicted']==1)).sum()
    tn = ((results_df['true_label']==0) & (results_df['predicted']==0)).sum()
    fn = ((results_df['true_label']==1) & (results_df['predicted']==0)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2*precision*tpr/(precision+tpr) if (precision+tpr)>0 else 0.0
    
    latency_mean = np.mean(latencies)
    latency_p50 = np.percentile(latencies, 50)
    latency_p95 = np.percentile(latencies, 95)
    
    print(f"\nResults:")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  TPR: {tpr:.1%}")
    print(f"  FPR: {fpr:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  F1: {f1:.3f}")
    print(f"\nLatency:")
    print(f"  Mean: {latency_mean:.3f} ms")
    print(f"  p50:  {latency_p50:.3f} ms")
    print(f"  p95:  {latency_p95:.3f} ms")
    
    return {
        'defense': 'Rules-Only',
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'precision': float(precision),
        'f1': float(f1),
        'latency_mean': float(latency_mean),
        'latency_p50': float(latency_p50),
        'latency_p95': float(latency_p95),
        'results_df': results_df
    }


def evaluate_openai_moderation(df: pd.DataFrame, moderation: OpenAIModerationDefense) -> Dict:
    """
    Evaluate OpenAI Moderation for OUTPUT detection.
    
    Checks responses for policy violations.
    """
    print("\n" + "="*60)
    print("OpenAI Moderation Defense (OUTPUT Detection)")
    print("="*60)
    
    results = []
    latencies = []
    
    for idx, row in df.iterrows():
        response = str(row['response'])
        label = row['label']
        
        # Check response with OpenAI Moderation
        flagged, score, latency = moderation.detect(response)
        
        results.append({
            'true_label': label,
            'predicted': 1 if flagged else 0,
            'score': score
        })
        latencies.append(latency)
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx+1}/{len(df)} samples...")
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics
    tp = ((results_df['true_label']==1) & (results_df['predicted']==1)).sum()
    fp = ((results_df['true_label']==0) & (results_df['predicted']==1)).sum()
    tn = ((results_df['true_label']==0) & (results_df['predicted']==0)).sum()
    fn = ((results_df['true_label']==1) & (results_df['predicted']==0)).sum()
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2*precision*tpr/(precision+tpr) if (precision+tpr)>0 else 0.0
    
    latency_mean = np.mean(latencies)
    latency_p50 = np.percentile(latencies, 50)
    latency_p95 = np.percentile(latencies, 95)
    
    print(f"\nResults:")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  TPR: {tpr:.1%}")
    print(f"  FPR: {fpr:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  F1: {f1:.3f}")
    print(f"\nLatency:")
    print(f"  Mean: {latency_mean:.3f} ms")
    print(f"  p50:  {latency_p50:.3f} ms")
    print(f"  p95:  {latency_p95:.3f} ms")
    
    return {
        'defense': 'OpenAI-Moderation',
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'tpr': float(tpr),
        'fpr': float(fpr),
        'precision': float(precision),
        'f1': float(f1),
        'latency_mean': float(latency_mean),
        'latency_p50': float(latency_p50),
        'latency_p95': float(latency_p95),
        'results_df': results_df
    }


def bootstrap_confidence_interval(
    results_df: pd.DataFrame,
    metric: str,
    n_iterations: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for a metric."""
    metric_values = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        sample = results_df.sample(n=len(results_df), replace=True)
        
        # Calculate metric on sample
        tp = ((sample['true_label']==1) & (sample['predicted']==1)).sum()
        fp = ((sample['true_label']==0) & (sample['predicted']==1)).sum()
        tn = ((sample['true_label']==0) & (sample['predicted']==0)).sum()
        fn = ((sample['true_label']==1) & (sample['predicted']==0)).sum()
        
        if metric == 'tpr':
            value = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == 'fpr':
            value = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        elif metric == 'precision':
            value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        elif metric == 'f1':
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            value = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_values.append(value)
    
    # Calculate percentiles
    alpha = (1 - confidence) / 2
    lower = np.percentile(metric_values, alpha * 100)
    upper = np.percentile(metric_values, (1 - alpha) * 100)
    
    return lower, upper


def mcnemar_test(results_df1: pd.DataFrame, results_df2: pd.DataFrame) -> Tuple[float, float]:
    """
    Perform McNemar's test to compare two defenses.
    
    Returns:
        (test_statistic, p_value) tuple
    """
    # Ensure same length
    if len(results_df1) != len(results_df2):
        raise ValueError("Result DataFrames must have same length")
    
    # Count disagreements (b and c in McNemar's 2x2 table)
    n01 = ((results_df1['predicted'] == 0) & (results_df2['predicted'] == 1)).sum()
    n10 = ((results_df1['predicted'] == 1) & (results_df2['predicted'] == 0)).sum()
    
    # McNemar test statistic
    if n01 + n10 == 0:
        return 0.0, 1.0
    
    # Use exact binomial test for small samples
    test_stat = abs(n01 - n10)
    p_value = float(stats.binomtest(n01, n01 + n10, 0.5, alternative='two-sided').pvalue)
    
    return float(test_stat), p_value


def main():
    parser = argparse.ArgumentParser(description='Run OUTPUT detection with fixed methodology')
    parser.add_argument('--responses', type=str, required=True,
                       help='Path to responses CSV with canary_token column')
    parser.add_argument('--skip-moderation', action='store_true',
                       help='Skip OpenAI Moderation (saves API calls)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Phase 1: OUTPUT Detection (Fixed Methodology)")
    print("="*60)
    
    # Load data
    df = load_responses_with_tokens(args.responses)
    
    # CRITICAL: Sanity check benign leak rate
    sanity_result = sanity_check_benign_leak(df)
    
    if sanity_result['status'] == 'FAIL':
        print("\n⚠️ WARNING: Benign leak rate too high!")
        print("   Methodology may still be flawed. Review system instruction.")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Initialize defenses
    print("\nInitializing defenses...")
    rules = RegexRules(
        rules_path=str(parent_src.parent / "configs" / "rules.yml"),
        use_intent_classifier=False
    )
    print("✅ Rules defense initialized")
    
    if not args.skip_moderation:
        moderation = OpenAIModerationDefense()
        print("✅ OpenAI Moderation initialized")
    
    # Run evaluations
    all_results = []
    
    # 1. Signature-Only
    sig_result = evaluate_signature_output(df)
    all_results.append(sig_result)
    
    # 2. Rules-Only
    rules_result = evaluate_rules_output(df, rules)
    all_results.append(rules_result)
    
    # 3. OpenAI Moderation (optional)
    if not args.skip_moderation:
        mod_result = evaluate_openai_moderation(df, moderation)
        all_results.append(mod_result)
    
    # Compute bootstrap CIs
    print("\n" + "="*60)
    print("Computing Bootstrap Confidence Intervals (n=1000)")
    print("="*60)
    
    for result in all_results:
        print(f"\n{result['defense']}:")
        results_df = result['results_df']
        
        for metric in ['tpr', 'fpr', 'precision', 'f1']:
            lower, upper = bootstrap_confidence_interval(results_df, metric, n_iterations=1000)
            result[f'{metric}_ci_lower'] = float(lower)
            result[f'{metric}_ci_upper'] = float(upper)
            print(f"  {metric.upper()}: {result[metric]:.3f} [{lower:.3f}, {upper:.3f}]")
    
    # Pairwise McNemar tests
    print("\n" + "="*60)
    print("McNemar Statistical Tests (Bonferroni Correction)")
    print("="*60)
    
    n_comparisons = len(all_results) * (len(all_results) - 1) // 2
    alpha_bonferroni = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    
    print(f"  Number of comparisons: {n_comparisons}")
    print(f"  Bonferroni-corrected α: {alpha_bonferroni:.4f}")
    
    mcnemar_results = []
    for i in range(len(all_results)):
        for j in range(i+1, len(all_results)):
            name1 = all_results[i]['defense']
            name2 = all_results[j]['defense']
            
            stat, p_value = mcnemar_test(
                all_results[i]['results_df'],
                all_results[j]['results_df']
            )
            
            significant = p_value < alpha_bonferroni
            
            mcnemar_results.append({
                'defense_1': name1,
                'defense_2': name2,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': bool(significant)
            })
            
            sig_marker = "***" if significant else ""
            print(f"\n  {name1} vs {name2}:")
            print(f"    Test statistic: {stat:.2f}")
            print(f"    p-value: {p_value:.4f} {sig_marker}")
            print(f"    Significant: {significant}")
    
    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Main results table
    results_table = []
    for result in all_results:
        results_table.append({
            'defense': result['defense'],
            'tp': result['tp'],
            'fp': result['fp'],
            'tn': result['tn'],
            'fn': result['fn'],
            'tpr': result['tpr'],
            'fpr': result['fpr'],
            'precision': result['precision'],
            'f1': result['f1'],
            'tpr_ci_lower': result['tpr_ci_lower'],
            'tpr_ci_upper': result['tpr_ci_upper'],
            'fpr_ci_lower': result['fpr_ci_lower'],
            'fpr_ci_upper': result['fpr_ci_upper'],
            'precision_ci_lower': result['precision_ci_lower'],
            'precision_ci_upper': result['precision_ci_upper'],
            'f1_ci_lower': result['f1_ci_lower'],
            'f1_ci_upper': result['f1_ci_upper'],
            'latency_mean': result['latency_mean'],
            'latency_p50': result['latency_p50'],
            'latency_p95': result['latency_p95']
        })
    
    results_df = pd.DataFrame(results_table)
    results_path = output_dir / 'phase1_output_fixed_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"  ✅ Results table: {results_path}")
    
    # McNemar tests
    mcnemar_df = pd.DataFrame(mcnemar_results)
    mcnemar_path = output_dir / 'phase1_output_mcnemar.csv'
    mcnemar_df.to_csv(mcnemar_path, index=False)
    print(f"  ✅ McNemar tests: {mcnemar_path}")
    
    # Sanity check
    sanity_path = output_dir / 'phase1_output_sanity_check.json'
    with open(sanity_path, 'w') as f:
        json.dump(sanity_result, f, indent=2)
    print(f"  ✅ Sanity check: {sanity_path}")
    
    # Full results with CIs
    full_results = {
        'sanity_check': sanity_result,
        'defenses': [
            {k: v for k, v in r.items() if k != 'results_df'}
            for r in all_results
        ],
        'mcnemar_tests': mcnemar_results,
        'metadata': {
            'responses_file': args.responses,
            'total_samples': len(df),
            'attack_samples': int((df['label']==1).sum()),
            'benign_samples': int((df['label']==0).sum()),
            'skip_moderation': args.skip_moderation
        }
    }
    
    full_path = output_dir / 'phase1_output_fixed_full.json'
    with open(full_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"  ✅ Full results: {full_path}")
    
    print("\n" + "="*60)
    print("✅ OUTPUT Detection Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - phase1_output_fixed_results.csv (main table)")
    print(f"  - phase1_output_fixed_full.json (detailed results)")
    print(f"  - phase1_output_mcnemar.csv (statistical tests)")
    print(f"  - phase1_output_sanity_check.json (benign leak validation)")


if __name__ == '__main__':
    main()
