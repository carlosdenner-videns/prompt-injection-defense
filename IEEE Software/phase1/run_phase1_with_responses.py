"""
Phase 1: Full Baseline Experiments with Real LLM Responses

This script runs comprehensive Phase 1 baseline evaluation using:
- Real LLM responses (generated from GPT-4/Claude)
- All 4 defenses: Signature, Rules, NeMo, OpenAI Moderation
- All enhancements: Bootstrap CIs, McNemar tests, error bars, cost analysis
- OUTPUT detection paradigm (checking if canary tokens leak in responses)

Usage:
    python run_phase1_with_responses.py --responses data/responses/test_gpt4_responses.csv
    python run_phase1_with_responses.py --responses data/responses/test_gpt4_responses.csv --skip-moderation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
for parent in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    env_file = parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        break

# Add src to path - use PARENT src directory for original defenses
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir / "src"))

from defenses import (
    SignatureOnlyDefense,
    RulesOnlyDefense,
    NeMoBaselineDefense,
    OpenAIModerationDefense
)


def load_responses_dataset(responses_path: str) -> pd.DataFrame:
    """Load dataset with real LLM responses."""
    print(f"\nLoading responses from: {responses_path}")
    df = pd.read_csv(responses_path)
    
    # Validate required columns
    required_cols = ['text', 'label']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Dataset missing required columns: {required_cols}")
    
    # Check for response column (should already have 'response' column from merge)
    if 'response' not in df.columns:
        response_cols = [col for col in df.columns if col.startswith('response_')]
        if not response_cols:
            raise ValueError("No response columns found! Expected 'response' column")
        
        print(f"  Found response columns: {response_cols}")
        
        # Use first response column
        response_col = response_cols[0]
        df = df.rename(columns={response_col: 'response'})
    else:
        print(f"  Using 'response' column")
    
    # Convert string labels to numeric if needed
    if df['label'].dtype == 'object':
        label_map = {'attack': 1, 'benign': 0}
        df['label'] = df['label'].map(label_map)
    
    # Remove rows with missing responses
    initial_count = len(df)
    df = df.dropna(subset=['response'])
    removed_count = initial_count - len(df)
    
    if removed_count > 0:
        print(f"  ⚠️  Removed {removed_count} rows with missing responses")
    
    # Count attacks and benign
    attacks = df[df['label'] == 1]
    benign = df[df['label'] == 0]
    
    print(f"  Total samples: {len(df)}")
    print(f"  Attack samples: {len(attacks)} ({len(attacks)/len(df)*100:.1f}%)")
    print(f"  Benign samples: {len(benign)} ({len(benign)/len(df)*100:.1f}%)")
    
    return df


def evaluate_defense(
    defense,
    df: pd.DataFrame,
    defense_name: str,
    canary_token: str = "DEADBEEF123456"
) -> Dict:
    """
    Evaluate a single defense on the dataset with real responses.
    
    Returns:
        Dictionary with metrics: TPR, FPR, Precision, F1, Latency, etc.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {defense_name}")
    print(f"{'='*60}")
    
    results = []
    latencies = []
    
    # Check if this is signature defense
    use_signature = isinstance(defense, SignatureOnlyDefense)
    
    for idx, row in df.iterrows():
        prompt = row['text']
        response = row['response']
        true_label = row['label']
        
        # For signature defense, inject token in prompt (simulating system prompt)
        if use_signature:
            modified_prompt = f"[SECURITY_TOKEN:{canary_token}]\n{prompt}"
            flagged, score, latency = defense.detect(modified_prompt, response)
        else:
            # For other defenses, use original prompt
            flagged, score, latency = defense.detect(prompt, response)
        
        results.append({
            'true_label': true_label,
            'predicted': 1 if flagged else 0,
            'score': score,
            'text': prompt[:100],  # Store first 100 chars for debugging
            'response': response[:100] if response else None
        })
        latencies.append(latency)
        
        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate confusion matrix
    tp = len(results_df[(results_df['true_label'] == 1) & (results_df['predicted'] == 1)])
    fp = len(results_df[(results_df['true_label'] == 0) & (results_df['predicted'] == 1)])
    tn = len(results_df[(results_df['true_label'] == 0) & (results_df['predicted'] == 0)])
    fn = len(results_df[(results_df['true_label'] == 1) & (results_df['predicted'] == 0)])
    
    # Calculate metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Latency statistics
    latency_p50 = np.percentile(latencies, 50)
    latency_p95 = np.percentile(latencies, 95)
    latency_mean = np.mean(latencies)
    
    # Print results
    print(f"\nResults for {defense_name}:")
    print(f"  True Positives:  {tp:4d}")
    print(f"  False Positives: {fp:4d}")
    print(f"  True Negatives:  {tn:4d}")
    print(f"  False Negatives: {fn:4d}")
    print(f"\nMetrics:")
    print(f"  TPR (Recall):    {tpr:.1%}")
    print(f"  FPR:             {fpr:.1%}")
    print(f"  Precision:       {precision:.1%}")
    print(f"  F1 Score:        {f1:.3f}")
    print(f"\nLatency:")
    print(f"  Mean:   {latency_mean:.3f} ms")
    print(f"  p50:    {latency_p50:.3f} ms")
    print(f"  p95:    {latency_p95:.3f} ms")
    
    return {
        'defense': defense_name,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'latency_mean': latency_mean,
        'latency_p50': latency_p50,
        'latency_p95': latency_p95,
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
        tp = len(sample[(sample['true_label'] == 1) & (sample['predicted'] == 1)])
        fp = len(sample[(sample['true_label'] == 0) & (sample['predicted'] == 1)])
        tn = len(sample[(sample['true_label'] == 0) & (sample['predicted'] == 0)])
        fn = len(sample[(sample['true_label'] == 1) & (sample['predicted'] == 0)])
        
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
    # Count disagreements
    n01 = len(results_df1[
        (results_df1['predicted'] == 0) & (results_df2['predicted'] == 1)
    ])
    n10 = len(results_df1[
        (results_df1['predicted'] == 1) & (results_df2['predicted'] == 0)
    ])
    
    # Use exact binomial test (works better for small samples)
    n = n01 + n10
    if n == 0:
        return 0.0, 1.0
    
    # Two-tailed exact test
    p_value = 2 * min(
        stats.binom.cdf(min(n01, n10), n, 0.5),
        1 - stats.binom.cdf(min(n01, n10) - 1, n, 0.5)
    )
    
    return n, p_value


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 experiments with real responses")
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to CSV with LLM responses"
    )
    parser.add_argument(
        "--skip-moderation",
        action="store_true",
        help="Skip OpenAI Moderation API (if no API key)"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset with responses
    df = load_responses_dataset(args.responses)
    
    # Initialize defenses
    print("\n" + "="*60)
    print("Initializing Defenses")
    print("="*60)
    
    defenses = {
        'Signature-Only': SignatureOnlyDefense(),
        'Rules-Only': RulesOnlyDefense(),
        'NeMo-Baseline': NeMoBaselineDefense()
    }
    
    if not args.skip_moderation:
        try:
            defenses['OpenAI-Moderation'] = OpenAIModerationDefense()
            print("✅ OpenAI Moderation API initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize OpenAI Moderation: {e}")
            print("   Running without OpenAI Moderation")
    
    # Evaluate each defense
    all_results = {}
    
    for name, defense in defenses.items():
        result = evaluate_defense(defense, df, name)
        all_results[name] = result
    
    # Calculate bootstrap CIs for all defenses
    print("\n" + "="*60)
    print("Calculating Bootstrap Confidence Intervals")
    print("="*60)
    
    for name, result in all_results.items():
        print(f"\n{name}:")
        for metric in ['tpr', 'fpr', 'precision', 'f1']:
            lower, upper = bootstrap_confidence_interval(
                result['results_df'], 
                metric,
                n_iterations=1000
            )
            result[f'{metric}_ci_lower'] = lower
            result[f'{metric}_ci_upper'] = upper
            print(f"  {metric.upper():10s}: {result[metric]:.3f} [{lower:.3f}, {upper:.3f}]")
    
    # Perform pairwise McNemar tests
    print("\n" + "="*60)
    print("Pairwise McNemar Statistical Tests")
    print("="*60)
    
    defense_names = list(all_results.keys())
    n_comparisons = len(defense_names) * (len(defense_names) - 1) // 2
    bonferroni_alpha = 0.05 / n_comparisons
    
    print(f"Bonferroni-corrected α = {bonferroni_alpha:.4f} ({n_comparisons} comparisons)")
    
    mcnemar_results = []
    for i, name1 in enumerate(defense_names):
        for name2 in defense_names[i+1:]:
            n, p_value = mcnemar_test(
                all_results[name1]['results_df'],
                all_results[name2]['results_df']
            )
            significant = p_value < bonferroni_alpha
            mcnemar_results.append({
                'defense1': name1,
                'defense2': name2,
                'n_disagreements': n,
                'p_value': p_value,
                'significant': significant
            })
            
            sig_mark = "***" if significant else ""
            print(f"\n{name1} vs {name2}:")
            print(f"  Disagreements: {n}")
            print(f"  p-value: {p_value:.4f} {sig_mark}")
    
    # Save results
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    # Main results table
    results_table = []
    for name, result in all_results.items():
        results_table.append({
            'Defense': name,
            'TPR': result['tpr'],
            'TPR_CI_Lower': result['tpr_ci_lower'],
            'TPR_CI_Upper': result['tpr_ci_upper'],
            'FPR': result['fpr'],
            'FPR_CI_Lower': result['fpr_ci_lower'],
            'FPR_CI_Upper': result['fpr_ci_upper'],
            'Precision': result['precision'],
            'F1': result['f1'],
            'Latency_Mean': result['latency_mean'],
            'TP': result['tp'],
            'FP': result['fp'],
            'TN': result['tn'],
            'FN': result['fn']
        })
    
    results_df = pd.DataFrame(results_table)
    results_file = output_dir / 'phase1_output_detection_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✅ Results saved to: {results_file}")
    
    # McNemar test results
    mcnemar_df = pd.DataFrame(mcnemar_results)
    mcnemar_file = output_dir / 'phase1_mcnemar_tests.csv'
    mcnemar_df.to_csv(mcnemar_file, index=False)
    print(f"✅ McNemar tests saved to: {mcnemar_file}")
    
    # Save full results as JSON
    results_json = {
        'experiment': 'Phase 1 - OUTPUT Detection with Real Responses',
        'dataset': args.responses,
        'n_samples': len(df),
        'defenses': {
            name: {
                'metrics': {k: v for k, v in result.items() 
                           if k != 'results_df' and not k.endswith('_df')}
            }
            for name, result in all_results.items()
        },
        'mcnemar_tests': mcnemar_results
    }
    
    json_file = output_dir / 'phase1_output_detection_full.json'
    with open(json_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✅ Full results saved to: {json_file}")
    
    print("\n" + "="*60)
    print("✅ Phase 1 Experiments Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
