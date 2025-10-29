"""
Enhanced Phase 1 experiment runner with test split evaluation and statistical tests.

This script runs the baseline experiments using the test split only (no data leakage).
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import defenses
from defenses.signature_only import SignatureOnlyDefense
from defenses.rules_only import RulesOnlyDefense
from defenses.nemo_baseline import NeMoBaselineDefense
from defenses.openai_moderation import OpenAIModerationDefense

# Import utilities
from data_utils import load_splits
from statistical_tests import pairwise_mcnemar_tests, format_mcnemar_results_table, format_mcnemar_results_latex


def bootstrap_confidence_interval(
    values: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        values: Array of values (predictions, labels, etc.)
        metric_func: Function to compute metric on resampled data
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (e.g., 0.95 for 95%)
        random_state: Random seed
        
    Returns:
        (lower_bound, point_estimate, upper_bound) tuple
    """
    np.random.seed(random_state)
    
    n = len(values)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, n, replace=True)
        resampled = metric_func(indices)
        bootstrap_metrics.append(resampled)
    
    # Calculate percentiles
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(bootstrap_metrics, lower_percentile)
    upper = np.percentile(bootstrap_metrics, upper_percentile)
    point = metric_func(np.arange(n))
    
    return lower, point, upper


def evaluate_defense(defense, df: pd.DataFrame, defense_name: str) -> Dict:
    """Evaluate a defense on a dataset."""
    print(f"\nEvaluating {defense_name}...")
    print("-" * 60)
    
    predictions = []
    latencies = []
    errors = 0
    
    for idx, row in df.iterrows():
        try:
            start_time = time.time()
            result = defense.detect(row['text'])
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            predictions.append(result)
            latencies.append(latency)
            
        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            predictions.append(0)  # Default to benign on error
            latencies.append(0)
            errors += 1
        
        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    predictions = np.array(predictions)
    y_true = df['label'].values
    
    # Calculate metrics
    tp = np.sum((predictions == 1) & (y_true == 1))
    fp = np.sum((predictions == 1) & (y_true == 0))
    tn = np.sum((predictions == 0) & (y_true == 0))
    fn = np.sum((predictions == 0) & (y_true == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Bootstrap confidence intervals
    print("  Computing confidence intervals...")
    
    def tpr_func(indices):
        pred_sample = predictions[indices]
        true_sample = y_true[indices]
        tp_sample = np.sum((pred_sample == 1) & (true_sample == 1))
        fn_sample = np.sum((pred_sample == 0) & (true_sample == 1))
        return tp_sample / (tp_sample + fn_sample) if (tp_sample + fn_sample) > 0 else 0
    
    def fpr_func(indices):
        pred_sample = predictions[indices]
        true_sample = y_true[indices]
        fp_sample = np.sum((pred_sample == 1) & (true_sample == 0))
        tn_sample = np.sum((pred_sample == 0) & (true_sample == 0))
        return fp_sample / (fp_sample + tn_sample) if (fp_sample + tn_sample) > 0 else 0
    
    def f1_func(indices):
        pred_sample = predictions[indices]
        true_sample = y_true[indices]
        tp_sample = np.sum((pred_sample == 1) & (true_sample == 1))
        fp_sample = np.sum((pred_sample == 1) & (true_sample == 0))
        fn_sample = np.sum((pred_sample == 0) & (true_sample == 1))
        
        prec = tp_sample / (tp_sample + fp_sample) if (tp_sample + fp_sample) > 0 else 0
        rec = tp_sample / (tp_sample + fn_sample) if (tp_sample + fn_sample) > 0 else 0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    tpr_ci = bootstrap_confidence_interval(y_true, tpr_func)
    fpr_ci = bootstrap_confidence_interval(y_true, fpr_func)
    f1_ci = bootstrap_confidence_interval(y_true, f1_func)
    
    avg_latency = np.mean(latencies) if latencies else 0
    
    results = {
        'defense': defense_name,
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'tpr': tpr,
        'tpr_ci_lower': tpr_ci[0],
        'tpr_ci_upper': tpr_ci[2],
        'fpr': fpr,
        'fpr_ci_lower': fpr_ci[0],
        'fpr_ci_upper': fpr_ci[2],
        'precision': precision,
        'f1': f1,
        'f1_ci_lower': f1_ci[0],
        'f1_ci_upper': f1_ci[2],
        'accuracy': accuracy,
        'avg_latency_ms': avg_latency,
        'total_samples': len(df),
        'errors': errors,
        'predictions': predictions.tolist()
    }
    
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"  TPR={tpr:.3f} [{tpr_ci[0]:.3f}, {tpr_ci[2]:.3f}]")
    print(f"  FPR={fpr:.3f} [{fpr_ci[0]:.3f}, {fpr_ci[2]:.3f}]")
    print(f"  F1={f1:.3f} [{f1_ci[0]:.3f}, {f1_ci[2]:.3f}]")
    print(f"  Avg Latency={avg_latency:.2f}ms")
    
    return results


def analyze_by_attack_family(df: pd.DataFrame, results: Dict[str, Dict]) -> pd.DataFrame:
    """Analyze performance by attack family."""
    print("\nAnalyzing performance by attack family...")
    
    family_stats = []
    
    # Get unique attack families
    attack_df = df[df['label'] == 1]
    families = attack_df['family'].unique()
    
    for family in families:
        family_df = attack_df[attack_df['family'] == family]
        family_indices = family_df.index.tolist()
        
        stat_dict = {
            'family': family,
            'count': len(family_df)
        }
        
        # Calculate TPR for each defense on this family
        for defense_name, defense_results in results.items():
            predictions = np.array(defense_results['predictions'])
            family_predictions = predictions[family_indices]
            
            # TPR = correct attack detections / total attacks in family
            tp = np.sum(family_predictions == 1)
            tpr = tp / len(family_df) if len(family_df) > 0 else 0
            
            stat_dict[f'{defense_name}_tpr'] = tpr
        
        family_stats.append(stat_dict)
    
    return pd.DataFrame(family_stats)


def main():
    print("="*80)
    print("Phase 1: Enhanced Baseline Experiments with Statistical Tests")
    print("="*80)
    
    # Load test split
    print("\nLoading test split...")
    try:
        train, dev, test, ood = load_splits('data/splits')
        print(f"  Test set: {len(test)} samples")
        print(f"  Attacks: {len(test[test['label'] == 1])}")
        print(f"  Benign: {len(test[test['label'] == 0])}")
    except Exception as e:
        print(f"Error loading splits: {e}")
        print("Run `python src/data_utils.py` first to create splits.")
        return
    
    # Initialize defenses
    print("\nInitializing defenses...")
    defenses = {
        'Signature-Only': SignatureOnlyDefense(),
        'Rules-Only': RulesOnlyDefense('configs/rules.yml'),
        'NeMo-Baseline': NeMoBaselineDefense(threshold=0.5),
        # 'OpenAI-Moderation': OpenAIModerationDefense()  # Skip due to rate limits
    }
    
    # Run evaluations
    all_results = {}
    predictions_dict = {}
    
    for defense_name, defense in defenses.items():
        results = evaluate_defense(defense, test, defense_name)
        all_results[defense_name] = results
        predictions_dict[defense_name] = np.array(results['predictions'])
    
    # Statistical significance tests
    print("\n" + "="*80)
    print("Statistical Significance Tests (McNemar)")
    print("="*80)
    
    y_true = test['label'].values
    mcnemar_results = pairwise_mcnemar_tests(y_true, predictions_dict)
    
    print("\n" + format_mcnemar_results_table(mcnemar_results))
    
    # Attack family analysis
    family_df = analyze_by_attack_family(test, all_results)
    
    print("\n" + "="*80)
    print("Performance by Attack Family")
    print("="*80)
    print(family_df.to_string(index=False))
    
    # Save results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    with open(results_dir / 'phase1_test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: results/phase1_test_results.json")
    
    # Save summary table
    summary_rows = []
    for defense_name, results in all_results.items():
        summary_rows.append({
            'Defense': defense_name,
            'TPR': f"{results['tpr']:.3f}",
            'TPR_CI': f"[{results['tpr_ci_lower']:.3f}, {results['tpr_ci_upper']:.3f}]",
            'FPR': f"{results['fpr']:.3f}",
            'FPR_CI': f"[{results['fpr_ci_lower']:.3f}, {results['fpr_ci_upper']:.3f}]",
            'F1': f"{results['f1']:.3f}",
            'F1_CI': f"[{results['f1_ci_lower']:.3f}, {results['f1_ci_upper']:.3f}]",
            'Latency_ms': f"{results['avg_latency_ms']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / 'phase1_test_summary.csv', index=False)
    print(f"  Saved: results/phase1_test_summary.csv")
    
    # Save McNemar results
    mcnemar_results.to_csv(results_dir / 'phase1_mcnemar_tests.csv', index=False)
    print(f"  Saved: results/phase1_mcnemar_tests.csv")
    
    # Save family analysis
    family_df.to_csv(results_dir / 'phase1_family_analysis.csv', index=False)
    print(f"  Saved: results/phase1_family_analysis.csv")
    
    # Save LaTeX tables
    with open(results_dir / 'phase1_test_table.tex', 'w') as f:
        f.write(format_mcnemar_results_latex(mcnemar_results))
    print(f"  Saved: results/phase1_test_table.tex")
    
    print("\n✅ Phase 1 enhanced experiments completed successfully!")
    print("\nKey findings:")
    print("  - Results based on held-out test set (400 samples)")
    print("  - Statistical significance assessed with McNemar tests")
    print("  - Performance analyzed by attack family")
    print("  - 95% confidence intervals computed via bootstrap")


if __name__ == '__main__':
    main()
