"""
Statistical significance tests for baseline comparison.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, binomtest, binom
from typing import List, Dict, Tuple
from itertools import combinations


def mcnemar(table, exact=True):
    """
    Perform McNemar's test on a 2x2 contingency table.
    
    Args:
        table: 2x2 array [[a, b], [c, d]] where:
            a = both correct
            b = method 1 correct, method 2 wrong
            c = method 1 wrong, method 2 correct  
            d = both wrong
        exact: If True, use exact binomial test; else use chi-square approximation
        
    Returns:
        Object with .statistic and .pvalue attributes
    """
    table = np.asarray(table)
    b = table[0, 1]  # Method 1 correct, method 2 wrong
    c = table[1, 0]  # Method 1 wrong, method 2 correct
    
    class McNemarResult:
        def __init__(self, statistic, pvalue):
            self.statistic = statistic
            self.pvalue = pvalue
    
    if exact or (b + c) < 25:
        # Exact binomial test
        n = b + c
        if n == 0:
            return McNemarResult(0, 1.0)
        
        # Two-sided test using scipy.stats.binomtest
        result = binomtest(int(b), int(n), 0.5, alternative='two-sided')
        p_value = result.pvalue
        statistic = b
    else:
        # Chi-square approximation with continuity correction
        statistic = (abs(b - c) - 1)**2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)
    
    return McNemarResult(statistic, p_value)


def mcnemar_test(
    y_true: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    method_a: str = "Method A",
    method_b: str = "Method B"
) -> Dict:
    """
    Perform McNemar's test to compare two classifiers.
    
    Tests the null hypothesis that the two methods have the same error rate.
    
    Args:
        y_true: Ground truth labels (0/1)
        predictions_a: Predictions from method A (0/1)
        predictions_b: Predictions from method B (0/1)
        method_a: Name of method A
        method_b: Name of method B
        
    Returns:
        Dictionary with test results
    """
    # Create contingency table
    # Rows: Method A correct/incorrect
    # Cols: Method B correct/incorrect
    correct_a = (predictions_a == y_true)
    correct_b = (predictions_b == y_true)
    
    # 2x2 contingency table
    both_correct = np.sum(correct_a & correct_b)
    a_correct_b_wrong = np.sum(correct_a & ~correct_b)
    a_wrong_b_correct = np.sum(~correct_a & correct_b)
    both_wrong = np.sum(~correct_a & ~correct_b)
    
    # McNemar's test focuses on the disagreement cells
    # [[both_correct, a_correct_b_wrong],
    #  [a_wrong_b_correct, both_wrong]]
    
    # Perform McNemar test
    contingency_table = [[both_correct, a_correct_b_wrong],
                        [a_wrong_b_correct, both_wrong]]
    
    # Use exact test if counts are small
    result = mcnemar(contingency_table, exact=True if min(a_correct_b_wrong, a_wrong_b_correct) < 25 else False)
    
    # Calculate accuracy for each method
    accuracy_a = np.mean(correct_a)
    accuracy_b = np.mean(correct_b)
    
    # Interpretation
    p_value = result.pvalue
    significant = p_value < 0.05
    
    if significant:
        if accuracy_a > accuracy_b:
            interpretation = f"{method_a} significantly better than {method_b}"
        else:
            interpretation = f"{method_b} significantly better than {method_a}"
    else:
        interpretation = "No significant difference"
    
    return {
        'method_a': method_a,
        'method_b': method_b,
        'accuracy_a': accuracy_a,
        'accuracy_b': accuracy_b,
        'both_correct': both_correct,
        'a_correct_b_wrong': a_correct_b_wrong,
        'a_wrong_b_correct': a_wrong_b_correct,
        'both_wrong': both_wrong,
        'statistic': result.statistic,
        'p_value': p_value,
        'significant': significant,
        'interpretation': interpretation
    }


def pairwise_mcnemar_tests(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """
    Perform pairwise McNemar tests for all method pairs.
    
    Args:
        y_true: Ground truth labels
        predictions_dict: Dictionary mapping method names to predictions
        
    Returns:
        DataFrame with all pairwise test results
    """
    results = []
    
    method_names = list(predictions_dict.keys())
    
    # Test all pairs
    for method_a, method_b in combinations(method_names, 2):
        result = mcnemar_test(
            y_true,
            predictions_dict[method_a],
            predictions_dict[method_b],
            method_a,
            method_b
        )
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Sort by p-value
    df = df.sort_values('p_value')
    
    return df


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[bool], float]:
    """
    Apply Bonferroni correction for multiple testing.
    
    Args:
        p_values: List of p-values
        alpha: Significance level
        
    Returns:
        (significant_flags, corrected_alpha) tuple
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in p_values]
    
    return significant, corrected_alpha


def format_mcnemar_results_table(results_df: pd.DataFrame, use_bonferroni: bool = True) -> str:
    """
    Format McNemar test results as a markdown table.
    
    Args:
        results_df: DataFrame from pairwise_mcnemar_tests
        use_bonferroni: Whether to apply Bonferroni correction
        
    Returns:
        Markdown formatted table
    """
    if use_bonferroni:
        significant, corrected_alpha = bonferroni_correction(results_df['p_value'].tolist())
        results_df = results_df.copy()
        results_df['significant_bonf'] = significant
    
    lines = []
    lines.append("| Method A | Method B | Acc(A) | Acc(B) | p-value | Significant | Interpretation |")
    lines.append("|----------|----------|--------|--------|---------|-------------|----------------|")
    
    for idx, row in results_df.iterrows():
        sig_marker = "✓" if (row['significant_bonf'] if use_bonferroni else row['significant']) else "✗"
        
        line = (
            f"| {row['method_a']} | {row['method_b']} | "
            f"{row['accuracy_a']:.3f} | {row['accuracy_b']:.3f} | "
            f"{row['p_value']:.4f} | {sig_marker} | {row['interpretation']} |"
        )
        lines.append(line)
    
    if use_bonferroni:
        lines.append("")
        lines.append(f"*Note: Bonferroni-corrected α = {corrected_alpha:.4f} ({len(results_df)} comparisons)*")
    
    return "\n".join(lines)


def format_mcnemar_results_latex(results_df: pd.DataFrame, use_bonferroni: bool = True) -> str:
    """
    Format McNemar test results as a LaTeX table.
    
    Args:
        results_df: DataFrame from pairwise_mcnemar_tests
        use_bonferroni: Whether to apply Bonferroni correction
        
    Returns:
        LaTeX formatted table
    """
    if use_bonferroni:
        significant, corrected_alpha = bonferroni_correction(results_df['p_value'].tolist())
        results_df = results_df.copy()
        results_df['significant_bonf'] = significant
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Pairwise McNemar Statistical Significance Tests}")
    lines.append("\\label{tab:mcnemar}")
    lines.append("\\begin{tabular}{llcccc}")
    lines.append("\\toprule")
    lines.append("Method A & Method B & Acc(A) & Acc(B) & $p$-value & Sig. \\\\")
    lines.append("\\midrule")
    
    for idx, row in results_df.iterrows():
        sig_marker = "$\\checkmark$" if (row['significant_bonf'] if use_bonferroni else row['significant']) else ""
        
        line = (
            f"{row['method_a'].replace('_', ' ')} & "
            f"{row['method_b'].replace('_', ' ')} & "
            f"{row['accuracy_a']:.3f} & "
            f"{row['accuracy_b']:.3f} & "
            f"{row['p_value']:.4f} & "
            f"{sig_marker} \\\\"
        )
        lines.append(line)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    if use_bonferroni:
        lines.append(f"\\\\[0.2cm]")
        lines.append(f"{{\\small Note: Bonferroni-corrected $\\alpha = {corrected_alpha:.4f}$ ({len(results_df)} comparisons)}}")
    
    lines.append("\\end{table}")
    
    return "\n".join(lines)


if __name__ == '__main__':
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Simulate ground truth
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0] * 10)
    
    # Simulate predictions from three methods
    # Method A: high accuracy
    predictions_a = y_true.copy()
    predictions_a[np.random.choice(len(y_true), 10, replace=False)] = 1 - predictions_a[np.random.choice(len(y_true), 10, replace=False)]
    
    # Method B: medium accuracy
    predictions_b = y_true.copy()
    predictions_b[np.random.choice(len(y_true), 20, replace=False)] = 1 - predictions_b[np.random.choice(len(y_true), 20, replace=False)]
    
    # Method C: low accuracy
    predictions_c = y_true.copy()
    predictions_c[np.random.choice(len(y_true), 30, replace=False)] = 1 - predictions_c[np.random.choice(len(y_true), 30, replace=False)]
    
    predictions_dict = {
        'Signature-Only': predictions_a,
        'Rules-Only': predictions_b,
        'NeMo-Baseline': predictions_c
    }
    
    print("Pairwise McNemar Tests")
    print("=" * 80)
    
    results_df = pairwise_mcnemar_tests(y_true, predictions_dict)
    
    print("\nMarkdown Table:")
    print(format_mcnemar_results_table(results_df))
    
    print("\n\nLaTeX Table:")
    print(format_mcnemar_results_latex(results_df))
