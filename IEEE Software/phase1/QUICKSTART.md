# Phase 1: Quick Start Guide

## One-Command Execution

For reviewers and replicators who want to run everything with minimal setup:

### Windows (PowerShell)

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Phase 1 experiments (without OpenAI Moderation)
python run_phase1_experiments.py --skip-moderation

# 4. Analyze results
python analyze_phase1_results.py
```

**Total time:** ~10 minutes  
**Output:** All results in `results/` directory

### With OpenAI Moderation API

```powershell
# Set API key
$env:OPENAI_API_KEY = "sk-your-key-here"

# Run all defenses (including OpenAI Moderation)
python run_phase1_experiments.py

# Analyze results
python analyze_phase1_results.py
```

**Total time:** ~20 minutes  
**Cost:** ~$0.50-1.00

## Expected Output Files

After running experiments and analysis:

```
results/
├── phase1_baseline_performance.csv      # Summary table
├── phase1_results_full.json             # Full metrics + CIs
├── phase1_baseline_table.tex            # LaTeX table
├── phase1_comparison_plot.png           # Bar charts
├── phase1_tradeoff_plot.png             # TPR vs FPR scatter
├── phase1_summary_report.txt            # Text summary
├── phase1_findings_snippet.tex          # LaTeX snippet
└── *_detailed.csv                       # Per-defense detailed results
```

## Troubleshooting

### "Module not found" error

```powershell
# Add src to Python path
$env:PYTHONPATH = "$PWD\src"
python run_phase1_experiments.py
```

### OpenAI API authentication error

```powershell
# Verify API key
echo $env:OPENAI_API_KEY

# Skip if no key
python run_phase1_experiments.py --skip-moderation
```

### Memory issues with 2,000 samples

Currently not configurable - use full dataset for reproducibility.
If issues persist, contact authors.

## Validation

Your results should match these ranges:

| Defense | TPR Range | FPR Range | F1 Range |
|---------|-----------|-----------|----------|
| Signature-Only | 75-85% | 0-1% | 0.85-0.92 |
| Rules-Only | 20-30% | 3-8% | 0.30-0.45 |
| NeMo-Baseline | 30-40% | 8-12% | 0.40-0.55 |
| OpenAI-Moderation | 40-60% | 1-3% | 0.60-0.75 |

If results differ significantly:
1. Check dataset has exactly 2,000 samples (1,000 attacks, 1,000 benign)
2. Verify random seed is 42
3. Ensure all dependencies are correct versions

## Next Steps

After Phase 1 completion:

1. **Review results** in `results/phase1_summary_report.txt`
2. **Copy LaTeX table** from `phase1_baseline_table.tex` to paper
3. **Include plots** in paper (phase1_comparison_plot.png, phase1_tradeoff_plot.png)
4. **Proceed to Phase 2** (not yet implemented)

## Contact

Questions? Open an issue or email the authors.
