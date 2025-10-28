# Quick Reference: Prompt Injection Defense Framework

## Installation & Setup

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install core dependencies
pip install pandas numpy pyyaml scipy scikit-learn statsmodels matplotlib seaborn

# Optional: For ML classifiers
pip install transformers torch
```

## Generate Dataset

```bash
python src/generate_dataset.py
# Output: data/prompts_base_large.csv (257 prompts)
#         data/prompts_large_aug.csv (312 with paraphrases)
```

## Run Experiments

### Single Experiment
```bash
python src/run_experiment.py \
    --data data/prompts_base_large.csv \
    --pipeline signature,rules,classifier \
    --threshold 0.5 \
    --out results/test
```

### All Configurations
```bash
python run_all_experiments.py
# Runs 12 different defense configurations
```

## Analyze Results

### Generate Pareto Frontier
```bash
python src/analyze_results.py \
    --results results \
    --output analysis_output
# Output: analysis_output/pareto_frontier.png
```

### Statistical Comparison
```bash
python src/analyze_results.py \
    --results results \
    --compare signature_only sig_rules sig_rules_clf \
    --output analysis_output
# Output: Bootstrap CIs + McNemar tests
```

## Available Defense Components

| Component | Description | Performance (TPR) |
|-----------|-------------|-------------------|
| `signature` | Token injection | 86.0% |
| `rules` | Regex patterns | 13.1% |
| `classifier` | Heuristic scorer | 0-13.1% |
| `protectai` | ML classifier (requires install) | TBD |
| `nemo` | Simple guardrails | 0% (needs tuning) |

## Best Configurations

1. **Production**: `signature,rules,classifier` (86.9% TPR, 0% FPR)
2. **Speed**: `signature` (86.0% TPR, 0.00ms latency)
3. **Oracle**: `signature,rules,classifier --oracle` (84.1% TPR, efficient)

## Key Results

- **Pareto Optimal**: sig_rules_clf
- **Zero false positives** across all configurations
- **Statistical significance**: No significant difference between top 3 (p > 0.05)
- **Latency**: All configs < 0.02ms median

## File Structure

```
data/
├── prompts_seed.csv              # Original 20 prompts
├── prompts_aug.csv               # Original augmented
├── prompts_base_large.csv        # New 257 base
└── prompts_large_aug.csv         # New 312 augmented

results/
├── signature_only/
├── sig_rules_clf/                # Best performer
└── [other experiments]/

analysis_output/
└── pareto_frontier.png           # Visualization

src/
├── run_experiment.py             # Main pipeline
├── generate_dataset.py           # Dataset generation
├── analyze_results.py            # Statistical analysis
└── defenses/
    ├── signature_proxy.py
    ├── rules.py
    ├── classifier_stub.py
    ├── llamaguard_adapter.py     # ProtectAI (ready)
    └── nemo_guardrails_adapter.py
```

## Next Steps

1. **Swap in ProtectAI**: Install transformers, use `--pipeline protectai`
2. **Expand dataset**: Add more agent/tool-based attacks
3. **Implement full NeMo**: Add OpenAI API key for LLM-based rails
4. **Test evasion**: Generate adversarial prompts targeting specific defenses
