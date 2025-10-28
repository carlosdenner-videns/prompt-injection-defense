# Cross-Model Validation for Prompt Injection Defenses

## Overview

This suite of scripts validates the generalizability of the **signature + classifier (sig+clf)** defense pipeline across multiple LLM vendors and models. It demonstrates that the defense mechanism performs consistently regardless of the underlying language model.

## Models Tested

### OpenAI
- **gpt-4o-mini**: Fast, cost-effective model
- **gpt-4o**: Latest flagship model

### Anthropic Claude
- **claude-3-haiku-20240307**: Fastest, most affordable
- **claude-3-5-sonnet-20241022**: Most capable, balanced

## Metrics Collected

For each model, we measure:
- **TPR (True Positive Rate)**: % of attacks correctly blocked
- **FPR (False Positive Rate)**: % of benign requests incorrectly blocked
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **Defense Latency**: Overhead added by defense mechanisms (ms)
- **LLM Latency**: Time for model response (ms)
- **Total Latency**: End-to-end request time (ms)

## Quick Start

### Prerequisites

1. **Environment Setup**
   ```bash
   # Ensure you have API keys in .env
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   ```

2. **Install Dependencies**
   ```bash
   pip install openai anthropic pandas matplotlib seaborn numpy python-dotenv
   ```

### Running the Full Pipeline

```bash
# Step 1: Run cross-model validation (collects data from all models)
python run_cross_model_validation.py --data data/prompts_hf_augmented.csv --max-samples 50

# Step 2: Analyze and aggregate results
python analyze_cross_model_results.py --input results/cross_model

# Step 3: Generate visualizations
python visualize_cross_model.py --input results/cross_model_summary.csv
```

### Quick Test (Minimal Cost)

For a quick test with minimal API costs:

```bash
# Test with only 20 samples (10 attacks + 10 benign)
python run_cross_model_validation.py --data data/prompts_hf_augmented.csv --max-samples 20 --rate-limit 1.0

# Analyze
python analyze_cross_model_results.py

# Visualize
python visualize_cross_model.py
```

## Output Files

### Validation Results
```
results/cross_model/
├── all_models_raw.csv                    # Combined raw results
├── gpt-4o-mini/
│   ├── predictions.csv                   # Per-prompt predictions
│   └── summary.csv                       # Model summary metrics
├── gpt-4o/
│   ├── predictions.csv
│   └── summary.csv
├── claude-haiku/
│   ├── predictions.csv
│   └── summary.csv
└── claude-sonnet/
    ├── predictions.csv
    └── summary.csv
```

### Analysis Outputs
```
results/
├── cross_model_summary.csv               # Main deliverable: cross-model comparison
└── cross_model_table.tex                 # LaTeX table for papers
```

### Visualizations
```
results/figures/
├── model_generalization.png              # Main figure (4-panel comparison)
├── performance_consistency.png           # Variance analysis
└── detailed_comparison_heatmap.png       # Metric heatmap
```

## Command-Line Options

### run_cross_model_validation.py
```bash
python run_cross_model_validation.py [OPTIONS]

Options:
  --data PATH              Input CSV with test prompts (default: data/prompts_hf_augmented.csv)
  --max-samples INT        Max samples per model (default: 100)
  --threshold FLOAT        Detection threshold (default: 0.5)
  --output PATH            Output directory (default: results/cross_model)
  --rate-limit FLOAT       Delay between API calls in seconds (default: 0.5)
```

### analyze_cross_model_results.py
```bash
python analyze_cross_model_results.py [OPTIONS]

Options:
  --input PATH             Input directory with model results (default: results/cross_model)
  --output PATH            Output CSV path (default: results/cross_model_summary.csv)
  --latex PATH             Output LaTeX table path (default: results/cross_model_table.tex)
```

### visualize_cross_model.py
```bash
python visualize_cross_model.py [OPTIONS]

Options:
  --input PATH             Input summary CSV (default: results/cross_model_summary.csv)
  --output-dir PATH        Output directory for figures (default: results/figures)
```

## Understanding the Results

### Model Generalization Figure

The main output figure (`model_generalization.png`) contains 4 panels:

**Panel A: TPR vs FPR**
- Scatter plot showing detection performance
- Closer to top-left corner = better (high TPR, low FPR)
- Gold star marks ideal performance (100% TPR, 0% FPR)

**Panel B: F1 Score Ranking**
- Horizontal bar chart of overall performance
- Higher F1 = better balance of precision and recall

**Panel C: Latency Breakdown**
- Shows defense overhead vs LLM response time
- Helps identify bottlenecks

**Panel D: Vendor Summary**
- Radar chart comparing OpenAI vs Anthropic
- Shows average performance across all metrics

### Interpreting Generalization

**Good Generalization** (Expected):
- Low variance in TPR/FPR across models
- F1 scores within 5-10% range
- Consistent defense latency (model-independent)

**Model-Specific Issues** (Unexpected):
- Large variance in TPR/FPR
- One vendor significantly outperforming
- High correlation between model and performance

## Cost Estimation

Approximate costs for different sample sizes (using current pricing as of Oct 2025):

| Samples/Model | Total Tests | OpenAI Cost | Anthropic Cost | Total    |
|---------------|-------------|-------------|----------------|----------|
| 20            | 80          | ~$0.02      | ~$0.03         | ~$0.05   |
| 50            | 200         | ~$0.05      | ~$0.08         | ~$0.13   |
| 100           | 400         | ~$0.10      | ~$0.15         | ~$0.25   |
| 500           | 2000        | ~$0.50      | ~$0.75         | ~$1.25   |

*Note: Costs are estimates based on token usage for short prompts/responses*

## Troubleshooting

### API Key Errors
```
⚠️ OPENAI_API_KEY not found. Skipping gpt-4o-mini
```
**Solution**: Ensure `.env` file exists in project root with valid API keys.

### Rate Limiting
```
Error: Rate limit exceeded
```
**Solution**: Increase `--rate-limit` to add more delay between calls:
```bash
python run_cross_model_validation.py --rate-limit 2.0
```

### Missing Dependencies
```
ModuleNotFoundError: No module named 'anthropic'
```
**Solution**: Install required packages:
```bash
pip install openai anthropic pandas matplotlib seaborn
```

### No Results Found
```
❌ Error: No summary.csv files found
```
**Solution**: Run validation script first:
```bash
python run_cross_model_validation.py
```

## Advanced Usage

### Testing Specific Models Only

Modify the `models` dictionary in `run_cross_model_validation.py` to include only desired models:

```python
self.models = {
    "gpt-4o-mini": {
        "vendor": "openai",
        "adapter_class": OpenAIAdapter,
        "kwargs": {"model": "gpt-4o-mini", "max_tokens": 150}
    },
    # Comment out others
}
```

### Custom Threshold Testing

Test different detection thresholds to find optimal settings:

```bash
for threshold in 0.3 0.4 0.5 0.6 0.7; do
    python run_cross_model_validation.py --threshold $threshold --output results/cross_model_t${threshold}
done
```

### Adding More Models

To add support for additional models (e.g., Google Gemini, Llama):

1. Create adapter in `src/defenses/[model]_adapter.py`
2. Add entry to `models` dictionary in `run_cross_model_validation.py`
3. Ensure adapter has `call_with_metadata()` method returning response with `content`, `latency_ms`, `total_tokens`

## Integration with Existing Experiments

This cross-model validation complements your existing experiments:

- **Baseline experiments**: Compare cross-model results to simulated baselines
- **Threshold tuning**: Use cross-model results to validate optimal thresholds
- **Cost-benefit analysis**: Compare real API costs to simulated performance

## Publication-Ready Outputs

The scripts generate publication-ready materials:

1. **LaTeX Table** (`cross_model_table.tex`):
   - Ready to \include in papers
   - Properly formatted with captions

2. **High-Resolution Figures**:
   - 300 DPI PNG images
   - Professional styling with clear labels
   - Suitable for journals and conferences

3. **CSV Data**:
   - Machine-readable results
   - Easy to import into statistical software (R, SPSS)

## Next Steps

After running cross-model validation:

1. **Compare to Baselines**: How do real models compare to simulations?
2. **Identify Weaknesses**: Which attack families are model-specific?
3. **Optimize Defense**: Can we reduce latency without sacrificing TPR?
4. **Scale Up**: Run on full dataset for comprehensive validation

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review console output for specific error messages
3. Verify API keys and network connectivity
4. Ensure all dependencies are installed

## Citation

If you use these scripts in your research, please cite:

```bibtex
@misc{cross_model_validation_2025,
  title={Cross-Model Validation of Prompt Injection Defenses},
  author={Your Name},
  year={2025},
  note={Demonstrates generalizability across OpenAI and Anthropic models}
}
```

---

**Last Updated**: October 28, 2025
