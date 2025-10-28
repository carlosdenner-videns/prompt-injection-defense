# Cross-Model Validation Quick Reference

## 🚀 One-Line Commands

```bash
# Quick test (10 samples, ~$0.03, 5 min)
python test_cross_model_pipeline.py

# Standard validation (100 samples, ~$0.25, 20 min)
python run_cross_model_validation.py --max-samples 100 && python analyze_cross_model_results.py && python visualize_cross_model.py

# Batch quick test
python batch_cross_model_validation.py --mode quick

# Batch full validation
python batch_cross_model_validation.py --mode full
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `run_cross_model_validation.py` | Main validation script |
| `analyze_cross_model_results.py` | Aggregate and analyze |
| `visualize_cross_model.py` | Generate figures |
| `test_cross_model_pipeline.py` | Quick smoke test |
| `batch_cross_model_validation.py` | Batch runner |
| `CROSS_MODEL_VALIDATION_GUIDE.md` | Full documentation |
| `CROSS_MODEL_IMPLEMENTATION_SUMMARY.md` | Technical details |

## 📊 Main Outputs

| Output | Description |
|--------|-------------|
| `results/cross_model_summary.csv` | ⭐ Main deliverable |
| `results/figures/model_generalization.png` | ⭐ Main figure |
| `results/cross_model/[model]/predictions.csv` | Per-model results |
| `results/cross_model_table.tex` | LaTeX table |

## 🎯 Models Tested

- **OpenAI**: gpt-4o-mini, gpt-4o
- **Anthropic**: claude-3-haiku, claude-3-5-sonnet

## 📈 Metrics

- **Performance**: TPR, FPR, Accuracy, F1
- **Latency**: Defense, LLM, Total (p50, p95)
- **Generalization**: Variance across models

## 💰 Cost Guide

| Samples | Cost | Time |
|---------|------|------|
| 10 | $0.03 | 5 min |
| 20 | $0.05 | 8 min |
| 50 | $0.13 | 15 min |
| 100 | $0.25 | 20 min |

## 🔧 Common Options

```bash
--max-samples N        # Number of prompts per model (default: 100)
--threshold T          # Detection threshold (default: 0.5)
--rate-limit S         # Delay between calls in seconds (default: 0.5)
--output DIR           # Output directory (default: results/cross_model)
```

## ⚡ Troubleshooting

| Problem | Solution |
|---------|----------|
| API key error | Add to `.env`: `OPENAI_API_KEY=...` |
| Rate limit | Increase `--rate-limit 2.0` |
| Missing deps | `pip install openai anthropic pandas matplotlib seaborn` |
| No results | Run validation first: `python run_cross_model_validation.py` |

## 📖 Documentation

- **User Guide**: `CROSS_MODEL_VALIDATION_GUIDE.md`
- **Implementation**: `CROSS_MODEL_IMPLEMENTATION_SUMMARY.md`
- **This card**: `CROSS_MODEL_QUICK_REFERENCE.md`

## ✅ Pre-Flight Checklist

Before running:
- [ ] API keys in `.env`
- [ ] Dependencies installed
- [ ] Data file exists: `data/prompts_hf_augmented.csv`
- [ ] Budget approved

## 🎓 Expected Results

**Good Generalization**:
- TPR variance < 5%
- FPR variance < 5%
- F1 scores: 85-95%
- Defense latency consistent (~5-20ms)

**✅ Conclusion**: Defense works across vendors!

---

**Last Updated**: October 28, 2025  
**Status**: ✅ Ready for testing
