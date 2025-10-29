# IEEE Software: Six-Phase Experimental Framework

This directory contains a comprehensive experimental framework for the IEEE Software article on pattern-based prompt injection detection.

## 📁 Contents

### ✅ Phase 1: Baselines and Prior Art Comparison (COMPLETE)

**Location:** `phase1/`

**Purpose:** Establish baseline performance using existing defenses and commercial APIs.

**Defenses Tested:**
- Signature-Only (canary tokens)
- Rules-Only (simple regex)
- NeMo-Baseline (prior art)
- OpenAI Moderation API (commercial comparator)

**Quick Start:**
```powershell
cd phase1
python validate_setup.py
pip install -r requirements.txt
python run_phase1_experiments.py --skip-moderation
python analyze_phase1_results.py
```

**Runtime:** ~10 minutes  
**Status:** ✅ Ready to run

**Documentation:**
- `phase1/README.md` - Comprehensive guide (350+ lines)
- `phase1/QUICKSTART.md` - One-page quick start
- `phase1/IMPLEMENTATION_SUMMARY.md` - What was created and how to use it

---

### 🔄 Phases 2-6 (Planned)

**Phase 2:** Simple Combinations  
**Phase 3:** Data-Driven Pattern Discovery  
**Phase 4:** Heuristic Classifier Optimization  
**Phase 5:** Real-World API Validation  
**Phase 6:** Ablation Studies

See `IEEE_SOFTWARE_PLAN.md` for complete roadmap.

---

## 🎯 Goals for IEEE Software Article

### Editor Requirements Addressed

✅ **"Compare to other prompt suggestions"**  
→ Phase 1 evaluates NeMo Guardrails and OpenAI Moderation API

✅ **"Establish credible reference points"**  
→ Four baseline defenses with expected TPR ranges

✅ **"Show where the problem stands"**  
→ Baseline performance table demonstrates need for better approaches

✅ **"Reproducibility for journal replication"**  
→ Standalone implementation with one-command execution (~10 min)

### Paper Contributions

**From Phase 1:**
- Baseline Performance Table (Table 1)
- Prior Art Comparison (Section 6.2)
- Motivation for data-driven approach (Section 1)

**From Future Phases:**
- Pattern discovery methodology (Phases 3-4)
- Real-world validation (Phase 5)
- Ablation studies (Phase 6)

---

## 📊 Expected Results Summary

| Phase | Key Metric | Expected Value | Paper Section |
|-------|------------|----------------|---------------|
| 1 | Baseline TPR | 20-60% | 4.1, 6.2 |
| 2 | Combined TPR | 40-50% | 4.1 |
| 3 | Patterns discovered | ~30+ | 3.2 |
| 4 | Optimized TPR | 85-92% | 4.1 |
| 5 | Real-world TPR | ~48% | 4.2 |
| 6 | Component contributions | Ablation Δ | 5.1 |

---

## 🚀 Getting Started

### Step 1: Validate Phase 1 Setup

```powershell
cd "IEEE Software\phase1"
python validate_setup.py
```

This checks that all files are in place and dependencies can be installed.

### Step 2: Run Phase 1 Experiments

```powershell
# Install dependencies
pip install -r requirements.txt

# Run experiments (without OpenAI Moderation)
python run_phase1_experiments.py --skip-moderation

# Analyze results
python analyze_phase1_results.py
```

### Step 3: Review Results

Check `phase1/results/` for:
- `phase1_baseline_performance.csv` - Summary table
- `phase1_baseline_table.tex` - LaTeX table
- `phase1_comparison_plot.png` - Visualizations
- `phase1_summary_report.txt` - Key findings

### Step 4: Integrate into Paper

Copy outputs to your IEEE Software manuscript:
- Table 1: Baseline performance (from `.tex` file)
- Figure 2: Comparison plot (from `.png` file)
- Section 4.1: Results (from summary report)

---

## 📝 Documentation

### For Running Experiments
- **`phase1/QUICKSTART.md`** - One-page execution guide
- **`phase1/README.md`** - Detailed documentation with expected results
- **`phase1/IMPLEMENTATION_SUMMARY.md`** - What was created and why

### For Understanding Design
- **`IEEE_SOFTWARE_PLAN.md`** - Complete 6-phase experimental roadmap
- **`IEEE_SOFTWARE_OUTLINE.md`** - Original paper outline

### For Reviewers
- **`phase1/README.md`** - Replication instructions (~10 min)
- **`phase1/validate_setup.py`** - Pre-run validation script

---

## 🔧 Technical Details

### Implementation Stats
- **Total code:** ~1,500+ lines (Phase 1 only)
- **Languages:** Python 3.9+
- **Dependencies:** pandas, numpy, scipy, scikit-learn
- **Optional:** openai (for Moderation API), matplotlib (for plots)

### Dataset
- **Source:** HuggingFace (deepset/prompt-injections + augmentation)
- **Size:** 2,000 samples (1,000 attacks + 1,000 benign)
- **Location:** `phase1/data/prompts_hf_augmented.csv`
- **Format:** CSV with columns [text, label]

### Defense Implementations
- **Signature-Only:** `phase1/src/defenses/signature_only.py` (90 lines)
- **Rules-Only:** `phase1/src/defenses/rules_only.py` (130 lines)
- **NeMo-Baseline:** `phase1/src/defenses/nemo_baseline.py` (130 lines)
- **OpenAI-Moderation:** `phase1/src/defenses/openai_moderation.py` (150 lines)

---

## ⚠️ Important Notes

### Before Running

1. **Check dataset is present:**  
   `phase1/data/prompts_hf_augmented.csv` should be ~660 KB

2. **Install dependencies:**  
   `pip install -r phase1/requirements.txt`

3. **Validate setup:**  
   `python phase1/validate_setup.py`

### Optional: OpenAI Moderation API

To test commercial API comparator:
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
python run_phase1_experiments.py  # Don't use --skip-moderation
```

**Cost:** ~$0.50-1.00 for 2,000 API calls  
**Runtime:** Adds ~10 minutes

### Reproducibility

All experiments use **seed=42** for reproducibility.  
Same dataset + same seed = identical results.

---

## 📧 Contact

For questions about Phase 1 implementation:
- Review documentation in `phase1/README.md`
- Run validation script: `phase1/validate_setup.py`
- Check `phase1/IMPLEMENTATION_SUMMARY.md` for troubleshooting

For questions about overall experimental plan:
- Review `IEEE_SOFTWARE_PLAN.md`

---

## 📄 License

MIT License - See `../LICENSE` file

---

## ✅ Quick Checklist

Before submitting to IEEE Software:

- [ ] Phase 1 experiments run successfully
- [ ] Results match expected ranges (see README.md)
- [ ] Baseline performance table generated
- [ ] Plots are publication-quality
- [ ] LaTeX table compiles in paper
- [ ] Summary report reviewed for key findings
- [ ] Replication instructions tested (<30 min)

---

**Last Updated:** October 29, 2025  
**Status:** Phase 1 complete, ready for execution

**Next Step:** Run `cd phase1; python validate_setup.py`
