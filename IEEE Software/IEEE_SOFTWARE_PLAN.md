# IEEE Software: Six-Phase Experimental Plan

## Overview

This repository contains a comprehensive experimental framework for the IEEE Software article:

> **"Pattern-Based Prompt Injection Detection: A Data-Driven Approach"**

The experiments are organized into **6 phases**, each building upon previous results to systematically demonstrate the value of the proposed methodology.

---

## Phase Structure

### ✅ Phase 1: Baselines and Prior Art Comparison (COMPLETED)

**Purpose:** Establish credible reference points showing what exists before your contributions.

**Experiments:**
1. **NeMo Guardrails baseline** - Traditional rule-based system (~30-35% TPR expected)
2. **OpenAI Moderation API** - Commercial content filter comparator (~40-60% TPR)
3. **Rules-Only** - Simple regex patterns (~20-25% TPR)
4. **Signature-Only** - Canary token detection (~80% TPR, limited scope)

**Key Output:** Baseline Performance Table for IEEE Software paper

**Status:** ✅ Complete - Standalone implementation in `phase1/`

**Quick Start:**
```powershell
cd "IEEE Software\phase1"
pip install -r requirements.txt
python run_phase1_experiments.py --skip-moderation
python analyze_phase1_results.py
```

**Expected Runtime:** ~10 minutes  
**Expected Outputs:** Results tables, plots, LaTeX tables in `phase1/results/`

---

### 🔄 Phase 2: Simple Combinations (PLANNED)

**Purpose:** Show that naive combinations provide incremental improvements but still fall short.

**Experiments:**
1. **Signature + Rules** - Combine token detection with pattern matching
2. **Signature + NeMo** - Combine token detection with weighted patterns
3. **Rules + Manual Heuristics** - Add domain-specific rules

**Expected Results:**
- TPR: ~40-50% (better than individual baselines)
- FPR: ~5-10%
- Finding: Simple combinations help but leave gap to ideal

**Key Output:** Combination Performance Table

**Status:** 🔄 Not yet implemented

---

### 🔄 Phase 3: Data-Driven Pattern Discovery (PLANNED)

**Purpose:** Demonstrate systematic pattern extraction from real attack dataset.

**Experiments:**
1. **Frequency Analysis** - Count pattern occurrences in 2K HuggingFace samples
2. **Statistical Filtering** - χ² test for pattern significance (p<0.01)
3. **Pattern Weighting** - Weight patterns by attack coverage
4. **Pattern Library Construction** - Build 3-tier weighted library

**Expected Results:**
- Discover ~30+ discriminative patterns
- Pattern frequency distribution (DAN: 51%, Ignore: 51%, etc.)
- Weighted scoring algorithm

**Key Output:** Pattern Discovery Methodology + Pattern Library

**Status:** 🔄 Not yet implemented (but analysis scripts exist in main repo)

---

### 🔄 Phase 4: Heuristic Classifier Optimization (PLANNED)

**Purpose:** Show systematic threshold tuning and statistical validation.

**Experiments:**
1. **Threshold Sweep** - Grid search t ∈ [0.1, 0.2, ..., 0.9]
2. **ROC Analysis** - TPR vs FPR curves
3. **Pareto Frontier** - Identify optimal configurations
4. **Bootstrap Validation** - 95% confidence intervals (n=1,000)
5. **McNemar Tests** - Statistical significance of improvements

**Expected Results:**
- Optimal threshold identification (t=0.1 for max F1, t=0.5 for 0% FPR)
- TPR: 85-92% (depending on threshold)
- FPR: 0-5%
- Statistical validation of improvements over Phase 1 baselines

**Key Output:** Threshold Optimization Methodology + Statistical Validation

**Status:** 🔄 Not yet implemented (but threshold tuning scripts exist)

---

### 🔄 Phase 5: Real-World API Validation (PLANNED)

**Purpose:** Validate simulated performance on actual LLM APIs (OpenAI GPT-4, Claude).

**Experiments:**
1. **OpenAI GPT-4 Testing** - 100 samples (50 attacks, 50 benign)
2. **Anthropic Claude Testing** - 100 samples
3. **Pre-LLM Blocking Analysis** - Measure % blocked before API call
4. **Cost Analysis** - Calculate API savings from pre-LLM filtering
5. **Attack Success Assessment** - Manual evaluation of LLM responses

**Expected Results:**
- Real TPR: ~48% (lower than simulated due to dataset bias)
- Real FPR: ~8% (higher than simulated due to benign diversity)
- Pre-LLM blocking: ~50% → $8K-11K annual savings
- Cross-vendor consistency: Similar performance on GPT-4 and Claude

**Key Output:** Real-World Validation Results + Cost-Benefit Analysis

**Status:** 🔄 Not yet implemented (but OpenAI adapter exists)

---

### 🔄 Phase 6: Ablation Studies and Sensitivity Analysis (PLANNED)

**Purpose:** Demonstrate robustness and identify critical components.

**Experiments:**
1. **Component Ablation** - Test Signature-only, Classifier-only, Rules-only
2. **Pattern Ablation** - Remove high/medium/low priority patterns
3. **Threshold Sensitivity** - Measure performance variance across thresholds
4. **Dataset Sensitivity** - Test on different attack distributions
5. **Cross-Model Generalization** - Validate across multiple LLMs

**Expected Results:**
- Component importance ranking
- Pattern category contributions (high-priority: +40% TPR, etc.)
- Threshold sensitivity curves
- Generalization evidence

**Key Output:** Ablation Study Results + Robustness Analysis

**Status:** 🔄 Not yet implemented

---

## Implementation Timeline

### Completed ✅
- **Phase 1:** Baselines and Prior Art Comparison (this submission)
  - 4 baseline defenses implemented
  - Evaluation framework complete
  - Results analysis pipeline ready
  - ~500 lines of production-quality code

### Remaining Work 🔄
- **Phase 2-6:** ~2-3 weeks of implementation
  - Leverage existing codebase (defenses, datasets, analysis scripts)
  - Primary work: Experiment orchestration and result synthesis
  - Total estimated effort: ~40-60 hours

---

## Repository Structure

```
IEEE Software/
├── phase1/                          # ✅ COMPLETE
│   ├── README.md                    # Comprehensive documentation
│   ├── QUICKSTART.md                # One-command execution guide
│   ├── requirements.txt             # Dependencies
│   ├── run_phase1_experiments.py    # Main experiment runner
│   ├── analyze_phase1_results.py    # Results analysis
│   ├── configs/
│   │   ├── experiment.json          # Experiment configuration
│   │   └── rules.yml                # Rule patterns
│   ├── data/
│   │   └── prompts_hf_augmented.csv # 2,000-sample dataset
│   ├── src/
│   │   └── defenses/                # Defense implementations
│   │       ├── signature_only.py
│   │       ├── rules_only.py
│   │       ├── nemo_baseline.py
│   │       └── openai_moderation.py
│   └── results/                     # Generated results (after run)
│
├── phase2/                          # 🔄 PLANNED
├── phase3/                          # 🔄 PLANNED
├── phase4/                          # 🔄 PLANNED
├── phase5/                          # 🔄 PLANNED
├── phase6/                          # 🔄 PLANNED
│
└── IEEE_SOFTWARE_PLAN.md           # This file
```

---

## Replication Instructions

### For Journal Reviewers

Each phase is **standalone** and **reproducible** with minimal setup:

1. Navigate to phase directory (e.g., `cd phase1`)
2. Install dependencies: `pip install -r requirements.txt`
3. Run experiments: `python run_phaseN_experiments.py`
4. Analyze results: `python analyze_phaseN_results.py`
5. Review outputs in `results/` directory

**Total time per phase:** ~10-30 minutes

### Full Reproduction

To reproduce all 6 phases:

```powershell
# Phase 1
cd "IEEE Software\phase1"
pip install -r requirements.txt
python run_phase1_experiments.py --skip-moderation
python analyze_phase1_results.py

# Phase 2-6 (when implemented)
# Similar pattern for each phase
```

**Total time (all phases):** ~2-3 hours

---

## Expected Paper Contributions

### From Phase 1 (Current)
- **Baseline Performance Table** (Table 1 in paper)
- **Prior Art Comparison** (Section 6.2)
- **Motivation for Data-Driven Approach** (Section 1)

### From Phase 2-3 (Future)
- **Pattern Discovery Methodology** (Section 3.2)
- **Pattern Library** (supplementary material)
- **Frequency Analysis Results** (Table 2 in paper)

### From Phase 4 (Future)
- **Threshold Optimization** (Section 3.4)
- **Statistical Validation** (bootstrap CIs, McNemar tests)
- **Performance Trade-off Analysis** (Figure 2, Panel B)

### From Phase 5 (Future)
- **Real-World Validation** (Section 4.2)
- **Cost-Benefit Analysis** (Section 5.3)
- **Cross-Vendor Generalization** (Table in Section 4.2)

### From Phase 6 (Future)
- **Ablation Studies** (Section 5.1)
- **Robustness Analysis** (Section 5.2)
- **Component Importance Ranking**

---

## Key Metrics Across Phases

| Phase | Primary Metric | Expected Value | Purpose |
|-------|---------------|----------------|---------|
| 1 | Baseline TPR | 20-60% | Establish reference points |
| 2 | Combined TPR | 40-50% | Show incremental improvement |
| 3 | Pattern Count | ~30+ | Demonstrate discovery |
| 4 | Optimized TPR | 85-92% | Show systematic optimization |
| 5 | Real-world TPR | ~48% | Validate on actual APIs |
| 6 | Ablation Δ | Component contributions | Identify critical parts |

---

## Contact

**For questions about Phase 1 (current submission):**
- Review `phase1/README.md` for detailed documentation
- Check `phase1/QUICKSTART.md` for one-command execution
- Open an issue for bugs or questions

**For questions about future phases:**
- See this plan for expected implementation timeline
- Future phases will follow same standalone structure

---

## License

MIT License - see `LICENSE` file

---

**Last Updated:** October 29, 2025  
**Status:** Phase 1 complete, Phases 2-6 planned
