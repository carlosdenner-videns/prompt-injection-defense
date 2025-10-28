# IEEE Software Paper - Experimental Design

**Title**: "Pattern-Based Prompt Injection Detection: From Dataset Analysis to Production Deployment"

**Date**: October 28, 2025  
**Status**: Experimental design for paper submission

---

## 🎯 Research Questions

**RQ1**: Can data-driven pattern discovery create effective prompt injection defenses without ML models?

**RQ2**: What is the performance gap between simulated testing and real-world LLM API deployment?

**RQ3**: How do heuristic defenses compare to published ML-based approaches?

**RQ4**: Do these defenses generalize across LLM vendors (OpenAI vs Anthropic)?

---

## 📊 Experimental Design

### Phase 1: Baseline Establishment (Compare to Prior Work)

**Purpose**: Establish credible comparison points

**Experiments**:

1. **Reproduce NeMo Guardrails (Rebedea 2023)**
   - Use their published patterns (from GitHub)
   - Test on your 2K dataset
   - Report: TPR, FPR, latency
   - **Expected**: ~30-40% TPR (based on their paper)

2. **Test Commercial Baseline (OpenAI Moderation API)**
   - Run subset (100 samples) through OpenAI moderation endpoint
   - Report: TPR, FPR, cost per request
   - **Expected**: ~60-70% TPR, higher cost

3. **Simple Rule-Based Baseline**
   - Use ONLY the rules.yml patterns (no classifier)
   - Report: TPR, FPR
   - **Expected**: ~20-30% TPR (high FP)

**Deliverable**: Table showing baseline performance landscape

---

### Phase 2: Data-Driven Pattern Discovery (Your Methodology)

**Purpose**: Show systematic improvement through your process

**Step 2.1: Dataset Analysis**
```bash
python analyze_patterns.py --data data/prompts_hf_augmented.csv --output analysis_output/pattern_frequencies.json
```

**Outputs**:
- Pattern frequency table (DAN 51%, ignore 51%, etc.)
- Word/bigram analysis
- Statistical significance (χ² test for each pattern)

**Step 2.2: Iterative Classifier Development**

Test 3 versions to show progression:

| Version | Description | Patterns | Expected TPR |
|---------|-------------|----------|--------------|
| V1 | Generic patterns (5 basic) | "ignore", "jailbreak", "DAN" | ~10-15% |
| V2 | Frequency-based (top 10) | Data-driven from analysis | ~40-50% |
| V3 | Weighted + combinations | Full system | ~58-60% |

**Run each version**:
```bash
# V1 - Generic
python run_experiment.py --classifier-version v1 --data data/prompts_hf_augmented.csv --out results/paper/clf_v1

# V2 - Frequency-based  
python run_experiment.py --classifier-version v2 --data data/prompts_hf_augmented.csv --out results/paper/clf_v2

# V3 - Full enhanced
python run_experiment.py --classifier-version v3 --data data/prompts_hf_augmented.csv --out results/paper/clf_v3
```

**Deliverable**: Figure showing iterative improvement (V1 → V2 → V3)

---

### Phase 3: Component Optimization & Combination

**Purpose**: Show how components combine (ablation study)

**Test Matrix** (threshold = 0.5 for all):

| Config ID | Components | Expected TPR | Expected FPR |
|-----------|------------|--------------|--------------|
| C1 | Signature only | ~81% | ~0% |
| C2 | Classifier only (V3) | ~26% | ~0% |
| C3 | Rules only | ~20% | ~10% |
| C4 | Sig + Clf | ~86% | ~0% |
| C5 | Sig + Rules | ~83% | ~3% |
| C6 | Sig + Rules + Clf | ~85% | ~3% |

**Run all configs**:
```bash
python run_paper_experiment.py --phase ablation --threshold 0.5 --data data/prompts_hf_augmented.csv
```

**Analysis**:
- Which combinations are Pareto-optimal?
- McNemar tests for statistical significance
- Bootstrap CIs for all metrics

**Deliverable**: 
- Table 3 in paper (component comparison)
- Figure 2B (Pareto frontier)

---

### Phase 4: Threshold Optimization

**Purpose**: Show trade-off space and optimal selection

**Sweep**: t ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}

**Focus on**: Sig + Clf (best performer from Phase 3)

```bash
python run_paper_experiment.py --phase threshold --pipeline sig,clf --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7
```

**Analysis**:
- Plot TPR/FPR curves with confidence bands
- Identify Pareto frontier points
- Select 2 optimal configs:
  - **High recall**: t=0.1 (max F1)
  - **High precision**: t=0.5 (zero FP)

**Deliverable**:
- Figure 2A (threshold sensitivity curves)
- Decision guide for practitioners

---

### Phase 5: Real-World API Validation

**Purpose**: Measure simulated-to-real gap (YOUR UNIQUE CONTRIBUTION)

**Setup**:
- Sample: 200 prompts (100 attacks, 100 benign)
- Models: GPT-4o-mini (OpenAI)
- Config: Sig + Clf at t=0.3 (balanced threshold)

**Metrics to measure**:
1. **Pre-LLM blocking rate** (defense TPR/FPR)
2. **Attack success rate** (manual assessment of LLM responses)
3. **Overall protection** (defense + LLM safety layer)
4. **Latency breakdown** (defense vs API call)
5. **Cost analysis** (API calls saved)

```bash
python run_paper_experiment.py --phase real-world --max-samples 200 --model gpt-4o-mini --threshold 0.3
```

**Manual Analysis** (critical!):
- Review all 200 responses
- Tag actual attack success (not just defense decision)
- Document failure modes

**Deliverable**:
- Table 4: Simulated vs Real comparison
- Figure 2C: Performance gap visualization
- Attack success examples (3-4 cases)

---

### Phase 6: Cross-Vendor Generalization

**Purpose**: Prove model-agnostic design (RQ4)

**Setup**:
- Sample: 100 prompts (50 attacks, 50 benign)
- Models: GPT-4o-mini, GPT-4o, Claude-3-Haiku, Claude-3-Sonnet
- Config: Sig + Clf at t=0.3

```bash
python run_paper_experiment.py --phase cross-vendor --max-samples 100 --models gpt-4o-mini,gpt-4o,claude-haiku,claude-sonnet
```

**Analysis**:
- Variance across models (σ²)
- McNemar tests for pairwise differences
- Latency comparison

**Deliverable**:
- Table 5: Cross-model consistency
- Claim: "No statistically significant difference (p>0.05)"

---

## 📈 Expected Results Summary

### Table 3: Component Performance (Simulated, 2K samples, t=0.5)

| Configuration | TPR | FPR | F1 | Latency | Use Case |
|---------------|-----|-----|----|---------| ---------|
| NeMo (baseline) | 35% | 3% | 0.51 | 0.02ms | Prior work |
| Signature only | 81% | 0% | 0.90 | 0.00ms | Fast baseline |
| Classifier V3 | 26% | 0% | 0.41 | 0.06ms | Standalone |
| **Sig + Clf (ours)** | **86%** | **0%** | **0.93** | 0.07ms | **Production** |
| Sig + Rules + Clf | 85% | 3% | 0.92 | 0.08ms | Defense-in-depth |

### Table 4: Real-World Validation (200 samples, OpenAI GPT-4o-mini)

| Metric | Simulated (t=0.5) | Real API (t=0.3) | Gap |
|--------|-------------------|------------------|-----|
| Defense TPR | 86% | 48% | -38% |
| Defense FPR | 0% | 8% | +8% |
| Attack Success | N/A | ~20% | - |
| Overall Protection | 86% | ~80%* | -6% |
| Cost per 1K requests | $0 | $12.30 | - |

*Defense blocks 48% + LLM safety catches ~60% of remainder

### Table 5: Cross-Vendor Generalization (100 samples, t=0.3)

| Model | Vendor | TPR | FPR | Latency (p50) |
|-------|--------|-----|-----|---------------|
| gpt-4o-mini | OpenAI | 48% | 8% | 2,507ms |
| gpt-4o | OpenAI | 50% | 8% | 6,270ms |
| claude-haiku | Anthropic | 46% | 8% | 1,690ms |
| claude-sonnet | Anthropic | 48% | 8% | 3,450ms |

**Variance**: σ(TPR) = 0.016, σ(FPR) = 0.000
**Conclusion**: No significant cross-model differences (ANOVA p=0.89)

---

## 🎯 Key Claims for Paper

### Claim 1: Systematic methodology works
**Evidence**: Classifier V1 (10%) → V2 (45%) → V3 (58%) through data-driven process

### Claim 2: Competitive with ML defenses
**Evidence**: Our 86% TPR vs Llama-Guard ~70% (different datasets, but comparable)

### Claim 3: Real-world gap is substantial
**Evidence**: 86% simulated → 48% real (38% drop), highlighting validation importance

### Claim 4: Cross-vendor generalization
**Evidence**: Consistent 48% TPR ±2% across GPT-4 and Claude (p>0.05)

### Claim 5: Production-ready economics
**Evidence**: 48% pre-LLM blocking → $8K annual savings on 1M requests

---

## 🔬 Statistical Rigor

**For ALL comparisons**:
1. Bootstrap 95% CIs (n=1,000 resamples)
2. McNemar tests for paired configs
3. ANOVA for cross-model variance
4. Effect sizes (Cohen's d) for improvements

**Significance threshold**: α = 0.05

**Small sample correction**: Use exact tests for n<30

---

## 📊 Figures for Paper (2 maximum)

### Figure 1: Methodology & Results Pipeline (1 panel)

**Flowchart showing**:
```
[2K Dataset]
    ↓
[Pattern Analysis] → "51% DAN, 51% ignore"
    ↓
[Classifier V1→V2→V3] → "10% → 45% → 58% TPR"
    ↓
[Component Combination] → "Sig+Clf: 86% TPR, 0% FP"
    ↓
[Real API Testing] → "48% TPR, 8% FP (gap: -38%)"
    ↓
[Cross-Vendor] → "Consistent across GPT-4 & Claude"
```

### Figure 2: Performance Analysis (4 panels)

**Panel A**: Threshold sensitivity (TPR/FPR vs threshold, error bars)
**Panel B**: Pareto frontier (FPR vs TPR, optimal configs highlighted)
**Panel C**: Simulated vs Real (bar chart, show gap)
**Panel D**: Cost-benefit (blocking rate vs annual savings)

---

## 🚀 Implementation Plan

### New Script: `run_paper_experiment.py`

```python
"""
Comprehensive experiment runner for IEEE Software paper.
Phases: baseline, discovery, ablation, threshold, real-world, cross-vendor
"""

import argparse

def run_baseline_comparison():
    """Phase 1: Compare to NeMo, OpenAI Moderation, simple rules"""
    pass

def run_iterative_development():
    """Phase 2: Test Classifier V1, V2, V3"""
    pass

def run_ablation_study():
    """Phase 3: Test all component combinations"""
    pass

def run_threshold_sweep():
    """Phase 4: Optimize threshold for Sig+Clf"""
    pass

def run_real_world_validation():
    """Phase 5: OpenAI API testing (200 samples)"""
    pass

def run_cross_vendor():
    """Phase 6: GPT-4 + Claude consistency"""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "discovery", "ablation", 
                                             "threshold", "real-world", "cross-vendor", "all"])
    parser.add_argument("--data", default="data/prompts_hf_augmented.csv")
    parser.add_argument("--output", default="results/paper")
    
    args = parser.parse_args()
    
    if args.phase == "all":
        # Run complete experiment suite (expect 2-3 hours + $2-5 API costs)
        run_baseline_comparison()
        run_iterative_development()
        run_ablation_study()
        run_threshold_sweep()
        run_real_world_validation()
        run_cross_vendor()
    else:
        # Run single phase
        phase_map[args.phase]()
```

---

## 📝 Execution Checklist

### Before Paper Experiments

- [ ] Archive current results to `results/exploratory/`
- [ ] Clean up `results/paper/` directory
- [ ] Verify dataset: 2,000 samples balanced
- [ ] Check API keys (OpenAI, Anthropic)
- [ ] Set random seed: 42 (all experiments)
- [ ] Document exact package versions (requirements.txt)

### Phase 1: Baseline (30 min, free)
- [ ] Test NeMo original patterns
- [ ] Test simple rules only
- [ ] Test signature only
- [ ] Generate Table 2 (baseline comparison)

### Phase 2: Discovery (1 hour, free)
- [ ] Run pattern analysis
- [ ] Implement Classifier V1 (generic)
- [ ] Implement Classifier V2 (frequency)
- [ ] Test V3 (full enhanced)
- [ ] Generate progression figure

### Phase 3: Ablation (30 min, free)
- [ ] Test all 6 component configs
- [ ] Compute bootstrap CIs
- [ ] McNemar tests for pairs
- [ ] Generate Table 3

### Phase 4: Threshold (20 min, free)
- [ ] Sweep t=0.1 to 0.7
- [ ] Plot curves with CIs
- [ ] Identify Pareto points
- [ ] Generate Figure 2A

### Phase 5: Real-World (1 hour, ~$1-2)
- [ ] Sample 200 prompts
- [ ] Run through OpenAI API
- [ ] Manual attack success assessment
- [ ] Cost analysis
- [ ] Generate Table 4, Figure 2C

### Phase 6: Cross-Vendor (30 min, ~$1-2)
- [ ] Test 4 models (100 samples each)
- [ ] Variance analysis (ANOVA)
- [ ] Latency comparison
- [ ] Generate Table 5

### Final Analysis (30 min, free)
- [ ] All statistical tests
- [ ] Generate both figures
- [ ] Create LaTeX tables
- [ ] Write results CSV summary

---

## 💰 Budget Estimate

**Total Runtime**: ~3-4 hours (mostly automated)

**API Costs**:
- Real-world validation (200 samples × 1 model): ~$1.50
- Cross-vendor (100 samples × 4 models): ~$2.00
- **Total**: ~$3.50

**Sample size justification**:
- 200 samples → 95% CI width ±7% (acceptable)
- 100 samples/model → 95% CI width ±10% (acceptable for consistency test)

---

## 🎓 Addressing EiC Concerns

### ✅ "Detailed technical account of novelty"
**Covered by**: Phase 2 (pattern discovery methodology), Phase 5 (real-world validation)

### ✅ "Scientific evaluation compared to other suggestions"
**Covered by**: Phase 1 (NeMo baseline), comparison to Llama-Guard in discussion

### ✅ "Give examples, not talk about"
**Covered by**: All phases include concrete numbers, manual attack examples in Phase 5

### ✅ "Link to evidence for unsubstantiated claims"
**Covered by**: Every claim mapped to specific experiment + table/figure

### ✅ "Sufficient technical detail for replication"
**Covered by**: Complete code (DOI), dataset (DOI), execution checklist

---

## 📚 Outputs for Paper

### Tables (5 total)
1. Dataset statistics
2. Baseline comparison (NeMo, rules, signature)
3. Component ablation (simulated)
4. Simulated vs real-world
5. Cross-vendor generalization

### Figures (2 total = 500 words deducted)
1. Methodology pipeline + results
2. Performance analysis (4 panels)

### Supplementary Material (online repository)
- Complete source code
- Raw experiment outputs (CSVs)
- Statistical analysis scripts
- Reproduction instructions

---

## 🎯 Success Criteria

**A successful paper experiment**:
1. ✅ Shows clear progression (baseline → optimized)
2. ✅ Compares to published work (NeMo, Llama-Guard)
3. ✅ Validates on real APIs (not just simulated)
4. ✅ Quantifies performance gaps (simulated vs real)
5. ✅ Tests generalization (cross-vendor)
6. ✅ Provides economic justification (cost savings)
7. ✅ Fully reproducible (<4 hours, <$5)

**Avoid**:
- ❌ Cherry-picked comparisons (no "toy baseline")
- ❌ Vague claims without evidence
- ❌ Missing statistical tests
- ❌ Only simulated results
- ❌ Single-model testing

---

**Next Steps**: 
1. Review this design
2. Implement `run_paper_experiment.py`
3. Execute Phase 1-6
4. Generate all tables/figures
5. Write paper using outline

**Timeline**: 1-2 weeks from start to submission-ready results
