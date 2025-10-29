# Phase 1: OUTPUT Detection with Real LLM Responses

## Overview

We are running **true OUTPUT detection** experiments by:

1. **Generating real LLM responses** for all test prompts (400 samples)
2. **Evaluating defenses** on whether they detect canary token leakage in responses
3. **Applying all enhancements** (bootstrap CIs, McNemar tests, error bars, cost analysis)

This gives us the **most realistic baseline** for prompt injection detection.

## Why OUTPUT Detection?

### OUTPUT Detection (checking responses)
- ✅ Tests if attacks **actually succeed** in making LLM leak secrets
- ✅ Realistic evaluation of defense effectiveness
- ✅ Measures real security risk (did the token leak?)
- ✅ Can use with any LLM (GPT-4, Claude, Llama, etc.)

### INPUT Detection (checking prompts only)
- ⚠️ Only detects **potential** attacks in prompts
- ⚠️ Doesn't verify if attacks succeed
- ⚠️ May have high false positives (harmless prompts flagged)
- ✅ Faster (no LLM call needed)
- ✅ Cheaper (no API costs)

## Phase 1 Timeline

### Step 1: Generate LLM Responses ⏳ IN PROGRESS
**Status:** Running (2% complete)
**Script:** `generate_llm_responses.py`
**Command:**
```bash
python generate_llm_responses.py --model gpt4 --split test --batch-size 10 --delay 1.0
```

**Progress:**
- Model: GPT-4 (gpt-4)
- Dataset: 400 test prompts (200 attacks, 200 benign)
- Rate limiting: 1 second delay between requests
- Expected time: ~60 minutes
- Expected cost: ~$0.40-$0.80
- Output: `data/responses/test_gpt4_responses.csv`

### Step 2: Run Phase 1 Experiments 📋 PENDING
**Script:** `run_phase1_with_responses.py`
**Command:**
```bash
python run_phase1_with_responses.py --responses data/responses/test_gpt4_responses.csv
```

**What it will do:**
- Load 400 test prompts + GPT-4 responses
- Evaluate 4 defenses:
  1. **Signature-Only**: Canary token detection (expected ~80% TPR)
  2. **Rules-Only**: Regex pattern matching (expected ~20-25% TPR)
  3. **NeMo-Baseline**: NeMo Guardrails patterns (expected ~30-35% TPR)
  4. **OpenAI-Moderation**: Commercial API filter (expected ~40-60% TPR)
- Calculate bootstrap confidence intervals (95%, n=1000)
- Perform pairwise McNemar statistical tests
- Save comprehensive results

### Step 3: Visualize Results 📊 PENDING
**Script:** `visualize_phase1_output_detection.py` (to be created)

Will generate:
- 4-panel performance plot with error bars (TPR, FPR, F1, Latency)
- ROC scatter plot with 2D confidence regions
- Confusion matrix heatmaps
- Attack family breakdown (7 families)

### Step 4: Cost Analysis 💰 PENDING
**Script:** `analyze_phase1_costs.py` (to be created)

Will analyze:
- Production deployment costs at 5 prevalence levels (0.1%-5%)
- False alarm rates per 10,000 requests
- Cost-benefit tradeoffs for each defense
- Recommended deployment thresholds

## Expected Results

### Hypothesis: OUTPUT Detection Shows Higher TPR

**Original Phase 1 (OUTPUT detection on full 2K dataset):**
- Signature-Only: 79.9% TPR
- Rules-Only: 19.8% TPR
- NeMo-Baseline: 13.0% TPR
- OpenAI-Moderation: 12.0% TPR

**Enhanced Phase 1 (INPUT detection on 400 test set):**
- Signature-Only: 25% TPR
- Rules-Only: 24% TPR
- NeMo-Baseline: 12.5% TPR
- OpenAI-Moderation: Not tested

**Expected from OUTPUT detection with real responses:**
- Signature-Only: **70-85% TPR** (high - canary tokens should leak)
- Rules-Only: **15-25% TPR** (low - only catches explicit patterns)
- NeMo-Baseline: **25-40% TPR** (medium - catches common attack patterns)
- OpenAI-Moderation: **40-65% TPR** (medium-high - commercial safety filter)

### Key Differences
- **Higher TPR** expected because we're checking if attacks **succeed** (token in response)
- **More realistic** because we see actual LLM behavior
- **Better comparison** to OpenAI Moderation (fair apple-to-apple comparison)

## Reproducibility

### Environment
- Python 3.13.7
- OpenAI API: gpt-4 model
- Test set: 400 samples (stratified split, seed=42)
- Rate limiting: 1.0s delay between requests
- Bootstrap: 1000 iterations, 95% CI
- Statistical tests: McNemar with Bonferroni correction

### Cost Estimates
- GPT-4 API: ~$0.03 per 1K input tokens, ~$0.06 per 1K output tokens
- 400 prompts × ~150 tokens/prompt = 60K input tokens ≈ $1.80
- 400 responses × ~200 tokens/response = 80K output tokens ≈ $4.80
- **Total: ~$6.60** (with retries and overhead)

### API Keys Required
- `OPENAI_API_KEY`: For GPT-4 response generation and OpenAI Moderation
- `ANTHROPIC_API_KEY`: Optional, for Claude comparison

## Files Created

### Data
- `data/responses/test_gpt4_responses.csv`: LLM responses for test set
- `data/responses/test_gpt4_summary.json`: Generation statistics

### Scripts
- `generate_llm_responses.py`: Response generation with rate limiting
- `run_phase1_with_responses.py`: Full Phase 1 experiment runner
- `visualize_phase1_output_detection.py`: Visualization generator (to create)
- `analyze_phase1_costs.py`: Cost analysis (to create)

### Results
- `results/phase1_output_detection_results.csv`: Main results table
- `results/phase1_output_detection_full.json`: Comprehensive results
- `results/phase1_mcnemar_tests.csv`: Statistical test results
- `results/phase1_output_detection_plots/`: Visualization outputs

## Next Steps After Phase 1

1. **Phase 2**: LLM-as-judge baselines (GPT-4, Claude)
2. **Phase 3**: NeMo Guardrails integration and tuning
3. **Phase 4**: Signature tuning and optimization
4. **Phase 5**: Hybrid approaches combining multiple defenses

## Progress Tracking

- [x] Create data splits (train/dev/test/OOD)
- [x] Label attack families (7 families)
- [x] Implement statistical tests (McNemar)
- [x] Add bootstrap confidence intervals
- [x] Create reproducibility documentation
- [⏳] Generate LLM responses (2% complete - 60min remaining)
- [ ] Run Phase 1 OUTPUT detection experiments
- [ ] Create visualizations with error bars
- [ ] Perform cost analysis
- [ ] Write up results for IEEE Software article
