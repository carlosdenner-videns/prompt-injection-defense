# Classifier Mode Guide

## Overview

The cross-model validation script now supports three classifier modes that determine **what** the heuristic classifier scores:

1. **`input`** - Scores the input prompt (model-independent)
2. **`output`** - Scores the LLM response (model-dependent)
3. **`both`** - Scores both and combines them

## Why This Matters

### The Original Problem

The initial implementation had a critical design flaw: it scored the **input prompt** before the LLM call, which meant:

- ✅ Same prompt → Same classifier score
- ❌ Zero variance across models (σ = 0%)
- ❌ "Perfect generalization" was actually a bug
- ❌ Not testing what we thought we were testing

### The Fix

By adding `--classifier-mode`, we can now:

- Test **input filtering** (pre-LLM defense)
- Test **output filtering** (post-LLM defense)
- Test **both** (layered defense)

## Mode Details

### Mode 1: `input` (Default)

**What it does:**
- Classifier scores the INPUT prompt before calling the LLM
- Can block requests before they reach the model (cost savings)
- Model-independent by design

**Scoring logic:**
```python
if args.classifier_mode == "input":
    clf_flag, clf_score, clf_dt = clf.detect(prompt, None)
```

**When to use:**
- Testing input-based defenses
- Cost-sensitive scenarios (block before LLM call)
- Establishing baseline with zero variance

**Expected variance:**
- 🔴 Zero variance across models (same input → same score)
- Not suitable for "generalization" claims

---

### Mode 2: `output`

**What it does:**
- Classifier scores the LLM's RESPONSE after the call
- Different models produce different responses
- Model-dependent (shows true generalization)

**Scoring logic:**
```python
elif args.classifier_mode == "output":
    clf_flag, clf_score, clf_dt = clf.detect("", response)
```

**When to use:**
- Testing model-agnostic generalization
- Detecting jailbreak responses
- Measuring true cross-model variance

**Expected variance:**
- 🟢 Some variance (different models → different responses → different scores)
- Suitable for generalization claims

---

### Mode 3: `both`

**What it does:**
- Scores both input AND output
- Combines scores with `max()` (most suspicious wins)
- Flags if either input or output is suspicious

**Scoring logic:**
```python
else:  # both
    fin_flag_i, s_i, dt_i = clf.detect(prompt, None)
    fin_flag_o, s_o, dt_o = clf.detect("", response)
    
    # Combine (explicit and deterministic)
    clf_score = max(s_i, s_o)
    clf_flag = (fin_flag_i or fin_flag_o)
    clf_dt = dt_i + dt_o
```

**When to use:**
- Layered defense (catch attacks in either direction)
- Maximum detection coverage
- Production deployments

**Expected variance:**
- 🟡 Some variance (depends on output component)
- Balances coverage and generalization

---

## Usage Examples

### Test All Three Modes (Small Sample)

```bash
# Input mode (zero variance expected)
python run_cross_model_validation.py \
    --classifier-mode input \
    --max-samples 20 \
    --output results/mode_input

# Output mode (some variance expected)
python run_cross_model_validation.py \
    --classifier-mode output \
    --max-samples 20 \
    --output results/mode_output

# Both mode (layered defense)
python run_cross_model_validation.py \
    --classifier-mode both \
    --max-samples 20 \
    --output results/mode_both
```

### Quick Comparison Test

```bash
# Run automated comparison (10 samples per mode)
python test_classifier_modes.py
```

This will test all three modes and show variance statistics.

### Full Validation (100 samples)

```bash
# For publication: use output or both mode
python run_cross_model_validation.py \
    --classifier-mode output \
    --max-samples 100 \
    --output results/cross_model_output
```

---

## Interpreting Results

### Input Mode Results

```
📊 Results for mode=input:
   Average classifier score variance: 0.000000
   Prompts with zero variance: 100.0%
```

**Interpretation:**
- ✅ Working as designed (input-based filtering)
- ❌ NOT suitable for "generalization across vendors" claims
- Use case: Cost-efficient pre-filtering

### Output Mode Results

```
📊 Results for mode=output:
   Average classifier score variance: 0.003542
   Prompts with zero variance: 23.0%
```

**Interpretation:**
- ✅ Shows model-dependent behavior
- ✅ Suitable for generalization claims
- 23% zero variance = some prompts produce similar responses across models (expected)
- 77% non-zero variance = different models → different scores (proof of model dependency)

### Both Mode Results

```
📊 Results for mode=both:
   Average classifier score variance: 0.002817
   Prompts with zero variance: 35.0%
```

**Interpretation:**
- ✅ Balanced approach
- More zero-variance than output (input component dominates)
- Better detection coverage (catches both input and output attacks)

---

## Best Practices

### For Research Papers

**Claim: "Defense generalizes across vendors"**
- ✅ Use `--classifier-mode output` or `both`
- ❌ Do NOT use `input` mode (zero variance is expected, not impressive)
- Report variance statistics to prove model-dependency

### For Production Deployments

**Goal: Maximum protection + cost efficiency**
- ✅ Use `--classifier-mode both`
- Blocks obvious attacks before LLM call (saves cost)
- Catches jailbreak responses (defense in depth)

### For Cost-Sensitive Scenarios

**Goal: Minimize API calls**
- ✅ Use `--classifier-mode input`
- Block suspicious prompts before calling expensive LLMs
- Accept that this is model-independent (by design)

---

## Technical Implementation

### Signature Proxy (Always Applied)

Regardless of classifier mode, the signature proxy **always**:
1. Injects token before LLM call
2. Checks for token in response

### Combined Scoring

Final score combines signature and classifier:
```python
combined_score = (sig_score * 0.2) + (clf_score * 0.8)
```

Where `clf_score` depends on the mode:
- `input`: Score of input prompt
- `output`: Score of LLM response
- `both`: max(input_score, output_score)

### Blocking Decision

```python
if combined_score >= threshold:
    # Block request
else:
    # Allow request
```

---

## Troubleshooting

### "Why is variance zero in output mode?"

Possible causes:
1. **Not enough samples** - Try more samples (variance emerges with scale)
2. **Models too similar** - Different GPT-4 variants may produce similar responses
3. **Prompt too simple** - Simple prompts → similar responses
4. **Classifier too coarse** - Binary patterns don't capture subtle differences

### "Why is TPR/FPR identical across models?"

Check the mode:
- `input` mode: **Expected** (same input → same score)
- `output` mode: **Not expected** (check classifier implementation)
- `both` mode: **Partially expected** (input component may dominate)

### "Performance is worse in output mode"

This is **normal** and **expected**:
- Input mode can catch obvious keyword attacks
- Output mode relies on response patterns (harder to detect)
- Use `both` mode for best coverage

---

## Future Work

### Potential Enhancements

1. **Weighted combination** in `both` mode:
   ```python
   clf_score = (s_i * 0.4) + (s_o * 0.6)  # Weighted blend
   ```

2. **Logistic regression** to combine scores:
   ```python
   from scipy.special import expit
   clf_score = expit(beta_0 + beta_1*s_i + beta_2*s_o)
   ```

3. **Adaptive mode** that learns optimal weights per model

4. **Confidence-based combination**:
   ```python
   if confidence_i > confidence_o:
       clf_score = s_i
   else:
       clf_score = s_o
   ```

---

## Summary

| Mode | Scores | Variance | Use Case |
|------|--------|----------|----------|
| `input` | Prompt before LLM | Zero (by design) | Cost savings, pre-filtering |
| `output` | Response after LLM | Some (proves generalization) | Research, generalization claims |
| `both` | Both (max) | Some (less than output) | Production, defense in depth |

**Key Insight:**
- Original bug: Used `input` mode, claimed generalization
- Fix: Added modes, can now test true model-agnostic behavior
- For research: Use `output` or `both` mode

---

## References

- `run_cross_model_validation.py` - Main validation script
- `test_classifier_modes.py` - Quick comparison test
- `debug_zero_variance.py` - Original bug discovery script
- `CROSS_MODEL_FINAL_SUMMARY.md` - Results from original (buggy) implementation
