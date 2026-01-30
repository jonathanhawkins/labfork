# Research Decision Framework

## Core Goal (Never Forget)

**Research Question:** Can explicit multi-layer prosody labels improve voice cloning when training data is limited?

**Success Metric:** Happy/Sad F0 separation + Emotion classification accuracy

## Decision Gates

Every technique must pass these gates before implementation:

### Gate 1: Relevance (5 seconds)
- Does it help prosody/emotion control? YES → continue, NO → reject
- Is it for our use case (limited data voice cloning)? YES → continue, NO → reject

### Gate 2: Testability (1 minute)
- Can we evaluate it in < 4 hours? YES → continue, NO → simplify or reject
- Do we have the data/compute? YES → continue, NO → defer

### Gate 3: Expected Impact (5 minutes)
- What metric does it improve? (F0 correlation, emotion accuracy, speaker similarity)
- Expected improvement: >10% → high priority, 5-10% → medium, <5% → low/reject

## Quick Eval Pipeline

**30-minute evaluation for any technique:**

```bash
# 1. Generate test samples (5 emotions × 3 samples = 15 files)
python inference/quick_eval.py --checkpoint <new_model> --output eval_results/

# 2. Auto-score
# - F0 separation: Happy > Sad by 30+ Hz? PASS/FAIL
# - Emotion accuracy: Qwen2-Audio classification ≥ 50%? PASS/FAIL
# - Speaker similarity: ECAPA-TDNN ≥ 0.7? PASS/FAIL

# 3. Compare to baseline
python evaluation/compare_to_baseline.py --new eval_results/ --baseline baseline_results/
```

## Research Budget (Hard Limits)

| Limit | Value | Reason |
|-------|-------|--------|
| New techniques per week | 3 | Focus over breadth |
| Eval deadline | 48 hours | No zombie implementations |
| Max research tasks | 5 pending | Clear the queue first |
| CLAUDE.md additions | 0 | Document elsewhere |

## Priority Categories

### P0 - Critical Path (Do Now)
- V7 LoRA training fix
- Quick eval pipeline
- Baseline comparison

### P1 - High Impact (This Week)
- Techniques with >10% expected improvement on core metrics
- Things that fix known failures (angry/neutral emotion accuracy)

### P2 - Medium Impact (Backlog)
- Nice-to-have improvements
- Alternative approaches to solved problems

### P3 - Research Only (Don't Implement)
- Interesting but not relevant to our goal
- Requires resources we don't have
- Already have a working solution

## Decision Log Template

When evaluating a technique:

```
Technique: [Name]
Paper: [ArXiv link]
Gate 1 (Relevance): PASS/FAIL - [reason]
Gate 2 (Testability): PASS/FAIL - [reason]
Gate 3 (Impact): HIGH/MED/LOW - [expected improvement]
Decision: IMPLEMENT / DEFER / REJECT
Owner: [who will test it]
Eval Deadline: [date]
```

## Current Technique Status

### Working (Keep)
- Prosody encoder with 4-layer conditioning (semantic, acoustic, rhythm, contour)
- Energy predictor auxiliary loss
- LoRA fine-tuning for limited data

### Partially Working (Fix)
- Emotion accuracy for angry/neutral (currently 0%)

### Researched But Not Evaluated
- 40+ techniques documented in CLAUDE.md
- Need quick eval to decide keep/reject

### Explicitly Rejected
(Add techniques here with reason)

## Weekly Review Checklist

1. [ ] How many techniques were tested this week?
2. [ ] What improved vs baseline?
3. [ ] What should we stop working on?
4. [ ] Is V7 verified? If not, why?
5. [ ] Are we closer to the research goal?

## The Golden Rule

**If you can't measure the improvement in 4 hours, don't implement it.**
