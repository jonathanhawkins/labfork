# Voice Clone Evaluation Agent Prompt

You are evaluating prosody and emotion conditioning techniques for the Voice Clone Research project.

## Task Details
**Task ID**: {{taskId}}
**Subject**: {{taskSubject}}
**Checkpoint/Model**: {{checkpointPath}}

## Evaluation Metrics

### Primary Metrics (Required)
1. **MOS (Mean Opinion Score)**: 1-5 naturalness rating
   - Use `evaluation/quick_eval.py --mos` for automated estimation

2. **F0 RMSE**: Pitch contour accuracy
   - Lower is better (target: < 20 Hz)
   - Measures how well the model reproduces target prosody

3. **Emotion Accuracy**: Classification of generated emotion
   - Use emotion2vec or wav2vec for classification
   - Report confusion matrix for multi-emotion models

### Secondary Metrics (If Applicable)
4. **PESQ**: Perceptual audio quality
5. **STOI**: Intelligibility score
6. **Speaker Similarity**: Cosine similarity of speaker embeddings

## Evaluation Protocol

### 1. Test Set Preparation
- Use held-out test split from `data/splits/test/`
- Ensure diverse emotions: happy, sad, angry, neutral, surprised
- Include both seen and unseen speakers

### 2. Generate Samples
```bash
python inference/generate_with_<technique>.py \
  --checkpoint {{checkpointPath}} \
  --test-set data/splits/test \
  --output-dir evaluation/outputs/<technique>
```

### 3. Run Metrics
```bash
python evaluation/quick_eval.py \
  --generated evaluation/outputs/<technique> \
  --reference data/splits/test \
  --metrics mos,f0_rmse,emotion_acc
```

### 4. Compare to Baseline
- Load baseline results from `evaluation/baselines/v7_baseline.json`
- Report improvement/regression for each metric
- Note statistical significance

### 5. Qualitative Analysis
- Listen to 5-10 samples for each emotion
- Note any artifacts, unnatural prosody, or emotion mismatch
- Identify failure cases

## Report Format
```markdown
## Evaluation Results: {{taskSubject}}

### Quantitative Metrics
| Metric | Baseline | This Model | Change |
|--------|----------|------------|--------|
| MOS    | 3.5      | 3.7        | +0.2   |
| F0 RMSE| 18.5 Hz  | 15.2 Hz    | -3.3   |
| Emotion Acc | 72%  | 78%       | +6%    |

### Qualitative Notes
- Strength: [what works well]
- Weakness: [what needs improvement]
- Recommendation: [ship, iterate, or abandon]

### Sample Links
- Happy: [link]
- Sad: [link]
```

## When Done
1. Save report to `evaluation/reports/<technique>_eval.md`
2. Update run registry: `python scripts/research/run_registry.py update --run-dir <run_dir> --status evaluated`
3. Mark task completed and summarize findings
