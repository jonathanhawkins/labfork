# Voice Clone Implementation Agent Prompt

You are working on the Voice Clone Research project. Your task is to implement a technique from a research paper or improve our prosody/emotion conditioning pipeline.

## Task Details
**Task ID**: {{taskId}}
**Subject**: {{taskSubject}}
**Description**: {{taskDescription}}

## Project Structure
- `training/` - Training scripts and configs (PyTorch)
- `inference/` - Generation scripts
- `backend/` - FastAPI prosody analysis
- `frontend/` - Next.js visualization and control
- `evaluation/` - Evaluation scripts (quick_eval.py)

## Implementation Guidelines

### 1. Read First
- Check `.skills/research-manager/MISSION.md` for current priorities
- Review existing implementations in `training/` for patterns
- Look at similar techniques already implemented

### 2. Code Standards
- Use PyTorch for training/inference code
- Follow existing config patterns in `training/config/`
- Create inference script in `inference/generate_with_<technique>.py`
- Use type hints and docstrings

### 3. Configuration
- Create YAML config in `training/config/<technique>.yaml`
- Include hyperparameters, paths, model settings
- Document all options in comments

### 4. Testing
- Test training runs for at least 10 steps
- Verify inference produces audio output
- Run `python evaluation/quick_eval.py` for metrics

### 5. Documentation
- Add docstring at top of training script
- Update CLAUDE.md if adding new commands
- Comment complex prosody/audio processing

## Research Output Policy
If this task produces experimental results:
1. Create run record: `python scripts/research/run_registry.py new --title "{{taskSubject}}" --task-id {{taskId}}`
2. Run evaluation: `python evaluation/quick_eval.py --checkpoint <ckpt> --run-dir <run_dir>`
3. Update run: `python scripts/research/run_registry.py update --run-dir <run_dir> --metrics-file <run_dir>/metrics.json`

## When Done
1. Mark task as completed: `TaskUpdate {{taskId}} status=completed`
2. Check TaskList for more pending tasks
3. Summarize what was implemented and any issues found
