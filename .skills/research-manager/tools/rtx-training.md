---
name: rtx-training
description: Remote training management on RTX 4090 via SSH
metadata:
  tags: training, gpu, rtx4090, remote, ssh, lora
---

# RTX 4090 Remote Training Management

Tools for managing training jobs on the Windows WSL RTX 4090 machine via Tailscale.

## Connection Details

- **Host**: `$REMOTE_GPU_USER@$REMOTE_GPU_HOST` (via Tailscale)
- **Project Path**: `~/dev/labfork`
- **Conda Environment**: `voice`
- **GPU**: RTX 4090 (24GB VRAM)

**Configure**: Set `REMOTE_GPU_HOST` and `REMOTE_GPU_USER` environment variables.

## Commands

### Check Status

View GPU utilization and running training jobs.

```bash
.skills/research-manager/rm rtx status
```

Output shows:
- Host connectivity (via Tailscale)
- GPU name, memory usage, utilization
- Running training processes

### Start Training

Launch a training job on the RTX 4090.

```bash
# Start LoRA training (default, recommended for small datasets)
.skills/research-manager/rm rtx train

# Use a different config
.skills/research-manager/rm rtx train --config rtx_4090_deepseek.yaml

# Run in foreground (blocks until complete)
.skills/research-manager/rm rtx train --foreground
```

### Training Configs

| Config | Use Case |
|--------|----------|
| `rtx_4090_lora.yaml` | LoRA training, <500 samples (RECOMMENDED) |
| `rtx_4090_deepseek.yaml` | Full fine-tuning, 500+ samples |

### Sync Files

Sync project files between local Mac and RTX 4090 machine.

```bash
# Push local changes to remote
.skills/research-manager/rm rtx sync

# Pull remote changes to local
.skills/research-manager/rm rtx sync --pull
```

**Note**: Automatically excludes node_modules, .next, venv, __pycache__, .git, checkpoints

### View Logs

Get recent training logs.

```bash
# Last 50 lines (default)
.skills/research-manager/rm rtx logs

# More lines
.skills/research-manager/rm rtx logs --lines 200
```

### Run Arbitrary Commands

Execute any command on the RTX 4090 machine.

```bash
# Check disk space
.skills/research-manager/rm rtx run df -h

# List checkpoints
.skills/research-manager/rm rtx run ls -la training/outputs

# Run nvidia-smi
.skills/research-manager/rm rtx run /usr/lib/wsl/lib/nvidia-smi

# Check conda environment
.skills/research-manager/rm rtx run conda list
```

## Training Workflow

```bash
# 1. Check RTX 4090 is available
.skills/research-manager/rm rtx status

# 2. Sync latest code
.skills/research-manager/rm rtx sync

# 3. Start training
.skills/research-manager/rm rtx train --config rtx_4090_lora.yaml

# 4. Monitor progress
.skills/research-manager/rm rtx logs

# 5. When complete, pull checkpoints
.skills/research-manager/rm rtx sync --pull
```

## Direct SSH Access

For more control, connect directly:

```bash
# Simple SSH
ssh $REMOTE_GPU_USER@$REMOTE_GPU_HOST

# Attach to training tmux session
ssh $REMOTE_GPU_USER@$REMOTE_GPU_HOST -t "tmux attach -t training"

# Start new tmux for monitoring
ssh $REMOTE_GPU_USER@$REMOTE_GPU_HOST -t "tmux new -s monitor"
```

## Troubleshooting

### Host Unreachable
- Check Tailscale is connected: `tailscale status`
- Verify connection: `tailscale ping $REMOTE_GPU_HOST`

### Training Fails to Start
- Check GPU memory: `.skills/research-manager/rm rtx run /usr/lib/wsl/lib/nvidia-smi`
- Ensure conda env exists: `.skills/research-manager/rm rtx run conda env list`

### Out of Memory
- Use LoRA config (3.4GB vs 22GB VRAM)
- Reduce batch size in config
- Check for other GPU processes

## Important Notes

1. **Always use LoRA for small datasets** (<500 samples) to prevent overfitting
2. **Sync before training** to ensure latest code is on remote
3. **Training runs in tmux** so it survives SSH disconnection
4. **Use /usr/lib/wsl/lib/nvidia-smi** not regular nvidia-smi in WSL
