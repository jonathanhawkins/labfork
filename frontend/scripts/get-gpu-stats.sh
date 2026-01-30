#!/bin/bash
# Script to fetch GPU stats from remote training machine
ssh -T -o ConnectTimeout=5 -o BatchMode=yes doc@${REMOTE_GPU_HOST} /usr/lib/wsl/lib/nvidia-smi 2>&1
