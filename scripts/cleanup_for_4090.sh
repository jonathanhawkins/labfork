#!/bin/bash
# Cleanup script for Mac - keep only essentials for local dev
# Training happens on RTX 4090, so we don't need experimental checkpoints

set -e

echo "🧹 Cleaning up Mac for 4090 remote training workflow..."
echo ""

# Show current disk usage
echo "📊 Current disk usage:"
du -h -d 1 . | sort -h | tail -10
echo ""

# Calculate what we'll free
CHECKPOINTS_SIZE=$(du -sh ./checkpoints 2>/dev/null | cut -f1)
OLD_MODELS_SIZE=$(du -sh ./models/checkpoints/voice_deepseek_v1 ./models/checkpoints/maskgct_prosody ./models/checkpoints/prosody_conditioned ./models/checkpoints/prosody_v5 ./models/checkpoints/prosody_v6 2>/dev/null | awk '{sum+=$1} END {print sum"M"}' || echo "0M")
SKILLS_SIZE=$(du -sh ./.skills/research-manager 2>/dev/null | cut -f1)

echo "🗑️  Will remove:"
echo "  - checkpoints/ (experimental models): ${CHECKPOINTS_SIZE}"
echo "  - Old training runs: ~18GB"
echo "  - .skills/research-manager cache: ${SKILLS_SIZE}"
echo "  - Total freed: ~74GB"
echo ""

read -p "Continue with cleanup? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

# Remove experimental research checkpoints (47GB)
echo "🗑️  Removing experimental checkpoints..."
rm -rf ./checkpoints

# Remove old training runs, keep only prosody_v7 (latest)
echo "🗑️  Removing old training checkpoints..."
cd models/checkpoints
rm -rf voice_deepseek_v1 maskgct_prosody prosody_conditioned prosody_v5 prosody_v6
cd ../..

# Remove research-manager cache
echo "🗑️  Removing research-manager cache..."
rm -rf ./.skills/research-manager

# Clean up inference outputs and temp files
echo "🗑️  Cleaning inference outputs..."
rm -f ./inference/*.wav 2>/dev/null || true
rm -f ./inference/*.json 2>/dev/null || true
rm -rf ./inference/outputs/* 2>/dev/null || true

# Clean up frontend build artifacts
echo "🗑️  Cleaning frontend build artifacts..."
rm -rf ./frontend/.next 2>/dev/null || true
rm -rf ./frontend/node_modules/.cache 2>/dev/null || true

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 New disk usage:"
du -h -d 1 . | sort -h | tail -10
echo ""
echo "🎯 Kept for local development:"
echo "  ✓ Whisper model (2.9GB)"
echo "  ✓ Qwen2-Audio (16GB)"
echo "  ✓ CSM-1B base model (18GB)"
echo "  ✓ Latest training checkpoint: prosody_v7 (105MB)"
echo ""
echo "🚀 Use /train-remote skill to train on RTX 4090"
