#!/bin/bash
# Voice Clone Pipeline - Linux/CUDA Setup Script
# Optimized for RTX 4090 with 24GB VRAM

set -e

echo "============================================"
echo "Voice Clone Pipeline - Linux/CUDA Setup"
echo "============================================"
echo ""

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. CUDA may not be installed."
    echo "Visit https://developer.nvidia.com/cuda-downloads for installation."
fi

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip nodejs npm ffmpeg git build-essential

# Create directories
echo "Creating directories..."
mkdir -p data/{raw,processed,labeled,splits}
mkdir -p models/{csm-1b,whisper,qwen2-audio,checkpoints}

# Setup Python virtual environment
echo "Setting up Python environment..."
cd backend

if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install training dependencies
cd ../training
pip install -r requirements.txt 2>/dev/null || pip install transformers accelerate peft wandb pyyaml tqdm

cd ..

# Setup frontend
echo "Setting up frontend..."
cd frontend
npm install

cd ..

# Create .env file
echo "Creating configuration..."
cat > .env << EOF
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000

# Model paths
CSM_MODEL_PATH=./models/csm-1b
WHISPER_MODEL=large-v3
QWEN_MODEL=Qwen/Qwen2-Audio-7B-Instruct

# Hardware
DEVICE=cuda
EOF

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To download models (28GB total), run:"
echo "  python scripts/download_models.py"
echo ""
echo "To start the pipeline:"
echo "  1. Terminal 1: cd backend && source venv/bin/activate && python main.py"
echo "  2. Terminal 2: cd frontend && npm run dev"
echo "  3. Open http://localhost:3000"
echo ""
