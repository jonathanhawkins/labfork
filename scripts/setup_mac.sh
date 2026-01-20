#!/bin/bash
# Voice Clone Pipeline - Mac Setup Script
# Optimized for M4 Pro with 64GB RAM

set -e

echo "============================================"
echo "Voice Clone Pipeline - Mac Setup"
echo "============================================"
echo ""

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install system dependencies
echo "Installing system dependencies..."
brew install python@3.11 node ffmpeg git || true

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

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Install training dependencies
cd ../training
pip install -r requirements.txt 2>/dev/null || pip install torch torchaudio transformers accelerate peft wandb pyyaml tqdm

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
DEVICE=mps
EOF

# Download models (optional - can be large)
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
