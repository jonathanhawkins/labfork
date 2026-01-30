#!/bin/bash
# AI Research Lab - macOS Setup Script
# Supports Intel and Apple Silicon Macs

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_success() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[X]${NC} $1"; }
print_info() { echo -e "${BLUE}[*]${NC} $1"; }
print_step() { echo -e "\n${BLUE}>>> $1${NC}"; }

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is for macOS only"
    exit 1
fi

ARCH=$(uname -m)
print_info "Detected architecture: $ARCH"

# Step 1: Check/Install Homebrew
print_step "Checking Homebrew..."
if ! command -v brew &> /dev/null; then
    print_info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add to PATH for Apple Silicon
    if [[ "$ARCH" == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    print_success "Homebrew is installed"
fi

# Step 2: Install Docker
print_step "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_info "Installing Docker Desktop..."
    brew install --cask docker
    print_warning "Please open Docker Desktop to complete installation"
    print_info "Waiting for Docker to start..."
    open -a Docker

    # Wait for Docker to be ready
    while ! docker info &> /dev/null; do
        sleep 2
    done
else
    if docker info &> /dev/null; then
        print_success "Docker is running"
    else
        print_warning "Docker is installed but not running"
        print_info "Starting Docker Desktop..."
        open -a Docker

        while ! docker info &> /dev/null; do
            sleep 2
        done
        print_success "Docker is now running"
    fi
fi

# Step 3: Install Node.js
print_step "Checking Node.js..."
if ! command -v node &> /dev/null; then
    print_info "Installing Node.js..."
    brew install node@20
    echo 'export PATH="/opt/homebrew/opt/node@20/bin:$PATH"' >> ~/.zprofile
    export PATH="/opt/homebrew/opt/node@20/bin:$PATH"
else
    NODE_VERSION=$(node -v)
    print_success "Node.js $NODE_VERSION is installed"
fi

# Step 4: Install Python
print_step "Checking Python..."
if ! command -v python3 &> /dev/null; then
    print_info "Installing Python 3.11..."
    brew install python@3.11
else
    PYTHON_VERSION=$(python3 --version)
    print_success "$PYTHON_VERSION is installed"
fi

# Step 5: Install Ollama
print_step "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    print_info "Installing Ollama..."
    brew install ollama
else
    print_success "Ollama is installed"
fi

# Start Ollama service
print_step "Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    brew services start ollama || ollama serve &
    sleep 3
fi
print_success "Ollama service is running"

# Step 6: Pull default model
print_step "Pulling Ollama model..."
MODEL="${OLLAMA_MODEL:-qwen3-coder:30b}"
if ! ollama list | grep -q "$MODEL"; then
    print_info "Pulling $MODEL (this may take a while)..."
    ollama pull "$MODEL"
else
    print_success "$MODEL is already pulled"
fi

# Step 7: Create environment file
print_step "Setting up environment..."
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    print_success "Created .env file from template"
    print_warning "Please edit .env to add your API keys"
else
    print_success ".env file already exists"
fi

# Step 8: Install frontend dependencies
print_step "Installing frontend dependencies..."
cd frontend
npm install
cd ..
print_success "Frontend dependencies installed"

# Step 9: Install backend dependencies
print_step "Installing backend dependencies..."
cd backend
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..
print_success "Backend dependencies installed"

# Step 10: Initialize database (if using local postgres)
print_step "Database setup..."
if command -v psql &> /dev/null; then
    print_info "PostgreSQL is available locally"
else
    print_info "PostgreSQL will run in Docker"
fi

# Final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start the lab:"
echo ""
echo "  Option 1: Docker (Recommended)"
echo "    docker compose up -d"
echo ""
echo "  Option 2: Local Development"
echo "    # Terminal 1 - Frontend"
echo "    cd frontend && npm run dev"
echo ""
echo "    # Terminal 2 - Backend"
echo "    cd backend && source venv/bin/activate && python main.py"
echo ""
echo "  Option 3: Quick Start"
echo "    ./scripts/start-local.sh"
echo ""
echo "Lab will be available at: http://localhost:3003"
echo ""

# Check for Apple Silicon optimizations
if [[ "$ARCH" == "arm64" ]]; then
    echo -e "${BLUE}Apple Silicon Detected:${NC}"
    echo "  - PyTorch will use MPS (Metal Performance Shaders) for GPU acceleration"
    echo "  - Ollama will use GPU by default"
    echo ""
fi
