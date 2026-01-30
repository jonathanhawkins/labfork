#!/bin/bash
# AI Research Lab - Linux Setup Script
# Supports Ubuntu, Debian, and RHEL-based distributions

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

# Detect package manager
detect_package_manager() {
    if command -v apt-get &> /dev/null; then
        echo "apt"
    elif command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v yum &> /dev/null; then
        echo "yum"
    elif command -v pacman &> /dev/null; then
        echo "pacman"
    else
        echo "unknown"
    fi
}

# Check for GPU
detect_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "nvidia"
            return
        fi
    fi
    echo "none"
}

PKG_MANAGER=$(detect_package_manager)
GPU=$(detect_gpu)
ARCH=$(uname -m)

print_info "Detected package manager: $PKG_MANAGER"
print_info "Detected architecture: $ARCH"
print_info "Detected GPU: $GPU"

# Step 1: Update system
print_step "Updating system packages..."
case $PKG_MANAGER in
    apt)
        sudo apt-get update
        ;;
    dnf)
        sudo dnf check-update || true
        ;;
    yum)
        sudo yum check-update || true
        ;;
    pacman)
        sudo pacman -Sy
        ;;
esac
print_success "System updated"

# Step 2: Install base dependencies
print_step "Installing base dependencies..."
case $PKG_MANAGER in
    apt)
        sudo apt-get install -y curl git wget build-essential
        ;;
    dnf|yum)
        sudo $PKG_MANAGER install -y curl git wget gcc gcc-c++ make
        ;;
    pacman)
        sudo pacman -S --noconfirm curl git wget base-devel
        ;;
esac
print_success "Base dependencies installed"

# Step 3: Install Docker
print_step "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_info "Installing Docker..."

    case $PKG_MANAGER in
        apt)
            # Remove old versions
            sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

            # Add Docker's official GPG key
            sudo apt-get install -y ca-certificates curl gnupg
            sudo install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            sudo chmod a+r /etc/apt/keyrings/docker.gpg

            # Add the repository
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
              $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
              sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

            # Install Docker
            sudo apt-get update
            sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            ;;
        dnf)
            sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
            sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            ;;
        *)
            print_error "Please install Docker manually for your distribution"
            print_info "See: https://docs.docker.com/engine/install/"
            exit 1
            ;;
    esac

    # Start Docker
    sudo systemctl start docker
    sudo systemctl enable docker

    # Add user to docker group
    sudo usermod -aG docker $USER
    print_warning "You may need to log out and back in for Docker group to take effect"
else
    if docker info &> /dev/null; then
        print_success "Docker is running"
    else
        print_info "Starting Docker..."
        sudo systemctl start docker
        print_success "Docker started"
    fi
fi

# Step 4: Install NVIDIA Container Toolkit (if GPU detected)
if [[ "$GPU" == "nvidia" ]]; then
    print_step "Setting up NVIDIA Container Toolkit..."

    if ! command -v nvidia-container-cli &> /dev/null; then
        case $PKG_MANAGER in
            apt)
                distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
                    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                sudo apt-get update
                sudo apt-get install -y nvidia-container-toolkit
                ;;
            dnf)
                curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
                sudo dnf install -y nvidia-container-toolkit
                ;;
        esac

        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi
    print_success "NVIDIA Container Toolkit configured"
fi

# Step 5: Install Node.js
print_step "Checking Node.js..."
if ! command -v node &> /dev/null; then
    print_info "Installing Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    case $PKG_MANAGER in
        apt)
            sudo apt-get install -y nodejs
            ;;
        dnf|yum)
            sudo $PKG_MANAGER install -y nodejs
            ;;
        pacman)
            sudo pacman -S --noconfirm nodejs npm
            ;;
    esac
else
    NODE_VERSION=$(node -v)
    print_success "Node.js $NODE_VERSION is installed"
fi

# Step 6: Install Python
print_step "Checking Python..."
if ! command -v python3 &> /dev/null; then
    print_info "Installing Python..."
    case $PKG_MANAGER in
        apt)
            sudo apt-get install -y python3 python3-pip python3-venv
            ;;
        dnf|yum)
            sudo $PKG_MANAGER install -y python3 python3-pip
            ;;
        pacman)
            sudo pacman -S --noconfirm python python-pip
            ;;
    esac
else
    PYTHON_VERSION=$(python3 --version)
    print_success "$PYTHON_VERSION is installed"
fi

# Step 7: Install Ollama
print_step "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    print_info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    print_success "Ollama is installed"
fi

# Start Ollama service
print_step "Starting Ollama service..."
sudo systemctl start ollama 2>/dev/null || ollama serve &
sleep 3
print_success "Ollama service is running"

# Step 8: Pull default model
print_step "Pulling Ollama model..."
MODEL="${OLLAMA_MODEL:-qwen3-coder:30b}"
if ! ollama list | grep -q "$MODEL"; then
    print_info "Pulling $MODEL (this may take a while)..."
    ollama pull "$MODEL"
else
    print_success "$MODEL is already pulled"
fi

# Step 9: Create environment file
print_step "Setting up environment..."
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    print_success "Created .env file from template"
    print_warning "Please edit .env to add your API keys"
else
    print_success ".env file already exists"
fi

# Step 10: Install frontend dependencies
print_step "Installing frontend dependencies..."
cd frontend
npm install
cd ..
print_success "Frontend dependencies installed"

# Step 11: Install backend dependencies
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

# Final summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To start the lab:"
echo ""

if [[ "$GPU" == "nvidia" ]]; then
    echo "  With GPU (Recommended):"
    echo "    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d"
    echo ""
fi

echo "  Without GPU:"
echo "    docker compose up -d"
echo ""
echo "  Local Development:"
echo "    ./scripts/start-local.sh"
echo ""
echo "Lab will be available at: http://localhost:3003"
echo ""

if [[ "$GPU" == "nvidia" ]]; then
    echo -e "${BLUE}NVIDIA GPU Detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi
