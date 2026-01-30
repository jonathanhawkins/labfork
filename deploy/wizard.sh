#!/bin/bash
# AI Research Lab - Interactive Setup Wizard
# Guides users through setting up their research lab

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# State
DEPLOYMENT_TYPE=""
DOMAIN=""
HARDWARE_TYPE=""
GPU_AVAILABLE=""
API_KEYS=()
LAB_NAME=""

# Utility functions
print_header() {
    clear
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}                                                            ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}     ${MAGENTA}🔬 AI Research Lab - Setup Wizard${NC}                       ${CYAN}║${NC}"
    echo -e "${CYAN}║${NC}                                                            ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() { echo -e "  ${GREEN}✓${NC} $1"; }
print_warning() { echo -e "  ${YELLOW}!${NC} $1"; }
print_error() { echo -e "  ${RED}✗${NC} $1"; }
print_info() { echo -e "  ${BLUE}→${NC} $1"; }

prompt() {
    local prompt="$1"
    local default="$2"
    local result

    if [[ -n "$default" ]]; then
        read -p "  $prompt [$default]: " result
        result="${result:-$default}"
    else
        read -p "  $prompt: " result
    fi

    echo "$result"
}

prompt_choice() {
    local prompt="$1"
    shift
    local options=("$@")
    local choice

    echo "  $prompt"
    echo ""

    for i in "${!options[@]}"; do
        echo "    $((i+1)). ${options[$i]}"
    done

    echo ""
    read -p "  Enter choice (1-${#options[@]}): " choice

    if [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 ]] && [[ "$choice" -le "${#options[@]}" ]]; then
        echo "${options[$((choice-1))]}"
    else
        echo ""
    fi
}

confirm() {
    local prompt="$1"
    local response

    read -p "  $prompt (y/n): " response
    [[ "$response" =~ ^[Yy] ]]
}

# Step 1: Welcome
step_welcome() {
    print_header

    echo "  Welcome to the AI Research Lab setup wizard!"
    echo ""
    echo "  This wizard will help you:"
    echo "    • Choose a deployment method"
    echo "    • Configure your hardware"
    echo "    • Set up API keys"
    echo "    • Create your first lab"
    echo ""
    echo "  Estimated time: 5-10 minutes"
    echo ""

    read -p "  Press Enter to continue..."
}

# Step 2: Deployment Type
step_deployment() {
    print_header
    print_step "Step 1: Choose Deployment Method"

    DEPLOYMENT_TYPE=$(prompt_choice "How would you like to deploy?" \
        "Docker (Local) - Recommended for development" \
        "Vercel + Remote GPU - Best for production" \
        "Local Development - No Docker required" \
        "Remote Server - Deploy to existing server")

    case "$DEPLOYMENT_TYPE" in
        *Docker*)
            DEPLOYMENT_TYPE="docker"
            print_success "Docker deployment selected"
            ;;
        *Vercel*)
            DEPLOYMENT_TYPE="vercel"
            print_success "Vercel deployment selected"
            ;;
        *Local*)
            DEPLOYMENT_TYPE="local"
            print_success "Local development selected"
            ;;
        *Remote*)
            DEPLOYMENT_TYPE="remote"
            print_success "Remote server deployment selected"
            ;;
        *)
            print_error "Invalid choice, defaulting to Docker"
            DEPLOYMENT_TYPE="docker"
            ;;
    esac
}

# Step 3: Domain Selection
step_domain() {
    print_header
    print_step "Step 2: Research Domain"

    echo "  What type of research will you be doing?"
    echo ""

    DOMAIN=$(prompt_choice "Select your primary research domain:" \
        "Voice Synthesis & Cloning" \
        "Speech Recognition" \
        "Natural Language Processing" \
        "Computer Vision" \
        "Reinforcement Learning" \
        "General Machine Learning")

    case "$DOMAIN" in
        *Voice*)
            DOMAIN="voice-synthesis"
            ;;
        *Speech*)
            DOMAIN="speech-recognition"
            ;;
        *Natural*)
            DOMAIN="nlp"
            ;;
        *Vision*)
            DOMAIN="computer-vision"
            ;;
        *Reinforcement*)
            DOMAIN="reinforcement-learning"
            ;;
        *)
            DOMAIN="machine-learning"
            ;;
    esac

    print_success "Domain set to: $DOMAIN"
}

# Step 4: Hardware Detection
step_hardware() {
    print_header
    print_step "Step 3: Hardware Configuration"

    echo "  Detecting your hardware..."
    echo ""

    # Detect OS
    local os="unknown"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        os="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        os="linux"
    fi
    print_info "Operating System: $os"

    # Detect architecture
    local arch=$(uname -m)
    print_info "Architecture: $arch"

    # Detect GPU
    GPU_AVAILABLE="none"

    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        local gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        GPU_AVAILABLE="nvidia"
        print_success "NVIDIA GPU: $gpu_name ($gpu_mem)"
    elif [[ "$arch" == "arm64" ]] && [[ "$os" == "macos" ]]; then
        GPU_AVAILABLE="apple-silicon"
        print_success "Apple Silicon GPU detected"
    else
        print_warning "No GPU detected - CPU-only mode"
    fi

    # Detect memory
    local memory
    if [[ "$os" == "macos" ]]; then
        memory=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    else
        memory=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
    fi
    print_info "System Memory: ${memory}GB"

    echo ""
    HARDWARE_TYPE="${GPU_AVAILABLE}-${memory}gb"

    # Recommendations
    echo "  Based on your hardware:"
    if [[ "$GPU_AVAILABLE" == "nvidia" ]]; then
        print_success "Full GPU acceleration available"
        print_info "Recommended model: qwen3-coder:30b"
    elif [[ "$GPU_AVAILABLE" == "apple-silicon" ]]; then
        print_success "MPS acceleration available"
        if [[ $memory -ge 48 ]]; then
            print_info "Recommended model: qwen3-coder:30b"
        else
            print_info "Recommended model: qwen3-coder:14b"
        fi
    else
        print_warning "Running on CPU - expect slower performance"
        print_info "Recommended model: qwen3-coder:7b"
    fi

    echo ""
    read -p "  Press Enter to continue..."
}

# Step 5: API Keys
step_api_keys() {
    print_header
    print_step "Step 4: API Keys (Optional)"

    echo "  API keys enhance functionality but aren't required."
    echo "  You can skip this step and add them later."
    echo ""

    if confirm "Would you like to configure API keys now?"; then
        echo ""

        # Anthropic
        local anthropic_key=$(prompt "Anthropic API Key (for Claude)" "")
        if [[ -n "$anthropic_key" ]]; then
            API_KEYS+=("ANTHROPIC_API_KEY=$anthropic_key")
            print_success "Anthropic API key set"
        fi

        # OpenAI
        local openai_key=$(prompt "OpenAI API Key (optional)" "")
        if [[ -n "$openai_key" ]]; then
            API_KEYS+=("OPENAI_API_KEY=$openai_key")
            print_success "OpenAI API key set"
        fi

        # Semantic Scholar
        local ss_key=$(prompt "Semantic Scholar API Key (optional)" "")
        if [[ -n "$ss_key" ]]; then
            API_KEYS+=("SEMANTIC_SCHOLAR_API_KEY=$ss_key")
            print_success "Semantic Scholar API key set"
        fi
    else
        print_info "Skipping API key configuration"
    fi
}

# Step 6: Lab Name
step_lab_name() {
    print_header
    print_step "Step 5: Create Your Lab"

    echo "  Let's create your first research lab!"
    echo ""

    LAB_NAME=$(prompt "Lab name" "My Research Lab")

    print_success "Lab name: $LAB_NAME"
}

# Step 7: Summary & Confirmation
step_summary() {
    print_header
    print_step "Step 6: Configuration Summary"

    echo "  Please review your configuration:"
    echo ""
    echo "    Deployment Method: $DEPLOYMENT_TYPE"
    echo "    Research Domain:   $DOMAIN"
    echo "    Hardware Type:     $HARDWARE_TYPE"
    echo "    GPU Available:     $GPU_AVAILABLE"
    echo "    Lab Name:          $LAB_NAME"
    echo "    API Keys:          ${#API_KEYS[@]} configured"
    echo ""

    if ! confirm "Proceed with installation?"; then
        print_warning "Installation cancelled"
        exit 0
    fi
}

# Step 8: Installation
step_install() {
    print_header
    print_step "Step 7: Installing"

    # Create .env file
    echo "  Creating environment file..."
    cp .env.example .env

    # Add API keys
    for key in "${API_KEYS[@]}"; do
        echo "$key" >> .env
    done

    # Set hardware-specific options
    if [[ "$GPU_AVAILABLE" == "nvidia" ]]; then
        echo "USE_GPU=true" >> .env
    fi

    print_success "Environment configured"

    # Run appropriate setup
    case "$DEPLOYMENT_TYPE" in
        docker)
            echo "  Starting Docker setup..."
            if [[ "$GPU_AVAILABLE" == "nvidia" ]]; then
                docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
            else
                docker compose up -d
            fi
            ;;
        local)
            echo "  Installing dependencies..."
            cd frontend && npm install && cd ..
            cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ..
            ;;
        vercel)
            echo "  Preparing Vercel deployment..."
            cd frontend && npm install && cd ..
            print_info "Run 'cd frontend && vercel' to deploy"
            ;;
        remote)
            print_info "Use deploy/deploy-runpod.sh or deploy/deploy-railway.sh"
            ;;
    esac

    print_success "Installation complete!"
}

# Step 9: Launch
step_launch() {
    print_header
    print_step "Step 8: Launch!"

    echo ""
    echo "  Your AI Research Lab is ready!"
    echo ""

    case "$DEPLOYMENT_TYPE" in
        docker)
            echo "  Lab is starting up..."
            sleep 5
            echo ""
            echo -e "  ${GREEN}Open your browser to:${NC}"
            echo -e "  ${CYAN}http://localhost:3003${NC}"
            ;;
        local)
            echo "  To start the lab, run these commands in separate terminals:"
            echo ""
            echo "    Terminal 1 (Frontend):"
            echo "      cd frontend && npm run dev"
            echo ""
            echo "    Terminal 2 (Backend):"
            echo "      cd backend && source venv/bin/activate && python main.py"
            echo ""
            echo -e "  Then open: ${CYAN}http://localhost:3003${NC}"
            ;;
        *)
            echo "  Follow the deployment instructions for your chosen method."
            ;;
    esac

    echo ""
    echo -e "  ${MAGENTA}Happy researching! 🔬${NC}"
    echo ""
}

# Main
main() {
    step_welcome
    step_deployment
    step_domain
    step_hardware
    step_api_keys
    step_lab_name
    step_summary
    step_install
    step_launch
}

main "$@"
