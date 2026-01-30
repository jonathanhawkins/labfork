#!/bin/bash
# AI Research Lab - Quick Setup Script
# Detects your OS and runs the appropriate setup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  AI Research Lab - Quick Setup${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Detect architecture
detect_arch() {
    local arch=$(uname -m)
    case $arch in
        x86_64)  echo "amd64" ;;
        aarch64) echo "arm64" ;;
        arm64)   echo "arm64" ;;
        *)       echo "unknown" ;;
    esac
}

# Check if Docker is installed and running
check_docker() {
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            return 0
        else
            print_warning "Docker is installed but not running"
            return 1
        fi
    else
        return 1
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

    # Check for Apple Silicon GPU
    if [[ "$(detect_os)" == "macos" ]] && [[ "$(detect_arch)" == "arm64" ]]; then
        echo "apple-silicon"
        return
    fi

    echo "none"
}

# Main setup flow
main() {
    print_header

    local os=$(detect_os)
    local arch=$(detect_arch)
    local gpu=$(detect_gpu)

    print_info "Detected OS: $os"
    print_info "Detected Architecture: $arch"
    print_info "Detected GPU: $gpu"
    echo ""

    # Check for required files
    if [[ ! -f "docker-compose.yml" ]]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi

    # Route to appropriate setup script
    case $os in
        macos)
            print_info "Running macOS setup..."
            ./deploy/setup-mac.sh
            ;;
        linux)
            print_info "Running Linux setup..."
            ./deploy/setup-linux.sh
            ;;
        windows)
            print_error "Please run deploy/setup-windows.ps1 from PowerShell"
            print_info "Example: powershell -ExecutionPolicy Bypass -File deploy/setup-windows.ps1"
            exit 1
            ;;
        *)
            print_error "Unsupported OS: $OSTYPE"
            print_info "Please see DEPLOYMENT.md for manual setup instructions"
            exit 1
            ;;
    esac
}

main "$@"
