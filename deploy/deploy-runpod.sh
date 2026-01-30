#!/bin/bash
# AI Research Lab - RunPod Deployment Script
# Deploys GPU compute workloads to RunPod

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

# Check for required environment variables
check_requirements() {
    if [[ -z "$RUNPOD_API_KEY" ]]; then
        print_error "RUNPOD_API_KEY environment variable is required"
        print_info "Get your API key from: https://www.runpod.io/console/user/settings"
        exit 1
    fi

    if ! command -v curl &> /dev/null; then
        print_error "curl is required"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        print_warning "jq is recommended for JSON parsing"
    fi
}

# List available GPU types
list_gpu_types() {
    print_step "Available GPU Types"

    curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' \
        -d '{"query":"query { gpuTypes { id displayName memoryInGb secureCloud communityCloud lowestPrice { minimumBidPrice uninterruptablePrice } } }"}' \
        | jq -r '.data.gpuTypes[] | "\(.id)\t\(.displayName)\t\(.memoryInGb)GB\t$\(.lowestPrice.uninterruptablePrice)/hr"' \
        | column -t -s $'\t'
}

# Create a pod
create_pod() {
    local gpu_type="${1:-NVIDIA RTX 4090}"
    local gpu_count="${2:-1}"
    local container_image="${3:-runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04}"
    local volume_size="${4:-50}"

    print_step "Creating RunPod Instance"
    print_info "GPU: $gpu_type x$gpu_count"
    print_info "Image: $container_image"
    print_info "Volume: ${volume_size}GB"

    local response=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' \
        -d "{
            \"query\": \"mutation { podFindAndDeployOnDemand(input: { cloudType: SECURE, gpuCount: $gpu_count, volumeInGb: $volume_size, containerDiskInGb: 20, minVcpuCount: 4, minMemoryInGb: 16, gpuTypeId: \\\"$gpu_type\\\", name: \\\"ai-research-lab\\\", imageName: \\\"$container_image\\\", dockerArgs: \\\"\\\", ports: \\\"8003/http,11434/http,3003/http\\\", volumeMountPath: \\\"/workspace\\\" }) { id imageName machine { podHostId } } }\"
        }")

    if command -v jq &> /dev/null; then
        local pod_id=$(echo "$response" | jq -r '.data.podFindAndDeployOnDemand.id')
        if [[ "$pod_id" != "null" ]] && [[ -n "$pod_id" ]]; then
            print_success "Pod created: $pod_id"
            echo "$pod_id"
            return 0
        else
            print_error "Failed to create pod"
            echo "$response" | jq .
            return 1
        fi
    else
        echo "$response"
    fi
}

# Get pod status
get_pod_status() {
    local pod_id="$1"

    curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"query { pod(input: { podId: \\\"$pod_id\\\" }) { id name runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } gpus { id gpuUtilPercent memoryUtilPercent } } } }\"}" \
        | jq .
}

# Wait for pod to be ready
wait_for_pod() {
    local pod_id="$1"
    local timeout="${2:-300}"
    local elapsed=0

    print_step "Waiting for pod to be ready..."

    while [[ $elapsed -lt $timeout ]]; do
        local status=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
            -H 'Content-Type: application/json' \
            -d "{\"query\":\"query { pod(input: { podId: \\\"$pod_id\\\" }) { id desiredStatus runtime { uptimeInSeconds } } }\"}" \
            | jq -r '.data.pod.runtime.uptimeInSeconds // "null"')

        if [[ "$status" != "null" ]] && [[ "$status" -gt 0 ]]; then
            print_success "Pod is ready!"
            return 0
        fi

        sleep 10
        elapsed=$((elapsed + 10))
        echo -n "."
    done

    print_error "Timeout waiting for pod"
    return 1
}

# Setup lab on pod
setup_lab_on_pod() {
    local pod_id="$1"

    print_step "Setting up AI Research Lab on pod..."

    # Get SSH connection info
    local ssh_info=$(curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"query { pod(input: { podId: \\\"$pod_id\\\" }) { runtime { ports { ip isIpPublic privatePort publicPort type } } } }\"}")

    if command -v jq &> /dev/null; then
        local ssh_ip=$(echo "$ssh_info" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .ip')
        local ssh_port=$(echo "$ssh_info" | jq -r '.data.pod.runtime.ports[] | select(.privatePort == 22) | .publicPort')

        if [[ -n "$ssh_ip" ]] && [[ -n "$ssh_port" ]]; then
            print_info "SSH: ssh root@$ssh_ip -p $ssh_port"
            print_info "Connecting and setting up lab..."

            ssh -o StrictHostKeyChecking=no -p "$ssh_port" "root@$ssh_ip" << 'ENDSSH'
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 5

# Clone repository
cd /workspace
git clone https://github.com/your-repo/voice-clone-pipeline.git lab
cd lab

# Setup environment
cp .env.example .env

# Pull model
ollama pull qwen3-coder:30b

# Start services
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

echo "Lab is ready!"
ENDSSH
        fi
    fi
}

# Terminate pod
terminate_pod() {
    local pod_id="$1"

    print_step "Terminating pod $pod_id..."

    curl -s "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\"$pod_id\\\" }) }\"}" \
        | jq .

    print_success "Pod terminated"
}

# Usage
usage() {
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  list-gpus              List available GPU types"
    echo "  create [gpu] [count]   Create a new pod"
    echo "  status <pod_id>        Get pod status"
    echo "  setup <pod_id>         Setup lab on pod"
    echo "  terminate <pod_id>     Terminate a pod"
    echo ""
    echo "Examples:"
    echo "  $0 create 'NVIDIA RTX 4090' 1"
    echo "  $0 status abc123"
    echo ""
}

# Main
main() {
    check_requirements

    case "$1" in
        list-gpus)
            list_gpu_types
            ;;
        create)
            create_pod "${2:-NVIDIA RTX 4090}" "${3:-1}"
            ;;
        status)
            if [[ -z "$2" ]]; then
                print_error "Pod ID required"
                exit 1
            fi
            get_pod_status "$2"
            ;;
        setup)
            if [[ -z "$2" ]]; then
                print_error "Pod ID required"
                exit 1
            fi
            wait_for_pod "$2"
            setup_lab_on_pod "$2"
            ;;
        terminate)
            if [[ -z "$2" ]]; then
                print_error "Pod ID required"
                exit 1
            fi
            terminate_pod "$2"
            ;;
        *)
            usage
            ;;
    esac
}

main "$@"
