#!/bin/bash
# AI Research Lab - Health Check Script
# Checks all services and reports their status

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3003}"
BACKEND_URL="${BACKEND_URL:-http://localhost:8003}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

# Status tracking
OVERALL_STATUS=0

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  AI Research Lab - Health Check${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

check_service() {
    local name="$1"
    local url="$2"
    local expected_code="${3:-200}"

    printf "%-20s" "$name"

    local start_time=$(date +%s%N)
    local response

    if response=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "$url" 2>/dev/null); then
        local end_time=$(date +%s%N)
        local latency=$(( (end_time - start_time) / 1000000 ))

        if [[ "$response" == "$expected_code" ]] || [[ "$response" == "200" ]]; then
            echo -e "${GREEN}UP${NC} (${latency}ms)"
            return 0
        else
            echo -e "${RED}DOWN${NC} (HTTP $response)"
            OVERALL_STATUS=1
            return 1
        fi
    else
        echo -e "${RED}DOWN${NC} (Connection failed)"
        OVERALL_STATUS=1
        return 1
    fi
}

check_port() {
    local name="$1"
    local host="$2"
    local port="$3"

    printf "%-20s" "$name"

    if nc -z -w 5 "$host" "$port" 2>/dev/null; then
        echo -e "${GREEN}UP${NC}"
        return 0
    else
        echo -e "${RED}DOWN${NC}"
        OVERALL_STATUS=1
        return 1
    fi
}

check_docker() {
    printf "%-20s" "Docker"

    if docker info &>/dev/null; then
        local running=$(docker compose ps --services --filter "status=running" 2>/dev/null | wc -l)
        local total=$(docker compose ps --services 2>/dev/null | wc -l)
        echo -e "${GREEN}UP${NC} ($running/$total containers)"
        return 0
    else
        echo -e "${RED}DOWN${NC}"
        OVERALL_STATUS=1
        return 1
    fi
}

check_gpu() {
    printf "%-20s" "GPU"

    if command -v nvidia-smi &>/dev/null; then
        if nvidia-smi &>/dev/null; then
            local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
            echo -e "${GREEN}AVAILABLE${NC} ($gpu_name, $gpu_util)"
            return 0
        fi
    fi

    # Check for Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]] && [[ "$(uname)" == "Darwin" ]]; then
        echo -e "${GREEN}AVAILABLE${NC} (Apple Silicon MPS)"
        return 0
    fi

    echo -e "${YELLOW}NOT DETECTED${NC}"
    return 0
}

check_ollama_models() {
    printf "%-20s" "Ollama Models"

    if command -v ollama &>/dev/null; then
        local models=$(ollama list 2>/dev/null | tail -n +2 | wc -l)
        if [[ "$models" -gt 0 ]]; then
            echo -e "${GREEN}$models models${NC}"
            return 0
        else
            echo -e "${YELLOW}No models${NC}"
            return 0
        fi
    else
        echo -e "${RED}Ollama not installed${NC}"
        return 1
    fi
}

check_disk_space() {
    printf "%-20s" "Disk Space"

    local free_space
    if [[ "$(uname)" == "Darwin" ]]; then
        free_space=$(df -h . | tail -1 | awk '{print $4}')
    else
        free_space=$(df -h . | tail -1 | awk '{print $4}')
    fi

    echo -e "${GREEN}$free_space free${NC}"
}

check_memory() {
    printf "%-20s" "Memory"

    if [[ "$(uname)" == "Darwin" ]]; then
        local total=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)"GB"}')
        echo -e "${GREEN}$total total${NC}"
    else
        local free=$(free -h | grep Mem | awk '{print $7}')
        local total=$(free -h | grep Mem | awk '{print $2}')
        echo -e "${GREEN}$free/$total available${NC}"
    fi
}

main() {
    print_header

    echo "System Resources:"
    echo "-----------------"
    check_memory
    check_disk_space
    check_gpu
    echo ""

    echo "Services:"
    echo "---------"
    check_docker
    check_service "Frontend" "$FRONTEND_URL/api/health"
    check_service "Backend" "$BACKEND_URL/health"
    check_service "Ollama" "$OLLAMA_URL/api/tags"
    check_ollama_models
    echo ""

    echo "Database:"
    echo "---------"
    check_port "PostgreSQL" "$DB_HOST" "$DB_PORT"
    echo ""

    # Summary
    echo "========================================="
    if [[ $OVERALL_STATUS -eq 0 ]]; then
        echo -e "Overall Status: ${GREEN}HEALTHY${NC}"
    else
        echo -e "Overall Status: ${RED}DEGRADED${NC}"
    fi
    echo "========================================="
    echo ""

    exit $OVERALL_STATUS
}

main "$@"
