#!/bin/bash
#
# Start the 4090 Compute Agent
#
# This script starts the compute agent that connects to Cloudflare Workers
# and executes inference tasks on the local RTX 4090 GPU.
#
# Usage:
#   ./start.sh              # Connect to production Workers
#   ./start.sh --local      # Connect to local Workers (localhost:8787)
#
# Environment variables:
#   WORKERS_BASE_URL - Production Workers URL
#   WORKERS_LOCAL_URL - Local Workers URL (default: http://localhost:8787)
#   DEVICE_NAME - Name for this device (default: RTX-4090-Primary)
#   DEFAULT_MODEL - Model to preload (default: Qwen/Qwen2.5-7B-Instruct)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default to local development
LOCAL_MODE="--local"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --production|--prod)
            LOCAL_MODE=""
            shift
            ;;
        --local)
            LOCAL_MODE="--local"
            shift
            ;;
        --workers-url)
            WORKERS_URL="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Activate venv if it exists
if [ -d "../../venv" ]; then
    source ../../venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "================================================"
echo "  4090 Compute Agent"
echo "================================================"
echo ""
echo "Starting agent..."
if [ -n "$LOCAL_MODE" ]; then
    echo "Mode: LOCAL (connecting to localhost:8787)"
else
    echo "Mode: PRODUCTION"
fi
echo ""

# Run the agent
if [ -n "$WORKERS_URL" ]; then
    python agent.py $LOCAL_MODE --workers-url "$WORKERS_URL"
else
    python agent.py $LOCAL_MODE
fi
