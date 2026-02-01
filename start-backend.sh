#!/bin/bash
# Start LabFork backend with remote GPU connection

cd "$(dirname "$0")/backend"

# Load environment variables
if [ -f ../.env ]; then
  export $(grep -v '^#' ../.env | grep -v '^$' | grep '=' | xargs)
fi

# Ensure remote GPU is set
export REMOTE_GPU_HOST="${REMOTE_GPU_HOST:-100.83.78.111}"
export REMOTE_GPU_USER="${REMOTE_GPU_USER:-doc}"

echo "Starting backend with GPU: $REMOTE_GPU_USER@$REMOTE_GPU_HOST"

# Activate venv and run
source venv/bin/activate
python main.py --port 8003
