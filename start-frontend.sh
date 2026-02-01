#!/bin/bash
# Start LabFork frontend

cd "$(dirname "$0")/frontend"

echo "Starting frontend on http://localhost:3003"
npm run dev -- -p 3003
