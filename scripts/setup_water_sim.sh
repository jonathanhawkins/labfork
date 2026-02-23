#!/bin/bash
# Setup script for Water Harvester Simulation on RTX 4090
# Run this on the 4090 server (doc@100.83.78.111)

set -e

echo "=== Water Harvester Simulation Setup ==="
echo ""

# Check if running on the right machine (WSL uses different nvidia-smi path)
NVIDIA_SMI="nvidia-smi"
if [ -f /usr/lib/wsl/lib/nvidia-smi ]; then
    NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
fi

if ! $NVIDIA_SMI &>/dev/null; then
    echo "Error: No NVIDIA GPU found. This script should run on the 4090 server."
    exit 1
fi

GPU_NAME=$($NVIDIA_SMI --query-gpu=name --format=csv,noheader)
echo "Detected GPU: $GPU_NAME"

# Create simulation directories
echo ""
echo "Creating simulation directories..."
mkdir -p ~/simulations/{water_harvester,templates,results}
mkdir -p ~/bin

# Install OpenFOAM if not present
if ! command -v blockMesh &>/dev/null; then
    echo ""
    echo "Installing OpenFOAM..."

    # Add OpenFOAM repository
    sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
    sudo add-apt-repository http://dl.openfoam.org/ubuntu
    sudo apt-get update

    # Install OpenFOAM v11
    sudo apt-get install -y openfoam11

    # Add to bashrc
    echo 'source /opt/openfoam11/etc/bashrc' >> ~/.bashrc
    source /opt/openfoam11/etc/bashrc

    echo "OpenFOAM installed successfully."
else
    echo "OpenFOAM already installed."
fi

# Create simulation template for water harvester
echo ""
echo "Creating simulation templates..."

cat > ~/simulations/templates/water_harvester_case/system/controlDict << 'EOF'
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}

application     interCondensatingEvaporatingFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         10;
deltaT          0.001;
writeControl    adjustableRunTime;
writeInterval   0.1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
adjustTimeStep  yes;
maxCo           0.5;
maxAlphaCo      0.5;
maxDeltaT       0.01;

functions
{
    fieldAverage
    {
        type            fieldAverage;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        fields
        (
            alpha.water
            {
                mean        on;
                prime2Mean  off;
                base        time;
            }
        );
    }
}
EOF

# Create quick estimation Python script
cat > ~/simulations/quick_estimate.py << 'EOF'
#!/usr/bin/env python3
"""
Quick analytical estimation for water harvester yield.
Based on research: Nature Communications 2024, Science 2017.
"""

import json
import sys
import math

def estimate(params):
    width = params.get('sorbent_width_cm', 30)
    depth = params.get('sorbent_depth_cm', 25)
    humidity = params.get('humidity_percent', 45)
    mirrors = params.get('mirror_count', 4)
    pattern = params.get('surface_pattern', 'flat')
    ambient_temp = params.get('temperature_ambient_c', 25)

    # Sorbent area in m²
    area = (width * depth) / 10000

    # Base collection rate from research (L/m²/day at 50% RH)
    base_rate = 3.5

    # Humidity factor (exponential relationship)
    humidity_factor = math.pow(humidity / 50, 1.5)

    # Mirror concentration factor
    mirror_factor = 1 + (mirrors * 0.15)

    # Surface pattern factor (beetle = 1.5x)
    surface_factor = 1.5 if pattern == 'beetle' else 1.0

    # Temperature delta estimation
    temp_delta = 20 + (mirrors * 10)
    temp_factor = 1 + (temp_delta / 100)

    # Calculate daily yield
    daily_yield = area * base_rate * humidity_factor * mirror_factor * surface_factor * temp_factor

    # Efficiency
    theoretical_max = area * humidity * 0.001 * 24
    efficiency = min(95, (daily_yield / max(theoretical_max, 0.1)) * 100)

    # Condensation rate
    condensation_rate = (daily_yield * 1000) / (area * 24)

    return {
        'collection_rate_ml_per_hour': round((daily_yield * 1000) / 24, 1),
        'daily_yield_liters': round(daily_yield, 2),
        'efficiency_percent': round(efficiency),
        'peak_temperature_c': ambient_temp + temp_delta,
        'condensation_rate_g_per_m2_hour': round(condensation_rate, 1),
    }

if __name__ == '__main__':
    if len(sys.argv) > 1:
        params = json.loads(sys.argv[1])
    else:
        params = {}

    result = estimate(params)
    print(json.dumps(result, indent=2))
EOF

chmod +x ~/simulations/quick_estimate.py

# Create simulation runner script
cat > ~/bin/run_water_sim << 'EOF'
#!/bin/bash
# Run water harvester simulation
# Usage: run_water_sim <case_name> <params_json>

CASE_NAME=$1
PARAMS=$2

if [ -z "$CASE_NAME" ]; then
    echo "Usage: run_water_sim <case_name> <params_json>"
    exit 1
fi

source /opt/openfoam11/etc/bashrc 2>/dev/null || source /opt/openfoam/etc/bashrc

CASE_DIR=~/simulations/water_harvester/$CASE_NAME
mkdir -p $CASE_DIR

# Copy template
cp -r ~/simulations/templates/water_harvester_case/* $CASE_DIR/ 2>/dev/null || true

cd $CASE_DIR

# Run quick estimate first
if [ -n "$PARAMS" ]; then
    echo "Quick estimate:"
    python3 ~/simulations/quick_estimate.py "$PARAMS"
fi

# For full simulation, we'd run:
# blockMesh
# setFields
# interCondensatingEvaporatingFoam

echo ""
echo "Simulation case prepared at: $CASE_DIR"
echo "Run 'blockMesh && interCondensatingEvaporatingFoam' to execute."
EOF

chmod +x ~/bin/run_water_sim

# Create status check script
cat > ~/bin/sim_status << 'EOF'
#!/bin/bash
# Check simulation infrastructure status

echo "=== Simulation Infrastructure Status ==="
echo ""

# GPU (WSL uses different path)
echo "GPU:"
NVIDIA_SMI="nvidia-smi"
[ -f /usr/lib/wsl/lib/nvidia-smi ] && NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"
$NVIDIA_SMI --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

# OpenFOAM
echo ""
echo "OpenFOAM:"
if command -v blockMesh &>/dev/null; then
    echo "  Installed: Yes"
    blockMesh -help 2>&1 | head -1
else
    echo "  Installed: No"
fi

# Simulation directories
echo ""
echo "Simulation directories:"
ls -la ~/simulations/ 2>/dev/null || echo "  Not found"

# Running simulations
echo ""
echo "Running simulations:"
ps aux | grep -E "Foam|openfoam" | grep -v grep || echo "  None"
EOF

chmod +x ~/bin/sim_status

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Available commands:"
echo "  sim_status         - Check simulation infrastructure"
echo "  run_water_sim      - Run water harvester simulation"
echo ""
echo "Quick test:"
echo "  python3 ~/simulations/quick_estimate.py '{\"humidity_percent\": 45, \"mirror_count\": 4}'"
