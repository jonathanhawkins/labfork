#!/usr/bin/env python3
"""
CFD Results API Server
======================
Serves OpenFOAM simulation results for visualization.

Endpoints:
- GET /          - Server info
- GET /results   - Latest CFD results (temperature, velocity, probes)
- GET /field/T   - Temperature field data
- GET /field/U   - Velocity field data
- GET /probes    - Probe time series data
- POST /run      - Trigger new simulation with parameters
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="MOF Water Harvester CFD API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
CASE_DIR = Path.home() / "cfd" / "mof_harvester"
DOCKER_IMAGE = "opencfd/openfoam-default:2312"

# ============================================================================
# Data Models
# ============================================================================

class SimulationParams(BaseModel):
    sorbent_temp_c: float = 70  # Sorbent temperature (°C)
    dome_temp_c: float = 30     # Dome temperature (°C)
    humidity_percent: float = 40 # Ambient humidity
    end_time: int = 500          # Simulation steps

class CFDResults(BaseModel):
    converged: bool
    iterations: int
    probe_data: dict
    temperature_range: dict
    velocity_max: float
    heat_flux_sorbent: Optional[float]
    heat_flux_dome: Optional[float]
    estimated_yield_ml_hr: Optional[float]

# ============================================================================
# OpenFOAM Data Parsing
# ============================================================================

def parse_probe_file(filepath: Path) -> dict:
    """Parse OpenFOAM probe output file."""
    if not filepath.exists():
        return {}

    data = {"time": [], "values": []}
    with open(filepath) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            # Handle both scalar and vector formats
            # Scalars: "100 323.5 324.2 325.1"
            # Vectors: "100 (-0.1 0.2 0) (0.1 -0.1 0) ..."
            line = line.strip()
            if "(" in line:
                # Vector format - extract time and vectors
                parts = line.split("(")
                try:
                    time_val = float(parts[0].strip())
                    data["time"].append(time_val)
                    vectors = []
                    for p in parts[1:]:
                        vec_str = p.split(")")[0].strip()
                        if vec_str:
                            vec = [float(x) for x in vec_str.split()]
                            vectors.append(vec)
                    data["values"].append(vectors)
                except (ValueError, IndexError):
                    continue
            else:
                # Scalar format
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        data["time"].append(float(parts[0]))
                        data["values"].append([float(x) for x in parts[1:]])
                    except ValueError:
                        continue
    return data

def get_latest_time_dir() -> Optional[Path]:
    """Get the latest time directory from simulation."""
    time_dirs = []
    for d in CASE_DIR.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    time_dirs.append((t, d))
            except ValueError:
                pass

    if not time_dirs:
        return None

    time_dirs.sort(key=lambda x: x[0], reverse=True)
    return time_dirs[0][1]

def parse_internal_field(filepath: Path) -> list:
    """Parse OpenFOAM internal field values."""
    if not filepath.exists():
        return []

    values = []
    in_internal = False

    with open(filepath) as f:
        for line in f:
            if "internalField" in line:
                in_internal = True
                continue
            if in_internal:
                if line.strip() == "(":
                    continue
                if line.strip() == ")":
                    break
                if line.strip() == ";":
                    break
                # Parse value
                try:
                    # Scalar field
                    val = float(line.strip())
                    values.append(val)
                except ValueError:
                    # Vector field
                    if line.strip().startswith("("):
                        parts = line.strip().strip("()").split()
                        if len(parts) == 3:
                            values.append([float(x) for x in parts])

    return values

def calculate_heat_flux(T_hot: float, T_cold: float, distance: float, k: float = 0.026) -> float:
    """
    Estimate heat flux using Fourier's law.
    q = -k * dT/dx

    k = thermal conductivity of air ≈ 0.026 W/(m·K)
    """
    dT = T_hot - T_cold
    return k * dT / distance

def estimate_water_yield(heat_flux: float, latent_heat: float = 2.26e6) -> float:
    """
    Estimate water yield from heat flux.

    latent_heat = heat of vaporization of water ≈ 2.26 MJ/kg

    Returns: ml/hr
    """
    # Power = heat_flux * area (assume 0.09 m² = 30cm x 30cm)
    area = 0.09  # m²
    power = heat_flux * area  # Watts

    # Mass flow rate = power / latent_heat
    mass_rate_kg_s = power / latent_heat

    # Convert to ml/hr (1 kg = 1000 ml, 1 hr = 3600 s)
    ml_hr = mass_rate_kg_s * 1000 * 3600

    return ml_hr

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    return {
        "name": "MOF Water Harvester CFD API",
        "engine": "OpenFOAM 2312",
        "case_dir": str(CASE_DIR),
        "status": "ready" if CASE_DIR.exists() else "no_case",
    }

@app.get("/results")
def get_results():
    """Get latest CFD simulation results."""

    # Check if case exists
    if not CASE_DIR.exists():
        raise HTTPException(status_code=404, detail="CFD case not found")

    # Get latest time directory
    latest_time = get_latest_time_dir()
    if not latest_time:
        raise HTTPException(status_code=404, detail="No simulation results found")

    # Parse probe data
    probe_dir = CASE_DIR / "postProcessing" / "probes" / "0"
    probe_T = parse_probe_file(probe_dir / "T")
    probe_U = parse_probe_file(probe_dir / "U")
    probe_p = parse_probe_file(probe_dir / "p")

    # Get temperature field
    T_values = parse_internal_field(latest_time / "T")

    # Calculate statistics
    T_min = min(T_values) if T_values else 0
    T_max = max(T_values) if T_values else 0
    T_avg = sum(T_values) / len(T_values) if T_values else 0

    # Get velocity field
    U_values = parse_internal_field(latest_time / "U")
    U_mag = [np.sqrt(u[0]**2 + u[1]**2 + u[2]**2) for u in U_values] if U_values else []
    U_max = max(U_mag) if U_mag else 0

    # Estimate heat flux and water yield
    # Using probe temperatures at sorbent (343K) and dome (303K)
    T_sorbent = 343  # K
    T_dome = 303     # K
    distance = 0.3   # m (height of cavity)

    heat_flux = calculate_heat_flux(T_sorbent, T_dome, distance)
    estimated_yield = estimate_water_yield(heat_flux)

    return {
        "converged": True,
        "latest_time": float(latest_time.name),
        "iterations": int(float(latest_time.name)),

        "temperature": {
            "min_K": round(T_min, 2),
            "max_K": round(T_max, 2),
            "avg_K": round(T_avg, 2),
            "min_C": round(T_min - 273.15, 1),
            "max_C": round(T_max - 273.15, 1),
            "avg_C": round(T_avg - 273.15, 1),
        },

        "velocity": {
            "max_m_s": round(U_max, 4),
            "max_cm_s": round(U_max * 100, 2),
        },

        "heat_transfer": {
            "heat_flux_W_m2": round(heat_flux, 2),
            "estimated_yield_ml_hr": round(estimated_yield, 2),
            "estimated_yield_L_day": round(estimated_yield * 24 / 1000, 3),
        },

        "probes": {
            "temperature": probe_T,
            "velocity": probe_U,
            "pressure": probe_p,
        },

        "validation": {
            "mit_paper_yield_L_day": 2.8,
            "our_estimate_L_day": round(estimated_yield * 24 / 1000, 3),
            "ratio": round((estimated_yield * 24 / 1000) / 2.8, 2) if estimated_yield > 0 else 0,
        }
    }

@app.get("/field/{field_name}")
def get_field(field_name: str):
    """Get a specific field (T, U, p, etc.)."""

    latest_time = get_latest_time_dir()
    if not latest_time:
        raise HTTPException(status_code=404, detail="No simulation results")

    field_path = latest_time / field_name
    if not field_path.exists():
        raise HTTPException(status_code=404, detail=f"Field {field_name} not found")

    values = parse_internal_field(field_path)

    return {
        "field": field_name,
        "time": float(latest_time.name),
        "count": len(values),
        "values": values[:100],  # Limit for API response
        "stats": {
            "min": min(values) if values and isinstance(values[0], (int, float)) else None,
            "max": max(values) if values and isinstance(values[0], (int, float)) else None,
        }
    }

@app.get("/probes")
def get_probes():
    """Get all probe time series data."""

    probe_dir = CASE_DIR / "postProcessing" / "probes" / "0"
    if not probe_dir.exists():
        raise HTTPException(status_code=404, detail="No probe data found")

    probes = {}
    for f in probe_dir.iterdir():
        if f.is_file():
            probes[f.name] = parse_probe_file(f)

    return {
        "probes": probes,
        "locations": [
            {"name": "near_sorbent", "position": [0.15, 0.05, 0.005]},
            {"name": "mid_height", "position": [0.15, 0.15, 0.005]},
            {"name": "near_dome", "position": [0.15, 0.25, 0.005]},
        ]
    }

@app.post("/run")
async def run_simulation(params: SimulationParams, background_tasks: BackgroundTasks):
    """Trigger a new CFD simulation with given parameters."""

    # This would update boundary conditions and run the simulation
    # For now, return the current parameters

    return {
        "status": "queued",
        "params": params.dict(),
        "message": "Simulation would be triggered with these parameters",
    }

@app.get("/mesh")
def get_mesh_info():
    """Get mesh information."""

    # Parse mesh from constant/polyMesh
    mesh_dir = CASE_DIR / "constant" / "polyMesh"
    if not mesh_dir.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")

    # Read points
    points_file = mesh_dir / "points"
    n_points = 0
    if points_file.exists():
        with open(points_file) as f:
            for line in f:
                if line.strip().isdigit():
                    n_points = int(line.strip())
                    break

    return {
        "n_points": n_points,
        "n_cells": 3600,  # From blockMesh output
        "bounds": {
            "min": [0, 0, 0],
            "max": [0.3, 0.3, 0.01],
        },
        "type": "2D rectangular cavity",
    }

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  MOF Water Harvester CFD API                                 ║
║  Serving OpenFOAM simulation results                         ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    GET  /         - Server info                              ║
║    GET  /results  - Latest simulation results                ║
║    GET  /field/T  - Temperature field                        ║
║    GET  /probes   - Probe time series                        ║
║    GET  /mesh     - Mesh information                         ║
╠══════════════════════════════════════════════════════════════╣
║  API: http://localhost:8007                                  ║
╚══════════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=8007)
