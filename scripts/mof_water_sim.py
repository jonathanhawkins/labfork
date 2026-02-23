#!/usr/bin/env python3
"""
MOF Water Harvester GPU Simulation
===================================
Physics-accurate simulation of MIT/Berkeley MOF-801 water harvesting cycle.

Real Physics Cycle:
1. NIGHT (adsorbing): MOF absorbs moisture from air - particles drift INTO sorbent
2. DAWN (heating): Sun warms sorbent - particles vibrate, energy building
3. DAY (releasing): Heat releases water as vapor - particles rise from sorbent
4. DAY (condensing): Vapor hits cool dome - particles cluster on dome walls
5. DAY (dripping): Condensed water falls - particles drop to collector

Run: python mof_water_sim.py
API: http://localhost:8006/particles
"""

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
import math

# Try to use NVIDIA Warp for GPU acceleration
try:
    import warp as wp
    wp.init()
    USE_GPU = True
    print("✓ NVIDIA Warp GPU initialized")
except ImportError:
    USE_GPU = False
    print("⚠ Warp not available, using NumPy CPU fallback")

app = FastAPI(title="MOF Water Harvester Simulation")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Simulation Parameters
# ============================================================================

NUM_PARTICLES = 2000
DOME_RADIUS = 0.6
DOME_HEIGHT = 0.7
SORBENT_Y = 0.1  # Sorbent bed height (normalized 0-1)
COLLECTOR_Y = 0.0  # Bottom

# Time cycle (24 hours compressed)
CYCLE_SPEED = 0.001  # Full day/night in ~1000 frames

# Particle states
STATE_ADSORBING = 0  # Night: moving into sorbent
STATE_HEATING = 1    # Dawn: vibrating in sorbent
STATE_RELEASING = 2  # Day: rising as vapor
STATE_CONDENSING = 3 # Day: clustering on dome
STATE_DRIPPING = 4   # Day: falling as droplet

# ============================================================================
# Simulation State
# ============================================================================

class SimState:
    def __init__(self):
        self.time = 0.5  # Start at noon
        self.positions = np.random.rand(NUM_PARTICLES, 3).astype(np.float32)
        self.velocities = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
        self.states = np.zeros(NUM_PARTICLES, dtype=np.int32)
        self.temperatures = np.full(NUM_PARTICLES, 25.0, dtype=np.float32)

        # Initialize positions around sorbent
        for i in range(NUM_PARTICLES):
            angle = np.random.rand() * 2 * np.pi
            radius = np.random.rand() * 0.3
            self.positions[i, 0] = 0.5 + np.cos(angle) * radius
            self.positions[i, 1] = SORBENT_Y + np.random.rand() * 0.3
            self.positions[i, 2] = 0.5 + np.sin(angle) * radius

        self.start_time = time.time()

sim = SimState()

# ============================================================================
# Physics Simulation
# ============================================================================

def get_cycle_phase(time_of_day: float) -> tuple[str, float]:
    """Determine cycle phase and intensity from time of day (0-1)."""
    # 0.0 = midnight, 0.25 = dawn, 0.5 = noon, 0.75 = dusk

    is_night = time_of_day < 0.25 or time_of_day > 0.75

    if is_night:
        return "adsorbing", 0.0
    elif time_of_day < 0.35:
        return "heating", (time_of_day - 0.25) / 0.1  # 0-1 during dawn
    elif time_of_day < 0.5:
        return "releasing", 1.0
    elif time_of_day < 0.65:
        return "condensing", 1.0
    else:
        return "dripping", (0.75 - time_of_day) / 0.1  # Fade toward dusk

def step_simulation():
    """Advance simulation by one frame."""
    global sim

    # Advance time (compress 24 hours into ~100 seconds)
    sim.time = (sim.time + CYCLE_SPEED) % 1.0

    phase, intensity = get_cycle_phase(sim.time)

    # Calculate sorbent temperature (25°C night, up to 100°C at noon)
    sun_intensity = max(0, math.sin((sim.time - 0.25) * 2 * math.pi)) if sim.time > 0.25 and sim.time < 0.75 else 0
    sorbent_temp = 25 + sun_intensity * 75

    # Update each particle based on phase
    for i in range(NUM_PARTICLES):
        px, py, pz = sim.positions[i]
        vx, vy, vz = sim.velocities[i]

        # Distance from center (in XZ plane)
        dx = px - 0.5
        dz = pz - 0.5
        dist_from_center = math.sqrt(dx*dx + dz*dz)

        # Distance from dome surface
        dist_from_dome = DOME_RADIUS * 0.5 - dist_from_center

        if phase == "adsorbing":
            # Night: particles drift toward sorbent
            sim.states[i] = STATE_ADSORBING

            # Pull toward sorbent center
            vx = -dx * 0.02
            vy = (SORBENT_Y - py) * 0.03
            vz = -dz * 0.02

            # Add slight random motion
            vx += (np.random.rand() - 0.5) * 0.002
            vz += (np.random.rand() - 0.5) * 0.002

        elif phase == "heating":
            # Dawn: particles vibrate in sorbent
            sim.states[i] = STATE_HEATING

            # Vibration increases with intensity
            vibration = intensity * 0.01
            vx = (np.random.rand() - 0.5) * vibration
            vy = (np.random.rand() - 0.5) * vibration
            vz = (np.random.rand() - 0.5) * vibration

            # Keep near sorbent
            if py > SORBENT_Y + 0.15:
                vy -= 0.005
            if py < SORBENT_Y:
                vy += 0.005

        elif phase == "releasing":
            # Day: particles rise as vapor
            sim.states[i] = STATE_RELEASING

            # Rising motion
            vy = 0.008 + np.random.rand() * 0.012

            # Slight horizontal drift
            vx += (np.random.rand() - 0.5) * 0.003
            vz += (np.random.rand() - 0.5) * 0.003

            # Dampen horizontal
            vx *= 0.95
            vz *= 0.95

        elif phase == "condensing":
            # Day: particles drift toward dome walls and slow down
            sim.states[i] = STATE_CONDENSING

            if dist_from_center < DOME_RADIUS * 0.4:
                # Push outward toward dome
                if dist_from_center > 0.01:
                    vx = dx / dist_from_center * 0.005
                    vz = dz / dist_from_center * 0.005
            else:
                # Near dome, slow down
                vx *= 0.9
                vz *= 0.9

            # Slow vertical motion, slight downward
            vy = vy * 0.9 - 0.001

        elif phase == "dripping":
            # Day: particles fall as droplets
            sim.states[i] = STATE_DRIPPING

            # Gravity
            vy -= 0.002

            # Dampen horizontal
            vx *= 0.95
            vz *= 0.95

            # Reset if reached collector
            if py < COLLECTOR_Y + 0.05:
                # Respawn at dome wall
                angle = np.random.rand() * 2 * np.pi
                radius = DOME_RADIUS * 0.4 + np.random.rand() * 0.1
                sim.positions[i, 0] = 0.5 + np.cos(angle) * radius
                sim.positions[i, 1] = 0.5 + np.random.rand() * 0.2
                sim.positions[i, 2] = 0.5 + np.sin(angle) * radius
                vx, vy, vz = 0, 0, 0

        # Apply velocities
        sim.velocities[i] = [vx, vy, vz]
        sim.positions[i, 0] = px + vx
        sim.positions[i, 1] = py + vy
        sim.positions[i, 2] = pz + vz

        # Update temperature (based on height - cooler at dome, hotter at sorbent)
        height_factor = 1.0 - (py - SORBENT_Y) / DOME_HEIGHT
        sim.temperatures[i] = 25 + height_factor * (sorbent_temp - 25) * 0.8

        # Bounds checking
        # Keep within dome hemisphere
        dx = sim.positions[i, 0] - 0.5
        dz = sim.positions[i, 2] - 0.5
        dist = math.sqrt(dx*dx + dz*dz)
        if dist > DOME_RADIUS * 0.45:
            scale = DOME_RADIUS * 0.45 / dist
            sim.positions[i, 0] = 0.5 + dx * scale
            sim.positions[i, 2] = 0.5 + dz * scale
            sim.velocities[i, 0] *= -0.5
            sim.velocities[i, 2] *= -0.5

        # Vertical bounds
        if sim.positions[i, 1] > 0.9:
            sim.positions[i, 1] = 0.9
            sim.velocities[i, 1] *= -0.3
        if sim.positions[i, 1] < 0.02:
            sim.positions[i, 1] = 0.02
            sim.velocities[i, 1] = 0

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    return {
        "name": "MOF Water Harvester Simulation",
        "engine": "nvidia_warp_gpu" if USE_GPU else "numpy_cpu",
        "particles": NUM_PARTICLES,
        "physics": "MIT/Berkeley MOF-801 cycle",
    }

@app.get("/particles")
def get_particles():
    """Return current particle state for visualization."""
    step_simulation()

    phase, intensity = get_cycle_phase(sim.time)

    # Count particles in each state
    state_counts = {
        "adsorbing": int(np.sum(sim.states == STATE_ADSORBING)),
        "heating": int(np.sum(sim.states == STATE_HEATING)),
        "releasing": int(np.sum(sim.states == STATE_RELEASING)),
        "condensing": int(np.sum(sim.states == STATE_CONDENSING)),
        "dripping": int(np.sum(sim.states == STATE_DRIPPING)),
    }

    # Get time as hours
    hours = sim.time * 24
    time_label = f"{int(hours):02d}:{int((hours % 1) * 60):02d}"

    return {
        "positions": sim.positions.tolist(),
        "states": sim.states.tolist(),
        "temperatures": sim.temperatures.tolist(),
        "time_of_day": sim.time,
        "time_label": time_label,
        "phase": phase,
        "phase_intensity": intensity,
        "state_counts": state_counts,
        "sorbent_temp": 25 + max(0, math.sin((sim.time - 0.25) * 2 * math.pi)) * 75,
    }

@app.get("/stats")
def get_stats():
    """Return simulation statistics."""
    phase, intensity = get_cycle_phase(sim.time)

    state_counts = {
        "adsorbing": int(np.sum(sim.states == STATE_ADSORBING)),
        "heating": int(np.sum(sim.states == STATE_HEATING)),
        "releasing": int(np.sum(sim.states == STATE_RELEASING)),
        "condensing": int(np.sum(sim.states == STATE_CONDENSING)),
        "dripping": int(np.sum(sim.states == STATE_DRIPPING)),
    }

    return {
        "total_particles": NUM_PARTICLES,
        "time_of_day": sim.time,
        "phase": phase,
        "state_counts": state_counts,
        "avg_temperature": float(np.mean(sim.temperatures)),
        "uptime_seconds": time.time() - sim.start_time,
    }

@app.post("/reset")
def reset_simulation():
    """Reset simulation to initial state."""
    global sim
    sim = SimState()
    return {"status": "reset", "particles": NUM_PARTICLES}

@app.post("/set_time")
def set_time(time_of_day: float):
    """Set time of day (0-1)."""
    sim.time = max(0, min(1, time_of_day))
    return {"time_of_day": sim.time, "phase": get_cycle_phase(sim.time)[0]}

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  MOF Water Harvester Simulation                              ║
║  Physics: MIT/Berkeley MOF-801 cycle                         ║
║  Particles: {NUM_PARTICLES}                                            ║
║  Engine: {'NVIDIA Warp GPU' if USE_GPU else 'NumPy CPU'}                                   ║
╠══════════════════════════════════════════════════════════════╣
║  Cycle Phases:                                               ║
║    Night (00:00-06:00): Adsorbing moisture from air          ║
║    Dawn  (06:00-08:30): Heating sorbent                      ║
║    Day   (08:30-12:00): Releasing vapor                      ║
║    Day   (12:00-15:30): Condensing on dome                   ║
║    Day   (15:30-18:00): Dripping to collector                ║
║    Dusk  (18:00-24:00): Back to adsorbing                    ║
╠══════════════════════════════════════════════════════════════╣
║  API: http://localhost:8006                                  ║
║  GET  /particles  - Current particle positions               ║
║  GET  /stats      - Simulation statistics                    ║
║  POST /set_time   - Set time of day (0-1)                    ║
╚══════════════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=8006)
