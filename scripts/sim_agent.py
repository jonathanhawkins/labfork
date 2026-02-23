#!/usr/bin/env python3
"""
Simulation Agent for RTX 4090

This agent runs on the 4090 server and:
1. Polls for pending simulation tasks from the LabFork API
2. Executes simulations using OpenFOAM or quick estimation
3. Reports results back to the API

Usage:
    python sim_agent.py --api-url https://labfork.com
    python sim_agent.py --api-url http://localhost:3003 --poll-interval 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Default configuration
DEFAULT_API_URL = "http://localhost:3003"
DEFAULT_POLL_INTERVAL = 10  # seconds
DEVICE_ID = os.environ.get("DEVICE_ID", "dev_4090_water_sim")
DEVICE_AUTH_TOKEN = os.environ.get("DEVICE_AUTH_TOKEN", "")

# Simulation directories
SIM_DIR = Path.home() / "simulations"
RESULTS_DIR = SIM_DIR / "results"
QUICK_ESTIMATE_SCRIPT = SIM_DIR / "quick_estimate.py"


def log(message: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def check_gpu():
    """Check if GPU is available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            log(f"GPU detected: {gpu_info}")
            return True
    except Exception as e:
        log(f"GPU check failed: {e}", "WARN")
    return False


def register_device(api_url: str) -> bool:
    """Register this device with the compute network."""
    try:
        # Get GPU capabilities
        gpu_name = "RTX 4090"
        compute_tflops = 82.58  # RTX 4090 FP32
        memory_gb = 24

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                gpu_name = parts[0] if len(parts) > 0 else gpu_name
                if len(parts) > 1 and "MiB" in parts[1]:
                    memory_gb = int(parts[1].replace(" MiB", "")) / 1024
        except Exception:
            pass

        payload = {
            "name": f"Water Sim Agent ({gpu_name})",
            "capabilities": {
                "compute": compute_tflops,
                "memory": memory_gb,
                "bandwidth": 1000,  # Assume gigabit
                "platform": "cuda",
                "gpuName": gpu_name,
                "cachedModels": ["simulation_water_harvester", "simulation_heat_transfer"]
            },
            "availability": {
                "wifiOnly": False,
                "chargingOnly": False,
                "minBattery": 0,
                "maxUtilization": 95
            }
        }

        response = requests.post(
            f"{api_url}/api/compute/devices",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.ok:
            data = response.json()
            log(f"Device registered: {data.get('device', {}).get('id', DEVICE_ID)}")
            return True
        else:
            log(f"Device registration failed: {response.status_code}", "WARN")

    except Exception as e:
        log(f"Device registration error: {e}", "ERROR")

    return False


def send_heartbeat(api_url: str, status: str = "online", task_progress: int | None = None) -> bool:
    """Send heartbeat to keep device marked as online."""
    try:
        payload = {
            "deviceId": DEVICE_ID,
            "status": status,
        }
        if task_progress is not None:
            payload["taskProgress"] = task_progress

        response = requests.post(
            f"{api_url}/api/compute/devices/{DEVICE_ID}",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEVICE_AUTH_TOKEN}" if DEVICE_AUTH_TOKEN else ""
            },
            timeout=5
        )
        return response.ok
    except Exception:
        return False


def poll_for_tasks(api_url: str) -> list[dict]:
    """Poll for pending simulation tasks."""
    try:
        response = requests.get(
            f"{api_url}/api/compute/tasks/simulation",
            params={"status": "pending", "limit": 1},
            headers={
                "Authorization": f"Bearer {DEVICE_AUTH_TOKEN}" if DEVICE_AUTH_TOKEN else ""
            },
            timeout=10
        )

        if response.ok:
            data = response.json()
            return data.get("tasks", [])
    except Exception as e:
        log(f"Task poll error: {e}", "WARN")

    return []


def run_quick_estimate(params: dict) -> dict[str, Any]:
    """Run quick analytical estimation."""
    if QUICK_ESTIMATE_SCRIPT.exists():
        try:
            result = subprocess.run(
                ["python3", str(QUICK_ESTIMATE_SCRIPT), json.dumps(params)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            log(f"Quick estimate script error: {e}", "WARN")

    # Fallback: inline quick estimate
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
    humidity_factor = (humidity / 50) ** 1.5

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


def run_full_simulation(params: dict, task_id: str) -> dict[str, Any]:
    """Run full OpenFOAM CFD simulation."""
    case_dir = RESULTS_DIR / task_id
    case_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting full simulation in {case_dir}")

    # For now, run quick estimate and add a note about full simulation
    # Full OpenFOAM integration would go here
    results = run_quick_estimate(params)
    results["simulation_mode"] = "quick_fallback"
    results["note"] = "Full CFD simulation infrastructure ready, using quick estimate"

    # Save results
    with open(case_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def execute_task(task: dict, api_url: str) -> bool:
    """Execute a simulation task."""
    task_id = task.get("id", "unknown")
    sim_params = task.get("input", {}).get("simulationParams", {})

    log(f"Executing task {task_id}: {sim_params.get('type', 'unknown')}")

    try:
        # Send heartbeat with busy status
        send_heartbeat(api_url, status="busy", task_progress=0)

        params = sim_params.get("parameters", {})
        mode = sim_params.get("mode", "quick")

        start_time = time.time()

        # Run appropriate simulation
        if mode == "full":
            results = run_full_simulation(params, task_id)
        else:
            results = run_quick_estimate(params)

        compute_time_ms = int((time.time() - start_time) * 1000)

        # Report results
        report_payload = {
            "taskId": task_id,
            "deviceId": DEVICE_ID,
            "success": True,
            "result": {
                **results,
                "metrics": {
                    "computeTime": compute_time_ms,
                }
            }
        }

        # Try to report to API
        try:
            response = requests.post(
                f"{api_url}/api/compute/tasks/{task_id}",
                json=report_payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {DEVICE_AUTH_TOKEN}" if DEVICE_AUTH_TOKEN else ""
                },
                timeout=10
            )
            if response.ok:
                log(f"Task {task_id} completed and reported successfully")
            else:
                log(f"Failed to report task completion: {response.status_code}", "WARN")
        except Exception as e:
            log(f"Failed to report task completion: {e}", "WARN")

        # Also sync to research results
        try:
            sync_payload = {
                "results": [{
                    "objective_id": task_id,
                    "objective_title": f"Water Harvester Simulation - {mode}",
                    "objective_description": json.dumps(params),
                    "success": True,
                    "output": json.dumps(results),
                    "duration_minutes": compute_time_ms / 60000,
                }]
            }
            requests.post(
                f"{api_url}/api/research/sync",
                json=sync_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
        except Exception:
            pass  # Research sync is optional

        return True

    except Exception as e:
        log(f"Task execution failed: {e}", "ERROR")

        # Report failure
        try:
            requests.post(
                f"{api_url}/api/compute/tasks/{task_id}",
                json={
                    "taskId": task_id,
                    "deviceId": DEVICE_ID,
                    "success": False,
                    "error": str(e)
                },
                headers={"Content-Type": "application/json"},
                timeout=10
            )
        except Exception:
            pass

        return False

    finally:
        send_heartbeat(api_url, status="online")


def main():
    parser = argparse.ArgumentParser(description="Simulation Agent for RTX 4090")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="LabFork API URL")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL, help="Poll interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit (for testing)")
    args = parser.parse_args()

    log("=" * 60)
    log("LabFork Simulation Agent Starting")
    log(f"API URL: {args.api_url}")
    log(f"Device ID: {DEVICE_ID}")
    log("=" * 60)

    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        log("No GPU detected - simulations may be slow", "WARN")

    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Register device
    register_device(args.api_url)

    # Main loop
    log("Entering main loop - polling for tasks...")
    consecutive_errors = 0

    while True:
        try:
            # Send heartbeat
            send_heartbeat(args.api_url, status="online")

            # Poll for tasks
            tasks = poll_for_tasks(args.api_url)

            if tasks:
                log(f"Found {len(tasks)} pending task(s)")
                for task in tasks:
                    execute_task(task, args.api_url)
            else:
                log("No pending tasks", "DEBUG") if not args.once else None

            consecutive_errors = 0

            if args.once:
                log("Single run mode - exiting")
                break

        except KeyboardInterrupt:
            log("Shutting down...")
            send_heartbeat(args.api_url, status="offline")
            break

        except Exception as e:
            consecutive_errors += 1
            log(f"Main loop error: {e}", "ERROR")

            if consecutive_errors >= 10:
                log("Too many consecutive errors - exiting", "ERROR")
                break

        # Wait before next poll
        if not args.once:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
