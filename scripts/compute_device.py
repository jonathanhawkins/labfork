#!/usr/bin/env python3
"""
Compute Device Client for RTX 4090

Registers with Cloudflare Workers API and polls for compute tasks.
Runs as a daemon to keep the 4090 connected to the LabFork compute network.

Usage:
    python compute_device.py --api-url https://labfork-agents.workers.dev
    python compute_device.py --api-url http://localhost:8787  # Local dev
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

# Configuration
CONFIG_FILE = Path.home() / ".labfork_device.json"
HEARTBEAT_INTERVAL = 30  # seconds
TASK_POLL_INTERVAL = 10  # seconds when idle


def get_ollama_models():
    """Get list of models available in Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            models = []
            for line in result.stdout.strip().split("\n")[1:]:  # Skip header
                if line.strip():
                    model_name = line.split()[0]
                    models.append(model_name.split(":")[0])  # Remove :latest tag
            return models if models else ["qwen3-coder-32k"]
    except Exception:
        pass
    return ["qwen3-coder-32k"]


def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        # Try WSL path first, then standard
        nvidia_smi = "/usr/lib/wsl/lib/nvidia-smi"
        if not os.path.exists(nvidia_smi):
            nvidia_smi = "nvidia-smi"

        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            name = parts[0] if len(parts) > 0 else "Unknown GPU"
            memory_mb = int(parts[1]) if len(parts) > 1 else 24000

            # Estimate TFLOPS for known GPUs
            tflops_map = {
                "RTX 4090": 82.6,
                "RTX 4080": 48.7,
                "RTX 3090": 35.6,
                "A100": 156.0,
                "H100": 267.0,
            }
            compute = 50.0  # default
            for gpu, tflops in tflops_map.items():
                if gpu in name:
                    compute = tflops
                    break

            return {
                "compute": compute,
                "memory": memory_mb / 1024,  # Convert to GB
                "models": get_ollama_models(),
                "gpuName": name,
            }
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")

    # Default RTX 4090 capabilities
    return {
        "compute": 82.6,
        "memory": 24,
        "models": get_ollama_models(),
        "gpuName": "RTX 4090",
    }


def load_config():
    """Load saved device configuration"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return None


def save_config(config):
    """Save device configuration"""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {CONFIG_FILE}")


def register_device(api_url: str):
    """Register this device with the compute network"""
    capabilities = get_gpu_info()
    try:
        hostname = subprocess.run(
            ["hostname"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or "unknown"
    except Exception:
        hostname = "unknown"

    # Use actual GPU name for device name
    gpu_short = capabilities["gpuName"].replace("NVIDIA GeForce ", "").replace(" ", "-")

    payload = {
        "name": f"{gpu_short}-{hostname}",
        "platform": "cuda",
        "capabilities": capabilities,
    }

    print(f"Registering device with {api_url}/api/compute/devices...")
    print(f"  Name: {payload['name']}")
    print(f"  Capabilities: {capabilities['compute']} TFLOPS, {capabilities['memory']}GB")

    try:
        resp = requests.post(
            f"{api_url}/api/compute/devices",
            json=payload,
            timeout=30
        )

        if resp.status_code == 200:
            data = resp.json()
            config = {
                "device_id": data["device"]["id"],
                "auth_token": data["authToken"],
                "api_url": api_url,
                "registered_at": datetime.now().isoformat(),
            }
            save_config(config)
            print(f"✓ Device registered: {data['device']['id']} (tier: {data['device']['tier']})")
            return config
        else:
            print(f"✗ Registration failed: {resp.status_code} - {resp.text}")
            return None
    except Exception as e:
        print(f"✗ Registration error: {e}")
        return None


def send_heartbeat(config: dict, status: str = "online"):
    """Send heartbeat to the compute network"""
    try:
        resp = requests.patch(
            f"{config['api_url']}/api/compute/devices/{config['device_id']}",
            json={"status": status},
            headers={"Authorization": f"Bearer {config['auth_token']}"},
            timeout=10
        )

        if resp.status_code == 200:
            data = resp.json()
            return data.get("task")  # Returns assigned task if any
        else:
            print(f"Heartbeat failed: {resp.status_code}")
            return None
    except Exception as e:
        print(f"Heartbeat error: {e}")
        return None


def poll_for_task(config: dict):
    """Poll for available work"""
    try:
        resp = requests.post(
            f"{config['api_url']}/api/compute/tasks/assign",
            json={"deviceId": config["device_id"]},
            headers={"Authorization": f"Bearer {config['auth_token']}"},
            timeout=10
        )

        if resp.status_code == 200:
            data = resp.json()
            return data.get("task")
        return None
    except Exception as e:
        print(f"Poll error: {e}")
        return None


def execute_task(task: dict) -> dict:
    """Execute a compute task using Ollama"""
    task_type = task.get("type", "inference")
    task_input = task.get("input", {})
    task_config = task.get("config", {})

    print(f"  Executing task: {task['id']} ({task_type})")

    try:
        # LLM-based tasks: inference, assessment, planning, execution, summarization, etc.
        llm_task_types = [
            "inference", "assessment", "planning", "execution",
            "draft_generation", "draft_verification", "summarization", "classification"
        ]

        if task_type in llm_task_types:
            model = task_config.get("model", "qwen3-coder-32k:latest")

            # Build prompt from input
            system_prompt = task_input.get("systemPrompt", "")
            user_prompt = task_input.get("prompt", "")

            # For simple inference, prompt might be at top level
            if not user_prompt and isinstance(task_input, str):
                user_prompt = task_input

            # Combine prompts
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            else:
                full_prompt = user_prompt

            if not full_prompt.strip():
                return {"success": False, "error": "No prompt provided"}

            print(f"    Model: {model}, Prompt length: {len(full_prompt)} chars")

            result = subprocess.run(
                ["ollama", "run", model],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Ollama exited with code {result.returncode}",
                    "model": model,
                }

            return {
                "success": True,
                "output": result.stdout,
                "model": model,
                "taskType": task_type,
            }

        elif task_type == "simulation":
            # Run water harvester simulation
            params = task_input.get("params", {})
            result = subprocess.run(
                ["python3", os.path.expanduser("~/simulations/quick_estimate.py"), json.dumps(params)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Simulation exited with code {result.returncode}",
                }

            try:
                output = json.loads(result.stdout) if result.stdout.strip() else {}
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Invalid JSON output: {e}"}

            return {
                "success": True,
                "output": output,
            }

        elif task_type == "embedding":
            # Embedding tasks - use Ollama's embedding endpoint
            text = task_input.get("text", task_input.get("prompt", ""))
            model = task_config.get("model", "nomic-embed-text")

            result = subprocess.run(
                ["ollama", "run", model],
                input=f"Generate embeddings for: {text}",
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Ollama exited with code {result.returncode}",
                    "model": model,
                }

            return {
                "success": True,
                "output": result.stdout,
                "model": model,
                "note": "Text embedding via Ollama"
            }

        else:
            # Unknown type - try to handle as generic LLM task
            print(f"    Warning: Unknown task type '{task_type}', treating as inference")
            prompt = task_input.get("prompt", str(task_input))
            model = task_config.get("model", "qwen3-coder-32k:latest")

            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr or f"Ollama exited with code {result.returncode}",
                    "model": model,
                }

            return {
                "success": True,
                "output": result.stdout,
                "model": model,
                "taskType": task_type,
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Task timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def report_result(config: dict, task_id: str, result: dict):
    """Report task result to the compute network"""
    try:
        resp = requests.post(
            f"{config['api_url']}/api/compute/tasks/{task_id}/complete",
            json={
                "deviceId": config["device_id"],
                "success": result.get("success", False),
                "result": result,
                "error": result.get("error"),
            },
            headers={"Authorization": f"Bearer {config['auth_token']}"},
            timeout=30
        )

        if resp.status_code == 200:
            print(f"  ✓ Result reported for task {task_id}")
            return True
        else:
            print(f"  ✗ Failed to report result: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Report error: {e}")
        return False


def validate_config(config: dict) -> bool:
    """Check if config has all required fields"""
    required = ["device_id", "auth_token", "api_url"]
    return all(k in config and config[k] for k in required)


def main():
    parser = argparse.ArgumentParser(description="LabFork Compute Device Client")
    parser.add_argument("--api-url", default=None,
                       help="Cloudflare Workers API URL (default: use saved config or localhost)")
    parser.add_argument("--register", action="store_true",
                       help="Force re-registration")
    parser.add_argument("--status", action="store_true",
                       help="Show device status and exit")
    args = parser.parse_args()

    # Load existing config
    config = load_config()

    if args.status:
        if config and validate_config(config):
            print(f"Device ID: {config['device_id']}")
            print(f"API URL: {config['api_url']}")
            print(f"Registered: {config.get('registered_at', 'Unknown')}")
        else:
            print("Device not registered")
        return

    # Determine API URL: CLI arg > saved config > default
    api_url = args.api_url or (config.get("api_url") if config else None) or "http://localhost:8787"

    # Re-register if: no config, invalid config, forced, or API URL changed
    needs_register = (
        not config or
        not validate_config(config) or
        args.register or
        (args.api_url and args.api_url != config.get("api_url"))
    )

    if needs_register:
        config = register_device(api_url)
        if not config:
            print("Failed to register device. Exiting.")
            sys.exit(1)

    print(f"\n=== LabFork Compute Device Running ===")
    print(f"Device ID: {config['device_id']}")
    print(f"API URL: {config['api_url']}")
    print(f"Heartbeat interval: {HEARTBEAT_INTERVAL}s")
    print("")

    last_heartbeat = 0
    consecutive_failures = 0

    while True:
        try:
            current_time = time.time()
            task = None  # Reset each iteration to avoid stale reference

            # Send heartbeat and check for tasks
            if current_time - last_heartbeat >= HEARTBEAT_INTERVAL:
                task = send_heartbeat(config)
                last_heartbeat = current_time

                if task:
                    consecutive_failures = 0
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received task from heartbeat")
                    result = execute_task(task)
                    report_result(config, task["id"], result)
                elif consecutive_failures == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Heartbeat OK - no pending tasks")
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: {consecutive_failures} consecutive heartbeat failures")

            # If no task from heartbeat, actively poll for work
            if not task:
                task = poll_for_task(config)
                if task:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Received task from poll")
                    result = execute_task(task)
                    report_result(config, task["id"], result)

            # Sleep less when actively processing, more when idle
            time.sleep(TASK_POLL_INTERVAL if not task else 1)

        except KeyboardInterrupt:
            print("\nShutting down...")
            send_heartbeat(config, status="offline")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            consecutive_failures += 1
            time.sleep(min(10 * consecutive_failures, 60))  # Exponential backoff up to 60s


if __name__ == "__main__":
    main()
