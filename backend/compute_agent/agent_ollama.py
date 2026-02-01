#!/usr/bin/env python3
"""
4090 Compute Agent (Ollama Version)

Uses Ollama for inference instead of transformers, avoiding dependency issues.
Ollama is already running on the 4090.

Usage:
    python agent_ollama.py --workers-url https://labfork-agents.XXX.workers.dev
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
import uuid
import json
import subprocess
from typing import Dict, Any, Optional

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ComputeAgent")

# Device configuration
DEVICE_ID = f"rtx4090-{uuid.uuid4().hex[:8]}"
DEVICE_NAME = "RTX-4090-Primary"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:8b"


class OllamaInference:
    """Simple Ollama inference client."""

    def __init__(self, base_url: str = OLLAMA_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model

    async def generate(self, prompt: str, system: str = None) -> str:
        """Generate text using Ollama."""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            }
            if system:
                payload["system"] = system

            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("response", "")
                else:
                    error = await resp.text()
                    raise Exception(f"Ollama error: {error}")

    async def chat(self, messages: list) -> str:
        """Chat completion using Ollama."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("message", {}).get("content", "")
                else:
                    error = await resp.text()
                    raise Exception(f"Ollama error: {error}")


class ComputeAgent:
    """Compute agent that polls Workers for tasks and executes them via Ollama."""

    def __init__(self, workers_url: str, poll_interval: int = 5):
        self.workers_url = workers_url.rstrip("/")
        self.poll_interval = poll_interval
        self.device_id = DEVICE_ID
        self.running = False
        self.ollama = OllamaInference()

    async def register(self) -> bool:
        """Register this device with Workers."""
        try:
            # Get GPU info
            gpu_info = self._get_gpu_info()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.workers_url}/api/compute/devices/register",
                    json={
                        "id": self.device_id,
                        "name": DEVICE_NAME,
                        "tier": "power",
                        "platform": "cuda",
                        "capabilities": {
                            "compute": 82.6,  # RTX 4090 TFLOPS
                            "memory": 24,     # GB VRAM
                            "models": ["qwen3:8b", "llama3.1:8b", "mistral:7b"]
                        }
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(f"Registered with Workers: {data}")
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"Registration failed: {error}")
                        return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    def _get_gpu_info(self) -> dict:
        """Get GPU info via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                return {
                    "name": parts[0] if len(parts) > 0 else "Unknown",
                    "memory_total": int(parts[1]) if len(parts) > 1 else 0,
                    "memory_used": int(parts[2]) if len(parts) > 2 else 0,
                    "utilization": int(parts[3]) if len(parts) > 3 else 0
                }
        except:
            pass
        return {"name": "RTX 4090", "memory_total": 24576, "memory_used": 0, "utilization": 0}

    async def heartbeat(self) -> bool:
        """Send heartbeat to Workers."""
        try:
            gpu_info = self._get_gpu_info()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.workers_url}/api/compute/devices/{self.device_id}/heartbeat",
                    json={
                        "status": "online",
                        "gpuUtilization": gpu_info.get("utilization", 0),
                        "memoryUsed": gpu_info.get("memory_used", 0),
                        "memoryTotal": gpu_info.get("memory_total", 24576)
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    async def poll_tasks(self) -> Optional[dict]:
        """Poll for available tasks."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.workers_url}/api/compute/tasks/pending",
                    params={"deviceId": self.device_id, "tier": "power"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tasks = data.get("tasks", [])
                        if tasks:
                            return tasks[0]  # Get first available task
        except Exception as e:
            logger.warning(f"Poll failed: {e}")
        return None

    async def claim_task(self, task_id: str) -> bool:
        """Claim a task for execution."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.workers_url}/api/compute/tasks/{task_id}/claim",
                    json={"deviceId": self.device_id},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    return resp.status == 200
        except:
            return False

    async def complete_task(self, task_id: str, output: Any, error: str = None) -> bool:
        """Report task completion."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.workers_url}/api/compute/tasks/{task_id}/complete",
                    json={
                        "deviceId": self.device_id,
                        "output": output,
                        "error": error
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    return resp.status == 200
        except Exception as e:
            logger.error(f"Complete task failed: {e}")
            return False

    async def execute_task(self, task: dict) -> tuple:
        """Execute a compute task using Ollama."""
        task_type = task.get("type", "inference")
        input_data = task.get("input", {})

        if isinstance(input_data, str):
            input_data = json.loads(input_data)

        try:
            if task_type == "inference":
                messages = input_data.get("messages", [])
                if messages:
                    result = await self.ollama.chat(messages)
                else:
                    prompt = input_data.get("prompt", "")
                    system = input_data.get("system")
                    result = await self.ollama.generate(prompt, system)
                return {"response": result, "model": self.ollama.model}, None

            elif task_type == "assessment":
                # Project assessment task
                prompt = input_data.get("prompt", "Assess this project")
                result = await self.ollama.generate(prompt)
                return {"assessment": result}, None

            elif task_type == "planning":
                # Task planning
                prompt = input_data.get("prompt", "Create a plan")
                result = await self.ollama.generate(prompt)
                return {"plan": result}, None

            elif task_type == "execution":
                # Execute step
                prompt = input_data.get("prompt", "Execute this step")
                result = await self.ollama.generate(prompt)
                return {"result": result}, None

            else:
                # Generic inference
                prompt = input_data.get("prompt", str(input_data))
                result = await self.ollama.generate(prompt)
                return {"result": result}, None

        except Exception as e:
            logger.error(f"Execution error: {e}")
            return None, str(e)

    async def run(self):
        """Main agent loop."""
        logger.info("=" * 50)
        logger.info("  4090 Compute Agent (Ollama)")
        logger.info("=" * 50)
        logger.info(f"Device ID: {self.device_id}")
        logger.info(f"Workers URL: {self.workers_url}")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info("")

        # Register with Workers
        if not await self.register():
            logger.error("Failed to register with Workers. Retrying...")
            await asyncio.sleep(5)
            if not await self.register():
                logger.error("Registration failed. Check Workers URL.")
                return

        self.running = True
        heartbeat_counter = 0

        while self.running:
            try:
                # Send heartbeat every 6 polls (30 seconds)
                heartbeat_counter += 1
                if heartbeat_counter >= 6:
                    await self.heartbeat()
                    heartbeat_counter = 0

                # Poll for tasks
                task = await self.poll_tasks()

                if task:
                    task_id = task.get("id")
                    task_type = task.get("type")
                    logger.info(f"Found task: {task_id} ({task_type})")

                    # Claim the task
                    if await self.claim_task(task_id):
                        logger.info(f"Claimed task {task_id}, executing...")

                        # Execute
                        output, error = await self.execute_task(task)

                        # Report completion
                        if await self.complete_task(task_id, output, error):
                            if error:
                                logger.warning(f"Task {task_id} failed: {error}")
                            else:
                                logger.info(f"Task {task_id} completed successfully")
                        else:
                            logger.error(f"Failed to report completion for {task_id}")
                    else:
                        logger.warning(f"Failed to claim task {task_id}")

                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                logger.info("Agent shutdown requested")
                break
            except Exception as e:
                logger.error(f"Agent loop error: {e}")
                await asyncio.sleep(self.poll_interval)

        logger.info("Agent stopped")

    def stop(self):
        """Stop the agent."""
        self.running = False


async def main():
    parser = argparse.ArgumentParser(description="4090 Compute Agent (Ollama)")
    parser.add_argument("--workers-url", required=True, help="Workers API URL")
    parser.add_argument("--poll-interval", type=int, default=5, help="Poll interval in seconds")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    args = parser.parse_args()

    agent = ComputeAgent(args.workers_url, args.poll_interval)
    agent.ollama.model = args.model

    # Handle shutdown
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, agent.stop)

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
