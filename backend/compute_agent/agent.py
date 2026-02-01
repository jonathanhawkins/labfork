#!/usr/bin/env python3
"""
4090 Compute Agent

This agent runs on the RTX 4090 server and:
1. Registers with Cloudflare Workers as a 'power' tier compute device
2. Polls for pending compute tasks
3. Executes inference using local models
4. Reports results back to Workers

This replaces Workers AI with our own GPU infrastructure.

Usage:
    python agent.py [--local] [--model MODEL_ID]

Options:
    --local     Use local Workers development server (localhost:8787)
    --model     Default model to preload
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from typing import Dict, Any, Optional

import aiohttp

from config import get_config, Config
from inference import get_inference_engine, InferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ComputeAgent")


class ComputeAgent:
    """
    Compute agent that connects to Workers for distributed inference.

    Lifecycle:
    1. Register with Workers API
    2. Start heartbeat loop
    3. Poll for tasks
    4. Execute tasks using InferenceEngine
    5. Report results
    """

    def __init__(self, config: Config):
        self.config = config
        self.engine: InferenceEngine = get_inference_engine()
        self.device_id: Optional[str] = None
        self.running = False
        self.current_task: Optional[Dict[str, Any]] = None

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # Task counters
        self.tasks_completed = 0
        self.tasks_failed = 0

    @property
    def api_url(self) -> str:
        """Get the Workers API URL."""
        return f"{self.config.workers.api_url}/api/compute"

    async def start(self):
        """Start the compute agent."""
        logger.info("Starting compute agent...")
        logger.info(f"Workers API: {self.api_url}")

        # Create HTTP session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
        )

        try:
            # Register with Workers
            if not await self.register():
                logger.error("Failed to register with Workers")
                return

            self.running = True

            # Preload default model
            logger.info(f"Preloading model: {self.config.inference.default_model}")
            self.engine.load_model(self.config.inference.default_model)

            # Start main loop
            await self.main_loop()

        except Exception as e:
            logger.error(f"Agent error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the compute agent."""
        logger.info("Stopping compute agent...")
        self.running = False

        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None

        logger.info(
            f"Agent stopped. Tasks completed: {self.tasks_completed}, "
            f"Tasks failed: {self.tasks_failed}"
        )

    async def register(self) -> bool:
        """Register this device with Workers."""
        try:
            payload = {
                "name": self.config.device.name,
                "platform": self.config.device.platform,
                "capabilities": {
                    "compute": self.config.device.compute_tflops,
                    "memory": self.config.device.memory_gb,
                    "bandwidth": self.config.device.bandwidth_mbps,
                    "models": self.config.device.cached_models,
                },
            }

            async with self.session.post(
                f"{self.api_url}/devices",
                json=payload,
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Registration failed: {error}")
                    return False

                data = await response.json()
                self.device_id = data["device"]["id"]
                tier = data["device"]["tier"]

                logger.info(f"Registered as device {self.device_id} ({tier} tier)")
                return True

        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    async def heartbeat(self) -> Optional[Dict[str, Any]]:
        """
        Send heartbeat to Workers.

        Returns any assigned task from the response.
        """
        if not self.device_id:
            return None

        try:
            status = "busy" if self.current_task else "online"

            async with self.session.patch(
                f"{self.api_url}/devices/{self.device_id}",
                json={"status": status},
            ) as response:
                if response.status != 200:
                    logger.warning(f"Heartbeat failed: {response.status}")
                    return None

                data = await response.json()

                # Check if a task was assigned
                if data.get("task"):
                    return data["task"]

                return None

        except Exception as e:
            logger.error(f"Heartbeat error: {e}")
            return None

    async def poll_for_task(self) -> Optional[Dict[str, Any]]:
        """Poll for pending tasks."""
        if not self.device_id:
            return None

        try:
            async with self.session.get(
                f"{self.api_url}/tasks/pending",
                params={"deviceId": self.device_id, "limit": "1"},
            ) as response:
                if response.status != 200:
                    return None

                data = await response.json()
                tasks = data.get("tasks", [])

                if tasks:
                    return tasks[0]

                return None

        except Exception as e:
            logger.error(f"Poll error: {e}")
            return None

    async def claim_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Claim a task for execution."""
        try:
            async with self.session.post(
                f"{self.api_url}/tasks/{task_id}/claim",
                json={"deviceId": self.device_id},
            ) as response:
                if response.status == 409:
                    # Task already claimed by another device
                    logger.info(f"Task {task_id} already claimed")
                    return None

                if response.status != 200:
                    logger.warning(f"Failed to claim task {task_id}: {response.status}")
                    return None

                data = await response.json()
                return data.get("task")

        except Exception as e:
            logger.error(f"Claim error: {e}")
            return None

    async def complete_task(
        self,
        task_id: str,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        """Report task completion."""
        try:
            payload = {
                "deviceId": self.device_id,
                "success": success,
            }

            if result:
                payload["result"] = result

            if error:
                payload["error"] = error

            async with self.session.post(
                f"{self.api_url}/tasks/{task_id}/complete",
                json=payload,
            ) as response:
                if response.status != 200:
                    logger.warning(f"Failed to complete task {task_id}: {response.status}")
                else:
                    logger.info(f"Task {task_id} completed: {'success' if success else 'failed'}")

        except Exception as e:
            logger.error(f"Complete task error: {e}")

    async def execute_task(self, task: Dict[str, Any]):
        """Execute a compute task."""
        task_id = task["id"]
        task_type = task["type"]
        task_input = task.get("input", {})
        task_config = task.get("config", {})

        logger.info(f"Executing task {task_id} ({task_type})")
        self.current_task = task

        try:
            # Extract parameters
            prompt = task_input.get("prompt", "")
            messages = task_input.get("messages")
            system_prompt = task_input.get("systemPrompt")
            model_id = task_input.get("model") or task_config.get("modelId")
            max_tokens = task_config.get("maxTokens", 1024)
            temperature = task_config.get("temperature", 0.7)

            # Execute based on task type
            if task_type in ("inference", "assessment", "planning", "execution"):
                result = self.engine.inference(
                    prompt=prompt,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                    messages=messages,
                )

            elif task_type == "embedding":
                result = self.engine.generate_embedding(
                    text=prompt,
                    model_id=model_id,
                )

            else:
                # Default to inference
                result = self.engine.inference(
                    prompt=prompt,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            # Report result
            if result.success:
                await self.complete_task(
                    task_id,
                    success=True,
                    result=result.to_dict(),
                )
                self.tasks_completed += 1
            else:
                await self.complete_task(
                    task_id,
                    success=False,
                    error=result.error,
                )
                self.tasks_failed += 1

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            await self.complete_task(
                task_id,
                success=False,
                error=str(e),
            )
            self.tasks_failed += 1

        finally:
            self.current_task = None

    async def main_loop(self):
        """Main agent loop."""
        heartbeat_interval = self.config.agent.heartbeat_interval
        poll_interval = self.config.agent.poll_interval

        last_heartbeat = 0
        last_poll = 0

        logger.info("Entering main loop...")

        while self.running:
            now = time.time()

            # Heartbeat
            if now - last_heartbeat >= heartbeat_interval:
                task = await self.heartbeat()
                last_heartbeat = now

                # Check for task from heartbeat response
                if task and not self.current_task:
                    await self.execute_task(task)
                    continue

            # Poll for tasks (if not currently executing)
            if not self.current_task and now - last_poll >= poll_interval:
                task = await self.poll_for_task()
                last_poll = now

                if task:
                    # Claim and execute
                    claimed_task = await self.claim_task(task["id"])
                    if claimed_task:
                        await self.execute_task(claimed_task)
                        continue

            # Small sleep to prevent busy loop
            await asyncio.sleep(0.5)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="4090 Compute Agent")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Workers development server",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Default model to preload",
    )
    parser.add_argument(
        "--workers-url",
        type=str,
        help="Workers API URL",
    )
    args = parser.parse_args()

    # Configure
    config = get_config()

    if args.local:
        config.workers.use_local = True
        logger.info("Using local Workers server")

    if args.workers_url:
        if "localhost" in args.workers_url or "127.0.0.1" in args.workers_url:
            config.workers.local_url = args.workers_url
            config.workers.use_local = True
        else:
            config.workers.base_url = args.workers_url
            config.workers.use_local = False

    if args.model:
        config.inference.default_model = args.model

    # Create agent
    agent = ComputeAgent(config)

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def shutdown_handler():
        logger.info("Shutdown signal received")
        agent.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_handler)

    # Start agent
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
