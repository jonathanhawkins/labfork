#!/usr/bin/env python3
"""
Hybrid Agent Daemon for 4090

This daemon runs continuously on the 4090 and implements a two-tier work system:

TIER 0 (HIGHEST PRIORITY): Project Tasks
- Polls Workers API for project tasks from the tasks table (Firefly Network)
- These are the MAIN work driver - real project work like "Research MPPT algorithms"
- Claims and executes tasks, reports progress and completion

TIER 1 (HIGH PRIORITY): Workers Coordination Layer
- Polls Workers API for compute_tasks every 30 seconds
- Claims and executes distributed tasks (from labfork.com/watch coordination)
- Reports results back to Workers

TIER 2 (BACKGROUND): Local Independent Research
- When no Workers tasks available, falls back to local research
- Uses the existing orchestrator.js/manager.py patterns
- Spawns Ollama-based agents for autonomous research
- Syncs findings periodically

Architecture:
    Project Tasks (tasks table) ---------> Hybrid Agent (claims first)
                                               |
    Workers Cron (15min) --> compute_tasks --> (claims second)
                                               |
    Local Research Queue <--> (fallback when idle)
                                               |
    lab-api-server.py <-- State Updates --> /watch page

Usage:
    python hybrid-agent-daemon.py [--workers-url URL] [--poll-interval 30]
"""

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Try to import aiohttp, fallback to requests if not available
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_AIOHTTP = False

# ============================================================================
# Configuration
# ============================================================================

# Default Workers API endpoint (Cloudflare Workers)
DEFAULT_WORKERS_URL = os.environ.get(
    "WORKERS_API_URL",
    "https://labfork-workers.your-account.workers.dev"
)

# Local paths
PROJECT_ROOT = Path.home() / "dev" / "voice-clone-pipeline"
SKILLS_DIR = PROJECT_ROOT / ".skills" / "research-manager"
STATE_DIR = SKILLS_DIR / "state"
LABS_DIR = SKILLS_DIR / "labs"
OUTPUTS_DIR = STATE_DIR / "outputs"

# Ensure directories exist
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Device registration info
DEVICE_INFO = {
    "name": "rtx-4090-primary",
    "platform": "cuda",
    "capabilities": {
        "compute": 82.6,  # TFLOPS
        "memory": 24,     # GB
        "models": ["qwen3-coder:30b", "qwen3-coder-32k"],
        "gpuName": "RTX 4090"
    }
}

# Limits (matches orchestrator.js)
LIMITS = {
    "max_tasks_per_day": 100,  # Increased for autonomous operation
    "max_research_per_day": 20,
    "max_agent_minutes_per_day": 480,  # 8 hours
    "max_retries": 5,
    "retry_cooldown_minutes": 10,
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STATE_DIR / "hybrid-agent.log")
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# State Management
# ============================================================================

class AgentState:
    """Manages local agent state and syncs with lab-api-server"""

    def __init__(self):
        self.device_id: Optional[str] = None
        self.status = "idle"
        self.current_task: Optional[Dict] = None
        self.tasks_completed_today = 0
        self.research_sessions_today = 0
        self.agent_minutes_today = 0
        self.last_reset_date = datetime.now().date()
        self.load_state()

    def load_state(self):
        """Load persisted state from disk"""
        state_file = STATE_DIR / "hybrid-agent-state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                self.device_id = data.get("device_id")
                self.tasks_completed_today = data.get("tasks_completed_today", 0)
                self.research_sessions_today = data.get("research_sessions_today", 0)
                self.agent_minutes_today = data.get("agent_minutes_today", 0)
                last_reset = data.get("last_reset_date")
                if last_reset:
                    self.last_reset_date = datetime.fromisoformat(last_reset).date()

                # Reset daily counters if new day
                if datetime.now().date() > self.last_reset_date:
                    self.reset_daily_counters()
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def save_state(self):
        """Persist state to disk"""
        state_file = STATE_DIR / "hybrid-agent-state.json"
        data = {
            "device_id": self.device_id,
            "status": self.status,
            "tasks_completed_today": self.tasks_completed_today,
            "research_sessions_today": self.research_sessions_today,
            "agent_minutes_today": self.agent_minutes_today,
            "last_reset_date": self.last_reset_date.isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        state_file.write_text(json.dumps(data, indent=2))

    def reset_daily_counters(self):
        """Reset daily limits"""
        self.tasks_completed_today = 0
        self.research_sessions_today = 0
        self.agent_minutes_today = 0
        self.last_reset_date = datetime.now().date()
        self.save_state()
        logger.info("Daily counters reset")

    def can_accept_work(self) -> bool:
        """Check if we're within daily limits"""
        if datetime.now().date() > self.last_reset_date:
            self.reset_daily_counters()

        return (
            self.tasks_completed_today < LIMITS["max_tasks_per_day"] and
            self.agent_minutes_today < LIMITS["max_agent_minutes_per_day"]
        )

    def can_do_research(self) -> bool:
        """Check if we can start a new research session"""
        return (
            self.can_accept_work() and
            self.research_sessions_today < LIMITS["max_research_per_day"]
        )

    def update_agents_json(self, agent_name: str, task: str, status: str = "running"):
        """Update agents.json for lab-api-server visibility"""
        agents_file = STATE_DIR / "agents.json"
        try:
            agents = {}
            if agents_file.exists():
                agents = json.loads(agents_file.read_text())

            if status == "stopped":
                agents.pop(agent_name, None)
            else:
                agents[agent_name] = {
                    "status": status,
                    "task": task,
                    "type": "ollama",
                    "started_at": datetime.now().isoformat()
                }

            agents_file.write_text(json.dumps(agents, indent=2))
        except Exception as e:
            logger.warning(f"Failed to update agents.json: {e}")


# ============================================================================
# Workers API Client
# ============================================================================

class WorkersClient:
    """Client for Workers compute API"""

    # Default headers for all requests (Cloudflare blocks Python's default User-Agent)
    DEFAULT_HEADERS = {
        "User-Agent": "HybridAgentDaemon/1.0",
        "Accept": "application/json",
    }

    def __init__(self, base_url: str, state: AgentState):
        self.base_url = base_url.rstrip("/")
        self.state = state

    async def register_device(self) -> Optional[str]:
        """Register this device with Workers"""
        try:
            url = f"{self.base_url}/compute/devices"
            data = DEVICE_INFO

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            device_id = result.get("device", {}).get("id")
                            logger.info(f"Registered device: {device_id}")
                            return device_id
                        else:
                            logger.error(f"Registration failed: {resp.status}")
            else:
                headers = {**self.DEFAULT_HEADERS, "Content-Type": "application/json"}
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    device_id = result.get("device", {}).get("id")
                    logger.info(f"Registered device: {device_id}")
                    return device_id
        except Exception as e:
            logger.warning(f"Failed to register with Workers: {e}")
            return None

    async def heartbeat(self) -> Optional[Dict]:
        """Send heartbeat and check for assigned tasks"""
        if not self.state.device_id:
            return None

        try:
            url = f"{self.base_url}/compute/devices/{self.state.device_id}"
            data = {"status": "online" if self.state.status == "idle" else "busy"}

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.patch(url, json=data) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return result.get("task")
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={**self.DEFAULT_HEADERS, "Content-Type": "application/json"},
                    method="PATCH"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    return result.get("task")
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
            return None

    async def get_pending_tasks(self) -> List[Dict]:
        """Get pending tasks from Workers"""
        if not self.state.device_id:
            return []

        try:
            url = f"{self.base_url}/compute/tasks/pending?deviceId={self.state.device_id}&limit=1"

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return result.get("tasks", [])
            else:
                req = urllib.request.Request(url, headers=self.DEFAULT_HEADERS)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    return result.get("tasks", [])
        except Exception as e:
            logger.debug(f"Failed to get pending tasks: {e}")
            return []

    async def claim_task(self, task_id: str) -> Optional[Dict]:
        """Claim a task for execution"""
        try:
            url = f"{self.base_url}/compute/tasks/{task_id}/claim"
            data = {"deviceId": self.state.device_id}

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return result.get("task")
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={**self.DEFAULT_HEADERS, "Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    return result.get("task")
        except Exception as e:
            logger.error(f"Failed to claim task: {e}")
            return None

    async def complete_task(self, task_id: str, success: bool, result: Any = None, error: str = None):
        """Report task completion"""
        try:
            url = f"{self.base_url}/compute/tasks/{task_id}/complete"
            data = {
                "deviceId": self.state.device_id,
                "success": success,
                "result": result,
                "error": error
            }

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        if resp.status == 200:
                            logger.info(f"Task {task_id} completed: {'success' if success else 'failed'}")
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={**self.DEFAULT_HEADERS, "Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    logger.info(f"Task {task_id} completed: {'success' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Failed to complete task: {e}")

    async def sync_research_result(self, objective: Dict, results: Dict, lab_id: str = "voice-clone"):
        """Sync a research result to Workers"""
        try:
            url = f"{self.base_url}/research/sync"
            data = {
                "deviceId": self.state.device_id,
                "objective": objective,
                "results": results,
                "labId": lab_id
            }

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        if resp.status == 200:
                            logger.info(f"Research synced: {objective.get('title', 'unknown')}")
                            return True
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={**self.DEFAULT_HEADERS, "Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    logger.info(f"Research synced: {objective.get('title', 'unknown')}")
                    return True
        except Exception as e:
            logger.debug(f"Failed to sync research (Workers may be offline): {e}")
            return False

    async def get_research_objectives(self, lab_id: str = "voice-clone") -> List[Dict]:
        """Fetch research objectives from Workers"""
        try:
            url = f"{self.base_url}/research/objectives?labId={lab_id}"

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return result.get("objectives", [])
            else:
                req = urllib.request.Request(url, headers=self.DEFAULT_HEADERS)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    return result.get("objectives", [])
        except Exception as e:
            logger.debug(f"Failed to fetch research objectives: {e}")
            return []

    # =========================================================================
    # Project Tasks API (TIER 0 - Main Work Driver)
    # =========================================================================

    async def get_pending_project_tasks(self, project_id: str = None) -> List[Dict]:
        """Get pending project tasks from the tasks table (main work driver)"""
        if not self.state.device_id:
            return []

        try:
            url = f"{self.base_url}/tasks/pending?deviceId={self.state.device_id}&limit=3"
            if project_id:
                url += f"&projectId={project_id}"

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            return result.get("tasks", [])
            else:
                req = urllib.request.Request(url, headers=self.DEFAULT_HEADERS)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    return result.get("tasks", [])
        except Exception as e:
            logger.debug(f"Failed to get pending project tasks: {e}")
            return []

    async def claim_project_task(self, task_id: str, agent_name: str = None) -> Optional[Dict]:
        """Claim a project task for execution"""
        try:
            url = f"{self.base_url}/tasks/{task_id}/claim"
            data = {
                "deviceId": self.state.device_id,
                "agentName": agent_name or f"4090-{self.state.device_id[:8]}"
            }

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            logger.info(f"Claimed project task: {task_id}")
                            return result.get("task")
                        else:
                            error = await resp.text()
                            logger.warning(f"Failed to claim task {task_id}: {error}")
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={**self.DEFAULT_HEADERS, "Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    logger.info(f"Claimed project task: {task_id}")
                    return result.get("task")
        except Exception as e:
            logger.error(f"Failed to claim project task: {e}")
            return None

    async def update_project_task_progress(self, task_id: str, progress: int, status: str = None, result: str = None):
        """Update progress on a project task"""
        try:
            url = f"{self.base_url}/tasks/{task_id}/progress"
            data = {
                "deviceId": self.state.device_id,
                "progress": progress,
            }
            if status:
                data["status"] = status
            if result:
                data["result"] = result

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        if resp.status == 200:
                            logger.debug(f"Progress updated: {task_id} -> {progress}%")
            else:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(data).encode(),
                    headers={**self.DEFAULT_HEADERS, "Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    logger.debug(f"Progress updated: {task_id} -> {progress}%")
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")

    async def complete_project_task(self, task_id: str, result: str = None):
        """Mark a project task as completed"""
        try:
            # First update progress to 100%
            await self.update_project_task_progress(task_id, 100, result=result)

            # Then call the complete endpoint
            url = f"{self.base_url}/tasks/{task_id}/complete"

            if HAS_AIOHTTP:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={}) as resp:
                        if resp.status == 200:
                            logger.info(f"Project task completed: {task_id}")
                            return True
            else:
                req = urllib.request.Request(
                    url,
                    data=b"{}",
                    headers={**self.DEFAULT_HEADERS, "Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    logger.info(f"Project task completed: {task_id}")
                    return True
        except Exception as e:
            logger.error(f"Failed to complete project task: {e}")
            return False


# ============================================================================
# Local Research Manager
# ============================================================================

class LocalResearchManager:
    """Manages local research when no Workers tasks available"""

    def __init__(self, state: AgentState, workers_client: Optional['WorkersClient'] = None):
        self.state = state
        self.workers = workers_client
        self.lab_id = "voice-clone"  # Default lab
        self.research_objectives = self.load_research_objectives()

    def load_research_objectives(self) -> List[Dict]:
        """Load research objectives from lab config"""
        objectives = []

        # Check for pending research in the lab's state
        lab_state_dir = LABS_DIR / self.lab_id / "state"
        research_file = lab_state_dir / "research-queue.json"

        if research_file.exists():
            try:
                data = json.loads(research_file.read_text())
                objectives = data.get("objectives", [])
            except Exception as e:
                logger.warning(f"Failed to load research objectives: {e}")

        # Default objectives if none defined
        if not objectives:
            objectives = [
                {
                    "id": "prosody-analysis",
                    "title": "Analyze prosody conditioning approaches",
                    "description": "Research recent papers on prosody conditioning for TTS",
                    "priority": 5,
                    "status": "pending"
                },
                {
                    "id": "eval-metrics",
                    "title": "Evaluate voice quality metrics",
                    "description": "Implement and test objective voice quality metrics",
                    "priority": 4,
                    "status": "pending"
                }
            ]

        return [o for o in objectives if o.get("status") == "pending"]

    def get_next_research_task(self) -> Optional[Dict]:
        """Get the next research task to work on"""
        if not self.research_objectives:
            self.research_objectives = self.load_research_objectives()

        # Return highest priority pending objective
        pending = [o for o in self.research_objectives if o.get("status") == "pending"]
        if pending:
            return sorted(pending, key=lambda x: -x.get("priority", 0))[0]
        return None

    async def execute_research(self, objective: Dict) -> Dict:
        """Execute a research objective using local Ollama"""
        logger.info(f"Starting local research: {objective.get('title')}")

        start_time = time.time()
        agent_name = f"research-{objective.get('id', 'unknown')}"

        # Update agents.json for visibility
        self.state.update_agents_json(
            agent_name,
            f"RESEARCH: {objective.get('title')}",
            "running"
        )

        try:
            # Create task file for the agent
            task_content = f"""You are a research agent working on: {objective.get('title')}

OBJECTIVE:
{objective.get('description')}

INSTRUCTIONS:
1. Research this topic thoroughly
2. Summarize key findings
3. Suggest concrete next steps
4. Document any code changes or experiments needed

TIME LIMIT: 20 minutes

BEGIN RESEARCH:
"""
            task_file = OUTPUTS_DIR / f"{agent_name}-task.txt"
            task_file.write_text(task_content)

            log_file = OUTPUTS_DIR / f"{agent_name}.log"

            # Run the spawn-ollama-agent.sh script
            spawn_script = SKILLS_DIR / "spawn-ollama-agent.sh"
            if spawn_script.exists():
                process = subprocess.Popen(
                    ["bash", str(spawn_script), str(task_file), str(log_file)],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Wait with timeout (20 minutes)
                try:
                    stdout, stderr = process.communicate(timeout=1200)
                    success = process.returncode == 0
                except subprocess.TimeoutExpired:
                    process.kill()
                    logger.warning(f"Research timed out: {agent_name}")
                    success = False

                # Read results from log file
                results = ""
                if log_file.exists():
                    results = log_file.read_text()[-5000:]  # Last 5000 chars

                duration_minutes = (time.time() - start_time) / 60

                return {
                    "success": success,
                    "objective_id": objective.get("id"),
                    "duration_minutes": round(duration_minutes, 1),
                    "output": results,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                logger.error("spawn-ollama-agent.sh not found")
                return {
                    "success": False,
                    "error": "spawn script not found"
                }

        finally:
            # Update agents.json to remove this agent
            self.state.update_agents_json(agent_name, "", "stopped")
            self.state.research_sessions_today += 1
            self.state.save_state()

    def save_research_results(self, objective: Dict, results: Dict):
        """Save research results for later sync"""
        results_dir = LABS_DIR / self.lab_id / "state" / "research-results"
        results_dir.mkdir(parents=True, exist_ok=True)

        result_file = results_dir / f"{objective.get('id')}-{int(time.time())}.json"
        result_file.write_text(json.dumps({
            "objective": objective,
            "results": results,
            "saved_at": datetime.now().isoformat()
        }, indent=2))

        logger.info(f"Saved research results: {result_file.name}")

    async def sync_to_workers(self, objective: Dict, results: Dict):
        """Sync research results to Workers API"""
        if self.workers:
            await self.workers.sync_research_result(objective, results, self.lab_id)

    async def fetch_workers_objectives(self):
        """Fetch objectives from Workers and merge with local"""
        if self.workers:
            workers_objectives = await self.workers.get_research_objectives(self.lab_id)
            if workers_objectives:
                # Merge Workers objectives with local, preferring Workers
                local_ids = {o.get("id") for o in self.research_objectives}
                for obj in workers_objectives:
                    if obj.get("id") not in local_ids:
                        self.research_objectives.append(obj)

                # Re-sort by priority
                self.research_objectives.sort(key=lambda x: -x.get("priority", 0))
                logger.info(f"Merged {len(workers_objectives)} objectives from Workers")


# ============================================================================
# Project Task Executor (TIER 0 - Main Work Driver)
# ============================================================================

class ProjectTaskExecutor:
    """Executes project tasks from the main tasks table (Firefly Network tasks)"""

    def __init__(self, state: AgentState, workers: 'WorkersClient'):
        self.state = state
        self.workers = workers

    async def execute(self, task: Dict) -> Dict:
        """Execute a project task using local Ollama"""
        task_id = task.get("id")
        title = task.get("title", "Unknown Task")
        description = task.get("description", "")
        project_name = task.get("project_name", "Unknown Project")

        logger.info(f"Executing project task: {title}")
        logger.info(f"  Project: {project_name}")
        logger.info(f"  Description: {description[:100]}...")

        start_time = time.time()
        agent_name = f"project-{task_id[:8]}"

        # Update agents.json for visibility
        self.state.update_agents_json(
            agent_name,
            f"PROJECT: {title}",
            "running"
        )

        try:
            # Report 10% progress - starting
            await self.workers.update_project_task_progress(task_id, 10)

            # Create a comprehensive prompt for the task
            prompt = f"""You are an AI research agent working on the Firefly Network project.

PROJECT: {project_name}
TASK: {title}

DESCRIPTION:
{description}

INSTRUCTIONS:
1. Analyze this task carefully
2. Break it down into steps if needed
3. Research and provide concrete findings
4. Suggest implementation approaches
5. Document any code changes needed
6. Provide actionable next steps

Please work on this task now and provide your findings:
"""

            # Report 20% progress - prompt ready
            await self.workers.update_project_task_progress(task_id, 20)

            # Run inference using Ollama
            logger.info(f"Running Ollama inference for task: {task_id}")
            process = subprocess.run(
                ["ollama", "run", "qwen3-coder:30b", prompt],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for project tasks
            )

            # Report 80% progress - inference complete
            await self.workers.update_project_task_progress(task_id, 80)

            output = process.stdout
            if process.returncode != 0:
                error = process.stderr
                logger.error(f"Ollama error: {error}")
                return {
                    "success": False,
                    "error": f"Inference failed: {error[:500]}"
                }

            # Summarize the output
            summary = output[:2000] if len(output) > 2000 else output

            duration_minutes = (time.time() - start_time) / 60
            logger.info(f"Project task completed in {duration_minutes:.1f} minutes")

            # Report 100% progress
            await self.workers.update_project_task_progress(task_id, 100, result=summary)

            return {
                "success": True,
                "output": summary,
                "full_output": output,
                "duration_minutes": round(duration_minutes, 2),
                "task_id": task_id,
                "task_title": title
            }

        except subprocess.TimeoutExpired:
            logger.warning(f"Task timed out: {task_id}")
            return {
                "success": False,
                "error": "Task timed out after 10 minutes"
            }
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            self.state.update_agents_json(agent_name, "", "stopped")
            self.state.tasks_completed_today += 1
            self.state.agent_minutes_today += (time.time() - start_time) / 60
            self.state.save_state()


# ============================================================================
# Compute Task Executor (TIER 1)
# ============================================================================

class TaskExecutor:
    """Executes compute tasks from Workers (TIER 1 - coordination layer)"""

    def __init__(self, state: AgentState):
        self.state = state

    async def execute(self, task: Dict) -> Dict:
        """Execute a compute task"""
        task_type = task.get("type")
        task_input = task.get("input", {})

        logger.info(f"Executing task: {task.get('id')} (type: {task_type})")

        start_time = time.time()

        # Update agents.json
        self.state.update_agents_json(
            f"workers-{task.get('id', 'unknown')[:8]}",
            f"WORKERS TASK: {task_type}",
            "running"
        )

        try:
            if task_type == "inference":
                result = await self.run_inference(task_input)
            elif task_type == "assessment":
                result = await self.run_assessment(task_input)
            elif task_type == "embedding":
                result = await self.run_embedding(task_input)
            elif task_type == "classification":
                result = await self.run_classification(task_input)
            elif task_type == "summarization":
                result = await self.run_summarization(task_input)
            else:
                result = await self.run_generic(task_type, task_input)

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "output": result,
                "metrics": {
                    "computeTime": duration_ms,
                    "taskType": task_type
                }
            }

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

        finally:
            self.state.update_agents_json(
                f"workers-{task.get('id', 'unknown')[:8]}",
                "",
                "stopped"
            )
            self.state.tasks_completed_today += 1
            self.state.agent_minutes_today += (time.time() - start_time) / 60
            self.state.save_state()

    async def run_inference(self, input_data: Dict) -> str:
        """Run inference using local Ollama"""
        prompt = input_data.get("prompt", "")
        system_prompt = input_data.get("systemPrompt", "")

        # Use ollama CLI for inference
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        process = subprocess.run(
            ["ollama", "run", "qwen3-coder:30b", full_prompt],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        return process.stdout

    async def run_assessment(self, input_data: Dict) -> str:
        """Run project assessment using Ollama"""
        prompt = input_data.get("prompt", "")
        context = input_data.get("context", "")

        full_prompt = f"""You are an AI research assistant performing an assessment task.

Context: {context}

Task: {prompt}

Provide a brief, helpful assessment (2-3 sentences max)."""

        return await self.run_inference({"prompt": full_prompt})

    async def run_embedding(self, input_data: Dict) -> Dict:
        """Generate embeddings using Ollama

        Note: Returns a simplified hash-based embedding since Ollama
        doesn't have a dedicated embedding model installed.
        For production, install nomic-embed-text or mxbai-embed-large.
        """
        text = input_data.get("text", str(input_data))

        # Simple hash-based pseudo-embedding (deterministic, not semantic)
        # This ensures consistent results for testing
        import hashlib
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Convert to 768 floats between -1 and 1
        embedding = [(b / 127.5 - 1.0) for b in hash_bytes * 24][:768]

        return {
            "embedding": embedding,
            "model": "hash-based",
            "dimensions": 768,
            "text_length": len(text)
        }

    async def run_classification(self, input_data: Dict) -> Dict:
        """Run classification task using Ollama"""
        text = input_data.get("text", "")
        categories = input_data.get("categories", [])

        if not categories:
            categories = ["positive", "negative", "neutral"]

        prompt = f"""Classify the following text into ONE of these categories: {', '.join(categories)}

Text: "{text}"

Respond with ONLY the category name, nothing else."""

        result = await self.run_inference({"prompt": prompt})
        result_clean = result.strip().lower()

        # Find best matching category
        best_match = categories[0]
        for cat in categories:
            if cat.lower() in result_clean:
                best_match = cat
                break

        return {
            "category": best_match,
            "raw_response": result.strip(),
            "categories": categories
        }

    async def run_summarization(self, input_data: Dict) -> str:
        """Run summarization task using Ollama"""
        text = input_data.get("text", "")
        max_length = input_data.get("maxLength", 150)

        prompt = f"""Summarize the following text in {max_length} characters or less:

{text}

Summary:"""

        return await self.run_inference({"prompt": prompt})

    async def run_generic(self, task_type: str, input_data: Dict) -> str:
        """Run generic task using Ollama"""
        prompt = input_data.get("prompt", "")
        if not prompt:
            prompt = f"Process this {task_type} task: {json.dumps(input_data)}"

        return await self.run_inference({"prompt": prompt})


# ============================================================================
# Main Daemon Loop
# ============================================================================

class HybridAgentDaemon:
    """Main daemon that orchestrates project tasks, compute tasks, and local research"""

    def __init__(self, workers_url: str, poll_interval: int = 30):
        self.state = AgentState()
        self.workers = WorkersClient(workers_url, self.state)
        self.research = LocalResearchManager(self.state, self.workers)
        self.project_executor = ProjectTaskExecutor(self.state, self.workers)  # TIER 0
        self.compute_executor = TaskExecutor(self.state)  # TIER 1
        self.poll_interval = poll_interval
        self.running = True
        self.sync_interval = 300  # Sync research objectives every 5 minutes
        self.last_sync = 0

        # Handle signals for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received")
        self.running = False

    async def run(self):
        """Main daemon loop"""
        logger.info("Hybrid Agent Daemon starting...")
        logger.info(f"Workers URL: {self.workers.base_url}")
        logger.info(f"Poll interval: {self.poll_interval}s")

        # Try to register with Workers
        self.state.device_id = await self.workers.register_device()
        if self.state.device_id:
            logger.info(f"Registered with Workers: {self.state.device_id}")
        else:
            logger.warning("Running in offline mode (no Workers connection)")

        self.state.save_state()

        while self.running:
            try:
                await self.tick()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            # Wait before next tick
            await asyncio.sleep(self.poll_interval)

        logger.info("Hybrid Agent Daemon stopped")

    async def tick(self):
        """Single tick of the daemon loop"""

        # Check daily limits
        if not self.state.can_accept_work():
            logger.info("Daily limits reached, sleeping...")
            return

        # Send heartbeat (keeps device online)
        if self.state.device_id:
            await self.workers.heartbeat()

        # =====================================================================
        # TIER 0: Project Tasks (HIGHEST PRIORITY - Main Work Driver)
        # These are the Firefly Network tasks from the tasks table
        # =====================================================================
        if self.state.device_id and self.state.status == "idle":
            pending_project_tasks = await self.workers.get_pending_project_tasks()
            if pending_project_tasks:
                task = pending_project_tasks[0]
                logger.info(f"Found project task: {task.get('title')}")

                # Claim the task
                claimed = await self.workers.claim_project_task(task["id"])
                if claimed:
                    logger.info(f"Claimed project task: {claimed.get('title')}")
                    await self.process_project_task(claimed)
                    return

        # =====================================================================
        # TIER 1: Compute Tasks (Workers Coordination Layer)
        # These are compute_tasks dispatched by Workers workflows
        # =====================================================================
        if self.state.device_id and self.state.status == "idle":
            # Check for pending compute tasks to claim
            pending_compute_tasks = await self.workers.get_pending_tasks()
            if pending_compute_tasks:
                task = pending_compute_tasks[0]
                claimed = await self.workers.claim_task(task["id"])
                if claimed:
                    logger.info(f"Claimed compute task: {claimed.get('id')}")
                    await self.process_workers_task(claimed)
                    return

        # Periodic sync of research objectives from Workers
        current_time = time.time()
        if current_time - self.last_sync > self.sync_interval:
            await self.research.fetch_workers_objectives()
            self.last_sync = current_time

        # =====================================================================
        # TIER 2: Local Research (Background, Lower Priority)
        # Only when no project or compute tasks are available
        # =====================================================================
        if self.state.can_do_research() and self.state.status == "idle":
            research_task = self.research.get_next_research_task()
            if research_task:
                logger.info(f"Starting local research: {research_task.get('title')}")
                self.state.status = "working"

                results = await self.research.execute_research(research_task)
                self.research.save_research_results(research_task, results)

                # Sync results to Workers
                await self.research.sync_to_workers(research_task, results)

                self.state.status = "idle"
                logger.info(f"Local research completed: {research_task.get('id')}")
                return

        # Nothing to do, idle
        logger.debug("No work available, idling...")

    async def process_project_task(self, task: Dict):
        """Process a project task (TIER 0 - Main Work Driver)"""
        self.state.status = "working"
        self.state.current_task = task

        try:
            result = await self.project_executor.execute(task)

            if result.get("success"):
                # Mark task as completed
                await self.workers.complete_project_task(
                    task["id"],
                    result=result.get("output", "Task completed")
                )
                logger.info(f"Project task completed: {task.get('title')}")
            else:
                # Update with error status
                await self.workers.update_project_task_progress(
                    task["id"],
                    progress=0,
                    status="pending",  # Return to pending for retry
                    result=f"Error: {result.get('error')}"
                )
                logger.error(f"Project task failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error processing project task: {e}")
            # Try to update status
            try:
                await self.workers.update_project_task_progress(
                    task["id"],
                    progress=0,
                    status="pending",
                    result=f"Exception: {str(e)}"
                )
            except:
                pass

        finally:
            self.state.status = "idle"
            self.state.current_task = None
            self.state.save_state()

    async def process_workers_task(self, task: Dict):
        """Process a compute task from Workers (TIER 1)"""
        self.state.status = "working"
        self.state.current_task = task

        try:
            result = await self.compute_executor.execute(task)

            # Report result back to Workers
            await self.workers.complete_task(
                task["id"],
                success=result.get("success", False),
                result=result if result.get("success") else None,
                error=result.get("error")
            )

        finally:
            self.state.status = "idle"
            self.state.current_task = None
            self.state.save_state()


# ============================================================================
# Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hybrid Agent Daemon for 4090")
    parser.add_argument(
        "--workers-url",
        default=DEFAULT_WORKERS_URL,
        help="Workers API base URL"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between Workers API polls"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    daemon = HybridAgentDaemon(
        workers_url=args.workers_url,
        poll_interval=args.poll_interval
    )

    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
