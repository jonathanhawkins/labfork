#!/usr/bin/env python3
"""
Research Manager - Orchestrates AI agents in tmux sessions

Usage:
    python manager.py spawn --type codex --name "my-agent" --task "Do something"
    python manager.py status
    python manager.py read --name "my-agent"
    python manager.py kill --name "my-agent"
    python manager.py remind --in 5m --message "Check progress"
    python manager.py sleep --seconds 60
    python manager.py wait --agent "my-agent"
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal

# Import labs module for multi-lab support
import labs as labs_module

# Paths
SKILL_DIR = Path(__file__).parent
TEMPLATES_DIR = SKILL_DIR / "templates"

# Lab-aware state directories (resolved dynamically)
_current_lab_id: Optional[str] = None  # Can be overridden via --lab flag


def get_current_lab_id() -> str:
    """Get the current lab ID (from override, env, or active lab)."""
    global _current_lab_id
    return labs_module.get_effective_lab_id(_current_lab_id)


def get_state_dir() -> Path:
    """Get the state directory for the current lab."""
    lab_id = get_current_lab_id()
    lab = labs_module.get_lab(lab_id)
    if lab:
        return labs_module.get_lab_state_dir(lab_id)
    # Fallback to global state for backwards compatibility
    return SKILL_DIR / "state"


def get_agents_file() -> Path:
    return get_state_dir() / "agents.json"


def get_reminders_file() -> Path:
    return get_state_dir() / "reminders.json"


def get_outputs_dir() -> Path:
    outputs = get_state_dir() / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    return outputs


def get_proposals_file() -> Path:
    return get_state_dir() / "proposals.json"


def get_task_list_id() -> str:
    """Get the Claude Code task list ID for the current lab."""
    return labs_module.get_task_list_id(get_current_lab_id())


# Legacy aliases for backwards compatibility
STATE_DIR = SKILL_DIR / "state"  # Used only for initialization
AGENTS_FILE = STATE_DIR / "agents.json"  # Legacy - use get_agents_file()
REMINDERS_FILE = STATE_DIR / "reminders.json"  # Legacy
OUTPUTS_DIR = STATE_DIR / "outputs"  # Legacy - use get_outputs_dir()
PROPOSALS_FILE = STATE_DIR / "proposals.json"  # Legacy

# Ensure base directories exist
STATE_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


def load_agents() -> dict:
    """Load agent registry from file."""
    agents_file = get_agents_file()
    if agents_file.exists():
        return json.loads(agents_file.read_text())
    return {}


def save_agents(agents: dict):
    """Save agent registry to file."""
    agents_file = get_agents_file()
    agents_file.parent.mkdir(parents=True, exist_ok=True)
    agents_file.write_text(json.dumps(agents, indent=2, default=str))


def load_reminders() -> list:
    """Load reminders from file."""
    reminders_file = get_reminders_file()
    if reminders_file.exists():
        return json.loads(reminders_file.read_text())
    return []


def save_reminders(reminders: list):
    """Save reminders to file."""
    reminders_file = get_reminders_file()
    reminders_file.parent.mkdir(parents=True, exist_ok=True)
    reminders_file.write_text(json.dumps(reminders, indent=2, default=str))


def tmux_session_exists(name: str) -> bool:
    """Check if a tmux session exists."""
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        capture_output=True
    )
    return result.returncode == 0


def get_tmux_sessions() -> list[str]:
    """Get list of all tmux sessions."""
    result = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        return []
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def spawn_agent(
    agent_type: Literal["codex", "ollama", "opus"],
    name: str,
    task: str,
    working_dir: Optional[str] = None,
    auto_rename: bool = True
) -> dict:
    """
    Spawn a new agent in a tmux session.

    Args:
        agent_type: "codex" for OpenAI Codex, "ollama" for local Claude Code via Ollama
        name: Unique name for this agent
        task: The task/prompt to give the agent
        working_dir: Working directory for the agent (defaults to project root)
        auto_rename: Whether to run /rename command for Claude agents

    Returns:
        Agent info dict
    """
    # CRITICAL: Prepend rules and limits to every task
    task_prefix = """⚠️ CRITICAL RULES:
1. NEVER modify CLAUDE.md - document in docstrings, TaskUpdate, or config comments
2. STAY FOCUSED - complete the specific task, don't expand scope
3. LIMIT WEB SEARCHES - max 5 searches per task, be specific
4. TIME LIMIT - you have ~30 minutes, wrap up and report findings
5. NO ENDLESS LOOPS - if stuck after 3 attempts, report blockers and stop

TASK: """
    task = task_prefix + task

    # Validate name doesn't exist
    agents = load_agents()
    session_name = f"rm-{name}"  # rm = research manager prefix

    if name in agents and tmux_session_exists(session_name):
        raise ValueError(f"Agent '{name}' already exists and is running")

    # Set working directory
    if working_dir is None:
        working_dir = str(SKILL_DIR.parent.parent)  # Project root

    # Output file for this agent
    outputs_dir = get_outputs_dir()
    output_file = outputs_dir / f"{name}.log"

    # Initialize the output file with metadata
    output_file.write_text(f"""=== Research Manager Agent Log ===
Agent: {name}
Type: {agent_type}
Task: {task}
Started: {datetime.now().isoformat()}
Working Dir: {working_dir}
Session: {session_name}
{'=' * 40}

""")

    # Build the command based on agent type
    # Use 'script' command to provide a PTY for CLIs that need it
    # Set CLAUDE_CODE_TASK_LIST_ID so all agents share the same task list
    task_list_id = get_task_list_id()
    lab_id = get_current_lab_id()
    env_prefix = f'export CLAUDE_CODE_TASK_LIST_ID="{task_list_id}" && export CLAUDE_CODE_LAB_ID="{lab_id}" && '

    def ansi_c_quote(text: str) -> str:
        escaped = (
            text.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\r", "\\r")
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )
        return f"$'{escaped}'"

    def build_script_command(command: str, log_file: Path) -> str:
        return f'{env_prefix}script -q -a "{log_file}" -c {ansi_c_quote(command)}'

    def resolve_ollama_command(base_dir: str) -> str:
        """Pick a local Claude Code runner that uses Ollama (no Anthropic key)."""
        # Prefer repo script for portability
        repo_cmd = Path(base_dir) / "scripts" / "claude-free"
        if repo_cmd.exists():
            return str(repo_cmd)
        # Fallback to 4090 lab-manager if present
        lab_cmd = Path.home() / "bin" / "lab-manager"
        if lab_cmd.exists():
            return str(lab_cmd)
        # Check ~/bin/claude wrapper
        bin_claude = Path.home() / "bin" / "claude"
        if bin_claude.exists():
            return str(bin_claude)
        # Last resort: plain claude (expects env vars to point to Ollama)
        return "claude"

    def resolve_codex_command() -> Optional[str]:
        """Find codex CLI even if PATH is missing (e.g., NVM installs)."""
        env_path = os.environ.get("CODEX_PATH")
        if env_path:
            resolved = Path(env_path).expanduser()
            if resolved.exists():
                return str(resolved)
        # Check ~/bin/codex first
        bin_codex = Path.home() / "bin" / "codex"
        if bin_codex.exists():
            return str(bin_codex)
        found = shutil.which("codex")
        if found:
            return found
        nvm_dir = Path.home() / ".nvm" / "versions" / "node"
        if nvm_dir.exists():
            candidates = sorted(nvm_dir.glob("*/bin/codex"))
            if candidates:
                return str(candidates[-1])
        return None

    if agent_type == "codex":
        # Use spawn script to avoid quoting issues with multi-line prompts
        task_file = outputs_dir / f"{name}.task"
        task_file.write_text(task)
        spawn_script = SKILL_DIR / "spawn-codex-agent.sh"
        agent_cmd = f'bash "{spawn_script}" "{task_file}" "{output_file}"'
    elif agent_type in {"ollama", "opus"}:
        # Local Claude Code via Ollama (no Anthropic key). "opus" kept as alias for backward compatibility.
        # Use simplified spawn script to avoid quoting issues
        task_file = outputs_dir / f"{name}.task"
        task_file.write_text(task)
        spawn_script = SKILL_DIR / "spawn-ollama-agent.sh"
        agent_cmd = f'bash "{spawn_script}" "{task_file}" "{output_file}"'
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Create tmux session with the agent command
    # First create the session
    subprocess.run([
        "tmux", "new-session", "-d", "-s", session_name, "-c", working_dir
    ], check=True)

    # Send the command to the session
    subprocess.run([
        "tmux", "send-keys", "-t", session_name, agent_cmd, "Enter"
    ], check=True)

    # For local Claude Code agents, send /rename after CLI starts
    # DISABLED: This causes syntax errors when script command exits before CLI starts
    # if agent_type in {"ollama", "opus"} and auto_rename:
    #     # Give the CLI time to start
    #     time.sleep(2)
    #     rename_cmd = f"/rename RM:{name}"
    #     subprocess.run([
    #         "tmux", "send-keys", "-t", session_name, rename_cmd, "Enter"
    #     ], check=False)  # Don't fail if this doesn't work

    # Record agent info
    agent_info = {
        "name": name,
        "type": agent_type,
        "task": task,
        "session": session_name,
        "output_file": str(output_file),
        "working_dir": working_dir,
        "started_at": datetime.now().isoformat(),
        "status": "running"
    }

    agents[name] = agent_info
    save_agents(agents)

    print(f"Spawned {agent_type} agent '{name}' in tmux session '{session_name}'")
    print(f"Output: {output_file}")
    print(f"Attach with: tmux attach -t {session_name}")

    return agent_info


def get_agent_status(name: Optional[str] = None) -> dict | list:
    """
    Get status of agent(s).

    Args:
        name: Specific agent name, or None for all agents

    Returns:
        Agent info dict or list of all agents
    """
    agents = load_agents()

    # Update status based on tmux session existence
    # Don't overwrite "killed" status
    for agent_name, info in agents.items():
        if info.get("status") == "killed":
            continue  # Preserve killed status
        session_name = info.get("session", f"rm-{agent_name}")
        if tmux_session_exists(session_name):
            info["status"] = "running"
        else:
            info["status"] = "stopped"

    save_agents(agents)

    if name:
        if name not in agents:
            raise ValueError(f"Agent '{name}' not found")
        return agents[name]

    return list(agents.values())


def read_agent_output(name: str, tail: int = 100) -> str:
    """
    Read output from an agent.

    Args:
        name: Agent name
        tail: Number of lines from end to read (0 for all)

    Returns:
        Agent output text
    """
    agents = load_agents()
    if name not in agents:
        raise ValueError(f"Agent '{name}' not found")

    output_file = Path(agents[name]["output_file"])
    if not output_file.exists():
        return "(No output yet)"

    content = output_file.read_text()
    if tail > 0:
        lines = content.split("\n")
        content = "\n".join(lines[-tail:])

    return content


def send_to_agent(name: str, message: str):
    """
    Send a message/command to a running agent.

    Args:
        name: Agent name
        message: Text to send (will press Enter after)
    """
    agents = load_agents()
    if name not in agents:
        raise ValueError(f"Agent '{name}' not found")

    session_name = agents[name].get("session", f"rm-{name}")
    if not tmux_session_exists(session_name):
        raise ValueError(f"Agent '{name}' is not running")

    # Send the message
    subprocess.run([
        "tmux", "send-keys", "-t", session_name, message, "Enter"
    ], check=True)

    print(f"Sent to '{name}': {message}")


def kill_agent(name: str):
    """
    Kill an agent and its tmux session.

    Args:
        name: Agent name
    """
    agents = load_agents()
    if name not in agents:
        raise ValueError(f"Agent '{name}' not found")

    session_name = agents[name].get("session", f"rm-{name}")

    if tmux_session_exists(session_name):
        subprocess.run(["tmux", "kill-session", "-t", session_name], check=True)
        print(f"Killed tmux session '{session_name}'")

    # Update status
    agents[name]["status"] = "killed"
    agents[name]["killed_at"] = datetime.now().isoformat()
    save_agents(agents)

    print(f"Agent '{name}' killed")


def kill_all_agents():
    """Kill all managed agents."""
    agents = load_agents()
    for name in list(agents.keys()):
        try:
            kill_agent(name)
        except Exception as e:
            print(f"Error killing {name}: {e}")


def parse_duration(duration_str: str) -> timedelta:
    """
    Parse duration string like "5m", "1h", "30s" into timedelta.
    """
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])

    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError(f"Unknown duration unit: {unit}")


def create_reminder(message: str, trigger_at: Optional[str] = None, trigger_in: Optional[str] = None) -> dict:
    """
    Create a reminder.

    Args:
        message: Reminder message
        trigger_at: ISO timestamp or "HH:MM" for when to trigger
        trigger_in: Duration string like "5m", "1h" for relative trigger

    Returns:
        Reminder info dict
    """
    reminders = load_reminders()

    if trigger_in:
        delta = parse_duration(trigger_in)
        trigger_time = datetime.now() + delta
    elif trigger_at:
        # Try parsing as ISO format first
        try:
            trigger_time = datetime.fromisoformat(trigger_at)
        except ValueError:
            # Try as HH:MM for today
            trigger_time = datetime.strptime(trigger_at, "%H:%M").replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day
            )
    else:
        raise ValueError("Must specify either trigger_at or trigger_in")

    reminder = {
        "id": len(reminders) + 1,
        "message": message,
        "trigger_at": trigger_time.isoformat(),
        "created_at": datetime.now().isoformat(),
        "triggered": False
    }

    reminders.append(reminder)
    save_reminders(reminders)

    print(f"Reminder #{reminder['id']} set for {trigger_time}")
    print(f"Message: {message}")

    return reminder


def check_reminders() -> list[dict]:
    """
    Check for triggered reminders.

    Returns:
        List of reminders that have triggered since last check
    """
    reminders = load_reminders()
    now = datetime.now()
    triggered = []

    for reminder in reminders:
        if reminder["triggered"]:
            continue

        trigger_time = datetime.fromisoformat(reminder["trigger_at"])
        if now >= trigger_time:
            reminder["triggered"] = True
            reminder["triggered_at"] = now.isoformat()
            triggered.append(reminder)

    save_reminders(reminders)

    return triggered


def list_reminders() -> list[dict]:
    """List all reminders."""
    return load_reminders()


def sleep_for(seconds: int):
    """
    Sleep for a duration, checking reminders periodically.

    Args:
        seconds: Number of seconds to sleep
    """
    print(f"Sleeping for {seconds} seconds...")
    start = time.time()

    while time.time() - start < seconds:
        # Check reminders every 5 seconds
        triggered = check_reminders()
        for r in triggered:
            print(f"\n[REMINDER #{r['id']}] {r['message']}")

        remaining = seconds - (time.time() - start)
        sleep_time = min(5, remaining)
        if sleep_time > 0:
            time.sleep(sleep_time)

    print("Woke up!")


def wait_for_agent(name: str, timeout: int = 3600, poll_interval: int = 10):
    """
    Wait for an agent to complete (tmux session to end).

    Args:
        name: Agent name
        timeout: Maximum seconds to wait
        poll_interval: Seconds between checks
    """
    agents = load_agents()
    if name not in agents:
        raise ValueError(f"Agent '{name}' not found")

    session_name = agents[name].get("session", f"rm-{name}")
    print(f"Waiting for agent '{name}' to complete...")

    start = time.time()
    while time.time() - start < timeout:
        # Check if session still exists
        if not tmux_session_exists(session_name):
            print(f"Agent '{name}' completed!")
            return True

        # Check reminders
        triggered = check_reminders()
        for r in triggered:
            print(f"\n[REMINDER #{r['id']}] {r['message']}")

        time.sleep(poll_interval)

    print(f"Timeout waiting for agent '{name}'")
    return False


def wait_for_file(filepath: str, timeout: int = 3600, poll_interval: int = 5):
    """
    Wait for a file to exist.

    Args:
        filepath: Path to file
        timeout: Maximum seconds to wait
        poll_interval: Seconds between checks
    """
    print(f"Waiting for file '{filepath}'...")

    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(filepath):
            print(f"File '{filepath}' appeared!")
            return True

        # Check reminders
        triggered = check_reminders()
        for r in triggered:
            print(f"\n[REMINDER #{r['id']}] {r['message']}")

        time.sleep(poll_interval)

    print(f"Timeout waiting for file '{filepath}'")
    return False


def clear_agents():
    """Clear the agent registry (doesn't kill running agents)."""
    save_agents({})
    print("Agent registry cleared")


def clear_reminders():
    """Clear all reminders."""
    save_reminders([])
    print("Reminders cleared")


# =============================================================================
# Web Search and Research Tools
# =============================================================================

def search_web(query: str, papers_only: bool = False, max_results: int = 5):
    """
    Perform a web search using ddgr (DuckDuckGo CLI) or fallback.
    """
    if papers_only:
        query = f"{query} site:arxiv.org OR site:paperswithcode.com OR site:scholar.google.com"

    # Try using ddgr (DuckDuckGo CLI)
    try:
        result = subprocess.run(
            ["ddgr", "--np", "-n", str(max_results), query],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(result.stdout)
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: suggest using the search in an agent
    print(f"Web search query: {query}")
    print()
    print("To perform this search, spawn a local agent with web access:")
    print(f'  .skills/research-manager/rm spawn --type ollama --name "web-search" \\')
    print(f'    --task "Search the web for: {query}. Summarize the top {max_results} results."')
    print()
    print("Or install ddgr for CLI search: brew install ddgr")


def search_github(query: str, language: Optional[str] = None, sort: str = "stars", max_results: int = 10):
    """
    Search GitHub repositories using the gh CLI.
    """
    cmd = ["gh", "search", "repos", query, "--limit", str(max_results), f"--sort={sort}"]

    if language:
        cmd.extend(["--language", language])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"GitHub search: {query}")
            print("=" * 60)
            print(result.stdout)
        else:
            print(f"Error: {result.stderr}")
    except FileNotFoundError:
        print("Error: gh CLI not found. Install with: brew install gh")
    except subprocess.TimeoutExpired:
        print("Error: GitHub search timed out")


def search_arxiv(query: str, max_results: int = 5):
    """
    Search arxiv for papers using the API.
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read().decode('utf-8')

        # Parse XML
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        entries = root.findall("atom:entry", ns)

        print(f"ArXiv search: {query}")
        print("=" * 60)

        if not entries:
            print("No papers found")
            return

        for i, entry in enumerate(entries, 1):
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            link = entry.find("atom:id", ns).text
            published = entry.find("atom:published", ns).text[:10]
            summary = entry.find("atom:summary", ns).text.strip()[:200].replace("\n", " ")

            authors = entry.findall("atom:author/atom:name", ns)
            author_names = ", ".join([a.text for a in authors[:3]])
            if len(authors) > 3:
                author_names += f" et al. ({len(authors)} authors)"

            print(f"\n{i}. {title}")
            print(f"   Authors: {author_names}")
            print(f"   Date: {published}")
            print(f"   Link: {link}")
            print(f"   Summary: {summary}...")

    except Exception as e:
        print(f"Error searching arxiv: {e}")


def fetch_url(url: str, raw: bool = False):
    """
    Fetch and display content from a URL.
    """
    import urllib.request

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode('utf-8', errors='ignore')

        if raw:
            print(content[:10000])
        else:
            # Basic HTML stripping
            import re
            text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()[:5000]
            print(text)

    except Exception as e:
        print(f"Error fetching URL: {e}")


# =============================================================================
# RTX 4090 Remote Training Management
# =============================================================================

RTX_4090_HOST = f"{os.environ.get('REMOTE_GPU_USER', 'doc')}@{os.environ.get('REMOTE_GPU_HOST', '')}"
RTX_4090_PROJECT = os.environ.get("REMOTE_GPU_PROJECT", "~/dev/voice-clone-pipeline")
RTX_4090_CONDA = os.environ.get("REMOTE_GPU_CONDA", "voice")


def rtx_run(command: str, background: bool = False) -> str:
    """
    Run a command on the RTX 4090 machine.
    """
    full_cmd = f"source ~/miniconda3/bin/activate && conda activate {RTX_4090_CONDA} && cd {RTX_4090_PROJECT} && {command}"

    if background:
        # Use nohup and redirect to a log file
        ssh_cmd = f'ssh {RTX_4090_HOST} "nohup bash -c \'{full_cmd}\' > /tmp/training.log 2>&1 &"'
    else:
        ssh_cmd = f'ssh {RTX_4090_HOST} "{full_cmd}"'

    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True, timeout=300)
    return result.stdout + result.stderr


def rtx_status():
    """Check GPU and training status on RTX 4090."""
    print("RTX 4090 Status")
    print("=" * 60)

    # Check if reachable
    gpu_host = os.environ.get("REMOTE_GPU_HOST", "")
    if not gpu_host:
        print("Status: NOT CONFIGURED (set REMOTE_GPU_HOST env var)")
        return
    ping = subprocess.run(
        ["ping", "-c", "1", "-t", "5", gpu_host],
        capture_output=True
    )
    if ping.returncode != 0:
        print("Status: OFFLINE (host unreachable)")
        print("Tip: Make sure Tailscale is connected")
        return

    print("Status: ONLINE")
    print()

    # Get GPU info
    try:
        result = subprocess.run(
            f'ssh {RTX_4090_HOST} "/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"',
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"GPU: {gpu_info}")
    except:
        print("GPU: Unable to query")

    print()

    # Check for running training
    try:
        result = subprocess.run(
            f'ssh {RTX_4090_HOST} "pgrep -f train_ || echo none"',
            shell=True, capture_output=True, text=True, timeout=10
        )
        if "none" not in result.stdout:
            print("Training: RUNNING")
            print(f"  PIDs: {result.stdout.strip()}")
        else:
            print("Training: IDLE")
    except:
        print("Training: Unable to check")


def rtx_train(config: str = "rtx_4090_lora.yaml", background: bool = True):
    """Start training on RTX 4090."""
    cmd = f"cd training && python train_lora_deepseek.py --config config/{config}"

    print(f"Starting training on RTX 4090...")
    print(f"Config: {config}")
    print(f"Background: {background}")
    print()

    if background:
        # Create tmux session for training
        session_name = f"training-{int(time.time())}"
        ssh_cmd = f'ssh {RTX_4090_HOST} "source ~/miniconda3/bin/activate && conda activate {RTX_4090_CONDA} && cd {RTX_4090_PROJECT} && tmux new-session -d -s {session_name} \\"{cmd}\\""'

        result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Training started in tmux session: {session_name}")
            print(f"Attach with: ssh {RTX_4090_HOST} -t 'tmux attach -t {session_name}'")
        else:
            print(f"Error: {result.stderr}")
    else:
        result = rtx_run(cmd)
        print(result)


def rtx_sync(direction: str = "push"):
    """Sync project files with RTX 4090."""
    local_path = str(SKILL_DIR.parent.parent) + "/"
    remote_path = f"{RTX_4090_HOST}:{RTX_4090_PROJECT}/"

    excludes = "--exclude 'node_modules' --exclude '.next' --exclude 'venv' --exclude '__pycache__' --exclude '.git' --exclude 'checkpoints'"

    if direction == "push":
        cmd = f"rsync -avz --progress {excludes} {local_path} {remote_path}"
        print("Syncing local -> RTX 4090...")
    else:
        cmd = f"rsync -avz --progress {excludes} {remote_path} {local_path}"
        print("Syncing RTX 4090 -> local...")

    subprocess.run(cmd, shell=True)


# =============================================================================
# Testing and Validation Tools
# =============================================================================

def validate_code(module_path: str, verbose: bool = True) -> dict:
    """
    Validate Python code by attempting to import it.

    Args:
        module_path: Path to Python file or module
        verbose: Print detailed output

    Returns:
        Dict with 'success', 'errors', 'warnings'
    """
    result = {"success": True, "errors": [], "warnings": []}

    # First, run Python syntax check
    if verbose:
        print(f"Validating: {module_path}")
        print("-" * 40)

    # Syntax check
    check = subprocess.run(
        ["python3", "-m", "py_compile", module_path],
        capture_output=True, text=True
    )
    if check.returncode != 0:
        result["success"] = False
        result["errors"].append(f"Syntax error: {check.stderr}")
        if verbose:
            print(f"❌ Syntax error:\n{check.stderr}")
        return result

    if verbose:
        print("✅ Syntax OK")

    # Try importing
    module_dir = str(Path(module_path).parent)
    module_name = Path(module_path).stem

    import_check = subprocess.run(
        ["python3", "-c", f"import sys; sys.path.insert(0, '{module_dir}'); import {module_name}"],
        capture_output=True, text=True,
        cwd=module_dir,
        timeout=60
    )

    if import_check.returncode != 0:
        # Check if it's a real error or just a warning
        stderr = import_check.stderr
        if "Error" in stderr or "error" in stderr.lower():
            result["success"] = False
            result["errors"].append(f"Import error: {stderr}")
            if verbose:
                print(f"❌ Import error:\n{stderr}")
        else:
            result["warnings"].append(stderr)
            if verbose:
                print(f"⚠️  Warning: {stderr}")
    else:
        if verbose:
            print("✅ Import OK")

    return result


def run_tests(test_path: str = None, pattern: str = "test_*.py", verbose: bool = True) -> dict:
    """
    Run pytest tests.

    Args:
        test_path: Path to test file or directory (default: project tests/)
        pattern: Test file pattern
        verbose: Print detailed output

    Returns:
        Dict with 'success', 'passed', 'failed', 'output'
    """
    project_root = SKILL_DIR.parent.parent

    if test_path is None:
        # Default test locations
        for loc in ["tests", "test", "training/tests"]:
            test_dir = project_root / loc
            if test_dir.exists():
                test_path = str(test_dir)
                break

    if test_path is None:
        return {"success": False, "error": "No test directory found"}

    if verbose:
        print(f"Running tests: {test_path}")
        print("-" * 40)

    cmd = ["python3", "-m", "pytest", test_path, "-v", "--tb=short"]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))

    output = result.stdout + result.stderr

    # Parse results
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    errors = output.count(" ERROR")

    success = result.returncode == 0

    if verbose:
        print(output)
        print("-" * 40)
        if success:
            print(f"✅ Tests passed: {passed}")
        else:
            print(f"❌ Tests failed: {failed}, errors: {errors}")

    return {
        "success": success,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "output": output
    }


def quick_test(script_path: str, args: list = None, timeout: int = 30) -> dict:
    """
    Run a quick test of a script with optional arguments.

    Args:
        script_path: Path to Python script
        args: Additional arguments
        timeout: Timeout in seconds

    Returns:
        Dict with 'success', 'output', 'error'
    """
    cmd = ["python3", script_path]
    if args:
        cmd.extend(args)

    print(f"Running: {' '.join(cmd)}")
    print("-" * 40)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(SKILL_DIR.parent.parent)
        )

        output = result.stdout
        error = result.stderr
        success = result.returncode == 0

        if output:
            print(output[:2000])
        if error:
            print(f"STDERR:\n{error[:1000]}")

        print("-" * 40)
        if success:
            print("✅ Script completed successfully")
        else:
            print(f"❌ Script failed with code {result.returncode}")

        return {"success": success, "output": output, "error": error}

    except subprocess.TimeoutExpired:
        print(f"❌ Script timed out after {timeout}s")
        return {"success": False, "output": "", "error": f"Timeout after {timeout}s"}


# =============================================================================
# Autonomous Improvement Loop
# =============================================================================

def run_improvement_loop(
    config: str = "prosody_v5.yaml",
    max_iterations: int = 5,
    eval_threshold: float = 0.1,  # Stop if F0 correlation > threshold
):
    """
    Autonomous training improvement loop.

    Cycle:
    1. Run training on RTX 4090
    2. Wait for completion
    3. Run evaluation
    4. Spawn Codex to analyze results
    5. Extract improvement suggestions
    6. Apply fixes
    7. Repeat until threshold met or max iterations

    Args:
        config: Training config file
        max_iterations: Maximum improvement cycles
        eval_threshold: F0 correlation threshold to stop (success)
    """
    project_root = SKILL_DIR.parent.parent

    print("=" * 60)
    print("    AUTONOMOUS IMPROVEMENT LOOP")
    print("=" * 60)
    print(f"Config: {config}")
    print(f"Max iterations: {max_iterations}")
    print(f"Success threshold: F0 correlation > {eval_threshold}")
    print("=" * 60)

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration}/{max_iterations}")
        print(f"{'='*60}\n")

        # Step 1: Sync code to RTX 4090
        print("[1/6] Syncing code to RTX 4090...")
        rtx_sync(direction="push")

        # Step 2: Start training
        print(f"\n[2/6] Starting training with {config}...")
        rtx_train(config=config, background=True)

        # Step 3: Wait for training (poll every 2 minutes)
        print("\n[3/6] Waiting for training to complete...")
        print("      (Polling every 2 minutes)")
        training_complete = False
        wait_minutes = 0
        max_wait_minutes = 120  # 2 hours max

        while not training_complete and wait_minutes < max_wait_minutes:
            time.sleep(120)  # 2 minutes
            wait_minutes += 2

            # Check if training process still running
            result = subprocess.run(
                f'ssh {RTX_4090_HOST} "pgrep -f train_ || echo DONE"',
                shell=True, capture_output=True, text=True, timeout=30
            )
            if "DONE" in result.stdout:
                training_complete = True
                print(f"      Training completed after ~{wait_minutes} minutes")
            else:
                print(f"      Still training... ({wait_minutes} min elapsed)")

        if not training_complete:
            print(f"      WARNING: Training timeout after {max_wait_minutes} minutes")

        # Step 4: Pull results and run evaluation
        print("\n[4/6] Running evaluation...")
        # Pull latest checkpoints
        rtx_sync(direction="pull")

        # Run evaluation script if it exists
        eval_script = project_root / "inference" / "evaluate_model.py"
        if eval_script.exists():
            subprocess.run(
                ["python3", str(eval_script)],
                cwd=str(project_root),
                timeout=300
            )

        # Check results
        results = update_landing_results(version=f"v5_iter{iteration}")
        if results is None:
            print("      No evaluation results found")
            continue

        avg_f0 = results.get("avg_f0_correlation", -1)
        print(f"      F0 Correlation: {avg_f0:.3f} (threshold: {eval_threshold})")

        # Check if we've met the success threshold
        if avg_f0 > eval_threshold:
            print(f"\n{'='*60}")
            print(f"  SUCCESS! F0 correlation {avg_f0:.3f} > {eval_threshold}")
            print(f"  Stopping after {iteration} iterations")
            print(f"{'='*60}")
            return {"success": True, "iterations": iteration, "f0_correlation": avg_f0}

        # Step 5: Spawn Codex to analyze and suggest improvements
        print("\n[5/6] Spawning Codex for analysis...")

        # Build analysis prompt with current results
        pitch_info = results.get("pitch_by_emotion", {})
        happy_pitch = pitch_info.get("happy", 0)
        sad_pitch = pitch_info.get("sad", 0)

        analysis_prompt = f"""Analyze V5 prosody training iteration {iteration} results and suggest fixes.

SHARED TASK LIST: You share a task list with other agents (CLAUDE_CODE_TASK_LIST_ID="voice-clone-pipeline").
1. First call TaskList to see existing tasks
2. Use TaskUpdate to mark your analysis task in_progress
3. When done, TaskUpdate with findings summary

CURRENT RESULTS:
- F0 Correlation: {avg_f0:.3f} (target: > {eval_threshold})
- Happy pitch: {happy_pitch:.0f} Hz
- Sad pitch: {sad_pitch:.0f} Hz
- Pitch separation: {happy_pitch - sad_pitch:.0f} Hz (should be positive, higher = better)

ISSUE: {"Pitch inversion - happy should be higher than sad!" if happy_pitch < sad_pitch else "Low F0 correlation - prosody not transferring well"}

Review these files:
1. training/train_prosody_conditioned.py - V5 training code
2. training/config/prosody_v5.yaml - Current config

Suggest SPECIFIC code changes to fix the issue. Focus on:
- Loss weights and schedules
- Gradient flow (detachment)
- Curriculum sampling thresholds
- Early stopping criteria

Output your suggestions as a numbered list of specific changes."""

        codex_name = f"analyzer-iter{iteration}"
        spawn_agent("codex", codex_name, analysis_prompt)

        # Wait for Codex analysis (max 10 minutes)
        print("      Waiting for Codex analysis (max 10 min)...")
        wait_for_agent(codex_name, timeout=600, poll_interval=30)

        # Read Codex output
        analysis = read_agent_output(codex_name, tail=200)
        print(f"\n      Codex Analysis:\n{'-'*40}")
        print(analysis[-2000:] if len(analysis) > 2000 else analysis)
        print(f"{'-'*40}")

        # Kill Codex agent
        try:
            kill_agent(codex_name)
        except:
            pass

        # Step 6: Apply suggestions (spawn local Ollama agent to implement)
        print("\n[6/6] Spawning Ollama agent to implement fixes...")

        impl_prompt = f"""Implement the following improvements to V5 prosody training.

SHARED TASK LIST: You share a task list with other agents (CLAUDE_CODE_TASK_LIST_ID="voice-clone-pipeline").
1. First call TaskList to see existing tasks
2. Use TaskUpdate to mark your implementation task in_progress
3. When done, TaskUpdate with changes summary

CODEX ANALYSIS:
{analysis[-3000:]}

YOUR TASK:
1. Read the suggested changes
2. Edit training/train_prosody_conditioned.py and/or training/config/prosody_v5.yaml
3. After EACH edit, run: .skills/research-manager/rm validate training/train_prosody_conditioned.py
4. If validation fails, fix immediately
5. When done, TaskUpdate to mark completed, then output "IMPLEMENTATION COMPLETE"

Focus on the TOP 2-3 most impactful changes only. Don't over-engineer."""

        impl_name = f"implementer-iter{iteration}"
        spawn_agent("ollama", impl_name, impl_prompt)

        # Wait for implementation (max 15 minutes)
        print("      Waiting for implementation (max 15 min)...")
        wait_for_agent(impl_name, timeout=900, poll_interval=30)

        # Read implementation result
        impl_output = read_agent_output(impl_name, tail=100)
        print(f"\n      Implementation result:\n{'-'*40}")
        print(impl_output[-1000:] if len(impl_output) > 1000 else impl_output)
        print(f"{'-'*40}")

        # Kill local agent
        try:
            kill_agent(impl_name)
        except:
            pass

        print(f"\n      Iteration {iteration} complete. Starting next cycle...")

    print(f"\n{'='*60}")
    print(f"  Loop completed after {max_iterations} iterations")
    print(f"  Final F0 correlation: {avg_f0:.3f}")
    print(f"{'='*60}")

    return {"success": False, "iterations": max_iterations, "f0_correlation": avg_f0}


# =============================================================================
# Training Watchdog - Real-time monitoring to stop wasted epochs
# =============================================================================

def parse_training_logs(log_content: str) -> dict:
    """
    Parse training logs to extract metrics per epoch.

    Returns dict with epoch -> {loss, val_loss, happy_pitch, sad_pitch, etc.}
    """
    import re

    epochs = {}
    current_epoch = None

    for line in log_content.split('\n'):
        # Match epoch start
        epoch_match = re.search(r'Epoch (\d+)', line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in epochs:
                epochs[current_epoch] = {}

        if current_epoch is None:
            continue

        # Match loss values
        loss_match = re.search(r'loss[=:]?\s*([\d.]+)', line, re.IGNORECASE)
        if loss_match:
            epochs[current_epoch]['loss'] = float(loss_match.group(1))

        val_loss_match = re.search(r'val[_\s]?loss[=:]?\s*([\d.]+)', line, re.IGNORECASE)
        if val_loss_match:
            epochs[current_epoch]['val_loss'] = float(val_loss_match.group(1))

        # Match pitch values
        happy_match = re.search(r'happy[=:]?\s*([\d.]+)\s*Hz', line, re.IGNORECASE)
        if happy_match:
            epochs[current_epoch]['happy_pitch'] = float(happy_match.group(1))

        sad_match = re.search(r'sad[=:]?\s*([\d.]+)\s*Hz', line, re.IGNORECASE)
        if sad_match:
            epochs[current_epoch]['sad_pitch'] = float(sad_match.group(1))

        # Match separation
        sep_match = re.search(r'separation[=:]?\s*([-\d.]+)', line, re.IGNORECASE)
        if sep_match:
            epochs[current_epoch]['pitch_separation'] = float(sep_match.group(1))

        # Match F0 correlation
        f0_match = re.search(r'f0[_\s]?corr(?:elation)?[=:]?\s*([-\d.]+)', line, re.IGNORECASE)
        if f0_match:
            epochs[current_epoch]['f0_correlation'] = float(f0_match.group(1))

    return epochs


def detect_overfitting(epochs: dict, patience: int = 3) -> dict:
    """
    Analyze epoch metrics to detect overfitting.

    Returns:
        {
            'is_overfitting': bool,
            'reason': str,
            'best_epoch': int,
            'current_epoch': int,
            'wasted_epochs': int,
            'recommendation': str
        }
    """
    if len(epochs) < 2:
        return {'is_overfitting': False, 'reason': 'Not enough epochs'}

    sorted_epochs = sorted(epochs.keys())
    current_epoch = sorted_epochs[-1]

    result = {
        'is_overfitting': False,
        'reason': None,
        'best_epoch': 1,
        'current_epoch': current_epoch,
        'wasted_epochs': 0,
        'recommendation': None
    }

    # Check for pitch inversion (happy < sad)
    for epoch in sorted_epochs[-patience:]:
        data = epochs.get(epoch, {})
        happy = data.get('happy_pitch', 0)
        sad = data.get('sad_pitch', 0)
        if happy > 0 and sad > 0 and happy < sad:
            result['is_overfitting'] = True
            result['reason'] = f'Pitch inversion: happy ({happy:.0f}Hz) < sad ({sad:.0f}Hz)'
            result['recommendation'] = 'STOP NOW - pitch patterns corrupted'
            break

    # Check for val_loss not improving
    val_losses = [(e, epochs[e].get('val_loss', float('inf'))) for e in sorted_epochs if 'val_loss' in epochs[e]]
    if len(val_losses) >= patience + 1:
        best_epoch, best_loss = min(val_losses, key=lambda x: x[1])
        result['best_epoch'] = best_epoch

        # Check if no improvement for `patience` epochs
        recent_losses = [v for e, v in val_losses[-patience:]]
        if all(v >= best_loss for v in recent_losses):
            result['wasted_epochs'] = current_epoch - best_epoch
            if result['wasted_epochs'] >= patience:
                result['is_overfitting'] = True
                result['reason'] = f'Val loss not improving for {result["wasted_epochs"]} epochs'
                result['recommendation'] = f'STOP - best model was at epoch {best_epoch}'

    # Check for loss divergence (sudden spike)
    losses = [(e, epochs[e].get('loss', 0)) for e in sorted_epochs if 'loss' in epochs[e]]
    if len(losses) >= 3:
        recent = [l for e, l in losses[-3:]]
        if recent[-1] > recent[0] * 1.5:  # 50% increase
            result['is_overfitting'] = True
            result['reason'] = f'Loss diverging: {recent[0]:.4f} -> {recent[-1]:.4f}'
            result['recommendation'] = 'STOP - training unstable'

    return result


def training_watchdog(
    poll_interval: int = 120,  # 2 minutes
    patience: int = 3,
    auto_kill: bool = False,
    max_runtime: int = 7200,  # 2 hours
):
    """
    Monitor training in real-time and detect overfitting early.

    This watchdog:
    1. Polls training logs every `poll_interval` seconds
    2. Parses metrics (loss, pitch, etc.)
    3. Detects overfitting patterns
    4. Optionally kills training if `auto_kill=True`

    Args:
        poll_interval: Seconds between log checks
        patience: Epochs without improvement before flagging
        auto_kill: If True, automatically kill training on overfit
        max_runtime: Maximum watchdog runtime in seconds
    """
    print("=" * 60)
    print("    TRAINING WATCHDOG")
    print("=" * 60)
    print(f"Poll interval: {poll_interval}s")
    print(f"Patience: {patience} epochs")
    print(f"Auto-kill: {auto_kill}")
    print(f"Max runtime: {max_runtime}s ({max_runtime//60} min)")
    print("=" * 60)

    start_time = time.time()
    last_epoch = 0
    alerts_sent = 0

    while time.time() - start_time < max_runtime:
        # Get training logs
        try:
            result = subprocess.run(
                f'ssh {RTX_4090_HOST} "cat {RTX_4090_PROJECT}/training/*.log 2>/dev/null | tail -500"',
                shell=True, capture_output=True, text=True, timeout=30
            )
            log_content = result.stdout
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error fetching logs: {e}")
            time.sleep(poll_interval)
            continue

        # Check if training is still running
        pid_check = subprocess.run(
            f'ssh {RTX_4090_HOST} "pgrep -f train_ || echo STOPPED"',
            shell=True, capture_output=True, text=True, timeout=10
        )
        if "STOPPED" in pid_check.stdout:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training completed or stopped")
            break

        # Parse logs
        epochs = parse_training_logs(log_content)
        if not epochs:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for training metrics...")
            time.sleep(poll_interval)
            continue

        current_epoch = max(epochs.keys())

        # Only report when new epoch detected
        if current_epoch > last_epoch:
            last_epoch = current_epoch
            data = epochs.get(current_epoch, {})

            # Print status
            status = f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {current_epoch}"
            if 'loss' in data:
                status += f" | loss={data['loss']:.4f}"
            if 'val_loss' in data:
                status += f" | val={data['val_loss']:.4f}"
            if 'happy_pitch' in data and 'sad_pitch' in data:
                sep = data['happy_pitch'] - data['sad_pitch']
                status += f" | pitch_sep={sep:.0f}Hz"
            print(status)

            # Check for overfitting
            analysis = detect_overfitting(epochs, patience=patience)

            if analysis['is_overfitting']:
                alerts_sent += 1
                print(f"\n{'!'*60}")
                print(f"  OVERFITTING DETECTED!")
                print(f"  Reason: {analysis['reason']}")
                print(f"  Best epoch: {analysis['best_epoch']}")
                print(f"  Wasted epochs: {analysis['wasted_epochs']}")
                print(f"  Recommendation: {analysis['recommendation']}")
                print(f"{'!'*60}\n")

                if auto_kill:
                    print("Auto-kill enabled. Stopping training...")
                    subprocess.run(
                        f'ssh {RTX_4090_HOST} "pkill -f train_"',
                        shell=True, timeout=10
                    )
                    print("Training killed.")
                    return {
                        'stopped_early': True,
                        'reason': analysis['reason'],
                        'best_epoch': analysis['best_epoch'],
                        'stopped_at_epoch': current_epoch,
                        'wasted_epochs': analysis['wasted_epochs']
                    }

        time.sleep(poll_interval)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Watchdog finished")
    return {'stopped_early': False, 'final_epoch': last_epoch, 'alerts': alerts_sent}


def start_loop_agent():
    """
    Spawn an autonomous loop agent that continuously improves training.

    This agent runs the improvement loop and keeps iterating until
    the model meets quality thresholds.
    """
    loop_prompt = '''You are the Autonomous Improvement Loop agent. Your job is to continuously improve prosody training until it works.

IMPORTANT: You have a SHARED TASK LIST with all other agents (CLAUDE_CODE_TASK_LIST_ID="voice-clone-pipeline").
Use the Claude Code Task tools to coordinate work:

TASK TOOLS (Claude built-in - use directly):
- TaskCreate: Create tasks with subject, description, activeForm
- TaskUpdate: Update status (pending/in_progress/completed), add dependencies
- TaskList: See all tasks and their status
- TaskGet: Get full details of a specific task

ALWAYS START BY:
1. TaskList - check for existing tasks
2. TaskCreate - create a master task for this improvement session
3. TaskUpdate - mark tasks in_progress when starting, completed when done

YOUR RESEARCH MANAGER TOOLS:
1. Training: .skills/research-manager/rm rtx status/sync/train/logs
2. Watchdog: .skills/research-manager/rm watchdog --auto-kill (STOPS WASTED EPOCHS)
3. Evaluate: .skills/research-manager/rm update-results --version vX
4. Validate: .skills/research-manager/rm validate <file.py>
5. Agents: .skills/research-manager/rm spawn/read/kill/wait
6. Research: .skills/research-manager/rm papers/github/search (SEARCH FOR NEW IDEAS!)

CRITICAL - USE WATCHDOG TO SAVE GPU TIME:
Run this IMMEDIATELY after starting training:
  .skills/research-manager/rm watchdog --auto-kill --patience 3

The watchdog will auto-kill training when overfitting detected.
V4 wasted 5 epochs (best was epoch 10, ran until 15). Don't repeat!

CRITICAL - SEARCH FOR NEW RESEARCH:
Before implementing fixes, ALWAYS search for recent papers:
  .skills/research-manager/rm papers prosody emotion TTS
  .skills/research-manager/rm papers disentanglement speech synthesis
  .skills/research-manager/rm github prosody conditioning pytorch

Example: DeepSeek-R1 uses pure RL without SFT. Could similar ideas work for prosody?

LOOP CYCLE (TRACK WITH TASKS):
1. TaskCreate: "Iteration N: V5 Training" with description
2. Check RTX 4090: .skills/research-manager/rm rtx status
3. Sync code: .skills/research-manager/rm rtx sync
4. Start training: .skills/research-manager/rm rtx train --config prosody_v5.yaml
5. START WATCHDOG: .skills/research-manager/rm watchdog --auto-kill --patience 3
6. When watchdog exits, evaluate: .skills/research-manager/rm update-results --version vX
7. If failed (pitch inversion or low F0):
   a. TaskCreate: "Analyze iteration N failure"
   b. SEARCH PAPERS: .skills/research-manager/rm papers <relevant query>
   c. Spawn Codex: .skills/research-manager/rm spawn --type codex --name "analyzer" --task "..."
   d. Read analysis: .skills/research-manager/rm read --name analyzer
   e. TaskUpdate: Mark analysis task completed with findings
   f. Implement fixes (validate after!)
8. TaskUpdate: Mark iteration task completed
9. Loop back to step 1

SUCCESS CRITERIA:
- Happy pitch > Sad pitch (no inversion)
- F0 correlation > 0.1
- Emotion accuracy > 50%

STOP WHEN: All criteria met OR 5 iterations completed.

You have full autonomy. BEGIN by calling TaskList, then TaskCreate for your session. GO!'''

    return spawn_agent("ollama", "auto-improver", loop_prompt)


# =============================================================================
# Landing Page Results Update
# =============================================================================

def update_landing_results(eval_dir: str = None, version: str = None, description: str = None):
    """
    Update the landing page with latest evaluation results.

    Args:
        eval_dir: Directory containing eval_*.json files (default: inference/)
        version: Version name for the results (e.g., "v4", "v5")
        description: Short description of this version

    Returns:
        Dict with updated metrics
    """
    project_root = SKILL_DIR.parent.parent
    if eval_dir is None:
        eval_dir = project_root / "inference"
    else:
        eval_dir = Path(eval_dir)

    # Collect all evaluation results
    eval_files = list(eval_dir.glob("eval_*.json"))
    if not eval_files:
        print(f"No eval_*.json files found in {eval_dir}")
        return None

    print(f"Found {len(eval_files)} evaluation files:")

    results = {}
    emotions_correct = 0
    emotions_total = 0

    for f in eval_files:
        name = f.stem.replace("eval_", "")
        with open(f) as fp:
            data = json.load(fp)
            results[name] = data
            print(f"  - {name}: F0 corr={data.get('prosody', {}).get('f0_correlation', 'N/A'):.3f}")

            # Check emotion accuracy if present
            if 'emotion_accuracy' in data:
                if data['emotion_accuracy']:
                    emotions_correct += 1
                emotions_total += 1

    # Calculate aggregate metrics
    f0_correlations = [r.get('prosody', {}).get('f0_correlation', 0) for r in results.values() if 'prosody' in r]
    avg_f0_corr = sum(f0_correlations) / len(f0_correlations) if f0_correlations else 0

    speaker_sims = [r.get('speaker_similarity', {}).get('cosine_similarity', 0) for r in results.values() if 'speaker_similarity' in r]
    avg_speaker_sim = sum(speaker_sims) / len(speaker_sims) if speaker_sims else 0

    emotion_accuracy = emotions_correct / emotions_total if emotions_total > 0 else None

    print(f"\nAggregate Metrics:")
    print(f"  Avg F0 Correlation: {avg_f0_corr:.3f}")
    print(f"  Avg Speaker Similarity: {avg_speaker_sim:.3f}")
    if emotion_accuracy is not None:
        print(f"  Emotion Accuracy: {emotions_correct}/{emotions_total} ({emotion_accuracy*100:.0f}%)")

    # Extract pitch info per emotion if available
    pitch_info = {}
    for name, data in results.items():
        if 'prosody' in data:
            f0_mean = data['prosody'].get('f0_mean_gen') or data['prosody'].get('f0_mean_diff', 0) + 150
            pitch_info[name] = f0_mean

    if pitch_info:
        print(f"\nPitch by Emotion:")
        for emotion, pitch in sorted(pitch_info.items(), key=lambda x: -x[1]):
            print(f"  {emotion}: {pitch:.0f} Hz")

    # Generate update summary
    summary = {
        "version": version or "latest",
        "description": description or "Evaluation results",
        "avg_f0_correlation": round(avg_f0_corr, 3),
        "avg_speaker_similarity": round(avg_speaker_sim, 3),
        "emotion_accuracy": f"{emotions_correct}/{emotions_total}" if emotions_total > 0 else "N/A",
        "pitch_by_emotion": pitch_info,
        "eval_files": [f.name for f in eval_files],
    }

    # Save summary
    summary_file = eval_dir / "results_summary.json"
    with open(summary_file, 'w') as fp:
        json.dump(summary, fp, indent=2)
    print(f"\nSaved summary to: {summary_file}")

    # Generate code snippet for landing page
    print(f"\n{'='*60}")
    print("Copy this to frontend/app/page.tsx (researchJourney phase 5):")
    print(f"{'='*60}")
    print(f'''
    insight: "Result: {emotions_correct}/{emotions_total} emotion accuracy. F0 correlation: {avg_f0_corr:.3f}. Speaker similarity: {avg_speaker_sim*100:.0f}%."
''')

    if pitch_info:
        print(f"\nPitch Results (for Key Achievement section):")
        happy_pitch = pitch_info.get('happy', 0)
        sad_pitch = pitch_info.get('sad', 0)
        print(f"  Happy: {happy_pitch:.0f} Hz")
        print(f"  Sad: {sad_pitch:.0f} Hz")
        print(f"  Separation: {happy_pitch - sad_pitch:.0f} Hz")

    return summary


# =============================================================================
# Research Lead & Proposal System
# =============================================================================

RESEARCH_LEAD_PROMPT = """You are the Research Lead for the AI Research Lab. Your job is to synthesize research findings into a clear proposal for user approval.

## Your Task
Review the completed research tasks for Story {story_id} and create a PROPOSAL document.

## Completed Research Tasks
{research_tasks}

## Research Artifacts
Check these output files for detailed findings:
{artifact_files}

## Instructions
1. Read each research artifact carefully
2. Synthesize the findings into a coherent proposal
3. Write the proposal to: docs/{story_dir}/PROPOSAL.md

## Proposal Structure (REQUIRED)
Use this exact structure:

```markdown
# Research Proposal: {story_title}

**Story ID:** {story_id}
**Generated:** [current date]
**Status:** Pending Review

---

## Executive Summary
[2-3 paragraphs: What was researched, key findings, recommended path forward]

---

## Research Findings
[For each task: summary, key findings, relevant sources]

---

## Recommended Approach

### Primary Recommendation
[What technology/technique to use and why]

### Implementation Roadmap
[Table: Phase | Description | Dependencies | Complexity]

### Success Criteria
[Checkboxes with measurable outcomes]

---

## Alternatives Considered
[For each alternative: pros, cons, why not chosen]

---

## Open Questions
[Questions needing user input before proceeding]

---

## Risk Assessment
[Table: Risk | Likelihood | Impact | Mitigation]

---

## Estimated Effort
[T-shirt size and breakdown by component]

---

## Approval
- [ ] **Approved** - Proceed with implementation
- [ ] **Needs Revision** - Address feedback below
- [ ] **Rejected** - Do not proceed
```

## After Writing the Proposal
1. Call TaskCreate to create a "Review proposal for Story {story_id}" task
2. The proposal will be reviewed by the user before implementation begins

Be thorough but concise. Focus on actionable recommendations.
"""


def load_proposals() -> dict:
    """Load proposals state from file."""
    proposals_file = get_proposals_file()
    if proposals_file.exists():
        return json.loads(proposals_file.read_text())
    return {"proposals": {}}


def save_proposals(proposals: dict):
    """Save proposals state to file."""
    proposals_file = get_proposals_file()
    proposals_file.parent.mkdir(parents=True, exist_ok=True)
    proposals_file.write_text(json.dumps(proposals, indent=2, default=str))


def get_story_from_task(task: dict) -> Optional[str]:
    """Extract story ID from task subject (e.g., [S1] -> S1)."""
    import re
    subject = task.get("subject", "")
    match = re.search(r'\[S(\d+)\]', subject)
    if match:
        return f"S{match.group(1)}"
    return None


def get_story_research_status(story_id: str) -> dict:
    """
    Check research status for a story.

    Returns:
        {
            'story_id': str,
            'total_tasks': int,
            'completed_tasks': int,
            'pending_tasks': int,
            'in_progress_tasks': int,
            'is_complete': bool,
            'tasks': list of task summaries
        }
    """
    # Get tasks from Claude Code task list
    import os
    task_list_id = get_task_list_id()
    tasks_dir = Path.home() / ".claude" / "tasks" / task_list_id

    if not tasks_dir.exists():
        return {
            'story_id': story_id,
            'total_tasks': 0,
            'completed_tasks': 0,
            'pending_tasks': 0,
            'in_progress_tasks': 0,
            'is_complete': False,
            'tasks': []
        }

    story_tasks = []
    for task_file in tasks_dir.glob("*.json"):
        try:
            task = json.loads(task_file.read_text())
            if get_story_from_task(task) == story_id:
                # Check if it's a research task (not implementation)
                subject_lower = task.get("subject", "").lower()
                if any(kw in subject_lower for kw in ['research', 'explore', 'analyze', 'investigate', 'study', 'compare', 'evaluate']):
                    story_tasks.append({
                        'id': task.get('id'),
                        'subject': task.get('subject'),
                        'status': task.get('status'),
                        'description': task.get('description', '')[:200]
                    })
        except:
            continue

    completed = sum(1 for t in story_tasks if t['status'] == 'completed')
    pending = sum(1 for t in story_tasks if t['status'] == 'pending')
    in_progress = sum(1 for t in story_tasks if t['status'] == 'in_progress')

    return {
        'story_id': story_id,
        'total_tasks': len(story_tasks),
        'completed_tasks': completed,
        'pending_tasks': pending,
        'in_progress_tasks': in_progress,
        'is_complete': len(story_tasks) > 0 and completed == len(story_tasks),
        'tasks': story_tasks
    }


def get_research_artifacts(story_id: str) -> list:
    """Find research output files for a story."""
    project_root = SKILL_DIR.parent.parent
    artifacts = []

    # Check docs directory
    docs_dir = project_root / "docs"
    story_dirs = [
        docs_dir / story_id.lower(),
        docs_dir / f"story_{story_id.lower()}",
        docs_dir / "firefly",  # Current project uses this
    ]

    for story_dir in story_dirs:
        if story_dir.exists():
            for md_file in story_dir.glob("*.md"):
                if md_file.name != "PROPOSAL.md":
                    artifacts.append(str(md_file))

    # Check agent output logs
    outputs_dir = get_outputs_dir()
    for log_file in outputs_dir.glob("*.log"):
        content = log_file.read_text()
        if story_id in content:
            artifacts.append(str(log_file))

    return artifacts


def spawn_research_lead(story_id: str, story_title: str = None) -> dict:
    """
    Spawn a research lead agent to synthesize findings into a proposal.

    Args:
        story_id: The story ID (e.g., "S1")
        story_title: Optional title for the story

    Returns:
        Agent info dict
    """
    status = get_story_research_status(story_id)

    if not status['is_complete']:
        raise ValueError(f"Story {story_id} research not complete. {status['pending_tasks']} pending, {status['in_progress_tasks']} in progress.")

    artifacts = get_research_artifacts(story_id)

    # Build the prompt
    research_tasks_text = "\n".join([
        f"- Task #{t['id']}: {t['subject']} (Status: {t['status']})"
        for t in status['tasks']
    ])

    artifacts_text = "\n".join([f"- {a}" for a in artifacts]) if artifacts else "No artifact files found - check agent output logs"

    # Determine story directory
    story_dir = story_id.lower()
    if story_id.startswith("S"):
        # Use current project's naming convention
        story_dir = "firefly"

    prompt = RESEARCH_LEAD_PROMPT.format(
        story_id=story_id,
        story_title=story_title or f"Story {story_id}",
        research_tasks=research_tasks_text,
        artifact_files=artifacts_text,
        story_dir=story_dir
    )

    agent_name = f"research-lead-{story_id.lower()}-{int(time.time())}"

    # Spawn using Codex for quality synthesis
    agent_info = spawn_agent("codex", agent_name, prompt)

    # Record proposal status
    proposals = load_proposals()
    proposals["proposals"][story_id] = {
        "status": "generating",
        "agent": agent_name,
        "started_at": datetime.now().isoformat(),
        "story_title": story_title,
        "research_tasks": [t['id'] for t in status['tasks']],
    }
    save_proposals(proposals)

    return agent_info


def update_proposal_status(story_id: str, status: str, notes: str = None):
    """
    Update proposal status.

    Args:
        story_id: Story ID
        status: One of: generating, pending_review, approved, rejected, needs_revision
        notes: Optional reviewer notes
    """
    proposals = load_proposals()

    if story_id not in proposals["proposals"]:
        proposals["proposals"][story_id] = {}

    proposals["proposals"][story_id]["status"] = status
    proposals["proposals"][story_id]["updated_at"] = datetime.now().isoformat()

    if notes:
        proposals["proposals"][story_id]["notes"] = notes

    if status == "approved":
        proposals["proposals"][story_id]["approved_at"] = datetime.now().isoformat()

    save_proposals(proposals)
    print(f"Proposal for {story_id}: {status}")


def list_proposals() -> list:
    """List all proposals and their status."""
    proposals = load_proposals()
    return [
        {"story_id": k, **v}
        for k, v in proposals.get("proposals", {}).items()
    ]


def check_proposal_exists(story_id: str) -> bool:
    """Check if a proposal document exists for a story."""
    project_root = SKILL_DIR.parent.parent

    # Check common locations
    proposal_paths = [
        project_root / "docs" / story_id.lower() / "PROPOSAL.md",
        project_root / "docs" / f"story_{story_id.lower()}" / "PROPOSAL.md",
        project_root / "docs" / "firefly" / "PROPOSAL.md",
    ]

    return any(p.exists() for p in proposal_paths)


def rtx_logs(lines: int = 50):
    """Get recent training logs from RTX 4090."""
    # Check for training output in common locations
    locations = [
        "training/outputs/*/train.log",
        "/tmp/training.log",
        "training/*.log"
    ]

    for loc in locations:
        result = subprocess.run(
            f'ssh {RTX_4090_HOST} "tail -n {lines} {RTX_4090_PROJECT}/{loc} 2>/dev/null || true"',
            shell=True, capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            print(f"Logs from {loc}:")
            print("=" * 60)
            print(result.stdout)
            return

    print("No training logs found")


def main():
    parser = argparse.ArgumentParser(description="Research Manager - Orchestrate AI agents")

    # Global --lab flag to override active lab
    parser.add_argument("--lab", "-L", dest="lab_override", help="Override active lab for this command")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Lab management commands
    lab_parser = subparsers.add_parser("lab", help="Manage research labs")
    lab_subparsers = lab_parser.add_subparsers(dest="lab_command")

    lab_subparsers.add_parser("list", help="List all labs")

    lab_create_parser = lab_subparsers.add_parser("create", help="Create a new lab")
    lab_create_parser.add_argument("lab_id", help="Unique lab ID (lowercase, hyphens)")
    lab_create_parser.add_argument("--name", "-n", required=True, help="Display name")
    lab_create_parser.add_argument("--description", "-d", default="", help="Lab description")
    lab_create_parser.add_argument("--task-list", "-t", help="Task list ID (defaults to lab_id)")
    lab_create_parser.add_argument("--domain", help="Link to .domains/ config")

    lab_switch_parser = lab_subparsers.add_parser("switch", help="Switch active lab")
    lab_switch_parser.add_argument("lab_id", help="Lab ID to switch to")

    lab_status_parser = lab_subparsers.add_parser("status", help="Show lab status")
    lab_status_parser.add_argument("lab_id", nargs="?", help="Lab ID (current if omitted)")

    lab_info_parser = lab_subparsers.add_parser("info", help="Show detailed lab info")
    lab_info_parser.add_argument("lab_id", help="Lab ID")

    lab_delete_parser = lab_subparsers.add_parser("delete", help="Delete a lab")
    lab_delete_parser.add_argument("lab_id", help="Lab ID to delete")
    lab_delete_parser.add_argument("--force", "-f", action="store_true", help="Force delete even if running agents")

    lab_migrate_parser = lab_subparsers.add_parser("migrate", help="Migrate global state to a lab")
    lab_migrate_parser.add_argument("lab_id", help="Target lab ID")
    lab_migrate_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")

    # Spawn command
    spawn_parser = subparsers.add_parser("spawn", help="Spawn a new agent")
    spawn_parser.add_argument("--type", "-t", choices=["codex", "ollama", "opus"], required=True, help="Agent type")
    spawn_parser.add_argument("--name", "-n", required=True, help="Unique agent name")
    spawn_parser.add_argument("--task", required=True, help="Task/prompt for the agent")
    spawn_parser.add_argument("--dir", "-d", help="Working directory")

    # Status command
    status_parser = subparsers.add_parser("status", help="Get agent status")
    status_parser.add_argument("--name", "-n", help="Specific agent name")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Read command
    read_parser = subparsers.add_parser("read", help="Read agent output")
    read_parser.add_argument("--name", "-n", required=True, help="Agent name")
    read_parser.add_argument("--tail", "-t", type=int, default=100, help="Lines from end (0 for all)")

    # Send command
    send_parser = subparsers.add_parser("send", help="Send message to agent")
    send_parser.add_argument("--name", "-n", required=True, help="Agent name")
    send_parser.add_argument("--message", "-m", required=True, help="Message to send")

    # Kill command
    kill_parser = subparsers.add_parser("kill", help="Kill an agent")
    kill_parser.add_argument("--name", "-n", help="Agent name (omit for all)")

    # Remind command
    remind_parser = subparsers.add_parser("remind", help="Create a reminder")
    remind_parser.add_argument("--message", "-m", required=True, help="Reminder message")
    remind_parser.add_argument("--in", dest="trigger_in", help="Trigger in duration (e.g., 5m, 1h)")
    remind_parser.add_argument("--at", dest="trigger_at", help="Trigger at time (ISO or HH:MM)")

    # Reminders command
    subparsers.add_parser("reminders", help="List all reminders")

    # Check-reminders command
    subparsers.add_parser("check-reminders", help="Check and trigger due reminders")

    # Sleep command
    sleep_parser = subparsers.add_parser("sleep", help="Sleep for duration")
    sleep_parser.add_argument("--seconds", "-s", type=int, required=True, help="Seconds to sleep")

    # Wait command
    wait_parser = subparsers.add_parser("wait", help="Wait for condition")
    wait_parser.add_argument("--agent", "-a", help="Wait for agent to complete")
    wait_parser.add_argument("--file", "-f", help="Wait for file to exist")
    wait_parser.add_argument("--timeout", "-t", type=int, default=3600, help="Timeout in seconds")

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear state")
    clear_parser.add_argument("--agents", action="store_true", help="Clear agent registry")
    clear_parser.add_argument("--reminders", action="store_true", help="Clear reminders")

    # Sessions command (list tmux sessions)
    subparsers.add_parser("sessions", help="List all tmux sessions")

    # Info command
    subparsers.add_parser("info", help="Show Research Manager info and paths")

    # Limits command - show current usage vs limits
    subparsers.add_parser("limits", help="Show current usage vs daily limits")

    # Dashboard command
    subparsers.add_parser("dashboard", help="Show full dashboard with all state")

    # Web search command
    search_parser = subparsers.add_parser("search", help="Web search for research")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument("--papers", "-p", action="store_true", help="Focus on academic papers")
    search_parser.add_argument("--max", "-m", type=int, default=5, help="Max results")

    # GitHub search command
    gh_parser = subparsers.add_parser("github", help="Search GitHub repositories")
    gh_parser.add_argument("query", nargs="+", help="Search query")
    gh_parser.add_argument("--language", "-l", help="Filter by language (e.g., python)")
    gh_parser.add_argument("--sort", "-s", choices=["stars", "forks", "updated"], default="stars")
    gh_parser.add_argument("--max", "-m", type=int, default=10, help="Max results")

    # Papers command (arxiv search)
    papers_parser = subparsers.add_parser("papers", help="Search arxiv for papers")
    papers_parser.add_argument("query", nargs="+", help="Search query")
    papers_parser.add_argument("--max", "-m", type=int, default=5, help="Max results")

    # Fetch URL command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch and summarize a URL")
    fetch_parser.add_argument("url", help="URL to fetch")
    fetch_parser.add_argument("--raw", action="store_true", help="Show raw content")

    # Test and validation commands
    validate_parser = subparsers.add_parser("validate", help="Validate Python code syntax and imports")
    validate_parser.add_argument("path", help="Path to Python file to validate")

    test_parser = subparsers.add_parser("test", help="Run pytest tests")
    test_parser.add_argument("--path", "-p", help="Test file or directory")
    test_parser.add_argument("--pattern", default="test_*.py", help="Test file pattern")

    quicktest_parser = subparsers.add_parser("quicktest", help="Quick test a script")
    quicktest_parser.add_argument("script", help="Path to Python script")
    quicktest_parser.add_argument("--args", "-a", nargs="*", help="Script arguments")
    quicktest_parser.add_argument("--timeout", "-t", type=int, default=30, help="Timeout in seconds")

    # Update results command
    results_parser = subparsers.add_parser("update-results", help="Update landing page with evaluation results")
    results_parser.add_argument("--dir", "-d", help="Directory with eval_*.json files (default: inference/)")
    results_parser.add_argument("--version", "-v", help="Version name (e.g., v4, v5)")
    results_parser.add_argument("--description", help="Short description of this version")

    # Autonomous improvement loop commands
    loop_parser = subparsers.add_parser("loop", help="Run autonomous improvement loop")
    loop_parser.add_argument("--config", "-c", default="prosody_v5.yaml", help="Training config")
    loop_parser.add_argument("--max-iter", "-n", type=int, default=5, help="Max iterations")
    loop_parser.add_argument("--threshold", "-t", type=float, default=0.1, help="F0 correlation success threshold")

    # Start loop agent (spawns autonomous improver)
    subparsers.add_parser("start-loop", help="Spawn autonomous improvement loop agent")

    # Training watchdog
    watchdog_parser = subparsers.add_parser("watchdog", help="Monitor training and detect overfitting early")
    watchdog_parser.add_argument("--interval", "-i", type=int, default=120, help="Poll interval in seconds")
    watchdog_parser.add_argument("--patience", "-p", type=int, default=3, help="Epochs without improvement before alert")
    watchdog_parser.add_argument("--auto-kill", "-k", action="store_true", help="Automatically kill training on overfit")
    watchdog_parser.add_argument("--max-runtime", "-m", type=int, default=7200, help="Max watchdog runtime in seconds")

    # RTX 4090 commands
    rtx_parser = subparsers.add_parser("rtx", help="RTX 4090 remote training management")
    rtx_subparsers = rtx_parser.add_subparsers(dest="rtx_command")

    rtx_subparsers.add_parser("status", help="Check RTX 4090 GPU and training status")

    rtx_train_parser = rtx_subparsers.add_parser("train", help="Start training on RTX 4090")
    rtx_train_parser.add_argument("--config", "-c", default="rtx_4090_lora.yaml", help="Training config file")
    rtx_train_parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")

    rtx_sync_parser = rtx_subparsers.add_parser("sync", help="Sync files with RTX 4090")
    rtx_sync_parser.add_argument("--pull", action="store_true", help="Pull from remote (default: push)")

    rtx_logs_parser = rtx_subparsers.add_parser("logs", help="View training logs")
    rtx_logs_parser.add_argument("--lines", "-n", type=int, default=50, help="Number of lines")

    rtx_run_parser = rtx_subparsers.add_parser("run", help="Run arbitrary command on RTX 4090")
    rtx_run_parser.add_argument("command", nargs="+", help="Command to run")

    # Research Lead commands (decision support)
    lead_parser = subparsers.add_parser("lead", help="Research Lead - decision support tools")
    lead_parser.add_argument("lead_command", nargs="?", default="status",
                             choices=["status", "evaluate", "focus", "decide", "reject"],
                             help="Lead command (default: status)")
    lead_parser.add_argument("lead_args", nargs="*", help="Additional arguments for lead command")

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up stale agents and free memory")
    cleanup_parser.add_argument("--execute", "-e", action="store_true",
                                help="Actually clean up (default is dry run)")
    cleanup_parser.add_argument("--force", "-f", action="store_true",
                                help="Also clean old completed agent records")
    cleanup_parser.add_argument("--quiet", "-q", action="store_true",
                                help="Minimal output")

    # GC alias for cleanup
    subparsers.add_parser("gc", help="Alias for cleanup --execute")

    # Focus mode - halt for priority tasks
    focus_parser = subparsers.add_parser("focus", help="Focus mode - halt work for priority task")
    focus_parser.add_argument("focus_command", nargs="?", default="status",
                              choices=["set", "status", "clear"],
                              help="Focus command (default: status)")
    focus_parser.add_argument("focus_args", nargs="*", help="Arguments (description for set)")
    focus_parser.add_argument("--task-id", "-t", type=int, help="Task ID to track")
    focus_parser.add_argument("--reason", "-r", help="Why this is priority")

    # Halt alias for focus set
    halt_parser = subparsers.add_parser("halt", help="Alias for focus set (halt all work)")
    halt_parser.add_argument("description", help="What needs to be done")
    halt_parser.add_argument("--task-id", "-t", type=int, help="Task ID to track")

    # Proposal management commands
    proposal_parser = subparsers.add_parser("proposal", help="Manage research proposals")
    proposal_subparsers = proposal_parser.add_subparsers(dest="proposal_command")

    proposal_subparsers.add_parser("list", help="List all proposals")

    proposal_status_parser = proposal_subparsers.add_parser("status", help="Check story research status")
    proposal_status_parser.add_argument("story_id", help="Story ID (e.g., S1)")

    proposal_generate_parser = proposal_subparsers.add_parser("generate", help="Generate proposal for a story")
    proposal_generate_parser.add_argument("story_id", help="Story ID (e.g., S1)")
    proposal_generate_parser.add_argument("--title", "-t", help="Story title")

    proposal_approve_parser = proposal_subparsers.add_parser("approve", help="Approve a proposal")
    proposal_approve_parser.add_argument("story_id", help="Story ID (e.g., S1)")
    proposal_approve_parser.add_argument("--notes", "-n", help="Approval notes")

    proposal_reject_parser = proposal_subparsers.add_parser("reject", help="Reject a proposal")
    proposal_reject_parser.add_argument("story_id", help="Story ID (e.g., S1)")
    proposal_reject_parser.add_argument("--notes", "-n", help="Rejection reason")

    proposal_revise_parser = proposal_subparsers.add_parser("revise", help="Request revision for a proposal")
    proposal_revise_parser.add_argument("story_id", help="Story ID (e.g., S1)")
    proposal_revise_parser.add_argument("--notes", "-n", required=True, help="What needs to change")

    args = parser.parse_args()

    # Handle global --lab override
    global _current_lab_id
    if hasattr(args, 'lab_override') and args.lab_override:
        _current_lab_id = args.lab_override

    try:
        if args.command == "lab":
            active_lab_id = labs_module.get_active_lab_id()

            if args.lab_command == "list":
                labs = labs_module.list_labs()
                if not labs:
                    print("No labs configured")
                    print("\nCreate one with: ./rm lab create <id> --name 'Lab Name'")
                else:
                    print("\nResearch Labs")
                    print("=" * 60)
                    for lab in labs:
                        is_active = lab.get('id') == active_lab_id
                        marker = "*" if is_active else " "
                        status = "active" if lab.get('active', True) else "paused"
                        print(f"  {marker} {lab['id']}: {lab.get('name', 'Unnamed')}")
                        print(f"      Status: {status} | Tasks: {lab.get('taskListId', lab['id'])}")
                        if lab.get('description'):
                            print(f"      {lab['description'][:60]}")
                    print()
                    if active_lab_id:
                        print(f"  * = active lab ({active_lab_id})")

            elif args.lab_command == "create":
                try:
                    lab = labs_module.create_lab(
                        args.lab_id,
                        args.name,
                        args.description,
                        args.task_list,
                        args.domain
                    )
                    print(f"\nCreated lab: {lab['id']}")
                    print(f"  Name: {lab['name']}")
                    print(f"  Task List: {lab['taskListId']}")
                    print(f"\nSwitch to it with: ./rm lab switch {lab['id']}")
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            elif args.lab_command == "switch":
                try:
                    lab = labs_module.set_active_lab(args.lab_id)
                    print(f"\nSwitched to lab: {lab['id']}")
                    print(f"  Name: {lab['name']}")
                    print(f"  Task List: {lab['taskListId']}")
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            elif args.lab_command == "status":
                lab_id = args.lab_id or active_lab_id
                if not lab_id:
                    print("No active lab. Specify a lab ID or run: ./rm lab switch <id>")
                    sys.exit(1)
                try:
                    status = labs_module.get_lab_status(lab_id)
                    lab = status['lab']
                    is_active = status['isActive']

                    print(f"\nLab Status: {lab['id']}")
                    print("=" * 50)
                    print(f"  Name: {lab['name']}")
                    print(f"  Active: {'Yes *' if is_active else 'No'}")
                    print(f"  Task List: {lab.get('taskListId', lab['id'])}")
                    if lab.get('domain'):
                        print(f"  Domain: {lab['domain']}")
                    print()
                    print(f"  Agents: {status['agents']['running']} running / {status['agents']['total']} total")
                    print(f"  Proposals: {status['proposals']['pendingReview']} pending review / {status['proposals']['total']} total")
                    print()
                    print(f"  Tasks:")
                    print(f"    Pending:     {status['tasks']['pending']}")
                    print(f"    In Progress: {status['tasks']['in_progress']}")
                    print(f"    Completed:   {status['tasks']['completed']}")
                    print(f"    Total:       {status['tasks']['total']}")
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            elif args.lab_command == "info":
                lab = labs_module.get_lab(args.lab_id)
                if not lab:
                    print(f"Lab '{args.lab_id}' not found")
                    sys.exit(1)
                print(json.dumps(lab, indent=2))

            elif args.lab_command == "delete":
                try:
                    labs_module.delete_lab(args.lab_id, force=args.force)
                    print(f"Deleted lab: {args.lab_id}")
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            elif args.lab_command == "migrate":
                try:
                    result = labs_module.migrate_global_state_to_lab(args.lab_id, overwrite=args.overwrite)
                    print(f"Migrated state to lab '{args.lab_id}':")
                    print(f"  Files migrated: {result['migrated']}")
                    print(f"  Files skipped: {result['skipped']}")
                except ValueError as e:
                    print(f"Error: {e}")
                    sys.exit(1)

            else:
                print("Lab Commands:")
                print("  lab list                    - List all labs")
                print("  lab create <id> --name 'X'  - Create a new lab")
                print("  lab switch <id>             - Switch active lab")
                print("  lab status [id]             - Show lab status")
                print("  lab info <id>               - Show lab config (JSON)")
                print("  lab delete <id>             - Delete a lab")
                print("  lab migrate <id>            - Migrate global state to lab")

        elif args.command == "spawn":
            spawn_agent(args.type, args.name, args.task, args.dir)

        elif args.command == "status":
            status = get_agent_status(args.name)
            if args.json:
                print(json.dumps(status, indent=2))
            else:
                if isinstance(status, list):
                    if not status:
                        print("No agents registered")
                    for agent in status:
                        print(f"\n{agent['name']} ({agent['type']}):")
                        print(f"  Status: {agent['status']}")
                        print(f"  Task: {agent['task'][:50]}...")
                        print(f"  Session: {agent['session']}")
                        print(f"  Started: {agent['started_at']}")
                else:
                    print(json.dumps(status, indent=2))

        elif args.command == "read":
            output = read_agent_output(args.name, args.tail)
            print(output)

        elif args.command == "send":
            send_to_agent(args.name, args.message)

        elif args.command == "kill":
            if args.name:
                kill_agent(args.name)
            else:
                kill_all_agents()

        elif args.command == "remind":
            create_reminder(args.message, args.trigger_at, args.trigger_in)

        elif args.command == "reminders":
            reminders = list_reminders()
            if not reminders:
                print("No reminders")
            for r in reminders:
                status = "TRIGGERED" if r["triggered"] else "pending"
                print(f"#{r['id']} [{status}] @ {r['trigger_at']}: {r['message']}")

        elif args.command == "check-reminders":
            triggered = check_reminders()
            if triggered:
                for r in triggered:
                    print(f"[TRIGGERED] #{r['id']}: {r['message']}")
            else:
                print("No reminders triggered")

        elif args.command == "sleep":
            sleep_for(args.seconds)

        elif args.command == "wait":
            if args.agent:
                wait_for_agent(args.agent, args.timeout)
            elif args.file:
                wait_for_file(args.file, args.timeout)
            else:
                print("Must specify --agent or --file")
                sys.exit(1)

        elif args.command == "clear":
            if args.agents:
                clear_agents()
            if args.reminders:
                clear_reminders()
            if not args.agents and not args.reminders:
                print("Specify --agents and/or --reminders")

        elif args.command == "sessions":
            sessions = get_tmux_sessions()
            if sessions:
                print("Tmux sessions:")
                for s in sessions:
                    managed = " (managed)" if s.startswith("rm-") else ""
                    print(f"  {s}{managed}")
            else:
                print("No tmux sessions")

        elif args.command == "info":
            print("Research Manager Info")
            print("=" * 40)
            print(f"Skill Directory: {SKILL_DIR}")

            # Show lab info
            current_lab_id = get_current_lab_id()
            lab = labs_module.get_lab(current_lab_id)
            print()
            print("Current Lab:")
            print(f"  ID: {current_lab_id}")
            if lab:
                print(f"  Name: {lab.get('name', 'Unknown')}")
                print(f"  Task List: {lab.get('taskListId', current_lab_id)}")
                print(f"  State Dir: {get_state_dir()}")
            else:
                print(f"  (Using default/global state)")
                print(f"  State Dir: {STATE_DIR}")

            print()
            print("Current State Files:")
            print(f"  Agents: {get_agents_file()}")
            print(f"  Reminders: {get_reminders_file()}")
            print(f"  Outputs: {get_outputs_dir()}")
            print()
            print("Agent Types:")
            print("  codex  -> OpenAI Codex CLI")
            print("  ollama -> Local Claude Code via Ollama (scripts/claude-free)")
            print("  opus   -> Alias for ollama (kept for backward compatibility)")
            print()
            print("Lab Commands: ./rm lab --help")

        elif args.command == "limits":
            # Read cost and progress data (from current lab)
            state_dir = get_state_dir()
            cost_file = state_dir / "cost-tracking.json"
            progress_file = state_dir / "progress.json"

            cost_data = {"sessions": [], "totals": {}}
            progress_data = {"tasks": {}}

            if cost_file.exists():
                cost_data = json.loads(cost_file.read_text())
            if progress_file.exists():
                progress_data = json.loads(progress_file.read_text())

            today = datetime.now().strftime("%Y-%m-%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            # Calculate stats
            today_sessions = [s for s in cost_data.get("sessions", []) if s.get("date") == today]
            daily_cost = sum(s.get("estimatedCost", 0) for s in today_sessions)
            weekly_cost = sum(s.get("estimatedCost", 0) for s in cost_data.get("sessions", []) if s.get("date", "") >= week_ago)
            tasks_today = len([t for t in progress_data.get("tasks", {}).values() if t.get("completedAt", "").startswith(today)])
            research_today = len([s for s in today_sessions if s.get("type") == "research"])
            agent_minutes = sum(s.get("durationMinutes", 0) for s in today_sessions)

            print("Daily Limits & Usage")
            print("=" * 50)
            print()
            print("COST LIMITS:")
            print(f"  Daily:  ${daily_cost:.2f} / $150.00 {'⚠️ EXCEEDED!' if daily_cost >= 150 else '✓'}")
            print(f"  Weekly: ${weekly_cost:.2f} / $750.00 {'⚠️ EXCEEDED!' if weekly_cost >= 750 else '✓'}")
            print()
            print("TASK LIMITS:")
            print(f"  Tasks completed:  {tasks_today} / 20 {'⚠️ LIMIT!' if tasks_today >= 20 else '✓'}")
            print(f"  Research sessions: {research_today} / 5 {'⚠️ LIMIT!' if research_today >= 5 else '✓'}")
            print(f"  Agent runtime:    {agent_minutes:.0f} / 180 min {'⚠️ LIMIT!' if agent_minutes >= 180 else '✓'}")
            print()
            print("AGENT RULES (in task prefix):")
            print("  - Max 5 web searches per task")
            print("  - 30 minute time limit per agent")
            print("  - No scope expansion")
            print("  - Report blockers after 3 failed attempts")

        elif args.command == "search":
            query = " ".join(args.query)
            search_web(query, papers_only=args.papers, max_results=args.max)

        elif args.command == "github":
            query = " ".join(args.query)
            search_github(query, language=args.language, sort=args.sort, max_results=args.max)

        elif args.command == "papers":
            query = " ".join(args.query)
            search_arxiv(query, max_results=args.max)

        elif args.command == "fetch":
            fetch_url(args.url, raw=args.raw)

        elif args.command == "validate":
            result = validate_code(args.path)
            sys.exit(0 if result["success"] else 1)

        elif args.command == "test":
            result = run_tests(test_path=args.path, pattern=args.pattern)
            sys.exit(0 if result["success"] else 1)

        elif args.command == "quicktest":
            result = quick_test(args.script, args=args.args, timeout=args.timeout)
            sys.exit(0 if result["success"] else 1)

        elif args.command == "update-results":
            update_landing_results(
                eval_dir=args.dir,
                version=args.version,
                description=args.description
            )

        elif args.command == "loop":
            run_improvement_loop(
                config=args.config,
                max_iterations=args.max_iter,
                eval_threshold=args.threshold
            )

        elif args.command == "start-loop":
            start_loop_agent()
            print("\nAuto-improver agent spawned!")
            print("Monitor with: .skills/research-manager/rm read --name auto-improver")
            print("Attach with: tmux attach -t rm-auto-improver")

        elif args.command == "watchdog":
            result = training_watchdog(
                poll_interval=args.interval,
                patience=args.patience,
                auto_kill=args.auto_kill,
                max_runtime=args.max_runtime
            )
            if result.get('stopped_early'):
                print(f"\nTraining stopped early at epoch {result['stopped_at_epoch']}")
                print(f"Best model was at epoch {result['best_epoch']}")
                print(f"Saved {result['wasted_epochs']} wasted epochs!")
                sys.exit(0)
            else:
                sys.exit(0 if result.get('alerts', 0) == 0 else 1)

        elif args.command == "rtx":
            if args.rtx_command == "status":
                rtx_status()
            elif args.rtx_command == "train":
                rtx_train(config=args.config, background=not args.foreground)
            elif args.rtx_command == "sync":
                rtx_sync(direction="pull" if args.pull else "push")
            elif args.rtx_command == "logs":
                rtx_logs(lines=args.lines)
            elif args.rtx_command == "run":
                cmd = " ".join(args.command)
                result = rtx_run(cmd)
                print(result)
            else:
                print("RTX 4090 Commands:")
                print("  rtx status  - Check GPU and training status")
                print("  rtx train   - Start training (--config, --foreground)")
                print("  rtx sync    - Sync files (--pull to pull from remote)")
                print("  rtx logs    - View training logs (--lines)")
                print("  rtx run     - Run arbitrary command")

        elif args.command == "dashboard":
            print("=" * 60)
            print("         RESEARCH MANAGER DASHBOARD")
            print("=" * 60)

            # Lab info section
            current_lab_id = get_current_lab_id()
            lab = labs_module.get_lab(current_lab_id)
            print(f"\nLab: {current_lab_id}")
            if lab:
                print(f"     Name: {lab.get('name', 'Unknown')}")
                print(f"     Tasks: {lab.get('taskListId', current_lab_id)}")

            # Agents section
            print("\nAGENTS")
            print("-" * 40)
            agents = get_agent_status()
            if not agents:
                print("  No agents registered")
            else:
                for agent in agents:
                    status_icon = "🟢" if agent["status"] == "running" else "🔴"
                    print(f"  {status_icon} {agent['name']} ({agent['type']})")
                    print(f"     Task: {agent['task'][:50]}...")
                    print(f"     Session: {agent['session']}")
                    print(f"     Started: {agent['started_at']}")
                    print()

            # Reminders section
            print("\n⏰ REMINDERS")
            print("-" * 40)
            reminders = list_reminders()
            if not reminders:
                print("  No reminders set")
            else:
                for r in reminders:
                    status = "✓" if r["triggered"] else "○"
                    print(f"  [{status}] #{r['id']} @ {r['trigger_at']}")
                    print(f"      {r['message']}")

            # Tmux sessions
            print("\n🖥️  TMUX SESSIONS")
            print("-" * 40)
            sessions = get_tmux_sessions()
            if not sessions:
                print("  No tmux sessions")
            else:
                for s in sessions:
                    managed = " (managed)" if s.startswith("rm-") else ""
                    print(f"  • {s}{managed}")

            # Output files
            print("\n📄 OUTPUT FILES")
            print("-" * 40)
            if OUTPUTS_DIR.exists():
                logs = list(OUTPUTS_DIR.glob("*.log"))
                if logs:
                    for log in logs:
                        size = log.stat().st_size
                        print(f"  • {log.name} ({size} bytes)")
                else:
                    print("  No output files")
            else:
                print("  No outputs directory")

            print("\n" + "=" * 60)

        elif args.command == "lead":
            # Delegate to research-lead.py
            lead_script = SKILL_DIR / "research-lead.py"
            cmd = ["python3", str(lead_script), args.lead_command] + args.lead_args
            subprocess.run(cmd)

        elif args.command == "cleanup":
            # Delegate to cleanup.py
            cleanup_script = SKILL_DIR / "cleanup.py"
            cmd = ["python3", str(cleanup_script)]
            if args.execute:
                cmd.append("--execute")
            if args.force:
                cmd.append("--force")
            if args.quiet:
                cmd.append("--quiet")
            subprocess.run(cmd)

        elif args.command == "gc":
            # Shorthand for cleanup --execute
            cleanup_script = SKILL_DIR / "cleanup.py"
            subprocess.run(["python3", str(cleanup_script), "--execute"])

        elif args.command == "focus":
            # Delegate to focus.py
            focus_script = SKILL_DIR / "focus.py"
            cmd = ["python3", str(focus_script), args.focus_command]
            if args.focus_command == "set" and args.focus_args:
                cmd.append(args.focus_args[0])  # description
            if args.task_id:
                cmd.extend(["--task-id", str(args.task_id)])
            if hasattr(args, 'reason') and args.reason:
                cmd.extend(["--reason", args.reason])
            subprocess.run(cmd)

        elif args.command == "halt":
            # Shorthand for focus set
            focus_script = SKILL_DIR / "focus.py"
            cmd = ["python3", str(focus_script), "set", args.description]
            if args.task_id:
                cmd.extend(["--task-id", str(args.task_id)])
            subprocess.run(cmd)

        elif args.command == "proposal":
            if args.proposal_command == "list":
                proposals = list_proposals()
                if not proposals:
                    print("No proposals found")
                else:
                    print("\nResearch Proposals")
                    print("=" * 60)
                    for p in proposals:
                        status_icon = {
                            "generating": "🔄",
                            "pending_review": "📋",
                            "approved": "✅",
                            "rejected": "❌",
                            "needs_revision": "📝",
                        }.get(p.get("status", "unknown"), "❓")
                        print(f"\n{status_icon} {p['story_id']}: {p.get('story_title', 'Untitled')}")
                        print(f"   Status: {p.get('status', 'unknown')}")
                        if p.get('updated_at'):
                            print(f"   Updated: {p['updated_at']}")
                        if p.get('notes'):
                            print(f"   Notes: {p['notes']}")

            elif args.proposal_command == "status":
                story_id = args.story_id.upper()
                if not story_id.startswith("S"):
                    story_id = f"S{story_id}"
                status = get_story_research_status(story_id)
                print(f"\nResearch Status for {story_id}")
                print("=" * 50)
                print(f"Total research tasks: {status['total_tasks']}")
                print(f"Completed: {status['completed_tasks']}")
                print(f"In Progress: {status['in_progress_tasks']}")
                print(f"Pending: {status['pending_tasks']}")
                print(f"\nResearch Complete: {'✅ Yes' if status['is_complete'] else '❌ No'}")

                if status['tasks']:
                    print("\nTasks:")
                    for t in status['tasks']:
                        icon = {"completed": "✅", "in_progress": "🔄", "pending": "○"}.get(t['status'], "?")
                        print(f"  {icon} #{t['id']}: {t['subject'][:50]}")

                # Check if proposal exists
                if check_proposal_exists(story_id):
                    print(f"\n📄 Proposal document exists")
                elif status['is_complete']:
                    print(f"\n💡 Ready to generate proposal: .skills/research-manager/rm proposal generate {story_id}")

            elif args.proposal_command == "generate":
                story_id = args.story_id.upper()
                if not story_id.startswith("S"):
                    story_id = f"S{story_id}"
                try:
                    agent_info = spawn_research_lead(story_id, args.title)
                    print(f"\n✅ Research Lead agent spawned: {agent_info['name']}")
                    print(f"   Monitor: .skills/research-manager/rm read --name {agent_info['name']}")
                    print(f"   Session: tmux attach -t {agent_info['session']}")
                except ValueError as e:
                    print(f"\n❌ Cannot generate proposal: {e}")
                    sys.exit(1)

            elif args.proposal_command == "approve":
                story_id = args.story_id.upper()
                if not story_id.startswith("S"):
                    story_id = f"S{story_id}"
                update_proposal_status(story_id, "approved", args.notes)
                print(f"\n✅ Proposal for {story_id} APPROVED")
                print("   Implementation tasks can now be spawned.")

            elif args.proposal_command == "reject":
                story_id = args.story_id.upper()
                if not story_id.startswith("S"):
                    story_id = f"S{story_id}"
                update_proposal_status(story_id, "rejected", args.notes)
                print(f"\n❌ Proposal for {story_id} REJECTED")

            elif args.proposal_command == "revise":
                story_id = args.story_id.upper()
                if not story_id.startswith("S"):
                    story_id = f"S{story_id}"
                update_proposal_status(story_id, "needs_revision", args.notes)
                print(f"\n📝 Proposal for {story_id} needs revision")
                print(f"   Feedback: {args.notes}")

            else:
                print("Proposal Commands:")
                print("  proposal list              - List all proposals")
                print("  proposal status <story_id> - Check research status for story")
                print("  proposal generate <story_id> - Generate proposal from research")
                print("  proposal approve <story_id>  - Approve proposal")
                print("  proposal reject <story_id>   - Reject proposal")
                print("  proposal revise <story_id>   - Request revision")

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
