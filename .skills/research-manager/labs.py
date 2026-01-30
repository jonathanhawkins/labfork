#!/usr/bin/env python3
"""
Labs Management Module

Manages multiple research labs with independent task lists, agents, and state.
Each lab has its own namespace and can run in parallel.

Usage:
    from labs import get_active_lab, set_active_lab, list_labs, create_lab
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

# Base paths
SKILL_DIR = Path(__file__).parent
LABS_DIR = SKILL_DIR / "labs"
GLOBAL_STATE_DIR = SKILL_DIR / "state"
ACTIVE_LAB_FILE = GLOBAL_STATE_DIR / "active-lab.json"

# Default lab ID for backwards compatibility
DEFAULT_LAB_ID = "voice-clone"


def ensure_dirs():
    """Ensure required directories exist."""
    LABS_DIR.mkdir(exist_ok=True)
    GLOBAL_STATE_DIR.mkdir(exist_ok=True)


def get_labs_dir() -> Path:
    """Get the labs directory path."""
    ensure_dirs()
    return LABS_DIR


def get_lab_dir(lab_id: str) -> Path:
    """Get the directory for a specific lab."""
    return LABS_DIR / lab_id


def get_lab_state_dir(lab_id: str) -> Path:
    """Get the state directory for a specific lab."""
    lab_dir = get_lab_dir(lab_id)
    state_dir = lab_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_lab_config_path(lab_id: str) -> Path:
    """Get the config file path for a lab."""
    return get_lab_dir(lab_id) / "config.json"


def lab_exists(lab_id: str) -> bool:
    """Check if a lab exists."""
    return get_lab_config_path(lab_id).exists()


def get_lab(lab_id: str) -> Optional[Dict[str, Any]]:
    """
    Get lab configuration.

    Args:
        lab_id: Lab identifier

    Returns:
        Lab config dict or None if not found
    """
    config_path = get_lab_config_path(lab_id)
    if not config_path.exists():
        return None

    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def list_labs() -> List[Dict[str, Any]]:
    """
    List all configured labs.

    Returns:
        List of lab config dicts with id and basic info
    """
    ensure_dirs()
    labs = []

    for item in LABS_DIR.iterdir():
        if item.is_dir() and not item.name.startswith('_'):
            config_path = item / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    labs.append(config)
                except (json.JSONDecodeError, IOError):
                    # Include partial info for broken configs
                    labs.append({
                        "id": item.name,
                        "name": item.name,
                        "error": "Invalid config.json"
                    })

    return sorted(labs, key=lambda x: x.get("name", x.get("id", "")))


def create_lab(
    lab_id: str,
    name: str,
    description: str = "",
    task_list_id: Optional[str] = None,
    domain: Optional[str] = None,
    settings: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Create a new lab.

    Args:
        lab_id: Unique lab identifier (lowercase, hyphens)
        name: Display name
        description: Lab description
        task_list_id: Claude Code task list ID (defaults to lab_id)
        domain: Optional domain slug to link
        settings: Optional lab settings

    Returns:
        Created lab config

    Raises:
        ValueError: If lab already exists or invalid ID
    """
    # Validate lab_id
    import re
    if not re.match(r'^[a-z0-9-]+$', lab_id):
        raise ValueError(f"Invalid lab ID '{lab_id}'. Use lowercase letters, numbers, and hyphens only.")

    if lab_exists(lab_id):
        raise ValueError(f"Lab '{lab_id}' already exists.")

    # Create lab directory structure
    lab_dir = get_lab_dir(lab_id)
    lab_dir.mkdir(parents=True, exist_ok=True)

    state_dir = lab_dir / "state"
    state_dir.mkdir(exist_ok=True)

    outputs_dir = state_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    # Create config
    now = datetime.now().isoformat()
    config = {
        "id": lab_id,
        "name": name,
        "description": description,
        "taskListId": task_list_id or lab_id,
        "active": True,
        "domain": domain,
        "settings": settings or {
            "maxAgents": 3,
            "autoSpawn": True,
            "researchInterval": 30
        },
        "createdAt": now,
        "updatedAt": now
    }

    # Save config
    config_path = get_lab_config_path(lab_id)
    config_path.write_text(json.dumps(config, indent=2))

    # Initialize empty state files
    (state_dir / "agents.json").write_text("{}")
    (state_dir / "proposals.json").write_text('{"proposals": {}}')
    (state_dir / "progress.json").write_text('{"history": []}')

    return config


def update_lab(lab_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update lab configuration.

    Args:
        lab_id: Lab identifier
        updates: Fields to update

    Returns:
        Updated lab config

    Raises:
        ValueError: If lab doesn't exist
    """
    config = get_lab(lab_id)
    if config is None:
        raise ValueError(f"Lab '{lab_id}' not found.")

    # Apply updates
    config.update(updates)
    config["updatedAt"] = datetime.now().isoformat()

    # Save
    config_path = get_lab_config_path(lab_id)
    config_path.write_text(json.dumps(config, indent=2))

    return config


def delete_lab(lab_id: str, force: bool = False) -> bool:
    """
    Delete a lab and its state.

    Args:
        lab_id: Lab identifier
        force: Skip confirmation for non-empty labs

    Returns:
        True if deleted

    Raises:
        ValueError: If lab doesn't exist
    """
    if not lab_exists(lab_id):
        raise ValueError(f"Lab '{lab_id}' not found.")

    lab_dir = get_lab_dir(lab_id)

    # Check if lab has state (agents, tasks, etc.)
    state_dir = lab_dir / "state"
    if state_dir.exists() and not force:
        agents_file = state_dir / "agents.json"
        if agents_file.exists():
            agents = json.loads(agents_file.read_text())
            running = [a for a in agents.values() if a.get("status") == "running"]
            if running:
                raise ValueError(f"Lab '{lab_id}' has {len(running)} running agents. Use --force to delete anyway.")

    # If this was the active lab, clear it
    active = get_active_lab_id()
    if active == lab_id:
        clear_active_lab()

    # Delete the lab directory
    shutil.rmtree(lab_dir)

    return True


def get_active_lab_id() -> Optional[str]:
    """
    Get the currently active lab ID.

    Returns:
        Active lab ID or None
    """
    ensure_dirs()

    if not ACTIVE_LAB_FILE.exists():
        return None

    try:
        data = json.loads(ACTIVE_LAB_FILE.read_text())
        return data.get("active_lab_id")
    except (json.JSONDecodeError, IOError):
        return None


def get_active_lab() -> Optional[Dict[str, Any]]:
    """
    Get the currently active lab configuration.

    Returns:
        Active lab config or None
    """
    lab_id = get_active_lab_id()
    if lab_id is None:
        return None

    return get_lab(lab_id)


def set_active_lab(lab_id: str) -> Dict[str, Any]:
    """
    Set the active lab.

    Args:
        lab_id: Lab identifier to activate

    Returns:
        The activated lab config

    Raises:
        ValueError: If lab doesn't exist
    """
    config = get_lab(lab_id)
    if config is None:
        raise ValueError(f"Lab '{lab_id}' not found.")

    ensure_dirs()

    data = {
        "active_lab_id": lab_id,
        "activated_at": datetime.now().isoformat()
    }
    ACTIVE_LAB_FILE.write_text(json.dumps(data, indent=2))

    return config


def clear_active_lab():
    """Clear the active lab setting."""
    if ACTIVE_LAB_FILE.exists():
        ACTIVE_LAB_FILE.unlink()


def get_effective_lab_id(override: Optional[str] = None) -> str:
    """
    Get the effective lab ID to use.

    Priority:
    1. Override parameter
    2. CLAUDE_CODE_LAB_ID environment variable
    3. Active lab from state file
    4. Default lab (voice-clone)

    Args:
        override: Optional explicit lab ID

    Returns:
        Lab ID to use
    """
    if override:
        return override

    env_lab = os.environ.get("CLAUDE_CODE_LAB_ID")
    if env_lab:
        return env_lab

    active = get_active_lab_id()
    if active:
        return active

    return DEFAULT_LAB_ID


def get_effective_lab(override: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the effective lab configuration.

    Creates a default lab if none exists and no override specified.

    Args:
        override: Optional explicit lab ID

    Returns:
        Lab configuration dict
    """
    lab_id = get_effective_lab_id(override)

    config = get_lab(lab_id)
    if config:
        return config

    # If the lab doesn't exist, return a virtual config for backwards compatibility
    return {
        "id": lab_id,
        "name": lab_id.replace("-", " ").title(),
        "description": "Auto-generated lab",
        "taskListId": lab_id,
        "active": True,
        "settings": {
            "maxAgents": 3,
            "autoSpawn": True,
            "researchInterval": 30
        }
    }


def get_task_list_id(lab_id: Optional[str] = None) -> str:
    """
    Get the Claude Code task list ID for a lab.

    Args:
        lab_id: Optional lab ID (uses effective lab if not specified)

    Returns:
        Task list ID
    """
    lab = get_effective_lab(lab_id)
    return lab.get("taskListId", lab.get("id", DEFAULT_LAB_ID))


def migrate_global_state_to_lab(lab_id: str, overwrite: bool = False) -> Dict[str, int]:
    """
    Migrate global state files to a lab's state directory.

    Args:
        lab_id: Target lab ID
        overwrite: Whether to overwrite existing files

    Returns:
        Dict with counts of migrated files
    """
    if not lab_exists(lab_id):
        raise ValueError(f"Lab '{lab_id}' not found.")

    lab_state_dir = get_lab_state_dir(lab_id)

    # Files to migrate
    state_files = [
        "agents.json",
        "proposals.json",
        "progress.json",
        "focus.json",
        "research-state.json",
        "reminders.json",
        "cost-tracking.json"
    ]

    migrated = 0
    skipped = 0

    for filename in state_files:
        src = GLOBAL_STATE_DIR / filename
        dst = lab_state_dir / filename

        if not src.exists():
            continue

        if dst.exists() and not overwrite:
            skipped += 1
            continue

        shutil.copy2(src, dst)
        migrated += 1

    # Migrate outputs directory
    global_outputs = GLOBAL_STATE_DIR / "outputs"
    lab_outputs = lab_state_dir / "outputs"

    if global_outputs.exists():
        lab_outputs.mkdir(exist_ok=True)
        for item in global_outputs.iterdir():
            dst = lab_outputs / item.name
            if not dst.exists() or overwrite:
                shutil.copy2(item, dst)
                migrated += 1
            else:
                skipped += 1

    return {"migrated": migrated, "skipped": skipped}


def get_lab_status(lab_id: str) -> Dict[str, Any]:
    """
    Get detailed status for a lab.

    Args:
        lab_id: Lab identifier

    Returns:
        Status dict with task counts, agent info, etc.
    """
    config = get_lab(lab_id)
    if config is None:
        raise ValueError(f"Lab '{lab_id}' not found.")

    state_dir = get_lab_state_dir(lab_id)

    # Count agents
    agents_file = state_dir / "agents.json"
    agents = {}
    if agents_file.exists():
        try:
            agents = json.loads(agents_file.read_text())
        except:
            pass

    running_agents = len([a for a in agents.values() if a.get("status") == "running"])
    total_agents = len(agents)

    # Count proposals
    proposals_file = state_dir / "proposals.json"
    proposals = {"proposals": {}}
    if proposals_file.exists():
        try:
            proposals = json.loads(proposals_file.read_text())
        except:
            pass

    proposal_count = len(proposals.get("proposals", {}))
    pending_proposals = len([p for p in proposals.get("proposals", {}).values()
                            if p.get("status") == "pending_review"])

    # Count tasks from Claude Code task list
    task_list_id = config.get("taskListId", lab_id)
    tasks_dir = Path.home() / ".claude" / "tasks" / task_list_id

    task_counts = {"total": 0, "pending": 0, "in_progress": 0, "completed": 0}
    if tasks_dir.exists():
        for task_file in tasks_dir.glob("*.json"):
            if task_file.name.startswith("."):
                continue
            try:
                task = json.loads(task_file.read_text())
                task_counts["total"] += 1
                status = task.get("status", "pending")
                if status in task_counts:
                    task_counts[status] += 1
            except:
                pass

    # Check if active
    active_lab_id = get_active_lab_id()
    is_active = active_lab_id == lab_id

    return {
        "lab": config,
        "isActive": is_active,
        "agents": {
            "running": running_agents,
            "total": total_agents
        },
        "proposals": {
            "total": proposal_count,
            "pendingReview": pending_proposals
        },
        "tasks": task_counts
    }


if __name__ == "__main__":
    # Simple CLI for testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python labs.py <command> [args]")
        print("Commands: list, get <id>, create <id> <name>, active, set-active <id>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        labs = list_labs()
        active_id = get_active_lab_id()
        for lab in labs:
            marker = "*" if lab.get("id") == active_id else " "
            print(f"{marker} {lab.get('id')}: {lab.get('name')}")

    elif cmd == "get" and len(sys.argv) > 2:
        lab = get_lab(sys.argv[2])
        if lab:
            print(json.dumps(lab, indent=2))
        else:
            print(f"Lab '{sys.argv[2]}' not found")
            sys.exit(1)

    elif cmd == "create" and len(sys.argv) > 3:
        lab = create_lab(sys.argv[2], sys.argv[3])
        print(f"Created lab: {lab['id']}")

    elif cmd == "active":
        lab_id = get_active_lab_id()
        if lab_id:
            print(f"Active lab: {lab_id}")
        else:
            print("No active lab set")

    elif cmd == "set-active" and len(sys.argv) > 2:
        lab = set_active_lab(sys.argv[2])
        print(f"Active lab set to: {lab['id']}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
