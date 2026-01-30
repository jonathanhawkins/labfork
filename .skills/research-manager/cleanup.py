#!/usr/bin/env python3
"""
Agent Cleanup Tool

Identifies and removes stale agents to keep memory clean.
Can be run manually or via cron/launchd for periodic cleanup.

Usage:
    python cleanup.py              # Dry run - show what would be cleaned
    python cleanup.py --execute    # Actually clean up stale agents
    python cleanup.py --force      # Clean all non-running agents
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
STATE_DIR = SCRIPT_DIR / 'state'
AGENTS_FILE = STATE_DIR / 'agents.json'

# Thresholds for staleness
STALE_THRESHOLD_HOURS = 2  # Agents older than 2 hours with no activity
STUCK_KEYWORDS = [
    "Could you clarify",
    "What would you like",
    "Please provide",
    "I need more information",
    "Can you specify",
]
MANAGED_PREFIXES = (
    "rm-task-",
    "rm-web-researcher-",
)

# Maximum allowed claude processes (safety limit)
MAX_CLAUDE_PROCESSES = 10


def load_agents():
    """Load agents from state file."""
    if not AGENTS_FILE.exists():
        return {}
    with open(AGENTS_FILE) as f:
        return json.load(f)


def save_agents(agents):
    """Save agents to state file."""
    with open(AGENTS_FILE, 'w') as f:
        json.dump(agents, f, indent=2)


def get_tmux_sessions():
    """Get list of active tmux sessions."""
    try:
        result = subprocess.run(
            ['tmux', 'list-sessions', '-F', '#{session_name}'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return set(result.stdout.strip().split('\n'))
        return set()
    except Exception:
        return set()


def get_managed_sessions(agents: dict) -> set:
    """Get all tmux session names tracked in agents.json."""
    sessions = set()
    for agent_id, agent in agents.items():
        sessions.add(agent.get('session', f'rm-{agent_id}'))
    return sessions


def find_orphaned_tmux_sessions(agents: dict, tmux_sessions: set) -> list:
    """Find tmux sessions using managed prefixes but missing from agents.json."""
    managed_sessions = get_managed_sessions(agents)
    orphaned = []
    for session in tmux_sessions:
        if not session.startswith(MANAGED_PREFIXES):
            continue
        if session not in managed_sessions:
            orphaned.append({
                'id': None,
                'session': session,
                'reason': 'tmux session not in agents.json',
            })
    return orphaned


def check_agent_stuck(session_name):
    """Check if agent is stuck waiting for input."""
    try:
        result = subprocess.run(
            ['tmux', 'capture-pane', '-t', session_name, '-p'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            output = result.stdout
            for keyword in STUCK_KEYWORDS:
                if keyword.lower() in output.lower():
                    return True, keyword
        return False, None
    except Exception:
        return False, None


def kill_tmux_session(session_name):
    """Kill a tmux session."""
    try:
        subprocess.run(
            ['tmux', 'kill-session', '-t', session_name],
            capture_output=True, timeout=5
        )
        return True
    except Exception:
        return False


def get_tmux_pane_pids():
    """Get PIDs of all processes running in tmux panes."""
    pids = set()
    try:
        # Get all pane PIDs from all sessions
        result = subprocess.run(
            ['tmux', 'list-panes', '-a', '-F', '#{pane_pid}'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for pid in result.stdout.strip().split('\n'):
                if pid:
                    pids.add(int(pid))
                    # Also get child processes recursively
                    try:
                        child_result = subprocess.run(
                            ['pgrep', '-P', pid],
                            capture_output=True, text=True, timeout=5
                        )
                        if child_result.returncode == 0:
                            for child in child_result.stdout.strip().split('\n'):
                                if child:
                                    pids.add(int(child))
                                    # Grandchildren too
                                    gc_result = subprocess.run(
                                        ['pgrep', '-P', child],
                                        capture_output=True, text=True, timeout=5
                                    )
                                    if gc_result.returncode == 0:
                                        for gc in gc_result.stdout.strip().split('\n'):
                                            if gc:
                                                pids.add(int(gc))
                    except Exception:
                        pass
    except Exception:
        pass
    return pids


def get_claude_processes():
    """Get all running claude processes."""
    processes = []
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split('\n'):
            # Match "claude" at end of command (the actual process)
            if line.endswith(' claude') or '/claude ' in line:
                parts = line.split()
                if len(parts) >= 11:
                    try:
                        processes.append({
                            'pid': int(parts[1]),
                            'user': parts[0],
                            'cpu': float(parts[2]),
                            'mem': float(parts[3]),
                            'start': parts[8],
                            'cmd': ' '.join(parts[10:]),
                        })
                    except (ValueError, IndexError):
                        pass
    except Exception:
        pass
    return processes


def find_orphaned_claude_processes():
    """Find claude processes not attached to any tmux session."""
    tmux_pids = get_tmux_pane_pids()
    claude_procs = get_claude_processes()
    orphaned = []

    for proc in claude_procs:
        if proc['pid'] not in tmux_pids:
            orphaned.append(proc)

    return orphaned


def kill_process(pid):
    """Kill a process by PID."""
    try:
        subprocess.run(['kill', str(pid)], timeout=5, capture_output=True)
        return True
    except Exception:
        try:
            subprocess.run(['kill', '-9', str(pid)], timeout=5, capture_output=True)
            return True
        except Exception:
            return False


def analyze_agents():
    """Analyze all agents and categorize them."""
    agents = load_agents()
    tmux_sessions = get_tmux_sessions()
    now = datetime.now()

    results = {
        'stale': [],      # Old agents with no tmux session
        'stuck': [],      # Running but waiting for input
        'orphaned': [],   # tmux session exists but agent record is bad
        'orphaned_tmux': [],  # tmux session exists with managed prefix but no agent record
        'active': [],     # Legitimately running
        'completed': [],  # Marked as completed/killed
    }

    for agent_id, agent in agents.items():
        session_name = agent.get('session', f'rm-{agent_id}')
        status = agent.get('status', 'unknown')
        started_str = agent.get('started', '')

        # Parse start time
        try:
            started = datetime.fromisoformat(started_str)
            age_hours = (now - started).total_seconds() / 3600
        except:
            age_hours = 999  # Very old if we can't parse

        has_tmux = session_name in tmux_sessions

        if status in ('completed', 'killed', 'error'):
            # Check if tmux session still exists (shouldn't)
            if has_tmux:
                results['orphaned'].append({
                    'id': agent_id,
                    'session': session_name,
                    'status': status,
                    'age_hours': age_hours,
                    'reason': 'tmux exists for completed agent',
                })
            else:
                results['completed'].append({
                    'id': agent_id,
                    'status': status,
                    'age_hours': age_hours,
                })
        elif status == 'running':
            if not has_tmux:
                # Marked running but no tmux - stale
                results['stale'].append({
                    'id': agent_id,
                    'session': session_name,
                    'age_hours': age_hours,
                    'reason': 'no tmux session',
                })
            else:
                # Check if stuck
                is_stuck, keyword = check_agent_stuck(session_name)
                if is_stuck:
                    results['stuck'].append({
                        'id': agent_id,
                        'session': session_name,
                        'age_hours': age_hours,
                        'reason': f'waiting for input: "{keyword}"',
                    })
                elif age_hours > STALE_THRESHOLD_HOURS:
                    results['stale'].append({
                        'id': agent_id,
                        'session': session_name,
                        'age_hours': age_hours,
                        'reason': f'running > {STALE_THRESHOLD_HOURS}h',
                    })
                else:
                    results['active'].append({
                        'id': agent_id,
                        'session': session_name,
                        'age_hours': age_hours,
                    })
        else:
            # Unknown status
            if has_tmux:
                results['orphaned'].append({
                    'id': agent_id,
                    'session': session_name,
                    'status': status,
                    'age_hours': age_hours,
                    'reason': f'unknown status: {status}',
                })

    results['orphaned_tmux'] = find_orphaned_tmux_sessions(agents, tmux_sessions)

    # Find orphaned claude processes (not in any tmux session)
    orphaned_procs = find_orphaned_claude_processes()
    results['orphaned_processes'] = orphaned_procs

    return results, agents


def cleanup(execute=False, force=False, verbose=True):
    """Run cleanup process."""
    results, agents = analyze_agents()

    to_clean = results['stale'] + results['stuck'] + results['orphaned'] + results['orphaned_tmux']
    if force:
        # Also clean completed agents older than 24h
        old_completed = [a for a in results['completed'] if a['age_hours'] > 24]
        to_clean.extend([{'id': a['id'], 'reason': 'old completed record'} for a in old_completed])

    # Get orphaned processes
    orphaned_procs = results.get('orphaned_processes', [])

    if verbose:
        print("=" * 60)
        print("AGENT CLEANUP ANALYSIS")
        print("=" * 60)
        print(f"\n  Active agents:       {len(results['active'])}")
        print(f"  Stale agents:        {len(results['stale'])}")
        print(f"  Stuck agents:        {len(results['stuck'])}")
        print(f"  Orphaned agents:     {len(results['orphaned'])}")
        print(f"  Orphaned tmux:       {len(results['orphaned_tmux'])}")
        print(f"  Orphaned processes:  {len(orphaned_procs)}")
        print(f"  Completed agents:    {len(results['completed'])}")
        print(f"\n  To clean: {len(to_clean)} agents + {len(orphaned_procs)} processes")

    if not to_clean:
        if verbose:
            print("\n✓ Nothing to clean up!")
        return 0

    if verbose:
        print("\n" + "-" * 60)
        print("AGENTS TO CLEAN:")
        for agent in to_clean:
            reason = agent.get('reason', 'unknown')
            print(f"  • {agent['id']}: {reason}")

    if not execute:
        if verbose:
            print("\n" + "-" * 60)
            print("DRY RUN - Use --execute to actually clean up")
        return len(to_clean) + len(orphaned_procs)

    # Execute cleanup
    cleaned = 0
    for agent in to_clean:
        agent_id = agent.get('id')
        session = agent.get('session') or (f'rm-{agent_id}' if agent_id else None)

        # Kill tmux session if exists
        if session:
            kill_tmux_session(session)

        # Update agent status
        if agent_id and agent_id in agents:
            agents[agent_id]['status'] = 'killed'
            agents[agent_id]['killed_at'] = datetime.now().isoformat()
            agents[agent_id]['kill_reason'] = agent.get('reason', 'cleanup')

        cleaned += 1
        if verbose:
            print(f"  ✓ Cleaned: {agent_id}")

    # Save updated agents
    save_agents(agents)

    # Kill orphaned processes
    proc_cleaned = 0
    if orphaned_procs:
        if verbose:
            print("\n" + "-" * 60)
            print("ORPHANED PROCESSES TO KILL:")
            for proc in orphaned_procs:
                print(f"  • PID {proc['pid']}: CPU {proc['cpu']}%, started {proc['start']}")

        for proc in orphaned_procs:
            if execute:
                if kill_process(proc['pid']):
                    proc_cleaned += 1
                    if verbose:
                        print(f"  ✓ Killed PID {proc['pid']}")
                else:
                    if verbose:
                        print(f"  ✗ Failed to kill PID {proc['pid']}")

    if verbose:
        print(f"\n✓ Cleaned {cleaned} agents, {proc_cleaned} orphaned processes")

    return cleaned + proc_cleaned


def main():
    parser = argparse.ArgumentParser(description="Agent Cleanup Tool")
    parser.add_argument('--execute', '-e', action='store_true',
                        help='Actually clean up (default is dry run)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Also clean old completed agent records')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Minimal output')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    if args.json:
        results, _ = analyze_agents()
        print(json.dumps(results, indent=2, default=str))
        return 0

    cleanup(
        execute=args.execute,
        force=args.force,
        verbose=not args.quiet
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
