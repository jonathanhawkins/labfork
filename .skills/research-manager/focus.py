#!/usr/bin/env python3
"""
Focus Mode - Halt all work until a priority task is complete.

When focus mode is active:
- No new agents are spawned (orchestrator respects this)
- Existing non-priority agents can be killed
- A clear message shows what needs to be done
- Focus is released when the task is marked complete or manually cleared

Usage:
    python focus.py set "V7 verification" --task-id 6
    python focus.py status
    python focus.py clear
    python focus.py block   # Check if spawning is blocked
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
STATE_DIR = SCRIPT_DIR / 'state'
FOCUS_FILE = STATE_DIR / 'focus.json'


def load_focus():
    """Load focus state."""
    if FOCUS_FILE.exists():
        with open(FOCUS_FILE) as f:
            return json.load(f)
    return None


def save_focus(focus):
    """Save focus state."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(FOCUS_FILE, 'w') as f:
        json.dump(focus, f, indent=2)


def clear_focus():
    """Clear focus mode."""
    if FOCUS_FILE.exists():
        FOCUS_FILE.unlink()


def is_blocked():
    """Check if spawning is blocked by focus mode."""
    focus = load_focus()
    if not focus:
        return False, None

    # Check if task is complete (if task_id provided)
    if focus.get('task_id'):
        task_status = get_task_status(focus['task_id'])
        if task_status == 'completed':
            # Auto-clear focus when task is done
            clear_focus()
            return False, None

    return True, focus


def get_task_status(task_id):
    """Get status of a task from Claude Code task list."""
    import os
    tasks_dir = Path.home() / '.claude' / 'tasks'
    if not tasks_dir.exists():
        return None

    # Find most recent session
    sessions = sorted(tasks_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for session_dir in sessions:
        if not session_dir.is_dir():
            continue
        tasks_file = session_dir / 'tasks.json'
        if tasks_file.exists():
            with open(tasks_file) as f:
                data = json.load(f)
                for task in data.get('tasks', []):
                    if task.get('id') == str(task_id):
                        return task.get('status')
    return None


def cmd_set(args):
    """Set focus on a priority task."""
    focus = {
        'description': args.description,
        'task_id': args.task_id,
        'set_at': datetime.now().isoformat(),
        'set_by': 'manual',
        'reason': args.reason or 'Priority task',
    }
    save_focus(focus)

    print("=" * 60)
    print("🎯 FOCUS MODE ACTIVATED")
    print("=" * 60)
    print(f"\nPriority: {args.description}")
    if args.task_id:
        print(f"Task ID: #{args.task_id}")
    if args.reason:
        print(f"Reason: {args.reason}")
    print("\n⛔ All agent spawning is now BLOCKED")
    print("⛔ Research agents will not be created")
    print("\nTo clear: python focus.py clear")
    if args.task_id:
        print(f"Auto-clears when task #{args.task_id} is marked complete")

    return 0


def cmd_status(args):
    """Show current focus status."""
    focus = load_focus()

    print("=" * 60)
    print("FOCUS MODE STATUS")
    print("=" * 60)

    if not focus:
        print("\n✓ Focus mode is OFF")
        print("  Agents can be spawned normally")
        return 0

    print(f"\n🎯 FOCUS MODE ACTIVE")
    print(f"\nPriority: {focus['description']}")
    if focus.get('task_id'):
        status = get_task_status(focus['task_id'])
        print(f"Task: #{focus['task_id']} ({status or 'unknown'})")
        if status == 'completed':
            print("\n✓ Task is complete! Clearing focus mode...")
            clear_focus()
            return 0
    print(f"Set at: {focus['set_at']}")
    print(f"Reason: {focus.get('reason', 'N/A')}")

    print("\n⛔ Agent spawning is BLOCKED")
    print("\nTo clear: rm focus clear")

    return 0


def cmd_clear(args):
    """Clear focus mode."""
    focus = load_focus()
    if not focus:
        print("Focus mode is not active")
        return 0

    clear_focus()
    print("✓ Focus mode cleared")
    print("  Agents can now be spawned normally")
    return 0


def cmd_block(args):
    """Check if spawning is blocked (for orchestrator)."""
    blocked, focus = is_blocked()
    if args.json:
        print(json.dumps({
            'blocked': blocked,
            'focus': focus,
        }))
    else:
        if blocked:
            print(f"BLOCKED: {focus['description']}")
            sys.exit(1)
        else:
            print("OK")
            sys.exit(0)
    return 0 if not blocked else 1


def main():
    parser = argparse.ArgumentParser(description="Focus Mode - Priority Task Lock")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # set
    set_parser = subparsers.add_parser('set', help='Set focus on a priority task')
    set_parser.add_argument('description', help='What needs to be done')
    set_parser.add_argument('--task-id', '-t', type=int, help='Task ID to track (auto-clears when complete)')
    set_parser.add_argument('--reason', '-r', help='Why this is priority')

    # status
    subparsers.add_parser('status', help='Show focus status')

    # clear
    subparsers.add_parser('clear', help='Clear focus mode')

    # block (for orchestrator)
    block_parser = subparsers.add_parser('block', help='Check if blocked (exit 1 if blocked)')
    block_parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    if not args.command:
        # Default to status
        args.command = 'status'

    commands = {
        'set': cmd_set,
        'status': cmd_status,
        'clear': cmd_clear,
        'block': cmd_block,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
