#!/usr/bin/env python3
"""
Haiku-Supervised Lab Manager
============================
Uses Claude Haiku (~$0.25/1M tokens) to supervise FREE local Ollama lab manager.

Pattern: Haiku plans → Ollama executes (via tmux) → Haiku validates → repeat

Cost: ~$0.03-0.10 per 100 tasks (vs $50-100 with Opus agents)
"""

import json
import subprocess
import time
import argparse
import os
from pathlib import Path
from typing import Optional
from datetime import datetime

# Try to import anthropic, give helpful error if missing
try:
    import anthropic
except ImportError:
    print("❌ anthropic package not installed. Run: pip install anthropic")
    exit(1)

# Configuration
HAIKU_MODEL = "claude-3-5-haiku-latest"  # Cheapest Claude model
OLLAMA_SESSION = "lab-manager"  # tmux session name
PROJECT_ROOT = Path(__file__).parent.parent
TASKS_API = "http://localhost:3003/api/tasks"
MAX_ITERATIONS = 50
ITERATION_DELAY = 30  # seconds between supervision cycles

# Cost tracking
cost_tracker = {
    "input_tokens": 0,
    "output_tokens": 0,
    "iterations": 0,
    "tasks_completed": 0,
    "start_time": None
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp"""
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERR": "❌", "COST": "💰"}.get(level, "•")
    print(f"[{ts}] {prefix} {msg}")


def get_cost_report() -> str:
    """Calculate and return cost report"""
    # Haiku 3.5 pricing: $0.80 input, $4.00 output per 1M tokens
    input_cost = cost_tracker["input_tokens"] * 0.80 / 1_000_000
    output_cost = cost_tracker["output_tokens"] * 4.00 / 1_000_000
    total_cost = input_cost + output_cost

    elapsed = ""
    if cost_tracker["start_time"]:
        mins = (time.time() - cost_tracker["start_time"]) / 60
        elapsed = f" in {mins:.1f}min"

    return (
        f"Haiku: ${total_cost:.4f} ({cost_tracker['input_tokens']:,} in, {cost_tracker['output_tokens']:,} out) | "
        f"Ollama: $0.00 | "
        f"Iterations: {cost_tracker['iterations']} | "
        f"Tasks: {cost_tracker['tasks_completed']}{elapsed}"
    )


def ask_haiku(prompt: str, system: str = "") -> str:
    """Ask Haiku a question (cheap supervisor)"""
    client = anthropic.Anthropic()

    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=1024,
        system=system or "You are a concise task supervisor. Return JSON when asked.",
        messages=messages
    )

    # Track costs
    cost_tracker["input_tokens"] += response.usage.input_tokens
    cost_tracker["output_tokens"] += response.usage.output_tokens

    return response.content[0].text


def send_to_lab_manager(instruction: str) -> bool:
    """Send instruction to lab manager tmux session"""
    try:
        # Send the instruction
        subprocess.run(
            ["tmux", "send-keys", "-t", OLLAMA_SESSION, instruction, "Enter"],
            check=True,
            capture_output=True
        )
        # Send extra Enter to submit
        time.sleep(1)
        subprocess.run(
            ["tmux", "send-keys", "-t", OLLAMA_SESSION, "Enter"],
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        log(f"Failed to send to tmux: {e}", "ERR")
        return False


def get_lab_manager_output(lines: int = 100) -> str:
    """Capture recent output from lab manager tmux session"""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", OLLAMA_SESSION, "-p", "-S", f"-{lines}"],
            capture_output=True,
            text=True
        )
        return result.stdout
    except:
        return ""


def check_lab_manager_running() -> bool:
    """Check if lab manager tmux session exists"""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", OLLAMA_SESSION],
            capture_output=True
        )
        return result.returncode == 0
    except:
        return False


def start_lab_manager() -> bool:
    """Start the lab manager if not running"""
    if check_lab_manager_running():
        log("Lab manager already running", "OK")
        return True

    log("Starting lab manager...")
    try:
        # Kill any existing session
        subprocess.run(["tmux", "kill-session", "-t", OLLAMA_SESSION],
                      capture_output=True)
    except:
        pass

    # Start new session with claude-free
    script_path = PROJECT_ROOT / "scripts" / "claude-free"
    subprocess.run([
        "tmux", "new-session", "-d", "-s", OLLAMA_SESSION,
        "-x", "140", "-y", "40", str(script_path)
    ])

    # Wait for initialization
    log("Waiting for Claude Code to initialize...")
    time.sleep(15)

    # Send initial prompt
    init_prompt = """You are the LAB MANAGER. Your job is to execute tasks I give you.
When I give you a task:
1. Use TaskList to check current tasks
2. Execute the specific instruction I give
3. Report back what you did
4. Wait for my next instruction

Confirm you're ready by saying READY."""

    send_to_lab_manager(init_prompt)
    time.sleep(5)

    return check_lab_manager_running()


def get_pending_tasks() -> list:
    """Get pending tasks from the tasks API"""
    try:
        result = subprocess.run(
            ["curl", "-s", TASKS_API],
            capture_output=True,
            text=True
        )
        data = json.loads(result.stdout)
        tasks = data.get("tasks", [])
        return [t for t in tasks if t.get("status") == "pending"]
    except Exception as e:
        log(f"Failed to get tasks: {e}", "ERR")
        return []


def parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response"""
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None


def supervision_cycle(pending_tasks: list) -> dict:
    """
    One cycle of supervision:
    1. Haiku picks a task and creates instruction
    2. Send instruction to Ollama lab manager
    3. Wait for execution
    4. Haiku validates result
    """
    cost_tracker["iterations"] += 1

    # Get current lab manager state
    current_output = get_lab_manager_output(50)

    # Ask Haiku to plan
    plan_prompt = f"""You are supervising a local AI lab manager (running on Ollama).

PENDING TASKS:
{json.dumps([{"id": t["id"], "subject": t["subject"]} for t in pending_tasks[:5]], indent=2)}

CURRENT LAB MANAGER STATE (last 50 lines):
{current_output[-2000:]}

Based on the state, decide what instruction to give the lab manager.
Return JSON:
{{
  "action": "work_on_task" | "check_status" | "wait" | "done",
  "task_id": "id if working on task",
  "instruction": "specific instruction to send (under 200 words)",
  "reason": "why this action"
}}

Rules:
- If lab manager is still processing (shows "Jitterbugging" or similar), action="wait"
- If lab manager just finished something, action="check_status" with instruction to use TaskList
- If ready for work, pick highest priority pending task
- Keep instructions specific and actionable"""

    plan_response = ask_haiku(plan_prompt)
    plan = parse_json_response(plan_response)

    if not plan:
        log(f"Failed to parse plan: {plan_response[:100]}", "WARN")
        return {"action": "error", "reason": "Parse failed"}

    log(f"Action: {plan.get('action')} - {plan.get('reason', '')[:50]}")

    action = plan.get("action", "wait")

    if action == "wait":
        return {"action": "wait", "reason": plan.get("reason")}

    if action == "done":
        return {"action": "done", "reason": plan.get("reason")}

    # Send instruction to lab manager
    instruction = plan.get("instruction", "")
    if instruction:
        log(f"→ Sending: {instruction[:80]}...")
        if send_to_lab_manager(instruction):
            # Wait for lab manager to process
            time.sleep(60)  # Local model is slow

            # Check result
            new_output = get_lab_manager_output(50)

            # Ask Haiku to validate
            validate_prompt = f"""Did the lab manager complete the instruction?

INSTRUCTION SENT: {instruction}

LAB MANAGER OUTPUT (after instruction):
{new_output[-2000:]}

Return JSON:
{{
  "completed": true/false,
  "evidence": "what shows completion",
  "next_action": "continue" | "retry" | "skip"
}}"""

            validation = ask_haiku(validate_prompt)
            result = parse_json_response(validation)

            if result and result.get("completed"):
                cost_tracker["tasks_completed"] += 1
                log(f"Task completed: {result.get('evidence', '')[:50]}", "OK")

            return {
                "action": action,
                "task_id": plan.get("task_id"),
                "validation": result
            }

    return {"action": action, "reason": "No instruction generated"}


def run_supervisor(max_iterations: int = MAX_ITERATIONS, continuous: bool = False):
    """Main supervisor loop"""
    cost_tracker["start_time"] = time.time()

    log("=" * 60)
    log("HAIKU-SUPERVISED LAB MANAGER")
    log(f"Model: {HAIKU_MODEL} (supervisor) + Ollama (worker)")
    log(f"Max iterations: {max_iterations}")
    log("=" * 60)

    # Ensure lab manager is running
    if not start_lab_manager():
        log("Failed to start lab manager", "ERR")
        return

    iteration = 0
    consecutive_waits = 0

    while iteration < max_iterations:
        iteration += 1
        log(f"\n--- Iteration {iteration}/{max_iterations} ---")

        # Get pending tasks
        pending_tasks = get_pending_tasks()
        log(f"Pending tasks: {len(pending_tasks)}")

        if not pending_tasks and not continuous:
            log("No pending tasks. Stopping.", "OK")
            break

        # Run supervision cycle
        result = supervision_cycle(pending_tasks if pending_tasks else [{"id": "0", "subject": "Check status"}])

        if result.get("action") == "wait":
            consecutive_waits += 1
            if consecutive_waits > 5:
                log("Too many waits, lab manager may be stuck", "WARN")
                # Try to unstick
                send_to_lab_manager("Are you stuck? Use TaskList to see pending tasks.")
                consecutive_waits = 0
        else:
            consecutive_waits = 0

        if result.get("action") == "done":
            log("Supervisor thinks work is done", "OK")
            if not continuous:
                break

        # Cost report every 10 iterations
        if iteration % 10 == 0:
            log(get_cost_report(), "COST")

        # Delay before next cycle
        log(f"Waiting {ITERATION_DELAY}s before next cycle...")
        time.sleep(ITERATION_DELAY)

    # Final report
    log("\n" + "=" * 60)
    log("FINAL REPORT")
    log("=" * 60)
    log(get_cost_report(), "COST")


def main():
    parser = argparse.ArgumentParser(
        description="Haiku-supervised Ollama lab manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run 50 iterations then stop
  %(prog)s --continuous       # Run continuously until stopped
  %(prog)s --iterations 100   # Run 100 iterations
  %(prog)s --check            # Just check if lab manager is running
  %(prog)s --start            # Just start lab manager, don't supervise

Cost: ~$0.03-0.10 per 100 tasks (Haiku) + $0 (Ollama)
        """
    )

    parser.add_argument(
        "--continuous", "-c",
        action="store_true",
        help="Run continuously (until Ctrl+C)"
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Max supervision iterations (default: {MAX_ITERATIONS})"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Just check if lab manager is running"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Just start lab manager, don't supervise"
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=ITERATION_DELAY,
        help=f"Seconds between cycles (default: {ITERATION_DELAY})"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY not set")
        print("   export ANTHROPIC_API_KEY='sk-ant-...'")
        exit(1)

    global ITERATION_DELAY
    ITERATION_DELAY = args.delay

    if args.check:
        if check_lab_manager_running():
            print("✅ Lab manager is running")
            print(f"   Attach: tmux attach -t {OLLAMA_SESSION}")
        else:
            print("❌ Lab manager is not running")
            print(f"   Start:  python {__file__} --start")
        return

    if args.start:
        if start_lab_manager():
            print("✅ Lab manager started")
            print(f"   Attach: tmux attach -t {OLLAMA_SESSION}")
        else:
            print("❌ Failed to start lab manager")
        return

    # Run supervisor
    try:
        run_supervisor(
            max_iterations=args.iterations if not args.continuous else 999999,
            continuous=args.continuous
        )
    except KeyboardInterrupt:
        print("\n")
        log("Interrupted by user")
        log(get_cost_report(), "COST")


if __name__ == "__main__":
    main()
