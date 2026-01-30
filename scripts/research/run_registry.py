#!/usr/bin/env python3
"""
Research Run Registry

World-class, file-based registry for experiment runs.
Creates/updates run.json under outputs/research/runs/<run_id>/
"""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "outputs" / "research" / "runs"


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "run"


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def load_run(run_dir: Path) -> Dict[str, Any]:
    run_file = run_dir / "run.json"
    if not run_file.exists():
        raise FileNotFoundError(f"run.json not found in {run_dir}")
    return json.loads(run_file.read_text())


def save_run(run_dir: Path, data: Dict[str, Any]):
    run_dir.mkdir(parents=True, exist_ok=True)
    run_file = run_dir / "run.json"
    run_file.write_text(json.dumps(data, indent=2))


def resolve_run_dir(run_id: Optional[str], run_dir: Optional[str]) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    if not run_id:
        raise ValueError("Must provide --run-id or --run-dir")
    return RUNS_DIR / run_id


def cmd_new(args: argparse.Namespace):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = slugify(args.title or args.technique or args.task_id or "run")
    run_id = args.run_id or f"{ts}_{slug}"

    run_dir = RUNS_DIR / run_id
    if run_dir.exists() and not args.allow_existing:
        raise FileExistsError(f"Run dir already exists: {run_dir}")

    commit = get_git_commit()

    run = {
        "run_id": run_id,
        "title": args.title or "",
        "task_id": args.task_id or "",
        "technique": args.technique or "",
        "status": args.status or "running",
        "owner": args.owner or "",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "code": {
            "commit": commit or "",
        },
        "config": {
            "path": args.config or "",
        },
        "artifacts": {
            "checkpoint": args.checkpoint or "",
            "samples": [],
            "logs": [],
        },
        "metrics": {},
        "review": {
            "status": "pending",
            "reviewer": "",
            "reviewed_at": "",
            "reason": "",
        },
        "notes": args.notes or "",
    }

    save_run(run_dir, run)
    print(str(run_dir))


def cmd_update(args: argparse.Namespace):
    run_dir = resolve_run_dir(args.run_id, args.run_dir)
    run = load_run(run_dir)

    if args.status:
        run["status"] = args.status
    if args.owner:
        run["owner"] = args.owner
    if args.notes:
        run["notes"] = args.notes
    if args.config:
        run.setdefault("config", {})["path"] = args.config
    if args.checkpoint:
        run.setdefault("artifacts", {})["checkpoint"] = args.checkpoint

    if args.artifact:
        run.setdefault("artifacts", {}).setdefault("other", [])
        run["artifacts"]["other"].extend(args.artifact)

    if args.metrics_file:
        metrics_path = Path(args.metrics_file).expanduser().resolve()
        metrics = json.loads(metrics_path.read_text())
        run.setdefault("metrics", {})
        run["metrics"].update(metrics)

    run["updated_at"] = now_iso()
    save_run(run_dir, run)
    print(str(run_dir / "run.json"))


def cmd_review(args: argparse.Namespace):
    run_dir = resolve_run_dir(args.run_id, args.run_dir)
    run = load_run(run_dir)

    run.setdefault("review", {})
    run["review"]["status"] = args.status
    run["review"]["reviewer"] = args.reviewer or run["review"].get("reviewer", "")
    run["review"]["reviewed_at"] = now_iso()
    run["review"]["reason"] = args.reason or run["review"].get("reason", "")

    run["updated_at"] = now_iso()
    save_run(run_dir, run)
    print(str(run_dir / "run.json"))


def cmd_list(args: argparse.Namespace):
    if not RUNS_DIR.exists():
        print("No runs directory")
        return

    runs = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        run_file = run_dir / "run.json"
        if not run_file.exists():
            continue
        try:
            run = json.loads(run_file.read_text())
            runs.append(run)
        except json.JSONDecodeError:
            continue

    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)

    if args.json:
        print(json.dumps(runs, indent=2))
        return

    for run in runs[: args.limit]:
        status = run.get("status", "")
        review = run.get("review", {}).get("status", "")
        title = run.get("title") or run.get("technique") or run.get("run_id")
        print(f"{run.get('run_id')}: {title} | status={status} | review={review}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research run registry")
    sub = parser.add_subparsers(dest="command", required=True)

    new = sub.add_parser("new", help="Create a new run")
    new.add_argument("--title", help="Run title")
    new.add_argument("--task-id", help="Task ID")
    new.add_argument("--technique", help="Technique name")
    new.add_argument("--config", help="Config path")
    new.add_argument("--checkpoint", help="Checkpoint path")
    new.add_argument("--owner", help="Owner name")
    new.add_argument("--status", help="Status", default="running")
    new.add_argument("--notes", help="Notes")
    new.add_argument("--run-id", help="Override run ID")
    new.add_argument("--allow-existing", action="store_true", help="Allow existing run dir")
    new.set_defaults(func=cmd_new)

    update = sub.add_parser("update", help="Update an existing run")
    update.add_argument("--run-id", help="Run ID")
    update.add_argument("--run-dir", help="Run directory")
    update.add_argument("--status", help="Status")
    update.add_argument("--owner", help="Owner name")
    update.add_argument("--notes", help="Notes")
    update.add_argument("--config", help="Config path")
    update.add_argument("--checkpoint", help="Checkpoint path")
    update.add_argument("--metrics-file", help="JSON file with metrics")
    update.add_argument("--artifact", action="append", help="Additional artifact path (repeatable)")
    update.set_defaults(func=cmd_update)

    review = sub.add_parser("review", help="Review a run")
    review.add_argument("--run-id", help="Run ID")
    review.add_argument("--run-dir", help="Run directory")
    review.add_argument("--status", choices=["approved", "rejected", "pending"], required=True)
    review.add_argument("--reviewer", help="Reviewer name")
    review.add_argument("--reason", help="Reason for decision")
    review.set_defaults(func=cmd_review)

    list_cmd = sub.add_parser("list", help="List runs")
    list_cmd.add_argument("--limit", type=int, default=20, help="Max runs to show")
    list_cmd.add_argument("--json", action="store_true", help="Output JSON")
    list_cmd.set_defaults(func=cmd_list)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
