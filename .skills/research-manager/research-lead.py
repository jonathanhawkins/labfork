#!/usr/bin/env python3
"""
Research Lead - Decision Support Tool

Helps the manager/lead make informed decisions about research directions.

Commands:
    status      - Show current research status and priorities
    evaluate    - List techniques that need evaluation
    decide      - Interactive decision helper for a technique
    compare     - Compare two techniques head-to-head
    focus       - Show what to focus on this week
    reject      - Mark a technique as rejected with reason
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
STATE_DIR = SCRIPT_DIR / 'state'
DECISIONS_FILE = STATE_DIR / 'research-decisions.json'


def load_decisions():
    """Load research decisions from file."""
    if DECISIONS_FILE.exists():
        with open(DECISIONS_FILE) as f:
            return json.load(f)
    return {"approved": [], "rejected": [], "pending": [], "evaluated": []}


def save_decisions(decisions):
    """Save research decisions to file."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DECISIONS_FILE, 'w') as f:
        json.dump(decisions, f, indent=2)


def extract_techniques_from_configs():
    """Extract technique names from training config files."""
    config_dir = PROJECT_ROOT / 'training' / 'config'

    # Base/hardware configs to exclude (not research techniques)
    excluded = {
        'baseline_no_prosody',
        'm4_pro',
        'm4_pro_deepseek',
        'prosody_conditioned',
        'prosody_joint_training',
        'prosody_joint_v4',
        'prosody_v5',
        'prosody_v7_balanced',
        'rtx_4090',
        'rtx_4090_deepseek',
        'rtx_4090_lora',
        'rtx_4090_lora_450',
    }

    techniques = []
    if not config_dir.exists():
        return techniques

    for config_file in config_dir.glob('*.yaml'):
        name = config_file.stem
        if name not in excluded:
            # Format: "emo_film" -> "Emo Film"
            formatted = ' '.join(word.capitalize() for word in name.split('_'))
            techniques.append(formatted)

    return sorted(techniques)


def cmd_status(args):
    """Show current research status."""
    decisions = load_decisions()
    techniques = extract_techniques_from_configs()

    print("\n" + "=" * 60)
    print("RESEARCH STATUS")
    print("=" * 60)

    print(f"\nTotal techniques documented: {len(techniques)}")
    print(f"  Approved:  {len(decisions['approved'])}")
    print(f"  Rejected:  {len(decisions['rejected'])}")
    print(f"  Evaluated: {len(decisions['evaluated'])}")
    print(f"  Pending:   {len(techniques) - len(decisions['approved']) - len(decisions['rejected'])}")

    # Core goal reminder
    print("\n" + "-" * 60)
    print("CORE GOAL: Prosody labels → better voice cloning with limited data")
    print("SUCCESS METRICS:")
    print("  - F0 separation: Happy > Sad by 30+ Hz")
    print("  - Emotion accuracy: ≥ 50%")
    print("-" * 60)

    # Current priorities
    print("\nP0 - CRITICAL (Do Now):")
    print("  [ ] V7 LoRA training fix → verify end-to-end")
    print("  [ ] Run quick_eval.py on V7 checkpoint")

    print("\nP1 - HIGH IMPACT (This Week):")
    p1_items = [
        "Fix angry/neutral emotion accuracy (currently 0%)",
        "Evaluate top 3 promising techniques from research",
    ]
    for item in p1_items:
        print(f"  [ ] {item}")

    return 0


def cmd_evaluate(args):
    """List techniques that need evaluation."""
    decisions = load_decisions()
    techniques = extract_techniques_from_configs()

    evaluated_set = set(decisions.get('evaluated', []))
    approved_set = set(decisions.get('approved', []))
    rejected_set = set(decisions.get('rejected', []))

    unevaluated = [t for t in techniques if t not in evaluated_set and t not in approved_set and t not in rejected_set]

    print("\n" + "=" * 60)
    print("TECHNIQUES NEEDING EVALUATION")
    print("=" * 60)

    if not unevaluated:
        print("\nAll techniques have been evaluated!")
        return 0

    print(f"\n{len(unevaluated)} techniques need evaluation:\n")

    # Categorize by likely impact
    high_impact_keywords = ['emotion', 'prosody', 'f0', 'pitch', 'energy']
    medium_impact_keywords = ['speaker', 'disentangle', 'vad', 'intensity']

    high = []
    medium = []
    low = []

    for tech in unevaluated:
        tech_lower = tech.lower()
        if any(kw in tech_lower for kw in high_impact_keywords):
            high.append(tech)
        elif any(kw in tech_lower for kw in medium_impact_keywords):
            medium.append(tech)
        else:
            low.append(tech)

    if high:
        print("HIGH PRIORITY (emotion/prosody related):")
        for t in high[:5]:  # Limit to top 5
            print(f"  • {t}")
        if len(high) > 5:
            print(f"  ... and {len(high) - 5} more")

    if medium:
        print("\nMEDIUM PRIORITY (disentanglement/speaker related):")
        for t in medium[:5]:
            print(f"  • {t}")
        if len(medium) > 5:
            print(f"  ... and {len(medium) - 5} more")

    if low:
        print("\nLOW PRIORITY (other):")
        for t in low[:5]:
            print(f"  • {t}")
        if len(low) > 5:
            print(f"  ... and {len(low) - 5} more")

    print("\n" + "-" * 60)
    print("To evaluate a technique:")
    print("  1. Run: python inference/quick_eval.py --checkpoint <model>")
    print("  2. Run: python research-lead.py decide '<technique_name>'")
    print("-" * 60)

    return 0


def cmd_decide(args):
    """Interactive decision helper for a technique."""
    technique_name = args.technique
    decisions = load_decisions()

    print("\n" + "=" * 60)
    print(f"DECISION: {technique_name}")
    print("=" * 60)

    print("\nGate 1 - RELEVANCE:")
    print("  Does this help prosody/emotion control for voice cloning?")

    print("\nGate 2 - TESTABILITY:")
    print("  Can we evaluate it in < 4 hours with existing infra?")

    print("\nGate 3 - IMPACT:")
    print("  Expected improvement on F0 separation or emotion accuracy?")

    print("\n" + "-" * 60)

    # Interactive prompts
    relevance = input("Gate 1 - Relevant? (y/n/skip): ").strip().lower()
    if relevance == 'n':
        reason = input("Rejection reason: ").strip()
        decisions.setdefault('rejected', []).append({
            'name': technique_name,
            'reason': reason,
            'gate': 'relevance',
            'date': datetime.now().isoformat()
        })
        save_decisions(decisions)
        print(f"\n✗ REJECTED: {technique_name} (not relevant)")
        return 0

    testable = input("Gate 2 - Testable in 4h? (y/n/skip): ").strip().lower()
    if testable == 'n':
        reason = input("Deferral reason: ").strip()
        decisions.setdefault('pending', []).append({
            'name': technique_name,
            'reason': reason,
            'gate': 'testability',
            'date': datetime.now().isoformat()
        })
        save_decisions(decisions)
        print(f"\n⏸ DEFERRED: {technique_name} (not easily testable)")
        return 0

    impact = input("Gate 3 - Expected impact (high/medium/low): ").strip().lower()

    # Record evaluation
    eval_result = input("Has this been evaluated with quick_eval.py? (y/n): ").strip().lower()
    if eval_result == 'y':
        passed = input("Did it pass? (y/n): ").strip().lower()
        decisions.setdefault('evaluated', []).append(technique_name)

        if passed == 'y':
            decisions.setdefault('approved', []).append({
                'name': technique_name,
                'impact': impact,
                'date': datetime.now().isoformat()
            })
            save_decisions(decisions)
            print(f"\n✓ APPROVED: {technique_name} (impact: {impact})")
        else:
            reason = input("Why did it fail? ").strip()
            decisions.setdefault('rejected', []).append({
                'name': technique_name,
                'reason': reason,
                'gate': 'evaluation',
                'date': datetime.now().isoformat()
            })
            save_decisions(decisions)
            print(f"\n✗ REJECTED: {technique_name} (failed evaluation)")
    else:
        print(f"\n→ NEXT STEP: Run evaluation")
        print(f"  python inference/quick_eval.py --checkpoint <{technique_name}_model>")

    return 0


def cmd_focus(args):
    """Show what to focus on this week."""
    decisions = load_decisions()

    print("\n" + "=" * 60)
    print("WEEKLY FOCUS")
    print("=" * 60)

    print("\n🎯 THIS WEEK'S PRIORITIES:\n")

    print("1. COMPLETE V7 VERIFICATION")
    print("   - Sync code to RTX 4090")
    print("   - Run training with LoRA fix")
    print("   - Verify: Happy F0 > Sad F0")
    print("")

    print("2. EVALUATE TOP 3 TECHNIQUES")
    print("   Focus on emotion accuracy improvements:")
    techniques_to_eval = [
        "EmoKnob (direction vectors)",
        "Emo-FiLM (word-level emotion)",
        "Activation Steering (training-free)",
    ]
    for i, tech in enumerate(techniques_to_eval, 1):
        print(f"   {i}. {tech}")
    print("")

    print("3. DECIDE: KEEP OR REJECT")
    print("   After evaluation, update research-decisions.json")
    print("")

    print("❌ DO NOT:")
    print("   - Start new research tasks")
    print("   - Add to CLAUDE.md")
    print("   - Implement unevaluated techniques")

    print("\n" + "-" * 60)
    print("Run 'python research-lead.py status' for full status")
    print("-" * 60)

    return 0


def cmd_reject(args):
    """Mark a technique as rejected."""
    decisions = load_decisions()

    decisions.setdefault('rejected', []).append({
        'name': args.technique,
        'reason': args.reason,
        'gate': 'manual',
        'date': datetime.now().isoformat()
    })
    save_decisions(decisions)

    print(f"✗ REJECTED: {args.technique}")
    print(f"   Reason: {args.reason}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Research Lead - Decision Support Tool")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # status
    subparsers.add_parser('status', help='Show research status')

    # evaluate
    subparsers.add_parser('evaluate', help='List techniques needing evaluation')

    # decide
    decide_parser = subparsers.add_parser('decide', help='Interactive decision helper')
    decide_parser.add_argument('technique', help='Technique name to decide on')

    # focus
    subparsers.add_parser('focus', help='Show weekly focus')

    # reject
    reject_parser = subparsers.add_parser('reject', help='Mark technique as rejected')
    reject_parser.add_argument('technique', help='Technique name')
    reject_parser.add_argument('--reason', '-r', required=True, help='Rejection reason')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        'status': cmd_status,
        'evaluate': cmd_evaluate,
        'decide': cmd_decide,
        'focus': cmd_focus,
        'reject': cmd_reject,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
