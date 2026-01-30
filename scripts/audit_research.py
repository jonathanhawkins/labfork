#!/usr/bin/env python3
"""
Quick audit of all research implementations.
Categorizes and checks basic quality without running expensive evaluations.
"""

import os
import ast
from pathlib import Path
from collections import defaultdict
import sys

def count_loc(filepath):
    """Count lines of code (excluding comments/blanks)"""
    try:
        with open(filepath) as f:
            lines = f.readlines()
        return sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
    except:
        return 0

def check_imports(filepath):
    """Check if file can be parsed and has real imports"""
    try:
        with open(filepath) as f:
            tree = ast.parse(f.read())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        return len(imports) > 0, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, str(e)

def has_training_loop(filepath):
    """Check if file has a training loop"""
    try:
        with open(filepath) as f:
            content = f.read().lower()
        # Look for common training patterns
        patterns = ['for epoch in', 'while', 'optimizer.step', 'loss.backward', 'train_step']
        return any(p in content for p in patterns)
    except:
        return False

def categorize_approach(filename):
    """Categorize by research approach"""
    name = filename.lower()

    categories = {
        'Prosody Conditioning': ['prosody', 'hed', 'keyframe', 'contour'],
        'Emotion Control': ['emotion', 'emo', 'sentiment', 'affect'],
        'Disentanglement': ['disentangle', 'vevo', 'disco', 'learn2diss'],
        'Codec/Compression': ['codec', 'vq', 'quantiz', 'compress'],
        'Flow/Diffusion': ['flow', 'diffusion', 'ddgan', 'ddpm'],
        'RL/Optimization': ['rl', 'dpo', 'ppo', 'reward', 'preference'],
        'Style Transfer': ['style', 'lora', 'adapter', 'restyle'],
        'Multi-modal': ['multi', 'mpe', 'cross'],
    }

    for category, keywords in categories.items():
        if any(kw in name for kw in keywords):
            return category
    return 'Other'

def audit_training_scripts():
    """Audit all training scripts"""
    training_dir = Path('training')

    results = defaultdict(list)

    for script in sorted(training_dir.glob('*.py')):
        if script.name.startswith('__'):
            continue

        loc = count_loc(script)
        can_import, error = check_imports(script)
        has_loop = has_training_loop(script)
        category = categorize_approach(script.name)

        # Calculate quality score
        score = 0
        if loc > 50: score += 1
        if loc > 200: score += 1
        if can_import: score += 2
        if has_loop: score += 1

        quality = 'HIGH' if score >= 4 else 'MED' if score >= 2 else 'LOW'

        results[category].append({
            'name': script.name,
            'loc': loc,
            'can_import': can_import,
            'has_loop': has_loop,
            'quality': quality,
            'score': score,
            'error': error
        })

    return results

def print_audit_report(results):
    """Print organized audit report"""
    print("=" * 80)
    print("RESEARCH IMPLEMENTATION AUDIT")
    print("=" * 80)
    print()

    total_scripts = sum(len(scripts) for scripts in results.values())
    print(f"Total scripts found: {total_scripts}")
    print()

    # Summary by quality
    quality_counts = defaultdict(int)
    for scripts in results.values():
        for s in scripts:
            quality_counts[s['quality']] += 1

    print("Quality Distribution:")
    print(f"  HIGH (likely production-ready): {quality_counts['HIGH']}")
    print(f"  MED  (partial implementation): {quality_counts['MED']}")
    print(f"  LOW  (skeleton/placeholder):   {quality_counts['LOW']}")
    print()

    # By category
    print("=" * 80)
    print("BY CATEGORY")
    print("=" * 80)

    for category in sorted(results.keys()):
        scripts = results[category]
        high_quality = [s for s in scripts if s['quality'] == 'HIGH']

        print(f"\n{category} ({len(scripts)} total, {len(high_quality)} high-quality)")
        print("-" * 80)

        # Show high-quality ones first
        for script in sorted(scripts, key=lambda x: (-x['score'], x['name'])):
            status = "✓" if script['can_import'] else "✗"
            loop = "🔁" if script['has_loop'] else "  "

            print(f"  {status} {loop} [{script['quality']:3s}] {script['name']:40s} ({script['loc']:4d} LOC)")

            if script['error']:
                print(f"        ERROR: {script['error'][:60]}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    high_quality_scripts = []
    for category, scripts in results.items():
        for s in scripts:
            if s['quality'] == 'HIGH':
                high_quality_scripts.append((category, s))

    print(f"\nFound {len(high_quality_scripts)} HIGH-quality implementations worth evaluating:")
    print()

    for category, script in sorted(high_quality_scripts, key=lambda x: -x[1]['score'])[:15]:
        print(f"  • {script['name']:45s} ({category})")

    print("\n" + "=" * 80)
    print(f"NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Review the {len(high_quality_scripts)} high-quality implementations above
2. Pick 5-10 most relevant to your prosody/emotion goals
3. Run S4 evaluation on ONLY those finalists (save $$)
4. Archive/delete the LOW quality skeleton files

This avoids wasting money evaluating {quality_counts['LOW']} placeholder files!
""")

if __name__ == '__main__':
    results = audit_training_scripts()
    print_audit_report(results)
