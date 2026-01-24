#!/usr/bin/env python3
"""
Use local Ollama model to triage 129 research implementations.
FREE - no API costs!
"""

import json
import subprocess
from pathlib import Path

def ask_ollama(prompt, model="qwen2.5-coder:14b"):
    """Ask Ollama a question"""
    try:
        result = subprocess.run(
            ['ollama', 'run', model, prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"

def analyze_script(script_path):
    """Use Ollama to analyze a training script"""

    # Read first 100 lines to keep context manageable
    try:
        with open(script_path) as f:
            lines = f.readlines()[:100]
        code_sample = ''.join(lines)
    except:
        return None

    prompt = f"""Analyze this TTS training script and answer:

1. What prosody/emotion control technique does it implement?
2. Is it relevant for controllable voice cloning with keyframe prosody editing?
3. Rate relevance 1-10 (10=perfect fit for keyframe prosody control)

Script: {script_path.name}

```python
{code_sample}
```

Answer ONLY in this JSON format:
{{"technique": "brief description", "relevant": true/false, "score": 1-10, "reason": "one sentence why"}}
"""

    response = ask_ollama(prompt)

    # Try to parse JSON response
    try:
        # Extract JSON from response (Ollama sometimes adds extra text)
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass

    return None

def triage_all_scripts():
    """Analyze all training scripts with Ollama"""
    training_dir = Path('training')

    results = []

    print("🦙 Using Ollama (FREE) to triage 129 implementations...")
    print("Model: qwen2.5-coder:14b (GPT-4 class)")
    print("")

    # Focus on train_*.py scripts first (easier to analyze)
    train_scripts = sorted(training_dir.glob('train_*.py'))

    print(f"Analyzing {len(train_scripts)} training scripts...")
    print("-" * 80)

    for i, script in enumerate(train_scripts, 1):
        print(f"[{i}/{len(train_scripts)}] {script.name}...", end=' ', flush=True)

        analysis = analyze_script(script)

        if analysis:
            results.append({
                'script': script.name,
                **analysis
            })
            score = analysis.get('score', 0)
            relevant = "✓" if analysis.get('relevant') else "✗"
            print(f"{relevant} Score: {score}/10")
        else:
            print("⚠ Failed to parse")

    # Sort by score
    results.sort(key=lambda x: x.get('score', 0), reverse=True)

    # Print top 10
    print("\n" + "=" * 80)
    print("TOP 10 MOST RELEVANT (by local AI)")
    print("=" * 80)

    for i, r in enumerate(results[:10], 1):
        print(f"\n{i}. {r['script']} - Score: {r.get('score', 0)}/10")
        print(f"   Technique: {r.get('technique', 'N/A')}")
        print(f"   Reason: {r.get('reason', 'N/A')}")

    # Save full results
    output_file = 'evaluation/ollama_triage_results.json'
    Path('evaluation').mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Full results saved to: {output_file}")
    print(f"✓ Analyzed {len(results)} scripts using FREE local AI")
    print(f"✓ No API costs!")

if __name__ == '__main__':
    # Check if Ollama is available
    try:
        subprocess.run(['ollama', 'list'], capture_output=True, check=True)
    except:
        print("❌ Ollama not found. Run: brew install ollama")
        exit(1)

    triage_all_scripts()
