#!/usr/bin/env python3
"""
Frontend Designer - Generates demo pages for research techniques.

Usage:
    python3 .skills/frontend-designer/generate.py \
        --technique emoknob \
        --inference inference/generate_with_emoknob.py \
        --description "Direction vector emotion control"
"""

import argparse
import ast
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def parse_inference_script(script_path: Path) -> Dict:
    """Parse inference script to extract parameters."""
    if not script_path.exists():
        print(f"Warning: Inference script not found: {script_path}")
        return {"params": [], "description": ""}

    content = script_path.read_text()
    params = []

    # Find argparse arguments
    arg_pattern = r'add_argument\(["\']--(\w+)["\'].*?(?:type=(\w+))?.*?(?:default=([^,\)]+))?.*?(?:help=["\']([^"\']+)["\'])?'
    # Parameters to skip (internal/backend params, not user-facing controls)
    skip_params = {
        "output", "checkpoint", "model", "device", "json_output",  # System params
        "reference", "directions", "blend", "word_emotions",  # Advanced params (not for basic demo)
        "transition", "sweep_intensity", "sweep_emotions", "auto_emotion",  # Mode flags
        "intensities",  # Internal sweep values
    }
    for match in re.finditer(arg_pattern, content, re.DOTALL):
        name, type_str, default, help_text = match.groups()
        if name in skip_params:
            continue  # Skip internal params
        params.append({
            "name": name,
            "type": type_str or "str",
            "default": default,
            "help": help_text or name.replace("_", " ").title()
        })

    # Try to get description from docstring
    docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
    description = docstring_match.group(1).strip().split("\n")[0] if docstring_match else ""

    return {"params": params, "description": description}


def generate_controls(params: List[Dict]) -> str:
    """Generate React control components for each parameter."""
    controls = []

    for param in params:
        name = param["name"]
        ptype = param["type"]
        label = param["help"] or name.replace("_", " ").title()

        if name == "text":
            controls.append(f'''
        {{/* Text Input */}}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-2">
            {label}
          </label>
          <textarea
            value={{text}}
            onChange={{(e) => setText(e.target.value)}}
            className="w-full h-24 px-4 py-3 bg-background-card border border-border rounded-lg text-foreground focus:border-foreground-muted focus:outline-none resize-none"
            placeholder="Enter text to synthesize..."
          />
        </div>''')

        elif name == "emotion":
            controls.append(f'''
        {{/* Emotion Selector */}}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-3">
            {label}
          </label>
          <div className="flex flex-wrap gap-2">
            {{EMOTIONS.map((e) => (
              <button
                key={{e.id}}
                onClick={{() => setEmotion(e.id)}}
                className={{`px-4 py-2 rounded-lg text-sm font-medium transition-all ${{
                  emotion === e.id
                    ? "bg-foreground text-background"
                    : "bg-background-card border border-border text-foreground hover:border-foreground-muted"
                }}`}}
              >
                {{e.label}}
              </button>
            ))}}
          </div>
        </div>''')

        elif ptype == "float" or name in ["intensity", "strength", "scale", "weight"]:
            default = param.get("default", "0.5")
            try:
                default_val = float(default) if default else 0.5
            except:
                default_val = 0.5
            max_val = 2.0 if "intensity" in name or "scale" in name else 1.0

            controls.append(f'''
        {{/* {label} Slider */}}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium text-foreground">
              {label}
            </label>
            <span className="text-sm text-foreground-bright font-mono">
              {{{name}.toFixed(2)}}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="{max_val}"
            step="0.05"
            value={{{name}}}
            onChange={{(e) => set{name.title().replace("_", "")}(parseFloat(e.target.value))}}
            className="w-full h-2 bg-background-card rounded-lg appearance-none cursor-pointer"
          />
        </div>''')

        elif ptype == "int":
            controls.append(f'''
        {{/* {label} Input */}}
        <div className="mb-6">
          <label className="block text-sm font-medium text-foreground mb-2">
            {label}
          </label>
          <input
            type="number"
            value={{{name}}}
            onChange={{(e) => set{name.title().replace("_", "")}(parseInt(e.target.value))}}
            className="w-full px-4 py-2 bg-background-card border border-border rounded-lg text-foreground focus:border-foreground-muted focus:outline-none"
          />
        </div>''')

    return "\n".join(controls)


def generate_state(params: List[Dict]) -> str:
    """Generate React useState declarations."""
    states = ['const [text, setText] = useState("Hello, this is a test.");']

    for param in params:
        name = param["name"]
        if name == "text":
            continue

        ptype = param["type"]
        default = param.get("default", "")

        if name == "emotion":
            states.append(f'const [emotion, setEmotion] = useState("happy");')
        elif ptype == "float" or name in ["intensity", "strength", "scale"]:
            default_val = float(default) if default else 0.5
            setter = f"set{name.title().replace('_', '')}"
            states.append(f"const [{name}, {setter}] = useState({default_val});")
        elif ptype == "int":
            default_val = int(default) if default else 0
            setter = f"set{name.title().replace('_', '')}"
            states.append(f"const [{name}, {setter}] = useState({default_val});")
        else:
            setter = f"set{name.title().replace('_', '')}"
            states.append(f'const [{name}, {setter}] = useState("{default or ""}");')

    return "\n  ".join(states)


def generate_body_params(params: List[Dict]) -> str:
    """Generate the body parameters for fetch."""
    names = ["text"] + [p["name"] for p in params if p["name"] != "text"]
    return ", ".join(names)


def generate_page(technique: str, params: List[Dict], description: str) -> str:
    """Generate the full page component."""
    state = generate_state(params)
    controls = generate_controls(params)
    body_params = generate_body_params(params)

    # Determine icon based on technique
    icon_map = {
        "emoknob": "Sliders",
        "easv": "Sparkles",
        "emo-film": "Layers",
        "draw-speech": "PenTool",
        "chatterbox": "Wand2",
    }
    icon = icon_map.get(technique, "Zap")

    has_emotion = any(p["name"] == "emotion" for p in params)
    emotions_const = '''
const EMOTIONS = [
  { id: "neutral", label: "Neutral" },
  { id: "happy", label: "Happy" },
  { id: "sad", label: "Sad" },
  { id: "angry", label: "Angry" },
  { id: "fearful", label: "Fearful" },
];
''' if has_emotion else ""

    title = technique.replace("-", " ").replace("_", " ").title()

    return f'''"use client";

import React, {{ useState, useRef }} from "react";
import Link from "next/link";
import {{
  ArrowLeft,
  Play,
  Loader2,
  Download,
  Volume2,
  {icon},
}} from "lucide-react";
import {{ Button }} from "@/components/ui/button";
{emotions_const}
const API_BASE = "/api/demos/{technique}";

export default function {technique.replace("-", "").replace("_", "").title()}DemoPage() {{
  {state}
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleGenerate = async () => {{
    if (!text.trim()) return;
    setIsGenerating(true);
    setError(null);
    setAudioUrl(null);

    try {{
      const response = await fetch(API_BASE, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ {body_params} }}),
      }});

      if (!response.ok) {{
        const data = await response.json();
        throw new Error(data.error || "Generation failed");
      }}

      const data = await response.json();
      setAudioUrl(data.audioUrl);
      if (audioRef.current) {{
        audioRef.current.load();
        audioRef.current.play();
      }}
    }} catch (err) {{
      setError(err instanceof Error ? err.message : "Unknown error");
    }} finally {{
      setIsGenerating(false);
    }}
  }};

  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-2xl mx-auto">
        <Link
          href="/demos"
          className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-8"
        >
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back to Demos
        </Link>

        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-foreground/10">
              <{icon} className="w-5 h-5 text-foreground-bright" />
            </div>
            <h1 className="text-2xl font-bold text-foreground-bright">
              {title}
            </h1>
          </div>
          <p className="text-muted-foreground">
            {description}
          </p>
        </div>

{controls}

        <Button
          onClick={{handleGenerate}}
          disabled={{isGenerating || !text.trim()}}
          className="w-full mb-6"
        >
          {{isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Generate Speech
            </>
          )}}
        </Button>

        {{error && (
          <div className="p-4 mb-6 bg-red-500/10 border border-red-500/20 rounded-lg text-red-500 text-sm">
            {{error}}
          </div>
        )}}

        {{audioUrl && (
          <div className="p-6 bg-background-card border border-border rounded-lg">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-foreground">
                <Volume2 className="w-5 h-5 text-foreground-bright" />
                <span className="font-medium">Generated Audio</span>
              </div>
              <a
                href={{audioUrl}}
                download="{technique}_output.wav"
                className="text-sm text-foreground-bright hover:text-foreground flex items-center gap-1"
              >
                <Download className="w-4 h-4" />
                Download
              </a>
            </div>
            <audio ref={{audioRef}} src={{audioUrl}} controls className="w-full" />
          </div>
        )}}

        <div className="mt-8 text-xs text-muted-foreground">
          <p>Inference: <code className="text-foreground-bright">inference/generate_with_{technique.replace("-", "_")}.py</code></p>
        </div>
      </div>
    </div>
  );
}}
'''


def generate_api_route(technique: str, params: List[Dict]) -> str:
    """Generate the API route."""
    param_names = ["text"] + [p["name"] for p in params if p["name"] != "text"]
    destructure = ", ".join(param_names)

    args_list = []
    for name in param_names:
        args_list.append(f'"--{name}", params.{name}.toString()')

    args_str = ",\n      ".join(args_list)

    return f'''import {{ NextRequest, NextResponse }} from "next/server";
import {{ spawn }} from "child_process";
import {{ mkdir }} from "fs/promises";
import {{ join }} from "path";
import {{ randomUUID }} from "crypto";
import {{ existsSync }} from "fs";

const INFERENCE_SCRIPT = "inference/generate_with_{technique.replace("-", "_")}.py";
const OUTPUT_DIR = "public/generated";

interface GenerateRequest {{
  {"; ".join([f"{name}: string | number" for name in param_names])};
}}

export async function POST(request: NextRequest) {{
  try {{
    const params: GenerateRequest = await request.json();

    if (!params.text || typeof params.text !== "string") {{
      return NextResponse.json({{ error: "Text is required" }}, {{ status: 400 }});
    }}

    const outputId = randomUUID();
    const outputFilename = `{technique}_${{outputId}}.wav`;
    const outputPath = join(process.cwd(), OUTPUT_DIR, outputFilename);

    const outputDir = join(process.cwd(), OUTPUT_DIR);
    if (!existsSync(outputDir)) {{
      await mkdir(outputDir, {{ recursive: true }});
    }}

    const scriptPath = join(process.cwd(), "..", INFERENCE_SCRIPT);

    if (!existsSync(scriptPath)) {{
      // Demo mode - return mock response
      return NextResponse.json({{
        audioUrl: "/demo-audio-placeholder.wav",
        message: "Demo mode: inference script not found at " + scriptPath,
      }});
    }}

    const result = await runInference(scriptPath, params, outputPath);

    if (!result.success) {{
      return NextResponse.json({{ error: result.error }}, {{ status: 500 }});
    }}

    return NextResponse.json({{
      audioUrl: `/generated/${{outputFilename}}`,
      prosodyData: result.prosodyData,
    }});

  }} catch (error) {{
    console.error("{technique} API error:", error);
    return NextResponse.json(
      {{ error: error instanceof Error ? error.message : "Unknown error" }},
      {{ status: 500 }}
    );
  }}
}}

async function runInference(
  scriptPath: string,
  params: GenerateRequest,
  outputPath: string
): Promise<{{ success: boolean; error?: string; prosodyData?: object }}> {{
  return new Promise((resolve) => {{
    const args = [
      scriptPath,
      {args_str},
      "--output", outputPath,
    ];

    const proc = spawn("python3", args, {{
      cwd: join(process.cwd(), ".."),
      env: {{ ...process.env, PYTHONUNBUFFERED: "1" }},
    }});

    let stdout = "";
    let stderr = "";

    proc.stdout.on("data", (data) => {{ stdout += data.toString(); }});
    proc.stderr.on("data", (data) => {{ stderr += data.toString(); }});

    proc.on("close", (code) => {{
      if (code !== 0) {{
        resolve({{ success: false, error: stderr || `Exit code ${{code}}` }});
        return;
      }}
      resolve({{ success: true, prosodyData: {{}} }});
    }});

    proc.on("error", (err) => {{
      resolve({{ success: false, error: err.message }});
    }});

    setTimeout(() => {{
      proc.kill();
      resolve({{ success: false, error: "Timeout after 60s" }});
    }}, 60000);
  }});
}}
'''


def update_demos_index(technique: str, description: str, icon: str = "Zap"):
    """Update the demos index page to include the new demo."""
    index_path = FRONTEND_DIR / "app" / "demos" / "page.tsx"

    if not index_path.exists():
        print(f"Warning: Demos index not found: {index_path}")
        return

    content = index_path.read_text()

    # Check if already exists
    if f'id: "{technique}"' in content:
        print(f"Demo {technique} already in index")
        return

    # Find the demos array and add new entry
    title = technique.replace("-", " ").replace("_", " ").title()
    new_entry = f'''  {{
    id: "{technique}",
    title: "{title}",
    description: "{description}",
    icon: {icon},
    status: "ready",
    features: ["Generated by frontend-designer"],
    story: "S2"
  }},'''

    # Insert after the opening bracket of demos array
    pattern = r'(const demos = \[)'
    replacement = f'\\1\n{new_entry}'
    new_content = re.sub(pattern, replacement, content)

    index_path.write_text(new_content)
    print(f"Updated demos index with {technique}")


def main():
    parser = argparse.ArgumentParser(description="Generate frontend demo page")
    parser.add_argument("--technique", required=True, help="Technique name (e.g., emoknob)")
    parser.add_argument("--inference", required=True, help="Path to inference script")
    parser.add_argument("--description", default="", help="One-line description")
    parser.add_argument("--dry-run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    technique = args.technique.lower().replace(" ", "-")
    inference_path = PROJECT_ROOT / args.inference

    print(f"Generating demo page for: {technique}")
    print(f"Inference script: {inference_path}")

    # Parse inference script
    script_info = parse_inference_script(inference_path)
    params = script_info["params"]
    description = args.description or script_info["description"] or f"{technique} demo"

    print(f"Found {len(params)} parameters: {[p['name'] for p in params]}")

    # Generate files
    page_content = generate_page(technique, params, description)
    route_content = generate_api_route(technique, params)

    # Output paths
    page_dir = FRONTEND_DIR / "app" / "demos" / technique
    page_path = page_dir / "page.tsx"
    api_dir = FRONTEND_DIR / "app" / "api" / "demos" / technique
    route_path = api_dir / "route.ts"

    if args.dry_run:
        print("\n=== PAGE ===")
        print(page_content[:500] + "...")
        print("\n=== ROUTE ===")
        print(route_content[:500] + "...")
        return

    # Create directories and write files
    page_dir.mkdir(parents=True, exist_ok=True)
    api_dir.mkdir(parents=True, exist_ok=True)

    page_path.write_text(page_content)
    print(f"Created: {page_path}")

    route_path.write_text(route_content)
    print(f"Created: {route_path}")

    # Update demos index
    update_demos_index(technique, description)

    print(f"\nDone! Test at: http://localhost:3000/demos/{technique}")
    print("Run: cd frontend && npm run dev")


if __name__ == "__main__":
    main()
