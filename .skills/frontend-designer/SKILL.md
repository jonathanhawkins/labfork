---
name: frontend-designer
description: Creates demo pages for research techniques with full frontend integration
metadata:
  tags: frontend, react, nextjs, demo, integration
---

# Frontend Designer Skill

Creates working demo pages for research techniques. Takes an inference script and generates a complete frontend integration.

## When to Use

Invoke this skill when:
- A research technique has a working inference script
- You need to create a demo page for testing/showing a technique
- You want to integrate research into the product

## What It Creates

For technique `<name>`:
```
frontend/app/demos/<name>/page.tsx     # React page component
frontend/app/api/demos/<name>/route.ts # API endpoint
frontend/app/demos/page.tsx            # Updates index (adds new demo)
frontend/components/Navigation.tsx      # Adds nav link (if needed)
```

## Invocation

From the orchestrator or any agent:

```bash
# Method 1: Use the generator script
python3 .skills/frontend-designer/generate.py \
  --technique emoknob \
  --inference inference/generate_with_emoknob.py \
  --description "Direction vector emotion control with intensity"

# Method 2: Use Task tool to spawn frontend-designer agent
# (This agent has full context about the design system)
```

## Required Context

The skill needs to know:
1. **Technique name** - e.g., "emoknob", "emo-film"
2. **Inference script path** - To understand the parameters
3. **Description** - One-line description for the demo index

## Design System Context (Built-in)

The skill knows:
- Color tokens: `--background`, `--foreground`, `--accent`, etc.
- Components: Button, Slider, Input, Select, Card, Dialog
- Patterns: From existing pages like `/studio`, `/demos/easv`
- Typography: Monospace, muted text hierarchy

## Page Structure

Every demo page follows this structure:

```tsx
"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, Play, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function DemoPage() {
  // 1. State for inputs
  const [text, setText] = useState("...");
  const [param1, setParam1] = useState(0.5);

  // 2. State for generation
  const [isGenerating, setIsGenerating] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  // 3. Generate handler
  const handleGenerate = async () => {
    setIsGenerating(true);
    const res = await fetch("/api/demos/<technique>", {
      method: "POST",
      body: JSON.stringify({ text, param1 })
    });
    const data = await res.json();
    setAudioUrl(data.audioUrl);
    setIsGenerating(false);
  };

  return (
    <div className="min-h-screen bg-background p-8">
      {/* Back link */}
      {/* Header with icon */}
      {/* Input controls */}
      {/* Generate button */}
      {/* Audio player */}
      {/* Info box */}
    </div>
  );
}
```

## API Route Structure

```ts
import { NextRequest, NextResponse } from "next/server";
import { spawn } from "child_process";

export async function POST(request: NextRequest) {
  const { text, ...params } = await request.json();

  // Call Python inference script
  const result = await runInference(params);

  return NextResponse.json({
    audioUrl: result.audioUrl,
    prosodyData: result.prosody
  });
}
```

## Parameter Mapping

The skill analyzes the inference script to find:
- `--text` → Text input
- `--emotion` → Emotion selector (dropdown)
- `--intensity` → Slider (0-1 or 0-2)
- `--output` → Handled automatically
- Custom params → Appropriate control

## Verification

After generating, the skill:
1. Runs `npm run build` to check for errors
2. Starts dev server
3. Takes screenshot of the page
4. Reports success/failure

## Example Usage

```python
# In orchestrator after research task completes
if task_has_inference_script(task):
    run_skill("frontend-designer", {
        "technique": task.technique_name,
        "inference": f"inference/generate_with_{task.technique_name}.py",
        "description": task.one_line_description
    })
```

## Files in This Skill

```
.skills/frontend-designer/
├── SKILL.md           # This file
├── generate.py        # Main generator script
├── templates/
│   ├── page.tsx.j2    # Jinja2 template for page
│   └── route.ts.j2    # Jinja2 template for API
└── context/
    ├── design-system.md    # Color/component reference
    └── examples/           # Example pages for reference
```
