/**
 * Agent Demo API
 *
 * POST /api/agents/demo - Run a quick demo task with an AI agent
 *
 * This provides a simple way to test Ollama integration from the Lab page.
 */

import { NextRequest, NextResponse } from "next/server";

const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const DEFAULT_MODEL = process.env.OLLAMA_MODEL || "qwen3-coder-32k";

// Demo tasks for each agent type
const DEMO_TASKS: Record<string, { task: string; context: string }> = {
  "synergy-detector": {
    task: "Analyze potential synergies between attention mechanisms and diffusion models for speech synthesis",
    context: "We have implementations of multi-head attention from transformers and denoising diffusion probabilistic models (DDPM). Both are used in modern TTS systems.",
  },
  "pattern-recognizer": {
    task: "Identify common successful patterns in neural codec language models",
    context: "Recent papers show success with discrete audio tokens, multi-scale codebooks, and autoregressive decoding. Look for recurring architectural choices.",
  },
  "gap-analyzer": {
    task: "Find gaps in current voice cloning research that could be explored",
    context: "Current approaches include speaker embedding, reference audio encoding, and prosody transfer. Many focus on English only.",
  },
  "evolution-engine": {
    task: "Propose mutations to improve a basic attention mechanism for prosody modeling",
    context: "Current implementation uses standard multi-head attention with 8 heads, 512 dim. Prosody features include pitch, energy, duration.",
  },
  "transfer-agent": {
    task: "How could image diffusion techniques transfer to audio generation?",
    context: "Stable Diffusion uses U-Net with cross-attention for conditioning. Audio has different dimensionality and temporal structure.",
  },
  "lab-manager": {
    task: "Create a research plan for improving emotion recognition in voice cloning",
    context: "Current system can clone voice but emotion transfer is inconsistent. We have labeled emotional speech data.",
  },
};

// Agent system prompts (same as execute route)
const AGENT_PROMPTS: Record<string, string> = {
  "synergy-detector": `You are the Synergy Detector agent. Analyze research techniques and find beneficial combinations. Be concise - provide 3-5 key synergies with scores.`,
  "pattern-recognizer": `You are the Pattern Recognizer agent. Identify recurring successful patterns. Be concise - list 3-5 patterns with examples.`,
  "gap-analyzer": `You are the Gap Analyzer agent. Find unexplored areas and missing techniques. Be concise - list 3-5 gaps with potential impact.`,
  "evolution-engine": `You are the Evolution Engine agent. Propose mutations and improvements. Be concise - suggest 3-5 concrete modifications.`,
  "transfer-agent": `You are the Transfer Agent. Adapt techniques across domains. Be concise - provide 3-5 transfer opportunities with feasibility scores.`,
  "lab-manager": `You are the Lab Manager agent. Coordinate research and create plans. Be concise - provide a clear 5-step plan.`,
};

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    const body = await request.json();
    const { agent = "synergy-detector" } = body;

    const systemPrompt = AGENT_PROMPTS[agent];
    const demoTask = DEMO_TASKS[agent];

    if (!systemPrompt || !demoTask) {
      return NextResponse.json(
        { error: `Unknown agent: ${agent}`, availableAgents: Object.keys(AGENT_PROMPTS) },
        { status: 400 }
      );
    }

    // First check if Ollama is running
    try {
      const healthCheck = await fetch(`${OLLAMA_URL}/api/version`, {
        signal: AbortSignal.timeout(2000),
      });
      if (!healthCheck.ok) throw new Error("Ollama not responding");
    } catch {
      // Return simulated response if Ollama is offline
      return NextResponse.json({
        agent,
        task: demoTask.task,
        response: generateSimulatedResponse(agent, demoTask.task),
        duration_ms: 200,
        model: "simulated",
        timestamp: new Date().toISOString(),
        simulated: true,
        hint: "Start Ollama for real AI responses: ollama serve",
      });
    }

    // Build messages
    const messages = [
      { role: "system" as const, content: systemPrompt },
      {
        role: "user" as const,
        content: `Context:\n${demoTask.context}\n\nTask: ${demoTask.task}`,
      },
    ];

    // Call Ollama
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_MODEL,
        messages,
        stream: false,
        options: { temperature: 0.7, num_ctx: 8192 }, // Smaller context for demo
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      return NextResponse.json(
        { error: "Ollama request failed", details: errorText },
        { status: 502 }
      );
    }

    const data = await response.json();

    return NextResponse.json({
      agent,
      task: demoTask.task,
      response: data.message?.content || "",
      duration_ms: Date.now() - startTime,
      model: DEFAULT_MODEL,
      timestamp: new Date().toISOString(),
      simulated: false,
    });
  } catch (error) {
    console.error("Agent demo error:", error);
    return NextResponse.json(
      { error: "Internal server error", details: String(error) },
      { status: 500 }
    );
  }
}

// Simulated responses for when Ollama is offline
function generateSimulatedResponse(agent: string, task: string): string {
  const responses: Record<string, string> = {
    "synergy-detector": `## Synergy Analysis

**Task:** ${task}

### Detected Synergies:

1. **Attention + Diffusion Conditioning** (Score: 0.85)
   - Cross-attention can guide diffusion denoising steps
   - Example: Stable Diffusion's text conditioning approach

2. **Multi-scale Processing** (Score: 0.78)
   - Diffusion operates at multiple noise levels
   - Attention can weight features across scales

3. **Iterative Refinement** (Score: 0.72)
   - Both methods benefit from iterative processing
   - Attention weights can be refined during diffusion

*[Simulated response - start Ollama for real analysis]*`,

    "pattern-recognizer": `## Pattern Recognition

**Task:** ${task}

### Identified Patterns:

1. **Residual Quantization** - Used in EnCodec, DAC, SoundStream
2. **Multi-codebook Design** - 4-8 codebooks at different rates
3. **Causal Convolutions** - Maintain temporal ordering
4. **Grouped Convolutions** - Efficiency without quality loss
5. **Skip Connections** - Preserve fine details

*[Simulated response - start Ollama for real analysis]*`,

    "gap-analyzer": `## Gap Analysis

**Task:** ${task}

### Research Gaps:

1. **Multilingual Prosody** (Impact: High)
   - Most systems trained on English only
   - Tonal languages underexplored

2. **Emotional Consistency** (Impact: High)
   - Emotion often inconsistent across sentences
   - Need better long-range modeling

3. **Real-time Adaptation** (Impact: Medium)
   - Few systems can adapt in real-time
   - Streaming architectures needed

*[Simulated response - start Ollama for real analysis]*`,

    "evolution-engine": `## Evolution Proposals

**Task:** ${task}

### Proposed Mutations:

1. **Increase Attention Heads** (12 → 16)
   - Finer-grained prosody capture
   - Estimated improvement: +8%

2. **Add Relative Position Encoding**
   - Better temporal relationships
   - Estimated improvement: +12%

3. **Cross-layer Attention**
   - Connect shallow and deep features
   - Estimated improvement: +6%

*[Simulated response - start Ollama for real analysis]*`,

    "transfer-agent": `## Transfer Analysis

**Task:** ${task}

### Transfer Opportunities:

1. **U-Net Architecture** (Feasibility: 0.82)
   - Adapt 2D conv to 1D for audio
   - Keep skip connections

2. **Cross-Attention Conditioning** (Feasibility: 0.90)
   - Already used in audio diffusion
   - Text/speaker embeddings as condition

3. **Classifier-Free Guidance** (Feasibility: 0.85)
   - Works directly for audio
   - Controls generation diversity

*[Simulated response - start Ollama for real analysis]*`,

    "lab-manager": `## Research Plan

**Task:** ${task}

### 5-Step Plan:

1. **Data Audit** - Catalog emotional speech samples by category
2. **Baseline Metrics** - Measure current emotion accuracy
3. **Architecture Review** - Identify emotion encoding bottlenecks
4. **Implement Improvements** - Add emotion embeddings/conditioning
5. **Evaluation** - A/B test with human evaluators

**Timeline:** 2-3 weeks per step
**Priority:** High

*[Simulated response - start Ollama for real analysis]*`,
  };

  return responses[agent] || `[Simulated] Analyzing: ${task}\n\nStart Ollama for real results.`;
}

// GET endpoint to list demo tasks
export async function GET() {
  // Check Ollama status
  let ollamaOnline = false;
  try {
    const response = await fetch(`${OLLAMA_URL}/api/version`, {
      signal: AbortSignal.timeout(2000),
    });
    ollamaOnline = response.ok;
  } catch {
    ollamaOnline = false;
  }

  return NextResponse.json({
    agents: Object.keys(DEMO_TASKS).map((name) => ({
      name,
      task: DEMO_TASKS[name].task,
    })),
    ollamaStatus: ollamaOnline ? "online" : "offline",
    ollamaUrl: OLLAMA_URL,
    model: DEFAULT_MODEL,
  });
}
