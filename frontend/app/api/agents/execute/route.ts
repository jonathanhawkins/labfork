/**
 * Agent Task Execution API
 *
 * POST /api/agents/execute - Execute a task with an AI agent
 *
 * This wires meta-agents to Ollama for real task execution.
 */

import { NextRequest, NextResponse } from "next/server";

// Ollama endpoint
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const DEFAULT_MODEL = process.env.OLLAMA_MODEL || "qwen3-coder-32k";

// Agent system prompts
const AGENT_PROMPTS: Record<string, string> = {
  "synergy-detector": `You are the Synergy Detector agent for LabFork. Your role is to analyze research techniques and find beneficial combinations.

When given techniques or papers, you should:
1. Identify potential synergies between approaches
2. Explain WHY combining them could be beneficial
3. Rate the synergy potential (0-1 score)
4. Suggest specific ways to combine the techniques

Be concise but insightful. Focus on actionable synergies.`,

  "pattern-recognizer": `You are the Pattern Recognizer agent for LabFork. Your role is to identify recurring successful patterns across research.

When analyzing techniques or results, you should:
1. Identify common patterns that lead to success
2. Note which architectural choices recur in top-performing approaches
3. Flag anti-patterns to avoid
4. Suggest pattern-based improvements

Focus on extractable, reusable patterns.`,

  "gap-analyzer": `You are the Gap Analyzer agent for LabFork. Your role is to find unexplored areas and missing techniques.

When reviewing a research domain, you should:
1. Identify gaps in the current technique coverage
2. Note missing combinations that haven't been tried
3. Suggest experiments to fill the gaps
4. Prioritize by potential impact

Focus on high-value unexplored areas.`,

  "evolution-engine": `You are the Evolution Engine agent for LabFork. Your role is to evolve and improve techniques using genetic algorithm principles.

When given a technique or set of techniques, you should:
1. Propose mutations (small changes that might improve performance)
2. Suggest crossovers (combining elements from different techniques)
3. Evaluate fitness improvements
4. Track lineage of evolved variants

Focus on concrete, testable modifications.`,

  "transfer-agent": `You are the Transfer Agent for LabFork. Your role is to adapt techniques across different research domains.

When analyzing a technique for transfer, you should:
1. Extract the core abstract principles
2. Map concepts to the target domain
3. Assess transfer feasibility
4. Provide concrete implementation guidance

Focus on practical cross-domain adaptation.`,

  "lab-manager": `You are the Lab Manager agent for LabFork. Your role is to coordinate research activities and manage the lab.

You should:
1. Prioritize research tasks
2. Assign work to appropriate agents
3. Track progress and blockers
4. Synthesize findings across agents

Focus on efficient orchestration and clear communication.`,
};

interface ExecuteRequest {
  agent: string;
  task: string;
  context?: string;
  stream?: boolean;
}

interface AgentResult {
  agent: string;
  task: string;
  response: string;
  duration_ms: number;
  model: string;
  timestamp: string;
}

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    const body = (await request.json()) as ExecuteRequest;
    const { agent, task, context, stream = false } = body;

    if (!agent || !task) {
      return NextResponse.json(
        { error: "agent and task are required" },
        { status: 400 }
      );
    }

    // Get agent system prompt
    const systemPrompt = AGENT_PROMPTS[agent];
    if (!systemPrompt) {
      return NextResponse.json(
        {
          error: `Unknown agent: ${agent}`,
          availableAgents: Object.keys(AGENT_PROMPTS),
        },
        { status: 400 }
      );
    }

    // Build messages
    const messages = [
      { role: "system" as const, content: systemPrompt },
      {
        role: "user" as const,
        content: context ? `Context:\n${context}\n\nTask: ${task}` : task,
      },
    ];

    // Call Ollama
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: DEFAULT_MODEL,
        messages,
        stream,
        options: {
          temperature: 0.7,
          num_ctx: 32768,
        },
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();

      // Check if Ollama is offline
      if (response.status === 0 || !response.ok) {
        return NextResponse.json(
          {
            error: "Ollama is offline",
            hint: "Start Ollama with: ollama serve",
            fallback: true,
            // Return simulated response when Ollama is offline
            result: generateFallbackResponse(agent, task),
          },
          { status: 200 } // Return 200 with fallback
        );
      }

      return NextResponse.json(
        { error: "Ollama request failed", details: errorText },
        { status: 502 }
      );
    }

    // Handle streaming
    if (stream && response.body) {
      const encoder = new TextEncoder();
      const decoder = new TextDecoder();

      const readableStream = new ReadableStream({
        async start(controller) {
          const reader = response.body!.getReader();
          let fullContent = "";

          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const text = decoder.decode(value);
              const lines = text.split("\n").filter(Boolean);

              for (const line of lines) {
                try {
                  const data = JSON.parse(line);
                  if (data.message?.content) {
                    fullContent += data.message.content;
                    // Send chunk
                    controller.enqueue(
                      encoder.encode(
                        `data: ${JSON.stringify({
                          agent,
                          chunk: data.message.content,
                          done: data.done,
                        })}\n\n`
                      )
                    );
                  }

                  if (data.done) {
                    // Send final summary
                    const result: AgentResult = {
                      agent,
                      task,
                      response: fullContent,
                      duration_ms: Date.now() - startTime,
                      model: DEFAULT_MODEL,
                      timestamp: new Date().toISOString(),
                    };
                    controller.enqueue(
                      encoder.encode(
                        `data: ${JSON.stringify({ ...result, done: true })}\n\n`
                      )
                    );
                  }
                } catch {
                  // Skip malformed lines
                }
              }
            }
            controller.close();
          } catch (error) {
            controller.error(error);
          }
        },
      });

      return new Response(readableStream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    // Non-streaming response
    const data = await response.json();
    const result: AgentResult = {
      agent,
      task,
      response: data.message?.content || "",
      duration_ms: Date.now() - startTime,
      model: DEFAULT_MODEL,
      timestamp: new Date().toISOString(),
    };

    return NextResponse.json(result);
  } catch (error) {
    console.error("Agent execution error:", error);

    // Return fallback when Ollama connection fails
    if (error instanceof TypeError) {
      const body = await request.clone().json().catch(() => ({})) as Partial<ExecuteRequest>;
      return NextResponse.json({
        error: "Ollama connection failed",
        fallback: true,
        result: generateFallbackResponse(body.agent || "unknown", body.task || ""),
      });
    }

    return NextResponse.json(
      { error: "Internal server error", details: String(error) },
      { status: 500 }
    );
  }
}

// Generate simulated response when Ollama is offline
function generateFallbackResponse(agent: string, task: string): AgentResult {
  const responses: Record<string, string> = {
    "synergy-detector": `[Simulated] Analyzing synergies for: "${task.slice(0, 50)}..."

Potential synergies detected:
1. Cross-attention mechanisms could enhance feature extraction
2. Multi-scale processing patterns show promise
3. Estimated synergy score: 0.72

Note: This is a simulated response. Start Ollama for real analysis.`,

    "pattern-recognizer": `[Simulated] Recognizing patterns in: "${task.slice(0, 50)}..."

Patterns identified:
1. Encoder-decoder architecture recurring in top approaches
2. Residual connections improve gradient flow
3. Layer normalization consistently beneficial

Note: This is a simulated response. Start Ollama for real analysis.`,

    "gap-analyzer": `[Simulated] Analyzing gaps for: "${task.slice(0, 50)}..."

Gaps identified:
1. Limited exploration of hybrid approaches
2. Few studies on efficiency optimization
3. Cross-domain transfer underexplored

Note: This is a simulated response. Start Ollama for real analysis.`,

    "evolution-engine": `[Simulated] Evolving techniques for: "${task.slice(0, 50)}..."

Evolution proposals:
1. Mutation: Increase attention heads by 25%
2. Crossover: Combine with transformer-based approach
3. Estimated fitness improvement: +12%

Note: This is a simulated response. Start Ollama for real analysis.`,

    "transfer-agent": `[Simulated] Planning transfer for: "${task.slice(0, 50)}..."

Transfer analysis:
1. Core principle: Attention-based feature selection
2. Target domain mapping: Applicable with modifications
3. Transfer feasibility: High (0.85)

Note: This is a simulated response. Start Ollama for real analysis.`,

    "lab-manager": `[Simulated] Managing task: "${task.slice(0, 50)}..."

Task plan:
1. Assign to appropriate specialist agent
2. Monitor progress and collect results
3. Synthesize findings

Note: This is a simulated response. Start Ollama for real analysis.`,
  };

  return {
    agent,
    task,
    response: responses[agent] || `[Simulated] Processing: ${task}\n\nNote: Ollama is offline.`,
    duration_ms: 150, // Simulated fast response
    model: "fallback",
    timestamp: new Date().toISOString(),
  };
}

// GET endpoint to list available agents
export async function GET() {
  return NextResponse.json({
    agents: Object.keys(AGENT_PROMPTS).map((name) => ({
      name,
      description: AGENT_PROMPTS[name].split("\n")[0],
    })),
    ollamaUrl: OLLAMA_URL,
    defaultModel: DEFAULT_MODEL,
  });
}
