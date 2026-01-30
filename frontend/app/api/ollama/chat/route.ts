/**
 * Ollama Chat API
 *
 * POST /api/ollama/chat - Send messages to local Ollama
 *
 * Supports both streaming and non-streaming responses.
 * Uses Ollama's native API format.
 */

import { NextRequest, NextResponse } from "next/server";

// Ollama server URL - defaults to localhost
const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";

// Default model - can be overridden in request
const DEFAULT_MODEL = process.env.OLLAMA_MODEL || "qwen3-coder-32k";

interface OllamaMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

interface OllamaChatRequest {
  model?: string;
  messages: OllamaMessage[];
  stream?: boolean;
  options?: {
    temperature?: number;
    top_p?: number;
    num_ctx?: number;
  };
}

interface OllamaChatResponse {
  model: string;
  created_at: string;
  message: {
    role: string;
    content: string;
  };
  done: boolean;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  eval_count?: number;
  eval_duration?: number;
}

export async function POST(request: NextRequest) {
  try {
    const body = (await request.json()) as OllamaChatRequest;

    const { model = DEFAULT_MODEL, messages, stream = false, options = {} } = body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return NextResponse.json(
        { error: "messages array is required" },
        { status: 400 }
      );
    }

    // Build Ollama request
    const ollamaRequest = {
      model,
      messages,
      stream,
      options: {
        temperature: options.temperature ?? 0.7,
        top_p: options.top_p ?? 0.9,
        num_ctx: options.num_ctx ?? 32768,
      },
    };

    // Call Ollama
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(ollamaRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Ollama error:", errorText);

      // Check if Ollama is not running
      if (response.status === 0 || errorText.includes("ECONNREFUSED")) {
        return NextResponse.json(
          {
            error: "Ollama is not running",
            hint: "Start Ollama with: ollama serve",
            ollamaUrl: OLLAMA_URL,
          },
          { status: 503 }
        );
      }

      return NextResponse.json(
        { error: "Ollama request failed", details: errorText },
        { status: response.status }
      );
    }

    // Handle streaming response
    if (stream && response.body) {
      const encoder = new TextEncoder();
      const decoder = new TextDecoder();

      const readableStream = new ReadableStream({
        async start(controller) {
          const reader = response.body!.getReader();

          try {
            while (true) {
              const { done, value } = await reader.read();

              if (done) {
                controller.close();
                break;
              }

              // Ollama returns newline-delimited JSON
              const text = decoder.decode(value);
              const lines = text.split("\n").filter(Boolean);

              for (const line of lines) {
                try {
                  const data = JSON.parse(line) as OllamaChatResponse;
                  // Format as SSE
                  controller.enqueue(
                    encoder.encode(`data: ${JSON.stringify(data)}\n\n`)
                  );
                } catch {
                  // Skip malformed lines
                }
              }
            }
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

    // Handle non-streaming response
    const data = (await response.json()) as OllamaChatResponse;

    return NextResponse.json({
      model: data.model,
      message: data.message,
      usage: {
        prompt_tokens: data.prompt_eval_count,
        completion_tokens: data.eval_count,
        total_duration_ms: data.total_duration
          ? Math.round(data.total_duration / 1_000_000)
          : undefined,
      },
    });
  } catch (error) {
    console.error("Ollama chat error:", error);

    // Check for connection errors
    if (error instanceof TypeError && error.message.includes("fetch")) {
      return NextResponse.json(
        {
          error: "Cannot connect to Ollama",
          hint: "Make sure Ollama is running: ollama serve",
          ollamaUrl: OLLAMA_URL,
        },
        { status: 503 }
      );
    }

    return NextResponse.json(
      { error: "Internal server error", details: String(error) },
      { status: 500 }
    );
  }
}

// GET endpoint to check Ollama status
export async function GET() {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/version`);

    if (!response.ok) {
      return NextResponse.json(
        { status: "offline", error: "Ollama not responding" },
        { status: 503 }
      );
    }

    const version = await response.json();

    // Also get list of available models
    const modelsResponse = await fetch(`${OLLAMA_URL}/api/tags`);
    const modelsData = modelsResponse.ok ? await modelsResponse.json() : { models: [] };

    return NextResponse.json({
      status: "online",
      version: version.version,
      url: OLLAMA_URL,
      defaultModel: DEFAULT_MODEL,
      availableModels: modelsData.models?.map((m: { name: string }) => m.name) || [],
    });
  } catch (error) {
    return NextResponse.json(
      {
        status: "offline",
        error: "Cannot connect to Ollama",
        hint: "Start Ollama with: ollama serve",
        url: OLLAMA_URL,
      },
      { status: 503 }
    );
  }
}
