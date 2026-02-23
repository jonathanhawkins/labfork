/**
 * WebLLM Inference Engine
 *
 * Wrapper around WebLLM for browser-based LLM inference.
 * Handles model loading, caching, and inference execution.
 */

import * as webllm from "@mlc-ai/web-llm";
import type { TaskType, TaskResult, ComputeTask } from "./types";

/**
 * Available models for browser inference
 * Small models suitable for draft token generation
 */
export const AVAILABLE_MODELS = {
  // Draft models (for speculative decoding)
  "TinyLlama-1.1B": "TinyLlama-1.1B-Chat-v1.0-q4f16_1-MLC",
  "Phi-2": "Phi2-q4f16_1-MLC",
  "Gemma-2B": "gemma-2-2b-it-q4f16_1-MLC",
  "Qwen2-0.5B": "Qwen2-0.5B-Instruct-q4f16_1-MLC",
  "Qwen2-1.5B": "Qwen2-1.5B-Instruct-q4f16_1-MLC",

  // Standard models (for full inference on capable devices)
  "Llama-3.2-1B": "Llama-3.2-1B-Instruct-q4f16_1-MLC",
  "Llama-3.2-3B": "Llama-3.2-3B-Instruct-q4f16_1-MLC",
  "Phi-3-mini": "Phi-3-mini-4k-instruct-q4f16_1-MLC",
  "Mistral-7B": "Mistral-7B-Instruct-v0.3-q4f16_1-MLC",
} as const;

export type ModelId = keyof typeof AVAILABLE_MODELS;

/**
 * Model info for display
 */
export interface ModelInfo {
  id: ModelId;
  name: string;
  params: string;
  vram: string;
  description: string;
  tier: "draft" | "standard" | "power";
}

export const MODEL_INFO: Record<ModelId, ModelInfo> = {
  "TinyLlama-1.1B": {
    id: "TinyLlama-1.1B",
    name: "TinyLlama 1.1B",
    params: "1.1B",
    vram: "~1GB",
    description: "Fast draft model for speculative decoding",
    tier: "draft",
  },
  "Phi-2": {
    id: "Phi-2",
    name: "Phi-2",
    params: "2.7B",
    vram: "~2GB",
    description: "Microsoft's efficient small model",
    tier: "draft",
  },
  "Gemma-2B": {
    id: "Gemma-2B",
    name: "Gemma 2B",
    params: "2B",
    vram: "~2GB",
    description: "Google's efficient instruction model",
    tier: "draft",
  },
  "Qwen2-0.5B": {
    id: "Qwen2-0.5B",
    name: "Qwen2 0.5B",
    params: "0.5B",
    vram: "~0.5GB",
    description: "Ultra-small model for mobile/low-end devices",
    tier: "draft",
  },
  "Qwen2-1.5B": {
    id: "Qwen2-1.5B",
    name: "Qwen2 1.5B",
    params: "1.5B",
    vram: "~1GB",
    description: "Small but capable instruction model",
    tier: "draft",
  },
  "Llama-3.2-1B": {
    id: "Llama-3.2-1B",
    name: "Llama 3.2 1B",
    params: "1B",
    vram: "~1GB",
    description: "Meta's latest small model",
    tier: "draft",
  },
  "Llama-3.2-3B": {
    id: "Llama-3.2-3B",
    name: "Llama 3.2 3B",
    params: "3B",
    vram: "~3GB",
    description: "Meta's capable edge model",
    tier: "standard",
  },
  "Phi-3-mini": {
    id: "Phi-3-mini",
    name: "Phi-3 Mini",
    params: "3.8B",
    vram: "~3GB",
    description: "Microsoft's powerful small model",
    tier: "standard",
  },
  "Mistral-7B": {
    id: "Mistral-7B",
    name: "Mistral 7B",
    params: "7B",
    vram: "~5GB",
    description: "Full 7B model for power tier devices",
    tier: "power",
  },
};

/**
 * Engine loading progress
 */
export interface LoadProgress {
  stage: "init" | "downloading" | "loading" | "ready";
  progress: number; // 0-100
  message: string;
}

/**
 * Engine events
 */
export interface EngineEvents {
  loadProgress: (progress: LoadProgress) => void;
  modelLoaded: (modelId: ModelId) => void;
  error: (error: Error) => void;
}

/**
 * WebLLM Inference Engine
 * Singleton that manages model loading and inference
 */
export class WebLLMEngine {
  private static instance: WebLLMEngine | null = null;
  // Lazy-loaded Transformers.js embedding pipeline (shared across instances)
  // Lazy-loaded Transformers.js embedding pipeline
  private static embeddingPipeline: ((text: string, opts?: Record<string, unknown>) => Promise<{ data: Float32Array }>) | null = null;

  private engine: webllm.MLCEngineInterface | null = null;
  private currentModelId: ModelId | null = null;
  private isLoading: boolean = false;
  private listeners: Map<keyof EngineEvents, Set<Function>> = new Map();

  private constructor() {
    const eventKeys: (keyof EngineEvents)[] = ["loadProgress", "modelLoaded", "error"];
    eventKeys.forEach((key) => this.listeners.set(key, new Set()));
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): WebLLMEngine {
    if (!WebLLMEngine.instance) {
      WebLLMEngine.instance = new WebLLMEngine();
    }
    return WebLLMEngine.instance;
  }

  /**
   * Reset singleton (for testing)
   */
  public static resetInstance(): void {
    if (WebLLMEngine.instance) {
      WebLLMEngine.instance.unload();
      WebLLMEngine.instance = null;
    }
    // Also release the embedding pipeline to free memory
    WebLLMEngine.embeddingPipeline = null;
  }

  /**
   * Add event listener
   */
  public on<K extends keyof EngineEvents>(event: K, listener: EngineEvents[K]): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.add(listener);
    }
  }

  /**
   * Remove event listener
   */
  public off<K extends keyof EngineEvents>(event: K, listener: EngineEvents[K]): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.delete(listener);
    }
  }

  /**
   * Emit event
   */
  private emit<K extends keyof EngineEvents>(
    event: K,
    ...args: Parameters<EngineEvents[K]>
  ): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.forEach((listener) => {
        try {
          (listener as Function)(...args);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  /**
   * Check if WebGPU is available
   */
  public static async isWebGPUAvailable(): Promise<boolean> {
    if (typeof navigator === "undefined") return false;
    if (!navigator.gpu) return false;

    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  /**
   * Load a model
   */
  public async loadModel(modelId: ModelId): Promise<void> {
    if (this.isLoading) {
      throw new Error("Already loading a model");
    }

    if (this.currentModelId === modelId && this.engine) {
      console.log(`Model ${modelId} already loaded`);
      return;
    }

    this.isLoading = true;
    this.emit("loadProgress", { stage: "init", progress: 0, message: "Initializing..." });

    try {
      // Unload previous model if exists
      if (this.engine) {
        await this.unload();
      }

      const mlcModelId = AVAILABLE_MODELS[modelId];

      // Create progress callback
      const progressCallback = (report: webllm.InitProgressReport) => {
        let stage: LoadProgress["stage"] = "init";
        let progress = 0;

        if (report.text.includes("Download")) {
          stage = "downloading";
          // Parse progress from report.progress if available
          progress = Math.round((report.progress || 0) * 100);
        } else if (report.text.includes("Load")) {
          stage = "loading";
          progress = Math.round((report.progress || 0) * 100);
        }

        this.emit("loadProgress", {
          stage,
          progress,
          message: report.text,
        });
      };

      // Create the engine
      this.engine = await webllm.CreateMLCEngine(mlcModelId, {
        initProgressCallback: progressCallback,
      });

      this.currentModelId = modelId;
      this.emit("loadProgress", { stage: "ready", progress: 100, message: "Model ready" });
      this.emit("modelLoaded", modelId);

      console.log(`Model ${modelId} loaded successfully`);
    } catch (error) {
      console.error("Failed to load model:", error);
      this.emit("error", error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally {
      this.isLoading = false;
    }
  }

  /**
   * Unload current model
   */
  public async unload(): Promise<void> {
    if (this.engine) {
      try {
        await this.engine.unload();
      } catch (error) {
        console.error("Error unloading model:", error);
      }
      this.engine = null;
      this.currentModelId = null;
    }
  }

  /**
   * Check if a model is loaded
   */
  public isModelLoaded(): boolean {
    return this.engine !== null && this.currentModelId !== null;
  }

  /**
   * Get current model ID
   */
  public getCurrentModel(): ModelId | null {
    return this.currentModelId;
  }

  /**
   * Get loading status
   */
  public isModelLoading(): boolean {
    return this.isLoading;
  }

  /**
   * Execute inference for a compute task
   */
  public async executeTask(task: ComputeTask): Promise<TaskResult> {
    if (!this.engine) {
      throw new Error("No model loaded");
    }

    const startTime = Date.now();

    try {
      switch (task.type) {
        case "full_inference":
        case "shard_inference":
          return await this.executeInference(task, startTime);

        case "draft_tokens":
        case "draft_generation":
          return await this.executeDraftTokens(task, startTime);

        case "embedding":
          return await this.executeEmbedding(task, startTime);

        case "validation":
        case "draft_verification":
          return await this.executeValidation(task, startTime);

        default:
          throw new Error(`Unsupported task type: ${task.type}`);
      }
    } catch (error) {
      console.error("Task execution failed:", error);
      throw error;
    }
  }

  /**
   * Execute full inference task
   */
  private async executeInference(task: ComputeTask, startTime: number): Promise<TaskResult> {
    if (!this.engine) throw new Error("No model loaded");

    const prompt = task.input.prompt || "";
    const maxTokens = task.config.maxTokens || 100;
    const temperature = task.config.temperature || 0.7;

    const response = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature,
    });

    const computeTime = Date.now() - startTime;
    const outputText = response.choices[0]?.message?.content || "";
    const tokenCount = response.usage?.completion_tokens || outputText.split(/\s+/).length;

    return {
      text: outputText,
      metrics: {
        computeTime,
        tokensPerSecond: (tokenCount / computeTime) * 1000,
      },
    };
  }

  /**
   * Execute draft token generation (for speculative decoding)
   */
  private async executeDraftTokens(task: ComputeTask, startTime: number): Promise<TaskResult> {
    if (!this.engine) throw new Error("No model loaded");

    const prompt = task.input.prompt || "";
    // Generate draft tokens quickly (fewer tokens, higher speed)
    const draftCount = 8; // Typical draft count for speculative decoding

    const response = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      max_tokens: draftCount,
      temperature: 0.8, // Slightly higher temp for diversity
    });

    const computeTime = Date.now() - startTime;
    const outputText = response.choices[0]?.message?.content || "";

    return {
      text: outputText,
      metrics: {
        computeTime,
        tokensPerSecond: (draftCount / computeTime) * 1000,
      },
    };
  }

  /**
   * Execute embedding generation
   *
   * Attempts to use Transformers.js for real embeddings.
   * Falls back to returning null embedding with computeMode: 'mock'
   * instead of fake random vectors.
   */
  private async executeEmbedding(task: ComputeTask, startTime: number): Promise<TaskResult> {
    const text = task.input.prompt || task.input.text || "";

    // Try Transformers.js embedding model (loaded lazily)
    try {
      if (!WebLLMEngine.embeddingPipeline) {
        // Dynamically import Transformers.js — webpackIgnore prevents SSR bundling
        // Dynamic import to avoid SSR bundling
        const mod = await (Function('return import("@huggingface/transformers")')() as Promise<{ pipeline: Function }>);
        const { pipeline } = mod;
        WebLLMEngine.embeddingPipeline = await pipeline(
          "feature-extraction",
          "Xenova/all-MiniLM-L6-v2"
        );
      }

      const output = await WebLLMEngine.embeddingPipeline(text, {
        pooling: "mean",
        normalize: true,
      });

      const computeTime = Date.now() - startTime;
      // output.data is a Float32Array of 384 dimensions
      const embedding = Array.from(output.data as Float32Array);

      return {
        embedding,
        computeMode: "transformers",
        metrics: {
          computeTime,
          embeddingDimension: embedding.length,
        },
      };
    } catch (embeddingError) {
      console.warn(
        "Transformers.js embedding not available, returning null embedding:",
        embeddingError
      );
    }

    // Fallback: return null embedding with honest mock mode
    // instead of fake random vectors that pollute the index
    const computeTime = Date.now() - startTime;

    return {
      embedding: undefined,
      computeMode: "mock",
      text: "Embedding model not available — mock result",
      metrics: {
        computeTime,
      },
    };
  }

  /**
   * Execute validation task
   */
  private async executeValidation(task: ComputeTask, startTime: number): Promise<TaskResult> {
    // Validation tasks verify computation from other devices
    // For now, just return success
    const computeTime = Date.now() - startTime;

    return {
      metrics: {
        computeTime,
      },
    };
  }

  /**
   * Stream inference (for chat-like applications)
   */
  public async *streamInference(
    prompt: string,
    options: { maxTokens?: number; temperature?: number } = {}
  ): AsyncGenerator<string, void, unknown> {
    if (!this.engine) {
      throw new Error("No model loaded");
    }

    const { maxTokens = 100, temperature = 0.7 } = options;

    const asyncChunkGenerator = await this.engine.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      max_tokens: maxTokens,
      temperature,
      stream: true,
    });

    for await (const chunk of asyncChunkGenerator) {
      const content = chunk.choices[0]?.delta?.content;
      if (content) {
        yield content;
      }
    }
  }

  /**
   * Get recommended model for device tier
   */
  public static getRecommendedModel(
    tierOrTflops: "power" | "standard" | "crowd" | number
  ): ModelId {
    let tier: "power" | "standard" | "crowd";

    if (typeof tierOrTflops === "number") {
      if (tierOrTflops >= 5) tier = "power";
      else if (tierOrTflops >= 1) tier = "standard";
      else tier = "crowd";
    } else {
      tier = tierOrTflops;
    }

    switch (tier) {
      case "power":
        return "Mistral-7B"; // Full 7B model
      case "standard":
        return "Llama-3.2-3B"; // 3B model
      case "crowd":
      default:
        return "Qwen2-0.5B"; // Smallest model for mobile/browsers
    }
  }

  /**
   * Get all available models for a tier
   */
  public static getModelsForTier(tier: "power" | "standard" | "crowd"): ModelId[] {
    return (Object.keys(MODEL_INFO) as ModelId[]).filter(
      (id) => {
        const info = MODEL_INFO[id];
        if (tier === "power") return true; // Power tier can run all
        if (tier === "standard") return info.tier !== "power";
        return info.tier === "draft"; // Crowd only runs draft models
      }
    );
  }
}

/**
 * Export singleton getter
 */
export function getWebLLMEngine(): WebLLMEngine {
  return WebLLMEngine.getInstance();
}
