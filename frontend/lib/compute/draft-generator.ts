/**
 * Draft Token Generator
 *
 * Generates draft tokens using small WebLLM models on phones/browsers.
 * Used in speculative decoding to produce fast, low-quality drafts
 * that are verified by larger models on GPUs.
 */

import { getWebLLMEngine, type ModelId } from "./webllm-engine";
import type {
  DraftToken,
  DraftSequence,
  SpeculativeTask,
} from "./speculative-decoding";
import {
  generateDraftId,
  calculateAvgConfidence,
  tokensToText,
} from "./speculative-decoding";

/**
 * Draft generation options
 */
export interface DraftGenerationOptions {
  /** Context/prompt */
  context: string;
  /** Number of draft tokens to generate */
  draftCount: number;
  /** Model ID (should be small/fast model) */
  modelId: ModelId;
  /** Temperature for sampling */
  temperature: number;
  /** Device ID generating the draft */
  deviceId: string;
}

/**
 * Draft generation result
 */
export interface DraftGenerationResult {
  /** Generated draft sequence */
  draft: DraftSequence;
  /** Generation time in ms */
  generationTime: number;
  /** Tokens per second */
  tokensPerSecond: number;
}

/**
 * Draft Token Generator
 * Uses WebLLM with small models to generate draft tokens quickly
 */
export class DraftGenerator {
  private engine = getWebLLMEngine();
  private currentModelId: ModelId | null = null;

  /**
   * Initialize generator with a specific model
   */
  async initialize(modelId: ModelId): Promise<void> {
    if (this.currentModelId === modelId && this.engine.isModelLoaded()) {
      console.log(`Draft model ${modelId} already loaded`);
      return;
    }

    console.log(`Loading draft model: ${modelId}`);
    await this.engine.loadModel(modelId);
    this.currentModelId = modelId;
  }

  /**
   * Check if generator is ready
   */
  isReady(): boolean {
    return this.engine.isModelLoaded();
  }

  /**
   * Get current model
   */
  getCurrentModel(): ModelId | null {
    return this.currentModelId;
  }

  /**
   * Generate draft tokens
   */
  async generateDraft(
    options: DraftGenerationOptions
  ): Promise<DraftGenerationResult> {
    if (!this.isReady()) {
      throw new Error("Draft generator not initialized. Call initialize() first.");
    }

    if (this.currentModelId !== options.modelId) {
      await this.initialize(options.modelId);
    }

    const startTime = Date.now();

    try {
      // Generate tokens using WebLLM
      const tokens = await this.generateTokens(
        options.context,
        options.draftCount,
        options.temperature
      );

      const generationTime = Date.now() - startTime;
      const tokensPerSecond = (options.draftCount / generationTime) * 1000;

      // Build draft sequence
      const draft: DraftSequence = {
        draftId: generateDraftId(),
        tokens,
        context: options.context,
        modelId: options.modelId,
        generatedAt: new Date().toISOString(),
        deviceId: options.deviceId,
        avgConfidence: calculateAvgConfidence(tokens),
      };

      return {
        draft,
        generationTime,
        tokensPerSecond,
      };
    } catch (error) {
      console.error("Draft generation failed:", error);
      throw error;
    }
  }

  /**
   * Generate tokens using WebLLM streaming API
   */
  private async generateTokens(
    context: string,
    count: number,
    temperature: number
  ): Promise<DraftToken[]> {
    const tokens: DraftToken[] = [];
    let position = 0;

    // Use streaming API to get token-by-token output
    const stream = this.engine.streamInference(context, {
      maxTokens: count,
      temperature,
    });

    let accumulatedText = "";

    for await (const chunk of stream) {
      accumulatedText += chunk;

      // Estimate confidence based on chunk length and position
      // Longer chunks and earlier positions typically have higher confidence
      const confidence = this.estimateConfidence(chunk, position, count);

      // Note: WebLLM doesn't expose token IDs or logProbs directly in streaming mode
      // We estimate these values. For production, use models with logprobs support.
      const token: DraftToken = {
        tokenId: this.estimateTokenId(chunk),
        text: chunk,
        logProb: Math.log(confidence), // Approximate log prob from confidence
        confidence,
        position,
      };

      tokens.push(token);
      position++;

      if (tokens.length >= count) {
        break;
      }
    }

    // If we didn't get enough tokens, pad with empty tokens
    // This shouldn't happen in normal operation
    while (tokens.length < count) {
      tokens.push({
        tokenId: 0,
        text: "",
        logProb: -Infinity,
        confidence: 0,
        position: tokens.length,
      });
    }

    return tokens;
  }

  /**
   * Estimate confidence score for a token
   * Higher confidence for:
   * - Longer token chunks (more certain)
   * - Earlier positions (less accumulated error)
   * - Complete words vs fragments
   */
  private estimateConfidence(
    text: string,
    position: number,
    totalCount: number
  ): number {
    // Base confidence starts high and decays with position
    const positionFactor = 1 - (position / totalCount) * 0.3; // Max 30% decay

    // Longer chunks are typically more confident
    const lengthFactor = Math.min(1, text.length / 4); // Normalize around 4 chars

    // Complete words (ending with space) are more confident
    const completeFactor = text.trim().endsWith(" ") ? 1.1 : 1.0;

    const confidence = positionFactor * lengthFactor * completeFactor;

    // Clamp between 0.3 and 0.99
    return Math.max(0.3, Math.min(0.99, confidence));
  }

  /**
   * Estimate token ID from text chunk
   * This is a simple hash function - in production, use proper tokenizer
   */
  private estimateTokenId(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash) % 50000; // Assuming vocab size ~50k
  }

  /**
   * Generate multiple drafts in parallel (if supported by device)
   */
  async generateDrafts(
    options: DraftGenerationOptions[]
  ): Promise<DraftGenerationResult[]> {
    // For now, generate sequentially
    // In production, could use Web Workers for parallel generation
    const results: DraftGenerationResult[] = [];

    for (const option of options) {
      const result = await this.generateDraft(option);
      results.push(result);
    }

    return results;
  }

  /**
   * Unload current model
   */
  async unload(): Promise<void> {
    await this.engine.unload();
    this.currentModelId = null;
  }
}

/**
 * Singleton instance
 */
let draftGeneratorInstance: DraftGenerator | null = null;

/**
 * Get singleton draft generator instance
 */
export function getDraftGenerator(): DraftGenerator {
  if (!draftGeneratorInstance) {
    draftGeneratorInstance = new DraftGenerator();
  }
  return draftGeneratorInstance;
}

/**
 * Reset singleton (for testing)
 */
export function resetDraftGenerator(): void {
  if (draftGeneratorInstance) {
    draftGeneratorInstance.unload();
    draftGeneratorInstance = null;
  }
}
