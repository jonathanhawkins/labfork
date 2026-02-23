/**
 * Draft Token Verifier
 *
 * Verifies draft tokens using large models on GPUs.
 * Implements the acceptance/rejection algorithm for speculative decoding.
 */

import { getWebLLMEngine, type ModelId } from "./webllm-engine";
import type {
  DraftSequence,
  DraftToken,
  TokenVerification,
  VerificationResult,
} from "./speculative-decoding";
import {
  generateVerificationId,
  shouldAcceptToken,
  calculateSpeedupFactor,
  mergeFinalSequence,
  tokensToText,
  validateDraft,
} from "./speculative-decoding";

/**
 * Verification options
 */
export interface VerificationOptions {
  /** Draft sequence to verify */
  draft: DraftSequence;
  /** Verification model ID (should be larger/better model) */
  modelId: ModelId;
  /** Acceptance threshold (typically 0.8-1.0) */
  acceptanceThreshold: number;
  /** Device ID performing verification */
  deviceId: string;
  /** Temperature for sampling (should match draft) */
  temperature: number;
}

/**
 * Verification result with metrics
 */
export interface VerificationResultWithMetrics extends VerificationResult {
  /** Verification time in ms */
  verificationTime: number;
  /** Tokens verified per second */
  tokensPerSecond: number;
  /** Time saved vs sequential (estimated) */
  timeSaved: number;
}

/**
 * Batch verification options
 */
export interface BatchVerificationOptions {
  /** Multiple drafts to verify */
  drafts: DraftSequence[];
  /** Verification model ID */
  modelId: ModelId;
  /** Acceptance threshold */
  acceptanceThreshold: number;
  /** Device ID performing verification */
  deviceId: string;
  /** Temperature for sampling */
  temperature: number;
}

/**
 * Draft Token Verifier
 * Uses large models to verify draft tokens in parallel
 */
export class DraftVerifier {
  private engine = getWebLLMEngine();
  private currentModelId: ModelId | null = null;

  /**
   * Initialize verifier with a specific model
   */
  async initialize(modelId: ModelId): Promise<void> {
    if (this.currentModelId === modelId && this.engine.isModelLoaded()) {
      console.log(`Verification model ${modelId} already loaded`);
      return;
    }

    console.log(`Loading verification model: ${modelId}`);
    await this.engine.loadModel(modelId);
    this.currentModelId = modelId;
  }

  /**
   * Check if verifier is ready
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
   * Verify a draft sequence
   */
  async verifyDraft(
    options: VerificationOptions
  ): Promise<VerificationResultWithMetrics> {
    if (!this.isReady()) {
      throw new Error("Draft verifier not initialized. Call initialize() first.");
    }

    // Validate draft
    const validation = validateDraft(options.draft);
    if (!validation.valid) {
      throw new Error(`Invalid draft: ${validation.errors.join(", ")}`);
    }

    if (this.currentModelId !== options.modelId) {
      await this.initialize(options.modelId);
    }

    const startTime = Date.now();

    try {
      // Verify each token in the draft
      const tokenResults = await this.verifyTokens(
        options.draft,
        options.acceptanceThreshold,
        options.temperature
      );

      const verificationTime = Date.now() - startTime;
      const tokensPerSecond = (options.draft.tokens.length / verificationTime) * 1000;

      // Count accepted tokens
      const acceptedCount = tokenResults.filter((r) => r.accepted).length;
      const acceptanceRate = acceptedCount / tokenResults.length;

      // Merge final sequence
      const finalTokens = mergeFinalSequence(tokenResults, options.draft.tokens);
      const finalText = tokensToText(finalTokens);

      // Calculate speedup
      const speedupFactor = calculateSpeedupFactor(
        options.draft.tokens.length,
        acceptedCount
      );

      // Estimate time saved (assuming sequential generation at 50ms/token)
      const sequentialTime = acceptedCount * 50;
      const speculativeTime = verificationTime;
      const timeSaved = Math.max(0, sequentialTime - speculativeTime);

      const result: VerificationResultWithMetrics = {
        verificationId: generateVerificationId(),
        draftId: options.draft.draftId,
        tokenResults,
        acceptedCount,
        acceptanceRate,
        finalTokens,
        finalText,
        verifiedAt: new Date().toISOString(),
        deviceId: options.deviceId,
        speedupFactor,
        verificationTime,
        tokensPerSecond,
        timeSaved,
      };

      return result;
    } catch (error) {
      console.error("Draft verification failed:", error);
      throw error;
    }
  }

  /**
   * Verify tokens using the large model
   *
   * Algorithm:
   * 1. For each draft token, compute P_verify(token | context)
   * 2. Accept if P_verify / P_draft >= threshold
   * 3. On first rejection, sample replacement from verify model
   * 4. Discard all tokens after rejection
   */
  private async verifyTokens(
    draft: DraftSequence,
    threshold: number,
    temperature: number
  ): Promise<TokenVerification[]> {
    const results: TokenVerification[] = [];
    let context = draft.context;
    let shouldContinue = true;

    for (let i = 0; i < draft.tokens.length; i++) {
      if (!shouldContinue) {
        // After first rejection, don't verify remaining tokens
        break;
      }

      const draftToken = draft.tokens[i];

      // Generate verification token at this position
      const verifyResult = await this.generateVerificationToken(
        context,
        temperature
      );

      // Check acceptance
      const accepted = shouldAcceptToken(
        verifyResult.logProb,
        draftToken.logProb,
        threshold
      );

      const probRatio = Math.exp(verifyResult.logProb - draftToken.logProb);

      const verification: TokenVerification = {
        position: i,
        accepted,
        verifyLogProb: verifyResult.logProb,
        draftLogProb: draftToken.logProb,
        probRatio,
      };

      if (accepted) {
        // Accept draft token, continue with next
        context += draftToken.text;
      } else {
        // Reject: use verify model's token instead
        verification.replacementToken = verifyResult.token;
        context += verifyResult.token.text;
        shouldContinue = false; // Stop verification after first rejection
      }

      results.push(verification);
    }

    return results;
  }

  /**
   * Generate a single verification token
   */
  private async generateVerificationToken(
    context: string,
    temperature: number
  ): Promise<{ token: DraftToken; logProb: number }> {
    // Generate single token using verify model
    let generatedText = "";
    const stream = this.engine.streamInference(context, {
      maxTokens: 1,
      temperature,
    });

    for await (const chunk of stream) {
      generatedText += chunk;
      break; // Only take first chunk
    }

    // Estimate log prob (in production, use model with logprobs support)
    const logProb = this.estimateLogProb(generatedText);

    const token: DraftToken = {
      tokenId: this.estimateTokenId(generatedText),
      text: generatedText,
      logProb,
      confidence: Math.exp(logProb),
      position: 0, // Will be set by caller
    };

    return { token, logProb };
  }

  /**
   * Estimate log probability for a token
   * Higher probability for common patterns
   */
  private estimateLogProb(text: string): number {
    // Simple heuristic: common tokens have higher probability
    const length = text.trim().length;

    if (length === 0) return -10; // Very low prob for empty

    // Common single characters/words
    if (length === 1) return -0.5;
    if (length <= 4) return -1.0;
    if (length <= 8) return -1.5;

    return -2.0; // Longer chunks less likely
  }

  /**
   * Estimate token ID from text
   */
  private estimateTokenId(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash) % 50000;
  }

  /**
   * Verify multiple drafts in batch
   */
  async verifyBatch(
    options: BatchVerificationOptions
  ): Promise<VerificationResultWithMetrics[]> {
    if (!this.isReady()) {
      throw new Error("Draft verifier not initialized. Call initialize() first.");
    }

    if (this.currentModelId !== options.modelId) {
      await this.initialize(options.modelId);
    }

    const results: VerificationResultWithMetrics[] = [];

    // For now, verify sequentially
    // In production, could implement true parallel verification
    for (const draft of options.drafts) {
      const result = await this.verifyDraft({
        draft,
        modelId: options.modelId,
        acceptanceThreshold: options.acceptanceThreshold,
        deviceId: options.deviceId,
        temperature: options.temperature,
      });

      results.push(result);
    }

    return results;
  }

  /**
   * Get verification statistics for a batch
   */
  getBatchStats(results: VerificationResultWithMetrics[]): {
    avgAcceptanceRate: number;
    avgSpeedupFactor: number;
    totalTimeSaved: number;
    totalTokensVerified: number;
    totalTokensAccepted: number;
  } {
    if (results.length === 0) {
      return {
        avgAcceptanceRate: 0,
        avgSpeedupFactor: 0,
        totalTimeSaved: 0,
        totalTokensVerified: 0,
        totalTokensAccepted: 0,
      };
    }

    const totalAcceptanceRate = results.reduce((sum, r) => sum + r.acceptanceRate, 0);
    const totalSpeedupFactor = results.reduce((sum, r) => sum + r.speedupFactor, 0);
    const totalTimeSaved = results.reduce((sum, r) => sum + r.timeSaved, 0);
    const totalTokensVerified = results.reduce(
      (sum, r) => sum + r.tokenResults.length,
      0
    );
    const totalTokensAccepted = results.reduce((sum, r) => sum + r.acceptedCount, 0);

    return {
      avgAcceptanceRate: totalAcceptanceRate / results.length,
      avgSpeedupFactor: totalSpeedupFactor / results.length,
      totalTimeSaved,
      totalTokensVerified,
      totalTokensAccepted,
    };
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
let draftVerifierInstance: DraftVerifier | null = null;

/**
 * Get singleton draft verifier instance
 */
export function getDraftVerifier(): DraftVerifier {
  if (!draftVerifierInstance) {
    draftVerifierInstance = new DraftVerifier();
  }
  return draftVerifierInstance;
}

/**
 * Reset singleton (for testing)
 */
export function resetDraftVerifier(): void {
  if (draftVerifierInstance) {
    draftVerifierInstance.unload();
    draftVerifierInstance = null;
  }
}
