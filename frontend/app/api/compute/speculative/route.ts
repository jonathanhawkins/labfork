/**
 * Speculative Decoding API
 *
 * POST /api/compute/speculative
 * - Submit draft for verification
 * - Returns verification result with accepted tokens
 */

import { NextRequest, NextResponse } from "next/server";
import type {
  DraftSequence,
  VerificationResult,
} from "@/lib/compute/speculative-decoding";
import { validateDraft } from "@/lib/compute/speculative-decoding";

/**
 * Request body for draft verification
 */
interface VerifyDraftRequest {
  /** Draft sequence to verify */
  draft: DraftSequence;
  /** Verification model ID */
  verifyModelId: string;
  /** Acceptance threshold (0-1) */
  acceptanceThreshold: number;
  /** Device ID submitting verification request */
  deviceId: string;
  /** Temperature for sampling */
  temperature: number;
}

/**
 * Response for draft verification
 */
interface VerifyDraftResponse {
  /** Verification result */
  result: VerificationResult;
  /** Time taken for verification (ms) */
  verificationTime: number;
  /** Speedup achieved */
  speedupFactor: number;
  /** Credits awarded for verification */
  creditsAwarded: number;
}

/**
 * POST /api/compute/speculative
 * Verify a draft sequence
 */
export async function POST(req: NextRequest) {
  try {
    const body: VerifyDraftRequest = await req.json();

    // Validate request
    if (!body.draft) {
      return NextResponse.json(
        { error: "Missing draft in request body" },
        { status: 400 }
      );
    }

    if (!body.verifyModelId) {
      return NextResponse.json(
        { error: "Missing verifyModelId in request body" },
        { status: 400 }
      );
    }

    if (!body.deviceId) {
      return NextResponse.json(
        { error: "Missing deviceId in request body" },
        { status: 400 }
      );
    }

    // Validate draft structure
    const validation = validateDraft(body.draft);
    if (!validation.valid) {
      return NextResponse.json(
        {
          error: "Invalid draft sequence",
          details: validation.errors,
        },
        { status: 400 }
      );
    }

    // Set defaults
    const acceptanceThreshold = body.acceptanceThreshold ?? 0.8;
    const temperature = body.temperature ?? 0.8;

    // In production, this would:
    // 1. Queue the verification task
    // 2. Assign to a power-tier device with GPU
    // 3. Return task ID for polling
    // 4. Update stats when complete

    // For now, return mock verification result
    const startTime = Date.now();

    // Simulate verification delay (would be real GPU compute)
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Mock verification result
    const tokenResults = body.draft.tokens.map((token, idx) => {
      // Simulate acceptance with ~70% rate
      const accepted = Math.random() > 0.3;
      const verifyLogProb = token.logProb + (Math.random() - 0.5) * 0.2;
      const probRatio = Math.exp(verifyLogProb - token.logProb);

      return {
        position: idx,
        accepted,
        verifyLogProb,
        draftLogProb: token.logProb,
        probRatio,
        replacementToken: accepted
          ? undefined
          : {
              ...token,
              tokenId: token.tokenId + 1,
              text: token.text + "*", // Mock replacement
              logProb: verifyLogProb,
            },
      };
    });

    const acceptedCount = tokenResults.filter((r) => r.accepted).length;
    const acceptanceRate = acceptedCount / tokenResults.length;

    // Calculate speedup: (K + 1) / (1 + R) where R = rejections
    const rejectionCount = tokenResults.length - acceptedCount;
    const speedupFactor = Math.max(
      1.0,
      (tokenResults.length + 1) / (1 + rejectionCount)
    );

    // Build final token sequence
    const finalTokens = [];
    for (const result of tokenResults) {
      if (result.accepted) {
        finalTokens.push(
          body.draft.tokens.find((t) => t.position === result.position)!
        );
      } else if (result.replacementToken) {
        finalTokens.push(result.replacementToken);
      } else {
        break; // Stop at first rejection without replacement
      }
    }

    const finalText = finalTokens.map((t) => t.text).join("");
    const verificationTime = Date.now() - startTime;

    const result: VerificationResult = {
      verificationId: `verify_${Date.now().toString(36)}`,
      draftId: body.draft.draftId,
      tokenResults,
      acceptedCount,
      acceptanceRate,
      finalTokens,
      finalText,
      verifiedAt: new Date().toISOString(),
      deviceId: body.deviceId,
      speedupFactor,
    };

    // Calculate credits (based on tokens verified and speedup achieved)
    const baseCredits = 3; // Base for verification task
    const tokenBonus = acceptedCount * 0.1; // Bonus for accepted tokens
    const speedupBonus = (speedupFactor - 1) * 2; // Bonus for good speedup
    const creditsAwarded = Math.round((baseCredits + tokenBonus + speedupBonus) * 10) / 10;

    const response: VerifyDraftResponse = {
      result,
      verificationTime,
      speedupFactor,
      creditsAwarded,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Speculative decoding API error:", error);
    return NextResponse.json(
      {
        error: "Internal server error",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/compute/speculative
 * Get speculative decoding statistics
 */
export async function GET(req: NextRequest) {
  try {
    // In production, fetch real stats from database
    const stats = {
      totalDrafts: 1247,
      totalVerified: 1189,
      avgAcceptanceRate: 0.73,
      avgSpeedupFactor: 2.8,
      totalTokensGenerated: 9976,
      totalTokensAccepted: 7282,
      timeSaved: 245670, // ms
      activeDevices: {
        drafters: 42, // Crowd tier devices generating drafts
        verifiers: 8, // Power tier devices verifying
      },
    };

    return NextResponse.json(stats);
  } catch (error) {
    console.error("Speculative stats API error:", error);
    return NextResponse.json(
      {
        error: "Internal server error",
        details: error instanceof Error ? error.message : String(error),
      },
      { status: 500 }
    );
  }
}
