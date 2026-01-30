/**
 * Universal Paper Ingestion API
 *
 * POST - Ingest papers from any supported source
 * GET - Search across sources
 */

import { NextRequest, NextResponse } from "next/server";
import {
  ingest,
  ingestBatch,
  detectSourceType,
  searchAcrossSources,
  IngestionInput,
  IngestionOptions,
  SourceType,
} from "@/lib/research/ingestion-pipeline";

export const dynamic = "force-dynamic";

/**
 * POST /api/research/ingest - Ingest paper(s) from any source
 *
 * Body: {
 *   input: string | IngestionInput | IngestionInput[],
 *   options?: IngestionOptions
 * }
 * Returns: { success: boolean, result?: IngestionResult | IngestionResult[], error?: string }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { input, inputs, options = {} } = body;

    // Handle batch ingestion
    if (inputs && Array.isArray(inputs)) {
      const ingestionInputs: IngestionInput[] = inputs.map((item) => {
        if (typeof item === "string") {
          return {
            type: detectSourceType(item),
            value: item,
          };
        }
        return item as IngestionInput;
      });

      const batchResult = await ingestBatch(ingestionInputs, options as IngestionOptions);

      return NextResponse.json({
        success: batchResult.successful.length > 0,
        results: batchResult,
        summary: {
          total: batchResult.successful.length + batchResult.failed.length,
          successful: batchResult.successful.length,
          failed: batchResult.failed.length,
          deduplicated: batchResult.deduplicated.duplicates.length,
        },
      });
    }

    // Handle single ingestion
    if (!input) {
      return NextResponse.json(
        { success: false, error: "Input is required" },
        { status: 400 }
      );
    }

    // Convert string to IngestionInput
    let ingestionInput: IngestionInput;
    if (typeof input === "string") {
      ingestionInput = {
        type: detectSourceType(input),
        value: input,
      };
    } else {
      ingestionInput = input as IngestionInput;
    }

    // Ingest the paper
    const result = await ingest(ingestionInput, options as IngestionOptions);

    return NextResponse.json({
      success: result.success,
      result,
      detection: {
        type: ingestionInput.type,
        value: ingestionInput.value,
      },
    });
  } catch (error) {
    console.error("Error ingesting paper:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to ingest paper",
      },
      { status: 500 }
    );
  }
}

/**
 * GET /api/research/ingest - Search across sources or detect source type
 *
 * Query params:
 *   - query: Search query or input to detect
 *   - detect: If "true", only detect source type without searching
 *   - sources: Comma-separated list of sources to search
 *   - limit: Maximum results per source
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const query = searchParams.get("query");
    const detectOnly = searchParams.get("detect") === "true";
    const sourcesParam = searchParams.get("sources");
    const limitParam = searchParams.get("limit");

    if (!query) {
      return NextResponse.json(
        { success: false, error: "Query parameter is required" },
        { status: 400 }
      );
    }

    // Just detect source type
    if (detectOnly) {
      const sourceType = detectSourceType(query);
      return NextResponse.json({
        success: true,
        detection: {
          type: sourceType,
          value: query,
          isIdentifier: ["arxiv", "doi", "semantic-scholar"].includes(sourceType),
        },
      });
    }

    // Parse sources
    const sources: SourceType[] | undefined = sourcesParam
      ? (sourcesParam.split(",") as SourceType[])
      : undefined;

    // Parse limit
    const limit = limitParam ? parseInt(limitParam, 10) : 5;

    // Search across sources
    const results = await searchAcrossSources(query, {
      sources,
      limit,
    });

    // Count successful results with papers
    const successfulWithPapers = results.filter(
      (r) => r.success && (r.paper || r.papers)
    );

    return NextResponse.json({
      success: true,
      query,
      results,
      totalPapers: successfulWithPapers.length,
    });
  } catch (error) {
    console.error("Error searching papers:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Failed to search papers",
      },
      { status: 500 }
    );
  }
}
