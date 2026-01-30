/**
 * Transfer Mappings API
 *
 * GET /api/transfer/mappings - Get concept mappings between domains
 */

import { NextRequest, NextResponse } from "next/server";
import { createCrossDomainTransferAgent, STANDARD_DOMAINS } from "@/lib/meta/transfer";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const sourceDomainId = searchParams.get("source");
    const targetDomainId = searchParams.get("target");

    const graph = getGlobalGraph();
    const agent = createCrossDomainTransferAgent(graph);

    // If specific domains requested
    if (sourceDomainId && targetDomainId) {
      // Validate domains
      const sourceDomain = agent.getDomain(sourceDomainId);
      const targetDomain = agent.getDomain(targetDomainId);

      if (!sourceDomain || !targetDomain) {
        return NextResponse.json(
          {
            success: false,
            error: `Domain not found: ${!sourceDomain ? sourceDomainId : targetDomainId}`,
            availableDomains: STANDARD_DOMAINS.map((d) => ({
              id: d.id,
              name: d.name,
            })),
          },
          { status: 404 }
        );
      }

      // Get or create mapping
      const mapping = agent.findDomainAnalogies(sourceDomainId, targetDomainId);

      return NextResponse.json({
        success: true,
        mapping: {
          id: mapping.id,
          sourceDomain: {
            id: sourceDomainId,
            name: sourceDomain.name,
            description: sourceDomain.description,
            concepts: sourceDomain.concepts.map((c) => ({
              id: c.id,
              name: c.name,
              abstractionLevel: c.abstractionLevel,
            })),
          },
          targetDomain: {
            id: targetDomainId,
            name: targetDomain.name,
            description: targetDomain.description,
            concepts: targetDomain.concepts.map((c) => ({
              id: c.id,
              name: c.name,
              abstractionLevel: c.abstractionLevel,
            })),
          },
          conceptMappings: mapping.conceptMappings.map((cm) => ({
            source: {
              id: cm.sourceConceptId,
              name: cm.sourceConceptName,
            },
            target: {
              id: cm.targetConceptId,
              name: cm.targetConceptName,
            },
            mappingType: cm.mappingType,
            similarity: cm.similarity,
            justification: cm.justification,
            transformation: cm.transformation,
            confidence: cm.confidence,
          })),
          structuralSimilarity: mapping.structuralSimilarity,
          functionalSimilarity: mapping.functionalSimilarity,
          dataCompatibility: mapping.dataCompatibility,
          mappingStrength: mapping.mappingStrength,
          quality: mapping.quality,
          analogies: mapping.analogies.map((a) => ({
            id: a.id,
            sourcePattern: a.sourcePattern,
            targetPattern: a.targetPattern,
            description: a.description,
            strength: a.strength,
            examples: a.examples,
          })),
          challenges: mapping.challenges.map((ch) => ({
            type: ch.type,
            description: ch.description,
            severity: ch.severity,
            mitigations: ch.mitigations,
          })),
          createdAt: mapping.createdAt,
        },
      });
    }

    // Return all available mappings
    const allMappings = agent.getMappings();

    return NextResponse.json({
      success: true,
      mappings: allMappings.map((m) => ({
        id: m.id,
        sourceDomainId: m.sourceDomainId,
        targetDomainId: m.targetDomainId,
        mappingStrength: m.mappingStrength,
        quality: m.quality,
        conceptMappingCount: m.conceptMappings.length,
        analogyCount: m.analogies.length,
        challengeCount: m.challenges.length,
        createdAt: m.createdAt,
      })),
      count: allMappings.length,
      availableDomains: STANDARD_DOMAINS.map((d) => ({
        id: d.id,
        name: d.name,
        relatedDomains: d.relatedDomainIds,
      })),
    });
  } catch (error) {
    console.error("Transfer mappings error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

// Create/compute a mapping between domains
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { sourceDomainId, targetDomainId } = body;

    if (!sourceDomainId || !targetDomainId) {
      return NextResponse.json(
        {
          success: false,
          error: "sourceDomainId and targetDomainId are required",
        },
        { status: 400 }
      );
    }

    const graph = getGlobalGraph();
    const agent = createCrossDomainTransferAgent(graph);

    // Create mapping
    const mapping = agent.findDomainAnalogies(sourceDomainId, targetDomainId);

    return NextResponse.json({
      success: true,
      message: "Domain mapping computed",
      mapping: {
        id: mapping.id,
        sourceDomainId: mapping.sourceDomainId,
        targetDomainId: mapping.targetDomainId,
        mappingStrength: mapping.mappingStrength,
        quality: mapping.quality,
        conceptMappingCount: mapping.conceptMappings.length,
        analogyCount: mapping.analogies.length,
        challengeCount: mapping.challenges.length,
      },
    });
  } catch (error) {
    console.error("Transfer mappings POST error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
