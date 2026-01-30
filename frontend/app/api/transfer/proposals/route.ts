/**
 * Transfer Proposals API
 *
 * GET /api/transfer/proposals - Get transfer proposals
 * POST /api/transfer/proposals - Create new transfer proposal
 */

import { NextRequest, NextResponse } from "next/server";
import { createCrossDomainTransferAgent, TransferProposal } from "@/lib/meta/transfer";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";

// In-memory storage for proposals
const proposalStore = new Map<string, TransferProposal>();

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const targetDomain = searchParams.get("targetDomain");
    const sourceDomain = searchParams.get("sourceDomain");
    const status = searchParams.get("status");
    const minFeasibility = parseFloat(searchParams.get("minFeasibility") || "0");
    const limit = parseInt(searchParams.get("limit") || "50", 10);

    const graph = getGlobalGraph();
    const agent = createCrossDomainTransferAgent(graph);

    // Get proposals from agent
    let proposals = agent.getProposals();

    // Also include stored proposals
    for (const proposal of Array.from(proposalStore.values())) {
      if (!proposals.find((p) => p.id === proposal.id)) {
        proposals.push(proposal);
      }
    }

    // Filter by target domain
    if (targetDomain) {
      proposals = proposals.filter((p) => p.targetDomain === targetDomain);
    }

    // Filter by source domain
    if (sourceDomain) {
      proposals = proposals.filter((p) => p.sourceDomain === sourceDomain);
    }

    // Filter by status
    if (status) {
      proposals = proposals.filter((p) => p.status === status);
    }

    // Filter by minimum feasibility
    if (minFeasibility > 0) {
      proposals = proposals.filter(
        (p) => p.feasibility.overallScore >= minFeasibility
      );
    }

    // Sort by feasibility score
    proposals.sort((a, b) => b.feasibility.overallScore - a.feasibility.overallScore);

    // Limit results
    proposals = proposals.slice(0, limit);

    return NextResponse.json({
      success: true,
      proposals: proposals.map((p) => ({
        id: p.id,
        sourceTechnique: {
          id: p.sourceTechniqueId,
          name: p.sourceTechniqueName,
        },
        sourceDomain: p.sourceDomain,
        targetDomain: p.targetDomain,
        principle: {
          name: p.principle.name,
          level: p.principle.level,
          coreInsight: p.principle.coreInsight,
        },
        feasibility: {
          score: p.feasibility.overallScore,
          level: p.feasibility.level,
        },
        successProbability: p.successPrediction.probability,
        status: p.status,
        createdAt: p.createdAt,
        updatedAt: p.updatedAt,
      })),
      count: proposals.length,
    });
  } catch (error) {
    console.error("Transfer proposals GET error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { techniqueId, targetDomainId } = body;

    if (!techniqueId || !targetDomainId) {
      return NextResponse.json(
        {
          success: false,
          error: "techniqueId and targetDomainId are required",
        },
        { status: 400 }
      );
    }

    const graph = getGlobalGraph();
    const agent = createCrossDomainTransferAgent(graph);

    // Create transfer proposal
    const proposal = agent.createTransferProposal(techniqueId, targetDomainId);

    // Store in local cache
    proposalStore.set(proposal.id, proposal);

    return NextResponse.json({
      success: true,
      proposal: {
        id: proposal.id,
        sourceTechnique: {
          id: proposal.sourceTechniqueId,
          name: proposal.sourceTechniqueName,
        },
        sourceDomain: proposal.sourceDomain,
        targetDomain: proposal.targetDomain,
        principle: {
          id: proposal.principle.id,
          name: proposal.principle.name,
          level: proposal.principle.level,
          coreInsight: proposal.principle.coreInsight,
          components: proposal.principle.components,
          applicabilityConditions: proposal.principle.applicabilityConditions,
          counterIndications: proposal.principle.counterIndications,
          confidence: proposal.principle.confidence,
        },
        domainMapping: {
          id: proposal.domainMapping.id,
          mappingStrength: proposal.domainMapping.mappingStrength,
          quality: proposal.domainMapping.quality,
          conceptMappingCount: proposal.domainMapping.conceptMappings.length,
          analogyCount: proposal.domainMapping.analogies.length,
          challengeCount: proposal.domainMapping.challenges.length,
        },
        feasibility: {
          overallScore: proposal.feasibility.overallScore,
          level: proposal.feasibility.level,
          components: proposal.feasibility.components,
          risks: proposal.feasibility.risks,
          enablers: proposal.feasibility.enablers,
          adaptations: proposal.feasibility.adaptations,
          effort: proposal.feasibility.effort,
          recommendations: proposal.feasibility.recommendations,
        },
        implementationGuide: {
          overview: proposal.implementationGuide.overview,
          prerequisiteCount: proposal.implementationGuide.prerequisites.length,
          stepCount: proposal.implementationGuide.steps.length,
          steps: proposal.implementationGuide.steps,
          successCriteria: proposal.implementationGuide.successCriteria,
          pitfalls: proposal.implementationGuide.pitfalls,
        },
        successPrediction: {
          probability: proposal.successPrediction.probability,
          confidenceInterval: proposal.successPrediction.confidenceInterval,
          confidence: proposal.successPrediction.confidence,
          basis: proposal.successPrediction.basis,
          successFactors: proposal.successPrediction.successFactors,
          failureRisks: proposal.successPrediction.failureRisks,
          expectedOutcomes: proposal.successPrediction.expectedOutcomes,
        },
        status: proposal.status,
        createdAt: proposal.createdAt,
      },
    });
  } catch (error) {
    console.error("Transfer proposals POST error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
