/**
 * Transfer Analyze API
 *
 * POST /api/transfer/analyze - Analyze transfer potential between technique and domain
 */

import { NextRequest, NextResponse } from "next/server";
import { createCrossDomainTransferAgent, STANDARD_DOMAINS } from "@/lib/meta/transfer";
import { getGlobalGraph } from "@/lib/meta/knowledge-graph";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { techniqueId, targetDomainId, includeGuide = false } = body;

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

    // Check if target domain exists
    const domain = agent.getDomain(targetDomainId);
    if (!domain) {
      return NextResponse.json(
        {
          success: false,
          error: `Domain not found: ${targetDomainId}`,
          availableDomains: STANDARD_DOMAINS.map((d) => ({ id: d.id, name: d.name })),
        },
        { status: 404 }
      );
    }

    // Extract principles
    const principles = agent.extractAbstractPrinciples(techniqueId);

    // Assess feasibility
    const feasibility = agent.assessFeasibility(techniqueId, targetDomainId);

    // Predict success
    const prediction = agent.predictSuccess(techniqueId, targetDomainId, feasibility);

    // Build response
    const response: Record<string, unknown> = {
      success: true,
      techniqueId,
      targetDomain: {
        id: domain.id,
        name: domain.name,
        description: domain.description,
      },
      principles: principles.map((p) => ({
        id: p.id,
        name: p.name,
        level: p.level,
        coreInsight: p.coreInsight,
        confidence: p.confidence,
        componentCount: p.components.length,
      })),
      feasibility: {
        overallScore: feasibility.overallScore,
        level: feasibility.level,
        components: feasibility.components,
        riskCount: feasibility.risks.length,
        topRisks: feasibility.risks.slice(0, 3).map((r) => ({
          name: r.name,
          score: r.score,
          mitigation: r.mitigation,
        })),
        adaptationCount: feasibility.adaptations.length,
        effort: feasibility.effort,
        recommendations: feasibility.recommendations,
      },
      prediction: {
        probability: prediction.probability,
        confidenceInterval: prediction.confidenceInterval,
        successFactors: prediction.successFactors,
        failureRisks: prediction.failureRisks,
        expectedOutcomes: prediction.expectedOutcomes,
      },
    };

    // Include implementation guide if requested
    if (includeGuide) {
      const guide = agent.generateImplementationGuide(
        techniqueId,
        targetDomainId,
        feasibility
      );

      response.implementationGuide = {
        overview: guide.overview,
        prerequisites: guide.prerequisites,
        stepCount: guide.steps.length,
        steps: guide.steps.map((s) => ({
          step: s.step,
          title: s.title,
          estimatedHours: s.estimatedHours,
        })),
        successCriteria: guide.successCriteria,
        pitfallCount: guide.pitfalls.length,
      };
    }

    return NextResponse.json(response);
  } catch (error) {
    console.error("Transfer analyze error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const graph = getGlobalGraph();
    const agent = createCrossDomainTransferAgent(graph);

    const domains = agent.getDomains();

    return NextResponse.json({
      success: true,
      domains: domains.map((d) => ({
        id: d.id,
        name: d.name,
        description: d.description,
        conceptCount: d.concepts.length,
        dataModality: d.characteristics.dataModality,
        taskTypes: d.characteristics.taskTypes,
        relatedDomains: d.relatedDomainIds,
      })),
      count: domains.length,
    });
  } catch (error) {
    console.error("Transfer analyze GET error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}
