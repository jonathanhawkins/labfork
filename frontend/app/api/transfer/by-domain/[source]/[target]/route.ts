/**
 * Transfer By Domain API
 *
 * GET /api/transfer/by-domain/[source]/[target] - Get transfers between specific domains
 */

import { NextRequest, NextResponse } from "next/server";
import { createCrossDomainTransferAgent, STANDARD_DOMAINS } from "@/lib/meta/transfer";
import { getGlobalGraph, isTechniqueNode, TechniqueNode } from "@/lib/meta/knowledge-graph";

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ source: string; target: string }> }
) {
  try {
    const { source, target } = await params;

    const graph = getGlobalGraph();
    const agent = createCrossDomainTransferAgent(graph);

    // Validate domains
    const sourceDomain = agent.getDomain(source);
    const targetDomain = agent.getDomain(target);

    if (!sourceDomain) {
      return NextResponse.json(
        {
          success: false,
          error: `Source domain not found: ${source}`,
          availableDomains: STANDARD_DOMAINS.map((d) => ({
            id: d.id,
            name: d.name,
          })),
        },
        { status: 404 }
      );
    }

    if (!targetDomain) {
      return NextResponse.json(
        {
          success: false,
          error: `Target domain not found: ${target}`,
          availableDomains: STANDARD_DOMAINS.map((d) => ({
            id: d.id,
            name: d.name,
          })),
        },
        { status: 404 }
      );
    }

    // Get domain mapping
    const mapping = agent.findDomainAnalogies(source, target);

    // Find techniques from source domain
    const allNodes = graph.getAllNodes();
    const sourceTechniques = allNodes.filter((node): node is TechniqueNode => {
      if (!isTechniqueNode(node)) return false;
      const tags = node.tags.map((t) => t.toLowerCase());
      return (
        tags.includes(source) ||
        tags.includes(sourceDomain.name.toLowerCase()) ||
        sourceDomain.techniqueIds.includes(node.id)
      );
    });

    // Analyze transfer potential for each technique
    const transferOpportunities: Array<{
      techniqueId: string;
      techniqueName: string;
      feasibilityScore: number;
      feasibilityLevel: string;
      successProbability: number;
      topRisks: string[];
      effortDays: number;
    }> = [];

    // Limit analysis to top techniques to avoid timeout
    const techniquesToAnalyze = sourceTechniques.slice(0, 10);

    for (const technique of techniquesToAnalyze) {
      try {
        const feasibility = agent.assessFeasibility(technique.id, target);
        const prediction = agent.predictSuccess(technique.id, target, feasibility);

        transferOpportunities.push({
          techniqueId: technique.id,
          techniqueName: technique.name,
          feasibilityScore: feasibility.overallScore,
          feasibilityLevel: feasibility.level,
          successProbability: prediction.probability,
          topRisks: feasibility.risks.slice(0, 2).map((r) => r.name),
          effortDays: feasibility.effort.personDays,
        });
      } catch {
        // Skip techniques that can't be analyzed
        continue;
      }
    }

    // Sort by feasibility
    transferOpportunities.sort((a, b) => b.feasibilityScore - a.feasibilityScore);

    return NextResponse.json({
      success: true,
      sourceDomain: {
        id: sourceDomain.id,
        name: sourceDomain.name,
        description: sourceDomain.description,
        dataModality: sourceDomain.characteristics.dataModality,
        taskTypes: sourceDomain.characteristics.taskTypes,
      },
      targetDomain: {
        id: targetDomain.id,
        name: targetDomain.name,
        description: targetDomain.description,
        dataModality: targetDomain.characteristics.dataModality,
        taskTypes: targetDomain.characteristics.taskTypes,
      },
      domainMapping: {
        id: mapping.id,
        mappingStrength: mapping.mappingStrength,
        quality: mapping.quality,
        structuralSimilarity: mapping.structuralSimilarity,
        functionalSimilarity: mapping.functionalSimilarity,
        dataCompatibility: mapping.dataCompatibility,
        conceptMappingCount: mapping.conceptMappings.length,
        topConceptMappings: mapping.conceptMappings.slice(0, 5).map((cm) => ({
          source: cm.sourceConceptName,
          target: cm.targetConceptName,
          type: cm.mappingType,
          similarity: cm.similarity,
        })),
        analogies: mapping.analogies.map((a) => ({
          pattern: `${a.sourcePattern} -> ${a.targetPattern}`,
          strength: a.strength,
        })),
        challenges: mapping.challenges.map((ch) => ({
          type: ch.type,
          severity: ch.severity,
          mitigations: ch.mitigations,
        })),
      },
      transferOpportunities,
      sourceTechniqueCount: sourceTechniques.length,
      analyzedCount: transferOpportunities.length,
      recommendations: generateRecommendations(
        mapping,
        transferOpportunities
      ),
    });
  } catch (error) {
    console.error("Transfer by-domain error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 }
    );
  }
}

function generateRecommendations(
  mapping: { mappingStrength: number; quality: string; challenges: Array<{ type: string; severity: number }> },
  opportunities: Array<{ feasibilityLevel: string; feasibilityScore: number }>
): string[] {
  const recommendations: string[] = [];

  // Based on mapping quality
  if (mapping.quality === "excellent" || mapping.quality === "good") {
    recommendations.push(
      "Domain mapping is strong - direct technique transfer is viable"
    );
  } else if (mapping.quality === "moderate") {
    recommendations.push(
      "Moderate domain alignment - careful adaptation required"
    );
  } else {
    recommendations.push(
      "Weak domain alignment - consider alternative approaches"
    );
  }

  // Based on opportunities
  const highFeasibility = opportunities.filter(
    (o) => o.feasibilityLevel === "trivial" || o.feasibilityLevel === "straightforward"
  );
  if (highFeasibility.length > 0) {
    recommendations.push(
      `${highFeasibility.length} techniques have high transfer potential - prioritize these`
    );
  }

  // Based on challenges
  const severeChallengees = mapping.challenges.filter((c) => c.severity >= 4);
  if (severeChallengees.length > 0) {
    recommendations.push(
      `Address ${severeChallengees.length} severe challenges before transfer`
    );
  }

  // General recommendations
  if (opportunities.length > 0) {
    const avgFeasibility =
      opportunities.reduce((a, o) => a + o.feasibilityScore, 0) /
      opportunities.length;
    if (avgFeasibility > 0.6) {
      recommendations.push("Overall transfer outlook is positive");
    } else if (avgFeasibility > 0.4) {
      recommendations.push("Transfer is feasible with significant effort");
    } else {
      recommendations.push("Consider whether transfer is worth the effort");
    }
  }

  return recommendations;
}
